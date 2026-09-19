import os
import random
from pathlib import Path
import numpy as np
import scipy.io as sio
import scipy.sparse as sp
from scipy.sparse.linalg import spsolve
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from sklearn.metrics import roc_auc_score, precision_score, recall_score, f1_score
from scipy.ndimage import convolve, binary_dilation
from tqdm.auto import tqdm
from functools import partial

# Disable tqdm progress bars to keep the nohup.out log file clean
tqdm = partial(tqdm, disable=True)

# Force matplotlib to run headlessly without a display
import matplotlib
matplotlib.use('Agg') 
import matplotlib.pyplot as plt

# ==========================================
# 1. SETUP & DATASET MINING
# ==========================================
DATA_DIR = Path.cwd().parent / "data" / "hyperspectral_oil_spill"
device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Executing on device: {device}")

train_files = [f for f in DATA_DIR.glob("*.mat") if f.stem not in ["GM01", "GM02"]]
test_files = [f for f in DATA_DIR.glob("*.mat") if f.stem in ["GM01", "GM02"]]

def evaluate_band_retention(train_files):
    water_vapor_bands = set(list(range(104, 114)) + list(range(148, 168)) + list(range(221, 224)))
    sample_img = sio.loadmat(train_files[0])["img"]
    total_raw_bands = sample_img.shape[2]
    
    valid_bands = [b for b in range(total_raw_bands) if b not in water_vapor_bands]
    mask = np.array([[1, -2,  1], [-2, 4, -2], [1, -2,  1]], dtype=float)
    all_scores = []
    
    for file_path in train_files:
        img = sio.loadmat(file_path)["img"]
        H, W, _ = img.shape
        scores = []
        for b in valid_bands:
            total_noise = 0.0
            for r in range(1, H - 1, 64):
                end = min(r + 64, H - 1)
                stripe = img[r-1:end+1, :, b].astype(float)
                total_noise += np.abs(convolve(stripe, mask)[1:-1, 1:-1]).sum()
            sigma_n = total_noise * np.sqrt(np.pi / 2) / (6 * (H - 2) * (W - 2))
            scores.append(sigma_n)
        all_scores.append(scores)
        
    avg_sigma = np.mean(all_scores, axis=0)
    clean_indices = np.argsort(avg_sigma)[:144]
    clean_bands = np.sort([valid_bands[i] for i in clean_indices]).tolist()
    return clean_bands

print("Evaluating clean bands...")
CLEAN_BANDS = evaluate_band_retention(train_files)
print(f"Retained strictly {len(CLEAN_BANDS)} bands.")

class HSIPatchDataset(Dataset):
    def __init__(self, file_list, clean_bands, patch_size=11, augment=False):
        self.patch_size = patch_size
        self.augment = augment
        self.arrays = []
        self.items = []
        pad = patch_size // 2
        
        for idx, file_path in enumerate(file_list):
            mat = sio.loadmat(file_path)
            img = mat["img"][:, :, clean_bands].astype(np.float32)
            
            # img = (img - np.min(img)) / (np.max(img) - np.min(img) + 1e-8) # This is a mistake, dont use min max

            for b in range(img.shape[2]):
                band = img[:, :, b]
                p_low, p_high = np.percentile(band, (1, 99))
                img[:, :, b] = np.clip((band - p_low) / (p_high - p_low + 1e-8), 0, 1)
                
            img_padded = np.pad(img, ((pad, pad), (pad, pad), (0, 0)), mode='symmetric')
            self.arrays.append(img_padded)
            
            gt = mat["map"].astype(np.int64)
            oil_mask = (gt == 1)
            water_mask = (gt == 0)
            
            oil_coords = np.argwhere(oil_mask)
            n_oil = len(oil_coords)
            if n_oil == 0: continue
            
            dilated_oil = binary_dilation(oil_mask, iterations=3)
            hard_water_mask = dilated_oil & water_mask
            hard_water_coords = np.argwhere(hard_water_mask)
            easy_water_mask = water_mask & ~hard_water_mask
            easy_water_coords = np.argwhere(easy_water_mask)
            
            np.random.shuffle(hard_water_coords)
            np.random.shuffle(easy_water_coords)
            
            # FIX: Force the model to see open water by splitting the negative class
            half_oil = n_oil // 2
            water_sampled = np.vstack((
                hard_water_coords[:half_oil], 
                easy_water_coords[:(n_oil - half_oil)]
            ))
            
            for r, c in oil_coords: self.items.append((idx, r, c, 1.0))
            for r, c in water_sampled: self.items.append((idx, r, c, 0.0))

    def __len__(self): return len(self.items)

    def __getitem__(self, idx):
        file_idx, r, c, label = self.items[idx]
        patch = self.arrays[file_idx][r:r+self.patch_size, c:c+self.patch_size, :]
        if self.augment:
            k = random.choice([0, 1, 2, 3])
            if k > 0: patch = np.rot90(patch, k=k, axes=(0, 1))
            if random.random() > 0.5: patch = np.fliplr(patch)
            if random.random() > 0.5: patch = np.flipud(patch)
        patch_tensor = torch.from_numpy(patch.copy().transpose(2, 0, 1))
        return patch_tensor, torch.tensor(label, dtype=torch.float32)

print("Mining dataset patches...")
train_dataset = HSIPatchDataset(train_files, CLEAN_BANDS, augment=True)
val_dataset = HSIPatchDataset(test_files, CLEAN_BANDS, augment=False)
train_loader = DataLoader(train_dataset, batch_size=256, shuffle=True, num_workers=0)
val_loader = DataLoader(val_dataset, batch_size=256, shuffle=False, num_workers=0)

# ==========================================
# 2. MODEL DEFINITIONS
# ==========================================
class CoTNetLayer(nn.Module):
    def __init__(self, dim, kernel_size=3):
        super().__init__()
        self.kernel_size = kernel_size
        self.key_embed = nn.Sequential(
            nn.Conv2d(dim, dim, kernel_size=kernel_size, padding=1, bias=False),
            nn.BatchNorm2d(dim), nn.ReLU(inplace=True)
        )
        self.value_embed = nn.Sequential(nn.Conv2d(dim, dim, kernel_size=1, bias=False), nn.BatchNorm2d(dim))
        factor = 4
        self.attention_embed = nn.Sequential(
            nn.Conv2d(2 * dim, 2 * dim // factor, 1, bias=False),
            nn.BatchNorm2d(2 * dim // factor), nn.ReLU(inplace=True),
            nn.Conv2d(2 * dim // factor, kernel_size * kernel_size * dim, 1)
        )
    def forward(self, x):
        bs, c, h, w = x.shape
        k1 = self.key_embed(x) 
        v = self.value_embed(x).view(bs, c, -1) 
        y = torch.cat([k1, x], dim=1) 
        att = self.attention_embed(y).reshape(bs, c, self.kernel_size * self.kernel_size, h, w)
        att = att.mean(2, keepdim=False).view(bs, c, -1) 
        k2 = F.softmax(att, dim=-1) * v 
        k2 = k2.view(bs, c, h, w)
        return k1 + k2 

class True_SSTNet(nn.Module):
    def __init__(self, in_bands, embed_dim=128, num_classes=1):
        super().__init__()
        self.spectral_conv = nn.Sequential(
            nn.Conv3d(1, embed_dim, kernel_size=(7, 1, 1), padding=(3, 0, 0), bias=False),
            nn.BatchNorm3d(embed_dim), nn.ReLU(inplace=True),
            nn.Conv3d(embed_dim, embed_dim, kernel_size=(in_bands, 1, 1), bias=False),
            nn.BatchNorm3d(embed_dim), nn.ReLU(inplace=True)
        )
        self.cot1 = CoTNetLayer(dim=embed_dim)
        self.cot2 = CoTNetLayer(dim=embed_dim)
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.classifier = nn.Sequential(
            nn.Linear(embed_dim, 64), nn.ReLU(inplace=True),
            nn.Dropout(0.4), nn.Linear(64, num_classes)
        )
    def forward(self, x):
        x = x.unsqueeze(1) 
        x = self.spectral_conv(x).squeeze(2) 
        x = self.cot1(x)
        x = self.cot2(x)
        return self.classifier(self.pool(x).flatten(1))

class MoCoDataset(Dataset):
    def __init__(self, original_dataset):
        self.dataset = original_dataset
        self.dataset.augment = True
    def __len__(self): return len(self.dataset)
    def __getitem__(self, idx):
        q_patch, _ = self.dataset[idx]
        k_patch, _ = self.dataset[idx]
        return q_patch, k_patch

class MoCo(nn.Module):
    def __init__(self, base_encoder, in_bands, dim=64, K=8192, m=0.999, T=0.07):
        super(MoCo, self).__init__()
        self.K, self.m, self.T = K, m, T
        self.encoder_q = base_encoder(in_bands=in_bands, embed_dim=128, num_classes=dim)
        self.encoder_k = base_encoder(in_bands=in_bands, embed_dim=128, num_classes=dim)
        for param_q, param_k in zip(self.encoder_q.parameters(), self.encoder_k.parameters()):
            param_k.data.copy_(param_q.data)
            param_k.requires_grad = False
        self.register_buffer("queue", torch.randn(dim, K))
        self.queue = nn.functional.normalize(self.queue, dim=0)
        self.register_buffer("queue_ptr", torch.zeros(1, dtype=torch.long))

    @torch.no_grad()
    def _momentum_update_key_encoder(self):
        for param_q, param_k in zip(self.encoder_q.parameters(), self.encoder_k.parameters()):
            param_k.data = param_k.data * self.m + param_q.data * (1. - self.m)

    @torch.no_grad()
    def _dequeue_and_enqueue(self, keys):
        batch_size = keys.shape[0]
        ptr = int(self.queue_ptr)
        self.queue[:, ptr:ptr + batch_size] = keys.T
        self.queue_ptr[0] = (ptr + batch_size) % self.K

    def forward(self, im_q, im_k):
        q = nn.functional.normalize(self.encoder_q(im_q), dim=1)
        with torch.no_grad():
            self._momentum_update_key_encoder()
            k = nn.functional.normalize(self.encoder_k(im_k), dim=1)
        l_pos = torch.einsum('nc,nc->n', [q, k]).unsqueeze(-1)
        l_neg = torch.einsum('nc,ck->nk', [q, self.queue.clone().detach()])
        logits = torch.cat([l_pos, l_neg], dim=1) / self.T
        labels = torch.zeros(logits.shape[0], dtype=torch.long).to(q.device)
        self._dequeue_and_enqueue(k)
        return logits, labels

# ==========================================
# 3. STAGE 1: MOCO PRETRAINING (WITH IF/ELSE)
# ==========================================
PRETRAIN_EPOCHS = 40
PRETRAINED_WEIGHTS_PATH = "moco_pretrained_sstnet.pth"

moco_model = MoCo(True_SSTNet, in_bands=len(CLEAN_BANDS)).to(device)

if os.path.exists(PRETRAINED_WEIGHTS_PATH):
    print(f"\n--- Found '{PRETRAINED_WEIGHTS_PATH}' ---")
    print("Skipping Stage 1 Pretraining. Loading existing weights...")
    moco_model.load_state_dict(torch.load(PRETRAINED_WEIGHTS_PATH, map_location=device))
else:
    print(f"\n--- Starting Stage 1: MoCo Pretraining ({PRETRAIN_EPOCHS} Epochs) ---")
    moco_train_loader = DataLoader(MoCoDataset(train_dataset), batch_size=256, shuffle=True, num_workers=0, drop_last=True)
    criterion_moco = nn.CrossEntropyLoss()
    optimizer_moco = torch.optim.AdamW(moco_model.parameters(), lr=3e-4, weight_decay=1e-4)

    for epoch in range(PRETRAIN_EPOCHS):
        moco_model.train()
        total_loss = 0.0
        
        for im_q, im_k in moco_train_loader:
            im_q, im_k = im_q.to(device), im_k.to(device)
            optimizer_moco.zero_grad()
            logits, labels = moco_model(im_q, im_k)
            loss = criterion_moco(logits, labels)
            loss.backward()
            optimizer_moco.step()
            total_loss += loss.item()
            
        avg_loss = total_loss / len(moco_train_loader)
        print(f"MoCo Epoch {epoch+1}/{PRETRAIN_EPOCHS} | Avg Contrastive Loss: {avg_loss:.4f}")

    torch.save(moco_model.state_dict(), PRETRAINED_WEIGHTS_PATH)
    print(f"Stage 1 complete. Saved weights to {PRETRAINED_WEIGHTS_PATH}")

# ==========================================
# 4. STAGE 2: BCE FINE-TUNING
# ==========================================
model = moco_model.encoder_q
model.classifier[3] = nn.Linear(64, 1).to(device)

# FIX 1: Swapped to standard BCE for stable 50/50 balanced training
criterion = nn.BCEWithLogitsLoss() 
optimizer = torch.optim.AdamW(model.parameters(), lr=3e-4, weight_decay=1e-3)
FINETUNE_EPOCHS = 30 

print(f"\n--- Starting Stage 2: Supervised Fine-Tuning ({FINETUNE_EPOCHS} Epochs) ---")
for epoch in range(FINETUNE_EPOCHS):
    model.train()
    total_train_loss = 0.0
    for X, y in train_loader:
        X, y = X.to(device), y.to(device)
        optimizer.zero_grad()
        logits = model(X).squeeze(1)
        loss = criterion(logits, y)
        loss.backward()
        optimizer.step()
        total_train_loss += loss.item()
        
    model.eval()
    total_val_loss = 0.0
    all_preds, all_targets = [], []
    with torch.inference_mode():
        for X, y in val_loader:
            X, y = X.to(device), y.to(device)
            logits = model(X).squeeze(1)
            
            # Calculate validation loss
            val_loss = criterion(logits, y)
            total_val_loss += val_loss.item()
            
            probs = torch.sigmoid(logits).cpu().numpy()
            all_preds.extend(probs)
            all_targets.extend(y.cpu().numpy())
            
    preds_binary = (np.array(all_preds) > 0.5).astype(int)
    auc = roc_auc_score(all_targets, all_preds)
    f1 = f1_score(all_targets, preds_binary, zero_division=0)
    precision = precision_score(all_targets, preds_binary, zero_division=0)
    recall = recall_score(all_targets, preds_binary, zero_division=0)
    
    avg_train_loss = total_train_loss / len(train_loader)
    avg_val_loss = total_val_loss / len(val_loader)
    
    print(f"Epoch {epoch+1}/{FINETUNE_EPOCHS} | Train Loss: {avg_train_loss:.4f} | Val Loss: {avg_val_loss:.4f} | AUC: {auc:.4f} | F1: {f1:.4f} | Precision: {precision:.4f} | Recall: {recall:.4f}")

torch.save(model.state_dict(), "finetuned_sstnet.pth")
print("Stage 2 complete. Saved final weights to finetuned_sstnet.pth")

# ==========================================
# 5. POST-PROCESSING & SAVING IMAGES
# ==========================================
def apply_erw_optimization(prob_map, img_rgb, beta=710, gamma=1e-5):
    H, W = prob_map.shape
    N = H * W
    P_init = prob_map.flatten()
    gray_img = np.mean(img_rgb, axis=2)
    diff_h = gray_img[:, :-1] - gray_img[:, 1:]
    weights_h = np.exp(-beta * (diff_h ** 2)).flatten()
    diff_v = gray_img[:-1, :] - gray_img[1:, :]
    weights_v = np.exp(-beta * (diff_v ** 2)).flatten()
    idx = np.arange(N).reshape(H, W)
    edges_h_i, edges_h_j = idx[:, :-1].flatten(), idx[:, 1:].flatten()
    edges_v_i, edges_v_j = idx[:-1, :].flatten(), idx[1:, :].flatten()
    rows = np.concatenate((edges_h_i, edges_h_j, edges_v_i, edges_v_j))
    cols = np.concatenate((edges_h_j, edges_h_i, edges_v_j, edges_v_i))
    vals = np.concatenate((weights_h, weights_h, weights_v, weights_v))
    W_adj = sp.csr_matrix((vals, (rows, cols)), shape=(N, N))
    L = sp.diags(W_adj.sum(axis=1).A1) - W_adj
    A = L + gamma * sp.eye(N, format='csr')
    return spsolve(A, gamma * P_init).reshape((H, W))

def plot_full_scene(model, file_path, clean_bands, device, patch_size=11):
    mat = sio.loadmat(file_path)
    img, gt = mat["img"], mat["map"]
    rgb_raw = img[:, :, [29, 19, 9]].astype(np.float32)
    rgb_clean = np.where(rgb_raw < 0, 0, rgb_raw)
    rgb_img = np.zeros_like(rgb_clean)
    for c in range(3):
        channel = rgb_clean[:, :, c]
        p_low, p_high = np.percentile(channel[channel > 0], (1, 99))
        rgb_img[:, :, c] = np.clip((channel - p_low) / (p_high - p_low + 1e-8), 0, 1)
        
    img_clean = img[:, :, clean_bands].astype(np.float32)
    
    # Apply the (1, 99) percentile scaling for test scenes
    for b in range(img_clean.shape[2]):
        band = img_clean[:, :, b]
        p_low, p_high = np.percentile(band, (1, 99))
        img_clean[:, :, b] = np.clip((band - p_low) / (p_high - p_low + 1e-8), 0, 1)
        
    pad = patch_size // 2
    img_padded = np.pad(img_clean, ((pad, pad), (pad, pad), (0, 0)), mode='symmetric')

    H, W = gt.shape
    raw_prob_map = np.zeros((H, W))
    valid_coords = [(r, c) for r in range(H) for c in range(W) if gt[r, c] >= 0]
    
    batch_size = 512 
    model.eval()
    with torch.inference_mode():
        # Iterate over batches using range without tqdm to keep logs clean
        for i in range(0, len(valid_coords), batch_size):
            batch_coords = valid_coords[i:i+batch_size]
            batch = [img_padded[r:r+patch_size, c:c+patch_size, :].transpose(2, 0, 1) for r, c in batch_coords]
            batch_tensor = torch.tensor(np.array(batch), dtype=torch.float32).to(device)
            
            # FIX: Flatten model output here too
            probs = torch.sigmoid(model(batch_tensor).squeeze(1)).cpu().numpy()
            
            for (r, c), prob in zip(batch_coords, probs):
                raw_prob_map[r, c] = prob

    print(f"Applying Extended Random Walker Optimization on {file_path.stem}...")
    optimized_probs = apply_erw_optimization(raw_prob_map, rgb_img)
    predictions = (optimized_probs > 0.50).astype(int) 

    # FIX: Mask out the invalid background swath so ERW doesn't hallucinate
    valid_mask = (gt >= 0)
    predictions = (optimized_probs > 0.50).astype(int) * valid_mask
    
    # Calculate full-scene metrics
    all_probs = [optimized_probs[r, c] for r, c in valid_coords]
    all_targets = [gt[r, c] for r, c in valid_coords]
    preds_binary = (np.array(all_probs) > 0.50).astype(int)

    auc = roc_auc_score(all_targets, all_probs)
    precision = precision_score(all_targets, preds_binary, zero_division=0)
    recall = recall_score(all_targets, preds_binary, zero_division=0)
    f1 = f1_score(all_targets, preds_binary, zero_division=0)
    print(f"--- Full Scene Metrics for {file_path.stem} ---")
    print(f"AUC: {auc:.4f} | Precision: {precision:.4f} | Recall: {recall:.4f} | F1: {f1:.4f}\n")

    fig, axs = plt.subplots(1, 3, figsize=(18, 8))
    axs[0].imshow(rgb_img)
    axs[0].set_title(f"False-Color RGB ({file_path.stem})")
    axs[1].imshow(gt == 1, cmap='magma')
    axs[1].set_title("Ground Truth Mask")
    axs[2].imshow(predictions, cmap='magma')
    axs[2].set_title("Model Prediction (ERW Optimized)")
    for ax in axs: ax.axis("off")
    plt.tight_layout()
    
    output_filename = f"prediction_{file_path.stem}.png"
    plt.savefig(output_filename, dpi=300, bbox_inches='tight')
    plt.close(fig) 
    print(f"Saved prediction image to {output_filename}")

print("\n--- Processing final evaluations ---")
for test_file in test_files:
    plot_full_scene(model, test_file, CLEAN_BANDS, device)

print("All tasks completed successfully!")