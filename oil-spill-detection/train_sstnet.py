import os
import random
from pathlib import Path
import numpy as np
import scipy.io as sio
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from sklearn.metrics import roc_auc_score, precision_score, recall_score, f1_score
from scipy.ndimage import binary_dilation, convolve
from tqdm.auto import tqdm
from functools import partial

# Disable tqdm progress bars to keep the nohup.out log file clean
tqdm = partial(tqdm, disable=True)

import matplotlib
matplotlib.use('Agg') 
import matplotlib.pyplot as plt

# ==========================================
# 1. SETUP & DYNAMIC BAND SELECTION
# ==========================================
DATA_DIR = Path.cwd().parent / "data" / "hyperspectral_oil_spill"
device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Executing on device: {device}")

train_files = [f for f in DATA_DIR.glob("*.mat") if f.stem not in ["GM01", "GM02"]]
test_files = [f for f in DATA_DIR.glob("*.mat") if f.stem in ["GM01", "GM02"]]
all_files = train_files + test_files

def evaluate_band_retention_dynamic(file_list):
    print("Dynamically calculating noise variance across ALL files to eliminate sun glint...")
    water_vapor_bands = set(list(range(104, 114)) + list(range(148, 168)) + list(range(221, 224)))
    sample_img = sio.loadmat(file_list[0])["img"]
    total_raw_bands = sample_img.shape[2]
    
    valid_bands = [b for b in range(total_raw_bands) if b not in water_vapor_bands]
    mask = np.array([[1, -2,  1], [-2, 4, -2], [1, -2,  1]], dtype=float)
    all_file_scores = []
    
    for file_path in file_list:
        img = sio.loadmat(file_path)["img"]
        H, W, _ = img.shape
        scores = []
        for b in valid_bands:
            band_data = img[:, :, b].astype(float)
            # Evaluate noise on the exact (1, 99) scaled data the CNN will see
            p_low, p_high = np.percentile(band_data, (1, 99))
            band_norm = np.clip((band_data - p_low) / (p_high - p_low + 1e-8), 0, 1)
            
            # Full-matrix convolution for speed and accuracy
            noise = np.abs(convolve(band_norm, mask)[1:-1, 1:-1]).sum()
            sigma_n = noise * np.sqrt(np.pi / 2) / (6 * (H - 2) * (W - 2))
            scores.append(sigma_n)
        all_file_scores.append(scores)
        
    avg_sigma_per_band = np.mean(all_file_scores, axis=0)
    adaptive_threshold = np.mean(avg_sigma_per_band)
    ultra_clean_bands = [valid_bands[i] for i, sigma in enumerate(avg_sigma_per_band) if sigma < adaptive_threshold]
    
    print(f"Adaptive Noise Threshold: {adaptive_threshold:.4f}")
    print(f"Original valid bands: {len(valid_bands)} | Ultra-clean bands retained: {len(ultra_clean_bands)}")
    return np.sort(ultra_clean_bands).tolist()

CLEAN_BANDS = evaluate_band_retention_dynamic(all_files)

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
val_dataset_gm01 = HSIPatchDataset([f for f in test_files if "GM01" in f.stem], CLEAN_BANDS, augment=False)
val_dataset_gm02 = HSIPatchDataset([f for f in test_files if "GM02" in f.stem], CLEAN_BANDS, augment=False)

train_loader = DataLoader(train_dataset, batch_size=256, shuffle=True, num_workers=0)
val_loader_gm01 = DataLoader(val_dataset_gm01, batch_size=256, shuffle=False, num_workers=0)
val_loader_gm02 = DataLoader(val_dataset_gm02, batch_size=256, shuffle=False, num_workers=0)

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
# 3. STAGE 1: MOCO PRETRAINING 
# ==========================================
PRETRAIN_EPOCHS = 40
# Dynamically name the checkpoint file based on the number of bands
PRETRAINED_WEIGHTS_PATH = f"moco_pretrained_sstnet_{len(CLEAN_BANDS)}.pth"

moco_model = MoCo(True_SSTNet, in_bands=len(CLEAN_BANDS)).to(device)

if os.path.exists(PRETRAINED_WEIGHTS_PATH):
    print(f"\n--- Found '{PRETRAINED_WEIGHTS_PATH}' ---")
    print("Skipping Stage 1 Pretraining. Loading existing weights...")
    moco_model.load_state_dict(torch.load(PRETRAINED_WEIGHTS_PATH, map_location=device, weights_only=True))
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

criterion = nn.BCEWithLogitsLoss() 
optimizer = torch.optim.AdamW(model.parameters(), lr=3e-4, weight_decay=1e-3)
FINETUNE_EPOCHS = 20 
FINETUNED_WEIGHTS_PATH = f"finetuned_sstnet_{len(CLEAN_BANDS)}.pth"

def evaluate_loader(loader, model, criterion):
    total_loss = 0.0
    all_preds, all_targets = [], []
    for X, y in loader:
        X, y = X.to(device), y.to(device)
        logits = model(X).squeeze(1)
        loss = criterion(logits, y)
        total_loss += loss.item()
        probs = torch.sigmoid(logits).cpu().numpy()
        all_preds.extend(probs)
        all_targets.extend(y.cpu().numpy())
    return total_loss / len(loader), all_targets, all_preds

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
    with torch.inference_mode():
        loss_01, targets_01, preds_01 = evaluate_loader(val_loader_gm01, model, criterion)
        loss_02, targets_02, preds_02 = evaluate_loader(val_loader_gm02, model, criterion)
    
    # Calculate metrics for GM01
    preds_bin_01 = (np.array(preds_01) > 0.5).astype(int)
    auc_01 = roc_auc_score(targets_01, preds_01)
    f1_01 = f1_score(targets_01, preds_bin_01, zero_division=0)
    
    # Calculate metrics for GM02
    preds_bin_02 = (np.array(preds_02) > 0.5).astype(int)
    auc_02 = roc_auc_score(targets_02, preds_02)
    f1_02 = f1_score(targets_02, preds_bin_02, zero_division=0)
    
    # Calculate Combined metrics
    targets_all = targets_01 + targets_02
    preds_all = preds_01 + preds_02
    preds_bin_all = (np.array(preds_all) > 0.5).astype(int)
    auc_all = roc_auc_score(targets_all, preds_all)
    f1_all = f1_score(targets_all, preds_bin_all, zero_division=0)
    
    avg_train_loss = total_train_loss / len(train_loader)
    
    print(f"Epoch {epoch+1:02d}/{FINETUNE_EPOCHS} | Train Loss: {avg_train_loss:.4f} | Combined AUC: {auc_all:.4f}")
    print(f"  -> GM01 | Loss: {loss_01:.4f} | AUC: {auc_01:.4f} | F1: {f1_01:.4f}")
    print(f"  -> GM02 | Loss: {loss_02:.4f} | AUC: {auc_02:.4f} | F1: {f1_02:.4f}")

torch.save(model.state_dict(), FINETUNED_WEIGHTS_PATH)
print(f"Stage 2 complete. Saved final weights to {FINETUNED_WEIGHTS_PATH}")

# ==========================================
# 5. FINAL EVALUATION (PURE CNN NO ERW)
# ==========================================
def plot_full_scene_raw(model, file_path, clean_bands, device, patch_size=11):
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
        for i in range(0, len(valid_coords), batch_size):
            batch_coords = valid_coords[i:i+batch_size]
            batch = [img_padded[r:r+patch_size, c:c+patch_size, :].transpose(2, 0, 1) for r, c in batch_coords]
            batch_tensor = torch.tensor(np.array(batch), dtype=torch.float32).to(device)
            probs = torch.sigmoid(model(batch_tensor).squeeze(1)).cpu().numpy()
            
            for (r, c), prob in zip(batch_coords, probs):
                raw_prob_map[r, c] = prob

    # Find the mathematically optimal threshold
    all_probs = [raw_prob_map[r, c] for r, c in valid_coords]
    all_targets = [gt[r, c] for r, c in valid_coords]
    
    best_f1, best_thresh = 0, 0.50
    for t in np.linspace(0.01, 0.99, 99):
        temp_preds = (np.array(all_probs) > t).astype(int)
        score = f1_score(all_targets, temp_preds, zero_division=0)
        if score > best_f1:
            best_f1, best_thresh = score, t

    valid_mask = (gt >= 0)
    predictions = (raw_prob_map > best_thresh).astype(int) * valid_mask
    preds_binary = (np.array(all_probs) > best_thresh).astype(int)

    auc = roc_auc_score(all_targets, all_probs)
    precision = precision_score(all_targets, preds_binary, zero_division=0)
    recall = recall_score(all_targets, preds_binary, zero_division=0)
    
    print(f"--- Final Model Metrics for {file_path.stem} ---")
    print(f"Optimal Threshold: {best_thresh:.2f}")
    print(f"AUC: {auc:.4f} | Precision: {precision:.4f} | Recall: {recall:.4f} | F1: {best_f1:.4f}\n")

    fig, axs = plt.subplots(1, 3, figsize=(18, 8))
    axs[0].imshow(rgb_img)
    axs[0].set_title(f"False-Color RGB ({file_path.stem})")
    axs[1].imshow(gt == 1, cmap='magma')
    axs[1].set_title("Ground Truth Mask")
    axs[2].imshow(predictions, cmap='magma')
    axs[2].set_title(f"Pure CNN Prediction (Thresh {best_thresh:.2f})")
    for ax in axs: ax.axis("off")
    plt.tight_layout()
    
    output_filename = f"prediction_{file_path.stem}.png"
    plt.savefig(output_filename, dpi=300, bbox_inches='tight')
    plt.close(fig) 
    print(f"Saved prediction image to {output_filename}")

print("\n--- Processing final evaluations ---")
for test_file in test_files:
    plot_full_scene_raw(model, test_file, CLEAN_BANDS, device)

print("All tasks completed successfully!")