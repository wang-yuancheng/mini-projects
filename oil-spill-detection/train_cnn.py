import os
import random
from pathlib import Path
import numpy as np
import scipy.io as sio
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from sklearn.metrics import roc_auc_score, precision_score, recall_score, f1_score
from scipy.ndimage import binary_dilation, convolve
from tqdm.auto import tqdm
from functools import partial

# Disable tqdm progress bars to keep log files clean
tqdm = partial(tqdm, disable=True)

# ==========================================
# 1. SETUP & DYNAMIC BAND SELECTION
# ==========================================
DATA_DIR = Path.cwd().parent / "data" / "hyperspectral_oil_spill"
device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Executing on device: {device}")

# Strict Split: GM03 = Validation; GM01, GM02 = Unseen Test; Rest = Train
train_files = sorted([f for f in DATA_DIR.glob("*.mat") if f.stem not in ["GM01", "GM02", "GM03"]])
val_files = sorted([f for f in DATA_DIR.glob("*.mat") if f.stem == "GM03"])
test_files = sorted([f for f in DATA_DIR.glob("*.mat") if f.stem in ["GM01", "GM02"]])
all_files = sorted(list(DATA_DIR.glob("*.mat")))

print(f"Train scenes: {[f.stem for f in train_files]}")
print(f"Val scene:    {[f.stem for f in val_files]}")
print(f"Test scenes:  {[f.stem for f in test_files]}")

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
            p_low, p_high = np.percentile(band_data, (1, 99))
            band_norm = np.clip((band_data - p_low) / (p_high - p_low + 1e-8), 0, 1)
            
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
            if n_oil == 0: 
                continue
            
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
            
            for r, c in oil_coords: 
                self.items.append((idx, r, c, 1.0))
            for r, c in water_sampled: 
                self.items.append((idx, r, c, 0.0))

    def __len__(self): 
        return len(self.items)

    def __getitem__(self, idx):
        file_idx, r, c, label = self.items[idx]
        patch = self.arrays[file_idx][r:r+self.patch_size, c:c+self.patch_size, :]
        
        if self.augment:
            # 1. Spatial rotations and flips (Geometric invariance)
            k = random.choice([0, 1, 2, 3])
            if k > 0: 
                patch = np.rot90(patch, k=k, axes=(0, 1))
            if random.random() > 0.5: 
                patch = np.fliplr(patch)
            if random.random() > 0.5: 
                patch = np.flipud(patch)

            # 2. Spectral Illumination Scaling (±10% radiance shift)
            if random.random() > 0.5:
                scale = np.random.uniform(0.90, 1.10)
                patch = patch * scale

            # 3. Additive Gaussian Sensor Noise (Physically grounded sensor noise)
            if random.random() > 0.5:
                noise = np.random.normal(0, 0.005, patch.shape).astype(np.float32)
                patch = patch + noise

            # Keep values safely bounded in the normalized [0, 1] range
            patch = np.clip(patch, 0.0, 1.0)

        patch_tensor = torch.from_numpy(patch.copy().transpose(2, 0, 1))
        return patch_tensor, torch.tensor(label, dtype=torch.float32)

print("Mining dataset patches...")
train_dataset = HSIPatchDataset(train_files, CLEAN_BANDS, augment=True)
val_dataset_gm03 = HSIPatchDataset(val_files, CLEAN_BANDS, augment=False)

train_loader = DataLoader(train_dataset, batch_size=256, shuffle=True, num_workers=0)
val_loader_gm03 = DataLoader(val_dataset_gm03, batch_size=256, shuffle=False, num_workers=0)

# ==========================================
# 2. MODEL DEFINITIONS (Updated for Paper 2 MoCo architecture)
# ==========================================
class OilSpillCNN(nn.Module):
    def __init__(self, in_bands, mlp_dim=64):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(in_bands, 32, 3, padding=1), nn.ReLU(),
            nn.Conv2d(32, 32, 3, padding=1), nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(32, 64, 3, padding=1), nn.ReLU(),
            nn.Conv2d(64, 64, 3, padding=1), nn.ReLU(),
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten()
        )
        
        # The CNN MLP retained for BOTH Pretraining and Fine-Tuning
        self.mlp = nn.Sequential(
            nn.Linear(64, mlp_dim),
            nn.ReLU(inplace=True),
            nn.Linear(mlp_dim, mlp_dim)
        )
        
        # The final linear classifier (Dormant in pretraining, active in fine-tuning)
        self.classifier = nn.Linear(mlp_dim, 1)

    def forward(self, x, pretrain=False):
        x = self.features(x)  # <-- 1. CNN Backbone
        x = self.mlp(x)       # <-- 2. The MLP head (retained in both stages)
        
        if pretrain:
            return x          # <-- Pretraining: Returns CNN > MLP for Contrastive Loss
            
        return self.classifier(x) # <-- Fine-Tuning: Returns CNN > MLP > Linear Classifier


class MoCoDataset(Dataset):
    def __init__(self, original_dataset):
        self.dataset = original_dataset
        self.dataset.augment = True
    def __len__(self): 
        return len(self.dataset)
    def __getitem__(self, idx):
        q_patch, _ = self.dataset[idx]
        k_patch, _ = self.dataset[idx]
        return q_patch, k_patch


class MoCo(nn.Module):
    def __init__(self, base_encoder, in_bands, dim=64, K=8192, m=0.999, T=0.07):
        super(MoCo, self).__init__()
        self.K, self.m, self.T = K, m, T

        # Twin Encoders (Backbone + MLP are inside base_encoder)
        self.encoder_q = base_encoder(in_bands=in_bands, mlp_dim=dim)
        self.encoder_k = base_encoder(in_bands=in_bands, mlp_dim=dim)

        # Initialize Key network parameters to match Query network exactly
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
        # Compute query features using the MLP head
        q = self.encoder_q(im_q, pretrain=True)
        q = nn.functional.normalize(q, dim=1)

        # Compute key features using the MLP head
        with torch.no_grad():
            self._momentum_update_key_encoder()
            k = self.encoder_k(im_k, pretrain=True)
            k = nn.functional.normalize(k, dim=1)

        l_pos = torch.einsum('nc,nc->n', [q, k]).unsqueeze(-1)
        l_neg = torch.einsum('nc,ck->nk', [q, self.queue.clone().detach()])
        logits = torch.cat([l_pos, l_neg], dim=1) / self.T
        labels = torch.zeros(logits.shape[0], dtype=torch.long).to(q.device)
        self._dequeue_and_enqueue(k)
        return logits, labels

# ==========================================
# 3. STAGE 1: MOCO PRETRAINING (50 EPOCHS)
# ==========================================
PRETRAIN_EPOCHS = 50
PRETRAINED_WEIGHTS_PATH = f"moco_v2_pretrained_cnn_{len(CLEAN_BANDS)}_50ep.pth"

moco_model = MoCo(OilSpillCNN, in_bands=len(CLEAN_BANDS)).to(device)

if os.path.exists(PRETRAINED_WEIGHTS_PATH):
    print(f"\n--- Found '{PRETRAINED_WEIGHTS_PATH}' ---")
    print("Skipping Stage 1 Pretraining. Loading existing weights...")
    moco_model.load_state_dict(torch.load(PRETRAINED_WEIGHTS_PATH, map_location=device, weights_only=True))
else:
    print(f"\n--- Starting Stage 1: MoCo Pretraining ({PRETRAIN_EPOCHS} Epochs) ---")
    moco_train_loader = DataLoader(MoCoDataset(train_dataset), batch_size=256, shuffle=True, num_workers=0, drop_last=True)
    criterion_moco = nn.CrossEntropyLoss()
    optimizer_moco = torch.optim.AdamW(moco_model.parameters(), lr=3e-4, weight_decay=1e-4)
    scheduler_moco = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer_moco, T_max=PRETRAIN_EPOCHS, eta_min=1e-6)

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

        scheduler_moco.step()

        if (epoch + 1) % 5 == 0 or epoch == 0:
            print(f"MoCo Epoch {epoch+1:03d}/{PRETRAIN_EPOCHS} | Avg Contrastive Loss: {avg_loss:.4f}")

    torch.save(moco_model.state_dict(), PRETRAINED_WEIGHTS_PATH)
    print(f"Stage 1 complete. Saved weights to {PRETRAINED_WEIGHTS_PATH}")
# ==========================================
# 4. STAGE 2: SUPERVISED FINE-TUNING
# ==========================================
# Extract the query backbone encoder (which now includes the retained MLP)
model = moco_model.encoder_q

# Randomly initialize the final linear classifier since it was dormant during pretraining
nn.init.xavier_uniform_(model.classifier.weight)
nn.init.zeros_(model.classifier.bias)

criterion = nn.BCEWithLogitsLoss() 
optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4, weight_decay=1e-3)

FINETUNE_EPOCHS = 30
scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=FINETUNE_EPOCHS, eta_min=1e-6)

BEST_WEIGHTS_PATH = f"best_finetuned_cnn_{len(CLEAN_BANDS)}.pth"
LAST_WEIGHTS_PATH = f"last_finetuned_cnn_{len(CLEAN_BANDS)}.pth"

# ---------------------------------------------------------
# PRE-COMPUTE CALIBRATION DATA (To speed up per-epoch evaluation)
# ---------------------------------------------------------
print("\nPre-computing full-scene distributions for adaptive threshold calibration...")
calibration_files = train_files + val_files
calib_cache = {}

for f in calibration_files:
    mat = sio.loadmat(f)
    img, gt = mat["img"], mat["map"]
    
    img_clean = img[:, :, CLEAN_BANDS].astype(np.float32)
    for b in range(img_clean.shape[2]):
        band = img_clean[:, :, b]
        p_low, p_high = np.percentile(band, (1, 99))
        img_clean[:, :, b] = np.clip((band - p_low) / (p_high - p_low + 1e-8), 0, 1)
    
    intensity_map = np.mean(img_clean, axis=2)
    water_pixels = intensity_map[gt == 0]
    variance = np.std(water_pixels) if len(water_pixels) > 0 else 0.0
    
    pad = 11 // 2
    img_padded = np.pad(img_clean, ((pad, pad), (pad, pad), (0, 0)), mode='symmetric')
    
    H, W = gt.shape
    valid_coords = [(r, c) for r in range(H) for c in range(W) if gt[r, c] >= 0]
    # Subsample by 4 to maintain true distribution but vastly speed up epoch inference
    valid_coords = valid_coords[::4] 
    targets = np.array([gt[r, c] for r, c in valid_coords])
    
    calib_cache[f.stem] = {
        'variance': variance,
        'img_padded': img_padded,
        'coords': valid_coords,
        'targets': targets,
        'is_val': f.stem == "GM03"
    }

best_val_f1 = 0.0

print(f"\n--- Starting Stage 2: Supervised Fine-Tuning ({FINETUNE_EPOCHS} Epochs) ---")
print("Validation Monitor: GM03 (No Adaptive Thresholding)\n")

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
        
    # Evaluate strictly on GM03 using a fixed 0.5 threshold
    model.eval()
    val_loss = 0.0
    all_preds, all_targets = [], []
    with torch.inference_mode():
        for X, y in val_loader_gm03:
            X, y = X.to(device), y.to(device)
            logits = model(X).squeeze(1)
            v_loss = criterion(logits, y)
            val_loss += v_loss.item()
            
            probs = torch.sigmoid(logits).cpu().numpy()
            all_preds.extend(probs)
            all_targets.extend(y.cpu().numpy())
            
    all_preds = np.array(all_preds)
    all_targets = np.array(all_targets)
    preds_binary = (all_preds > 0.5).astype(int)
    
    val_auc = roc_auc_score(all_targets, all_preds)
    val_f1 = f1_score(all_targets, preds_binary, zero_division=0)
    val_precision = precision_score(all_targets, preds_binary, zero_division=0)
    val_recall = recall_score(all_targets, preds_binary, zero_division=0)
    
    avg_train_loss = total_train_loss / len(train_loader)
    avg_val_loss = val_loss / len(val_loader_gm03)
    
    # Checkpointing based on strict validation F1
    is_best = val_f1 > best_val_f1
    if is_best:
        best_val_f1 = val_f1
        torch.save(model.state_dict(), BEST_WEIGHTS_PATH)
        marker = " [★ BEST SAVED]"
    else:
        marker = ""

    scheduler.step()
    current_lr = scheduler.get_last_lr()[0]

    print(f"Epoch {epoch+1:02d}/{FINETUNE_EPOCHS} | Train Loss: {avg_train_loss:.4f} | Val Loss: {avg_val_loss:.4f} | "
          f"GM03 AUC: {val_auc:.4f} | GM03 F1: {val_f1:.4f} | Prec: {val_precision:.4f} | Rec: {val_recall:.4f}{marker}")

torch.save(model.state_dict(), LAST_WEIGHTS_PATH)
print(f"\nTraining complete.")
print(f"-> Peak GM03 Validation F1: {best_val_f1:.4f}")
print(f"-> Best weights saved to:   {BEST_WEIGHTS_PATH}")
print("\nReady for evaluation inside Jupyter Notebook.")