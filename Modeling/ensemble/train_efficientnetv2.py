#!/usr/bin/env python
# coding: utf-8

import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.amp import GradScaler
import pandas as pd
import numpy as np
from tqdm import tqdm
from sklearn.metrics import f1_score, accuracy_score
import matplotlib.pyplot as plt
from torch.cuda.amp import autocast, GradScaler
import cv2
import random
from torch.nn import functional as F
from torch.utils.data import Dataset
import albumentations as A
from albumentations.pytorch import ToTensorV2
import timm  # 변경된 부분: timm 사용
import wandb

# sweep을 위한 wandb 초기화 (코드 시작 부분에 추가)
run = wandb.init(project="document-classification", entity="imeanseo_")
config = wandb.config

# 디바이스 설정
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"🖥️ Device: {DEVICE}")

# Seed 고정
def set_seed(seed=42):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)

set_seed(42)
print("✅ 환경 설정 완료!")


# In[ ]:


#!pip install wandb
import wandb
wandb.login()


# In[ ]:


# 기존 하이퍼파라미터 설정부분 대체
IMG_H, IMG_W = config.img_size, config.img_size
BATCH_SIZE = config.batch_size
LR = config.learning_rate
WEIGHT_DECAY = config.weight_decay
EPOCHS = config.epochs


# In[5]:


# ============================
# Dataset 클래스 (기존 동일)
# ============================

class DocumentDataset(Dataset):
    def __init__(self, img_path, csv_path, img_h, img_w, is_train=True, use_mixup=False):
        self.img_path = img_path
        self.is_train = is_train
        self.use_mixup = use_mixup
        
        self.df = pd.read_csv(csv_path)
        self.num_classes = 17
        
        self.transform = A.Compose([
            A.Resize(height=img_h, width=img_w),
            A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
            ToTensorV2(),
        ])
        
        print(f"{'Train' if is_train else 'Val'} Dataset: {len(self.df)}개")

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        img_id = self.df.iloc[idx, 0]
        target = self.df.iloc[idx, 1]

        img_path = f"{self.img_path}/{img_id}"
        image = cv2.imread(img_path)

        if image is None:
            raise ValueError(f"이미지를 로드할 수 없습니다: {img_path}")

        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        if self.use_mixup and self.is_train and random.random() > 0.5:
            rand_idx = random.randint(0, len(self.df)-1)
            bg_id = self.df.iloc[rand_idx, 0]
            bg_target = self.df.iloc[rand_idx, 1]

            bg_img = cv2.imread(f"{self.img_path}/{bg_id}")
            if bg_img is None:
                y = F.one_hot(torch.tensor(target), num_classes=self.num_classes).float().contiguous()
            else:
                bg_img = cv2.cvtColor(bg_img, cv2.COLOR_BGR2RGB)
                bg_img = cv2.resize(bg_img, (image.shape[1], image.shape[0]))

                lam = np.random.beta(0.5, 0.5)
                image = (lam * image.astype(np.float32) + (1-lam) * bg_img.astype(np.float32)).astype(np.uint8)

                y = F.one_hot(torch.tensor(target), num_classes=self.num_classes).float().contiguous()
                bg_y = F.one_hot(torch.tensor(bg_target), num_classes=self.num_classes).float().contiguous()
                y = (lam * y + (1-lam) * bg_y).contiguous()
        else:
            y = F.one_hot(torch.tensor(target), num_classes=self.num_classes).float().contiguous()

        x = self.transform(image=image)['image']
        return x, y

print("✅ Dataset 클래스 정의 완료!")

# ============================
# EfficientNetV2 XL 모델 생성 (timm 라이브러리 사용)
# ============================

NUM_CLASSES = 17

model = timm.create_model('tf_efficientnetv2_xl', pretrained=True, num_classes=NUM_CLASSES).to(DEVICE)


# In[ ]:


# ============================
# 옵티마이저, 손실함수, 스케줄러
# ============================

model = timm.create_model('tf_efficientnetv2_xl', pretrained=True, num_classes=17).to(DEVICE)
optimizer = optim.AdamW(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
criterion = nn.CrossEntropyLoss()
scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer, T_0=10, T_mult=2, eta_min=1e-6)


# In[ ]:


# ============================
# DataLoader 준비
# ============================
# DataLoader 생성 부분 (config 기준으로)
train_dataset = DocumentDataset(
    img_path='/data/ephemeral/home/cv_data/train_aug_50',
    csv_path='/data/ephemeral/home/cv_data/train_aug_50.csv',
    img_h=IMG_H,
    img_w=IMG_W,
    is_train=True,
    use_mixup=True
)

val_dataset = DocumentDataset(
    img_path='/data/ephemeral/home/cv_data/train',
    csv_path='/data/ephemeral/home/cv_data/val_split.csv',
    img_h=IMG_H,
    img_w=IMG_W,
    is_train=False,
    use_mixup=False
)

train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=4)
val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=4)


print(f"\n✅ DataLoader 준비 완료!")
print(f"  - Train batches: {len(train_loader)}")
print(f"  - Val batches: {len(val_loader)}")


# In[ ]:


# ============================
# 학습 루프 (mixed precision 적용 포함)
# ============================
run = wandb.init()
config = wandb.config

EPOCHS = config.epochs

scaler = GradScaler()
patience = 1
counter = 0
best_f1 = 0.0
train_losses = []
train_f1s = []
train_accs = []
val_f1s = []

for epoch in range(EPOCHS):
    try:
        # Train
        model.train()
        epoch_loss = 0.0
        train_preds, train_targets = [], []
        
        for images, labels in tqdm(train_loader, desc=f"Epoch {epoch+1}/{EPOCHS}"):
            images, labels = images.to(DEVICE), labels.to(DEVICE)
            optimizer.zero_grad()
            
            with autocast():
                outputs = model(images)
                loss = criterion(outputs, labels)
            
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            
            epoch_loss += loss.item()
            
            preds = outputs.argmax(dim=1).cpu().numpy()
            train_preds.extend(preds)
            
            # one-hot 인코딩을 클래스 인덱스로 변환
            if labels.dim() > 1 and labels.shape[1] > 1:
                labels = labels.argmax(dim=1)
            train_targets.extend(labels.cpu().numpy())
        
        avg_loss = epoch_loss / len(train_loader)
        scheduler.step()
        
        train_f1 = f1_score(train_targets, train_preds, average='macro')
        train_acc = accuracy_score(train_targets, train_preds)
        train_f1s.append(train_f1)
        train_accs.append(train_acc)

        # Validation
        model.eval()
        val_preds, val_targets = [], []
        with torch.no_grad():
            for images, labels in tqdm(val_loader, desc="Validation", leave=False):
                images, labels = images.to(DEVICE), labels.to(DEVICE)
                outputs = model(images)
                preds = outputs.argmax(dim=1).cpu().numpy()
                val_preds.extend(preds)
                
                # ✅ validation에도 one-hot 인코딩 처리 추가
                if labels.dim() > 1 and labels.shape[1] > 1:
                    labels = labels.argmax(dim=1)
                val_targets.extend(labels.cpu().numpy())
        
        val_f1 = f1_score(val_targets, val_preds, average='macro')
        val_f1s.append(val_f1)

        print(f"Epoch {epoch+1}/{EPOCHS}: Train Loss={avg_loss:.4f}, Train F1={train_f1:.4f}, Train Acc={train_acc:.4f}, Val F1={val_f1:.4f}")

        # WandB 로깅
        wandb.log({
            "epoch": epoch + 1,  # ✅ epoch은 1부터 시작하도록 +1
            "train_loss": avg_loss,
            "train_f1": train_f1,
            "train_acc": train_acc,
            "val_f1": val_f1,
            "learning_rate": optimizer.param_groups[0]['lr']
        })

        # Best model 저장
        if val_f1 > best_f1:
            best_f1 = val_f1
            torch.save(model.state_dict(), 'best_efficientnetv2_xl.pth')
            print(f"  ✅ Best Model Saved! F1={best_f1:.4f}")
            counter = 0
        else:
            counter += 1
            print(f"  ⚠️ EarlyStopping counter: {counter} / {patience}")
            if counter >= patience:
                print("▶ Early stopping triggered!")
                break

    except Exception as e:
        print(f"\n❌ Epoch {epoch+1}에서 에러 발생: {e}")
        torch.save(model.state_dict(), f'efficientnetv2_xl_checkpoint_epoch_{epoch+1}.pth')
        raise

print(f"\n🎯 최종 Best Val F1: {best_f1:.4f}")

# ✅ 최종 best_val_f1도 로깅
wandb.log({'best_val_f1': best_f1})
wandb.finish()


# In[ ]:


# ============================
# 학습 로그 저장 및 시각화
# ============================

training_log = pd.DataFrame({
    'epoch': range(1, len(train_losses) + 1),
    'train_loss': train_losses,
    'train_f1': train_f1s,
    'train_acc': train_accs,
    'val_f1': val_f1s
})
training_log.to_csv('efficientnetv2_xl_training_log.csv', index=False)
print("✅ 학습 로그 저장 완료: efficientnetv2_xl_training_log.csv")

fig, axes = plt.subplots(1, 2, figsize=(14, 5))
axes[0].plot(train_losses, 'b-o', linewidth=2, markersize=4)
axes[0].set_title('Training Loss')
axes[0].set_xlabel('Epoch')
axes[0].set_ylabel('Loss')
axes[0].grid(True, alpha=0.3)

axes[1].plot(val_f1s, 'g-o', linewidth=2, markersize=4)
axes[1].axhline(y=best_f1, color='r', linestyle='--', label=f'Best: {best_f1:.4f}')
axes[1].set_title('Validation F1 Score')
axes[1].set_xlabel('Epoch')
axes[1].set_ylabel('F1 Score')
axes[1].legend()
axes[1].grid(True, alpha=0.3)

plt.tight_layout()
plt.show()

print("✅ 학습 결과 시각화 완료")

