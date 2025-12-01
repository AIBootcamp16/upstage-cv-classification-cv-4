#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
EfficientNet-B4 Training Script with WandB Sweep Support
"""
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import pandas as pd
from tqdm import tqdm
import albumentations as A
from albumentations.pytorch import ToTensorV2
import cv2
import os
from sklearn.metrics import f1_score, accuracy_score
import random
import argparse
import wandb
import timm  # ✅ timm 추가
from torch.cuda.amp import autocast, GradScaler

# ============================
# Argument Parser (Sweep용)
# ============================
def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--learning_rate', type=float, default=1e-4)
    parser.add_argument('--batch_size', type=int, default=16)
    parser.add_argument('--img_size', type=int, default=380)
    parser.add_argument('--weight_decay', type=float, default=1e-4)
    parser.add_argument('--epochs', type=int, default=30)
    return parser.parse_args()

# ============================
# Seed 고정
# ============================
def set_seed(seed=42):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)

# ============================
# Dataset
# ============================
class DocumentDataset(torch.utils.data.Dataset):
    def __init__(self, img_path, csv_path, img_h, img_w, is_train=True):
        self.img_path = img_path
        self.df = pd.read_csv(csv_path)
        self.is_train = is_train
        self.transform = A.Compose([
            A.Resize(height=img_h, width=img_w),
            A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
            ToTensorV2()
        ])
        
    def __len__(self):
        return len(self.df)
    
    def __getitem__(self, idx):
        id = self.df.iloc[idx, 0]
        label = self.df.iloc[idx, 1]
        img_path = f"{self.img_path}/{id}"
        img = cv2.imread(img_path)
        
        if img is None:
            raise ValueError(f"이미지를 로드할 수 없습니다: {img_path}")
        
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = self.transform(image=img)['image']
        return img, label

# ============================
# Model (EfficientNet-B4)
# ============================
class EfficientNetB4Classifier(nn.Module):
    def __init__(self, num_classes=17):
        super().__init__()
        self.model = timm.create_model('efficientnet_b4', pretrained=True, num_classes=num_classes)
    
    def forward(self, x):
        return self.model(x)

# ============================
# Main Training Function
# ============================
def main():
    # Parse arguments
    args = parse_args()
    
    # WandB 초기화
    run = wandb.init(project="document-classification", entity="imeanseo_")
    config = wandb.config
    
    # 하이퍼파라미터 설정
    IMG_H = IMG_W = config.img_size
    BATCH_SIZE = config.batch_size
    EPOCHS = config.epochs
    LR = config.learning_rate
    WEIGHT_DECAY = config.weight_decay
    
    # 디바이스 설정
    DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    set_seed(42)
    print(f"🖥️ Device: {DEVICE}")
    print("✅ 환경 설정 완료!")
    
    # 데이터셋 생성
    train_dataset = DocumentDataset(
        img_path='/data/ephemeral/home/cv_data/train_aug_50',
        csv_path='/data/ephemeral/home/cv_data/train_aug_50.csv',
        img_h=IMG_H,
        img_w=IMG_W,
        is_train=True
    )
    val_dataset = DocumentDataset(
        img_path='/data/ephemeral/home/cv_data/train',
        csv_path='/data/ephemeral/home/cv_data/val_split.csv',
        img_h=IMG_H,
        img_w=IMG_W,
        is_train=False
    )
    
    train_loader = torch.utils.data.DataLoader(
        train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=4
    )
    val_loader = torch.utils.data.DataLoader(
        val_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=4
    )
    
    print(f"Train: {len(train_dataset)}개, Val: {len(val_dataset)}개")
    
    # 모델, Optimizer, Loss, Scheduler
    model = EfficientNetB4Classifier(num_classes=17).to(DEVICE)
    optimizer = optim.AdamW(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
    criterion = nn.CrossEntropyLoss()
    scheduler = optim.lr_scheduler.CosineAnnealingWarmRestarts(
        optimizer, T_0=10, T_mult=2, eta_min=1e-6
    )
    scaler = GradScaler()
    
    # 학습 루프
    best_f1 = 0.0
    patience = 3
    counter = 0
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
                    
                    if labels.dim() > 1 and labels.shape[1] > 1:
                        labels = labels.argmax(dim=1)
                    val_targets.extend(labels.cpu().numpy())
            
            val_f1 = f1_score(val_targets, val_preds, average='macro')
            val_f1s.append(val_f1)

            print(f"Epoch {epoch+1}/{EPOCHS}: Train Loss={avg_loss:.4f}, Train F1={train_f1:.4f}, Train Acc={train_acc:.4f}, Val F1={val_f1:.4f}")

            # WandB 로깅
            wandb.log({
                "epoch": epoch + 1,
                "train_loss": avg_loss,
                "train_f1": train_f1,
                "train_acc": train_acc,
                "val_f1": val_f1,
                "learning_rate": optimizer.param_groups[0]['lr']
            })

            # Best model 저장
            if val_f1 > best_f1:
                best_f1 = val_f1
                torch.save(model.state_dict(), 'best_efficientb4.pth')
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
            torch.save(model.state_dict(), f'efficientb4_checkpoint_epoch_{epoch+1}.pth')
            raise
    
    print(f"\n🎯 최종 Best Val F1: {best_f1:.4f}")
    
    wandb.log({'best_val_f1': best_f1})
    wandb.finish()

if __name__ == '__main__':
    main()
