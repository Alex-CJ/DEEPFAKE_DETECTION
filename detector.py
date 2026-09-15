"""
GAN-BASED DEEPFAKE DETECTOR - University Assignment Version
Proper Train/Validation/Test splits with weight saving for easy evaluation
"""
def plot_individual_metrics(history):
    """Create separate plots for each metric for detailed analysis"""
    epochs = range(1, len(history['train_loss']) + 1)
    
    # Individual AUC plot
    plt.figure(figsize=(10, 6))
    plt.plot(epochs, history['val_auc'], 'g-', label='Validation AUC', linewidth=2.5, marker='o', markersize=6)
    plt.xlabel('Epoch', fontsize=13, fontweight='bold')
    plt.ylabel('AUC-ROC', fontsize=13, fontweight='bold')
    plt.title('Validation AUC-ROC Over Epochs', fontsize=15, fontweight='bold')
    plt.legend(fontsize=12)
    plt.grid(True, alpha=0.3, linestyle='--')
    plt.ylim([0, 1.05])
    plt.tight_layout()
    plt.savefig('plots/validation_auc.png', dpi=150, bbox_inches='tight')
    print("   📊 Validation AUC plot saved: plots/validation_auc.png")
    plt.close()
    
    # Individual F1 plot
    plt.figure(figsize=(10, 6))
    plt.plot(epochs, history['val_f1'], 'm-', label='Validation F1 Score', linewidth=2.5, marker='s', markersize=6)
    plt.xlabel('Epoch', fontsize=13, fontweight='bold')
    plt.ylabel('F1 Score', fontsize=13, fontweight='bold')
    plt.title('Validation F1 Score Over Epochs', fontsize=15, fontweight='bold')
    plt.legend(fontsize=12)
    plt.grid(True, alpha=0.3, linestyle='--')
    plt.ylim([0, 1.05])
    plt.tight_layout()
    plt.savefig('plots/validation_f1.png', dpi=150, bbox_inches='tight')
    print("   📊 Validation F1 plot saved: plots/validation_f1.png")
    plt.close()
    
    # Individual Accuracy plot
    plt.figure(figsize=(10, 6))
    plt.plot(epochs, history['val_acc'], 'b-', label='Validation Accuracy', linewidth=2.5, marker='^', markersize=6)
    plt.xlabel('Epoch', fontsize=13, fontweight='bold')
    plt.ylabel('Accuracy (%)', fontsize=13, fontweight='bold')
    plt.title('Validation Accuracy Over Epochs', fontsize=15, fontweight='bold')
    plt.legend(fontsize=12)
    plt.grid(True, alpha=0.3, linestyle='--')
    plt.tight_layout()
    plt.savefig('plots/validation_accuracy.png', dpi=150, bbox_inches='tight')
    print("   📊 Validation Accuracy plot saved: plots/validation_accuracy.png")
    plt.close()

"""
SETUP YOUR FOLDERS:
data/
├── train/
│   ├── real/     (your real training images)
│   └── fake/     (your fake training images)
├── val/
│   ├── real/     (your real validation images)
│   └── fake/     (your fake validation images)
└── test/
    ├── real/     (your real test images)
    └── fake/     (your fake test images)

USAGE:
1. Train once: python detector.py --mode train --data_dir data_testing --epochs 5 --batch_size 64
2. Teacher evaluates: python detector.py --mode evaluate

Installation:
pip install torch torchvision pillow tqdm matplotlib scikit-learn
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from torchvision.utils import save_image
from PIL import Image
import os
from tqdm import tqdm
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix, roc_auc_score
import numpy as np
import argparse
import json
import warnings
warnings.filterwarnings('ignore')


# ============================================================================
# NETWORK ARCHITECTURE
# ============================================================================
class SelfAttention(nn.Module):
    """Self-attention for better image quality"""
    def __init__(self, in_channels):
        super(SelfAttention, self).__init__()
        self.query = nn.Conv2d(in_channels, in_channels // 8, 1)
        self.key = nn.Conv2d(in_channels, in_channels // 8, 1)
        self.value = nn.Conv2d(in_channels, in_channels, 1)
        self.gamma = nn.Parameter(torch.zeros(1))
    
    def forward(self, x):
        batch, channels, height, width = x.size()
        query = self.query(x).view(batch, -1, height * width).permute(0, 2, 1)
        key = self.key(x).view(batch, -1, height * width)
        attention = torch.softmax(torch.bmm(query, key), dim=-1)
        value = self.value(x).view(batch, -1, height * width)
        out = torch.bmm(value, attention.permute(0, 2, 1))
        out = out.view(batch, channels, height, width)
        return self.gamma * out + x


class Generator(nn.Module):
    """Generator for GAN training"""
    def __init__(self, latent_dim=128):
        super(Generator, self).__init__()
        self.latent_dim = latent_dim
        
        self.fc = nn.Sequential(
            nn.Linear(latent_dim, 512 * 4 * 4),
            nn.BatchNorm1d(512 * 4 * 4),
            nn.ReLU(True)
        )
        
        self.conv = nn.Sequential(
            nn.ConvTranspose2d(512, 256, 4, 2, 1),
            nn.BatchNorm2d(256),
            nn.ReLU(True),
            nn.ConvTranspose2d(256, 128, 4, 2, 1),
            nn.BatchNorm2d(128),
            nn.ReLU(True),
            SelfAttention(128),
            nn.ConvTranspose2d(128, 64, 4, 2, 1),
            nn.BatchNorm2d(64),
            nn.ReLU(True),
            nn.ConvTranspose2d(64, 3, 4, 2, 1),
            nn.Tanh()
        )
    
    def forward(self, z):
        x = self.fc(z)
        x = x.view(-1, 512, 4, 4)
        return self.conv(x)


class Discriminator(nn.Module):
    """Discriminator - becomes our deepfake detector"""
    def __init__(self):
        super(Discriminator, self).__init__()
        
        norm = nn.utils.spectral_norm
        
        self.features = nn.Sequential(
            norm(nn.Conv2d(3, 64, 4, 2, 1)),
            nn.LeakyReLU(0.2),
            
            norm(nn.Conv2d(64, 128, 4, 2, 1)),
            nn.BatchNorm2d(128),
            nn.LeakyReLU(0.2),
            
            norm(nn.Conv2d(128, 256, 4, 2, 1)),
            nn.BatchNorm2d(256),
            nn.LeakyReLU(0.2),
            SelfAttention(256),
            
            norm(nn.Conv2d(256, 512, 4, 2, 1)),
            nn.BatchNorm2d(512),
            nn.LeakyReLU(0.2),
            nn.Dropout2d(0.3),
            
            norm(nn.Conv2d(512, 1024, 4, 2, 1)),
            nn.BatchNorm2d(1024),
            nn.LeakyReLU(0.2),
            nn.Dropout2d(0.3)
        )
        
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(1024 * 2 * 2, 256),
            nn.LeakyReLU(0.2),
            nn.Dropout(0.5),
            nn.Linear(256, 1),
            nn.Sigmoid()
        )
    
    def forward(self, x):
        features = self.features(x)
        return self.classifier(features)


# ============================================================================
# DATASET
# ============================================================================
class DeepfakeDataset(Dataset):
    """Load real/fake images from folder"""
    def __init__(self, data_dir, transform=None):
        self.transform = transform
        self.images = []
        self.labels = []
        
        real_dir = os.path.join(data_dir, 'real')
        fake_dir = os.path.join(data_dir, 'fake')
        
        # Load real images (label = 1)
        if os.path.exists(real_dir):
            for img_name in os.listdir(real_dir):
                if img_name.lower().endswith(('.jpg', '.png', '.jpeg')):
                    self.images.append(os.path.join(real_dir, img_name))
                    self.labels.append(1)
        
        # Load fake images (label = 0)
        if os.path.exists(fake_dir):
            for img_name in os.listdir(fake_dir):
                if img_name.lower().endswith(('.jpg', '.png', '.jpeg')):
                    self.images.append(os.path.join(fake_dir, img_name))
                    self.labels.append(0)
        
        print(f"   Loaded {sum(self.labels)} real, {len(self.labels)-sum(self.labels)} fake images")
    
    def __len__(self):
        return len(self.images)
    
    def __getitem__(self, idx):
        img = Image.open(self.images[idx]).convert('RGB')
        label = self.labels[idx]
        if self.transform:
            img = self.transform(img)
        return img, label


# ============================================================================
# TRAINING FUNCTIONS
# ============================================================================
def train_detector(train_dir, val_dir, num_epochs=2, batch_size=64, save_path='models'):
    """
    Train deepfake detector with ADVERSARIAL GAN training + Supervised fine-tuning
    """
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"🚀 Device: {device}")
    
    os.makedirs(save_path, exist_ok=True)
    os.makedirs('plots', exist_ok=True)
    os.makedirs('generated_images', exist_ok=True)
    
    # Data transforms
    train_transform = transforms.Compose([
        transforms.Resize((64, 64)),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomRotation(10),
        transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.1),
        transforms.ToTensor(),
        transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
    ])
    
    val_transform = transforms.Compose([
        transforms.Resize((64, 64)),
        transforms.ToTensor(),
        transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
    ])
    
    print("\n📂 Loading datasets...")
    print("Training set:")
    train_dataset = DeepfakeDataset(train_dir, train_transform)
    print("Validation set:")
    val_dataset = DeepfakeDataset(val_dir, val_transform)
    
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=2)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=2)
    
    # Initialize BOTH networks for GAN training
    generator = Generator(latent_dim=128).to(device)
    discriminator = Discriminator().to(device)
    
    criterion = nn.BCELoss()
    optimizer_G = optim.Adam(generator.parameters(), lr=0.0002, betas=(0.5, 0.999))
    optimizer_D = optim.Adam(discriminator.parameters(), lr=0.0001, betas=(0.5, 0.999), weight_decay=1e-4)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer_D, mode='max', factor=0.5, patience=3)
    
    # Training history
    history = {
        'train_loss': [],
        'train_acc': [],
        'train_auc': [],
        'train_f1': [],
        'val_loss': [],
        'val_acc': [],
        'val_auc': [],
        'val_f1': [],
        'g_loss': [],
        'd_loss': []
    }
    
    best_val_acc = 0
    
    print("\n" + "="*60)
    print("PHASE 1: ADVERSARIAL GAN TRAINING")
    print("="*60)
    
    for epoch in range(num_epochs):
        # ADVERSARIAL TRAINING PHASE
        generator.train()
        discriminator.train()
        train_loss = 0
        train_correct = 0
        train_total = 0
        train_probs = []
        train_labels_list = []
        train_preds = []
        epoch_g_loss = 0
        epoch_d_loss = 0
        
        for images, labels in tqdm(train_loader, desc=f"Phase1 Epoch {epoch+1}/{num_epochs} [GAN]"):
            batch_size_actual = images.size(0)
            images = images.to(device)
            real_labels = torch.ones(batch_size_actual, 1).to(device)
            fake_labels = torch.zeros(batch_size_actual, 1).to(device)
            
            # Train Discriminator
            optimizer_D.zero_grad()
            outputs_real = discriminator(images)
            d_loss_real = criterion(outputs_real, real_labels)
            
            z = torch.randn(batch_size_actual, 128).to(device)
            fake_images = generator(z)
            outputs_fake = discriminator(fake_images.detach())
            d_loss_fake = criterion(outputs_fake, fake_labels)
            
            d_loss = d_loss_real + d_loss_fake
            d_loss.backward()
            optimizer_D.step()
            epoch_d_loss += d_loss.item()
            
            # Train Generator
            optimizer_G.zero_grad()
            z = torch.randn(batch_size_actual, 128).to(device)
            fake_images = generator(z)
            outputs_fake = discriminator(fake_images)
            g_loss = criterion(outputs_fake, real_labels)
            g_loss.backward()
            optimizer_G.step()
            epoch_g_loss += g_loss.item()
            
            # Track metrics
            train_loss += d_loss_real.item()
            predicted = (outputs_real > 0.5).float()
            train_correct += (predicted == real_labels).sum().item()
            train_total += real_labels.size(0)
            train_probs.extend(outputs_real.detach().cpu().numpy().flatten())
            train_labels_list.extend(labels.numpy())
            train_preds.extend(predicted.cpu().numpy().flatten())
        
        train_loss /= len(train_loader)
        train_acc = 100 * train_correct / train_total
        train_auc = roc_auc_score(train_labels_list, train_probs)
        train_f1 = f1_score(train_labels_list, train_preds, zero_division=0)
        avg_g_loss = epoch_g_loss / len(train_loader)
        avg_d_loss = epoch_d_loss / len(train_loader)
        
        # VALIDATION
        discriminator.eval()
        val_loss = 0
        val_correct = 0
        val_total = 0
        val_probs = []
        val_labels_list = []
        val_preds = []
        
        with torch.no_grad():
            for images, labels in tqdm(val_loader, desc=f"Phase1 Epoch {epoch+1}/{num_epochs} [Val]", leave=False):
                images = images.to(device)
                labels_tensor = labels.float().unsqueeze(1).to(device)
                outputs = discriminator(images)
                loss = criterion(outputs, labels_tensor)
                val_loss += loss.item()
                predicted = (outputs > 0.5).float()
                val_correct += (predicted == labels_tensor).sum().item()
                val_total += labels_tensor.size(0)
                val_probs.extend(outputs.cpu().numpy().flatten())
                val_labels_list.extend(labels.numpy())
                val_preds.extend(predicted.cpu().numpy().flatten())
        
        val_loss /= len(val_loader)
        val_acc = 100 * val_correct / val_total
        val_auc = roc_auc_score(val_labels_list, val_probs)
        val_f1 = f1_score(val_labels_list, val_preds, zero_division=0)
        
        # Save history
        history['train_loss'].append(train_loss)
        history['train_acc'].append(train_acc)
        history['train_auc'].append(train_auc)
        history['train_f1'].append(train_f1)
        history['val_loss'].append(val_loss)
        history['val_acc'].append(val_acc)
        history['val_auc'].append(val_auc)
        history['val_f1'].append(val_f1)
        history['g_loss'].append(avg_g_loss)
        history['d_loss'].append(avg_d_loss)
        
        # Learning rate scheduling
        old_lr = optimizer_D.param_groups[0]['lr']
        scheduler.step(val_acc)
        new_lr = optimizer_D.param_groups[0]['lr']
        
        print(f"Phase1 Epoch {epoch+1}: G_Loss={avg_g_loss:.4f}, D_Loss={avg_d_loss:.4f} | "
              f"Train Acc={train_acc:.2f}%, AUC={train_auc:.4f}, F1={train_f1:.4f} | "
              f"Val Acc={val_acc:.2f}%, AUC={val_auc:.4f}, F1={val_f1:.4f}", end="")
        
        if new_lr != old_lr:
            print(f" | LR: {old_lr:.6f} → {new_lr:.6f}", end="")
        
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save({
                'epoch': epoch,
                'model_state_dict': discriminator.state_dict(),
                'generator_state_dict': generator.state_dict(),
                'val_acc': val_acc,
                'train_acc': train_acc
            }, os.path.join(save_path, 'best_detector.pth'))
            print(" ✓ BEST", end="")
        
        print()
        
        # Generate sample images
        generator.eval()
        with torch.no_grad():
            sample_z = torch.randn(16, 128).to(device)
            sample_images = generator(sample_z)
            save_image(sample_images, f'generated_images/epoch_{epoch+1}.png', 
                      nrow=4, normalize=True, value_range=(-1, 1))
        print(f"   🖼️  Generated images saved: generated_images/epoch_{epoch+1}.png")
        generator.train()
    
    print(f"\n✅ Phase 1 complete! Best val acc: {best_val_acc:.2f}%")
    
    # ========================================================================
    # PHASE 2: SUPERVISED FINE-TUNING - CHANGED TO 3 EPOCHS
    # ========================================================================
    print("\n" + "="*60)
    print("PHASE 2: SUPERVISED FINE-TUNING ON DATASET")
    print("="*60)
    
    # Load best discriminator from Phase 1
    checkpoint = torch.load(os.path.join(save_path, 'best_detector.pth'))
    discriminator.load_state_dict(checkpoint['model_state_dict'])
    
    optimizer_supervised = optim.Adam(discriminator.parameters(), lr=0.0001, weight_decay=1e-4)
    scheduler_supervised = optim.lr_scheduler.ReduceLROnPlateau(optimizer_supervised, mode='max', factor=0.5, patience=3)
    best_val_acc_phase2 = 0
    
    for epoch in range(3):  # CHANGED FROM 5 TO 3
        # SUPERVISED TRAINING
        discriminator.train()
        train_loss = 0
        train_correct = 0
        train_total = 0
        train_probs = []
        train_labels_list = []
        train_preds = []
        
        for images, labels in tqdm(train_loader, desc=f"Phase2 Epoch {epoch+1}/3 [Supervised]"):  # UPDATED DISPLAY
            images = images.to(device)
            labels_tensor = labels.float().unsqueeze(1).to(device)
            
            optimizer_supervised.zero_grad()
            outputs = discriminator(images)
            loss = criterion(outputs, labels_tensor)
            loss.backward()
            optimizer_supervised.step()
            
            train_loss += loss.item()
            predicted = (outputs > 0.5).float()
            train_correct += (predicted == labels_tensor).sum().item()
            train_total += labels_tensor.size(0)
            train_probs.extend(outputs.detach().cpu().numpy().flatten())
            train_labels_list.extend(labels.numpy())
            train_preds.extend(predicted.cpu().numpy().flatten())
        
        train_loss /= len(train_loader)
        train_acc = 100 * train_correct / train_total
        train_auc = roc_auc_score(train_labels_list, train_probs)
        train_f1 = f1_score(train_labels_list, train_preds, zero_division=0)
        
        # VALIDATION
        discriminator.eval()
        val_loss = 0
        val_correct = 0
        val_total = 0
        val_probs = []
        val_labels_list = []
        val_preds = []
        
        with torch.no_grad():
            for images, labels in tqdm(val_loader, desc=f"Phase2 Epoch {epoch+1}/3 [Val]", leave=False):  # UPDATED DISPLAY
                images = images.to(device)
                labels_tensor = labels.float().unsqueeze(1).to(device)
                outputs = discriminator(images)
                loss = criterion(outputs, labels_tensor)
                val_loss += loss.item()
                predicted = (outputs > 0.5).float()
                val_correct += (predicted == labels_tensor).sum().item()
                val_total += labels_tensor.size(0)
                val_probs.extend(outputs.cpu().numpy().flatten())
                val_labels_list.extend(labels.numpy())
                val_preds.extend(predicted.cpu().numpy().flatten())
        
        val_loss /= len(val_loader)
        val_acc = 100 * val_correct / val_total
        val_auc = roc_auc_score(val_labels_list, val_probs)
        val_f1 = f1_score(val_labels_list, val_preds, zero_division=0)
        
        # Update history
        history['train_loss'].append(train_loss)
        history['train_acc'].append(train_acc)
        history['train_auc'].append(train_auc)
        history['train_f1'].append(train_f1)
        history['val_loss'].append(val_loss)
        history['val_acc'].append(val_acc)
        history['val_auc'].append(val_auc)
        history['val_f1'].append(val_f1)
        history['g_loss'].append(0)
        history['d_loss'].append(0)
        
        old_lr = optimizer_supervised.param_groups[0]['lr']
        scheduler_supervised.step(val_acc)
        new_lr = optimizer_supervised.param_groups[0]['lr']
        
        print(f"Phase2 Epoch {epoch+1}: Train Loss={train_loss:.4f}, Acc={train_acc:.2f}%, AUC={train_auc:.4f}, F1={train_f1:.4f} | "
              f"Val Loss={val_loss:.4f}, Acc={val_acc:.2f}%, AUC={val_auc:.4f}, F1={val_f1:.4f}", end="")
        
        if new_lr != old_lr:
            print(f" | LR: {old_lr:.6f} → {new_lr:.6f}", end="")
        
        if val_acc > best_val_acc_phase2:
            best_val_acc_phase2 = val_acc
            torch.save({
                'epoch': num_epochs + epoch,
                'model_state_dict': discriminator.state_dict(),
                'val_acc': val_acc,
                'train_acc': train_acc
            }, os.path.join(save_path, 'best_detector.pth'))
            print(" ✓ BEST", end="")
        
        print()
    
    print(f"\n✅ Phase 2 complete! Best val acc: {best_val_acc_phase2:.2f}%")
    
    # Save final models
    torch.save(discriminator.state_dict(), os.path.join(save_path, 'final_detector.pth'))
    torch.save(generator.state_dict(), os.path.join(save_path, 'final_generator.pth'))
    
    with open(os.path.join(save_path, 'training_history.json'), 'w') as f:
        json.dump(history, f, indent=2)
    
    print(f"\n✅ All training complete!")
    print(f"   Phase 1 (GAN) best val acc: {best_val_acc:.2f}%")
    print(f"   Phase 2 (Supervised) best val acc: {best_val_acc_phase2:.2f}%")
    print(f"   Models saved in: {save_path}/")
    
    # Plot training curves at the very end
    plot_training_curves(history)
    
    return discriminator, history


def evaluate_model(model, dataloader, device, dataset_name="Test"):
    """Comprehensive evaluation with all metrics"""
    model.eval()
    all_preds = []
    all_labels = []
    all_probs = []
    total_loss = 0
    criterion = nn.BCELoss()
    
    print(f"\n📊 Evaluating on {dataset_name} set...")
    
    with torch.no_grad():
        for images, labels in tqdm(dataloader, desc=f"Evaluating {dataset_name}"):
            images = images.to(device)
            labels_tensor = labels.float().unsqueeze(1).to(device)
            
            outputs = model(images)
            loss = criterion(outputs, labels_tensor)
            total_loss += loss.item()
            
            probs = outputs.cpu().numpy().flatten()
            predicted = (outputs > 0.5).cpu().numpy().flatten()
            
            all_probs.extend(probs)
            all_preds.extend(predicted)
            all_labels.extend(labels.numpy())
    
    avg_loss = total_loss / len(dataloader)
    accuracy = accuracy_score(all_labels, all_preds)
    precision = precision_score(all_labels, all_preds, zero_division=0)
    recall = recall_score(all_labels, all_preds, zero_division=0)
    f1 = f1_score(all_labels, all_preds, zero_division=0)
    auc = roc_auc_score(all_labels, all_probs)
    cm = confusion_matrix(all_labels, all_preds)
    
    print(f"\n{dataset_name} Results:")
    print(f"  Loss:      {avg_loss:.4f}")
    print(f"  Accuracy:  {accuracy*100:.2f}%")
    print(f"  Precision: {precision*100:.2f}%")
    print(f"  Recall:    {recall*100:.2f}%")
    print(f"  F1 Score:  {f1*100:.2f}%")
    print(f"  AUC-ROC:   {auc:.4f}")
    print(f"\n  Confusion Matrix:")
    print(f"                Predicted")
    print(f"              Fake  Real")
    print(f"  Actual Fake  {cm[0][0]:4d}  {cm[0][1]:4d}")
    print(f"  Actual Real  {cm[1][0]:4d}  {cm[1][1]:4d}")
    
    metrics = {
        'loss': avg_loss,
        'accuracy': accuracy,
        'precision': precision,
        'recall': recall,
        'f1': f1,
        'auc': auc,
        'confusion_matrix': cm.tolist()
    }
    
    return metrics


def plot_training_curves(history):
    """Plot comprehensive training curves - 2x2 grid with all metrics"""
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))
    
    epochs = range(1, len(history['train_loss']) + 1)
    
    # Plot 1: Loss
    axes[0, 0].plot(epochs, history['train_loss'], 'b-', label='Training Loss', linewidth=2, marker='o', markersize=4)
    axes[0, 0].plot(epochs, history['val_loss'], 'r-', label='Validation Loss', linewidth=2, marker='s', markersize=4)
    axes[0, 0].set_xlabel('Epoch', fontsize=12, fontweight='bold')
    axes[0, 0].set_ylabel('Loss', fontsize=12, fontweight='bold')
    axes[0, 0].set_title('Training and Validation Loss', fontsize=14, fontweight='bold')
    axes[0, 0].legend(fontsize=11, loc='best')
    axes[0, 0].grid(True, alpha=0.3, linestyle='--')
    
    # Plot 2: Accuracy
    axes[0, 1].plot(epochs, history['train_acc'], 'b-', label='Training Accuracy', linewidth=2, marker='o', markersize=4)
    axes[0, 1].plot(epochs, history['val_acc'], 'r-', label='Validation Accuracy', linewidth=2, marker='s', markersize=4)
    axes[0, 1].set_xlabel('Epoch', fontsize=12, fontweight='bold')
    axes[0, 1].set_ylabel('Accuracy (%)', fontsize=12, fontweight='bold')
    axes[0, 1].set_title('Training and Validation Accuracy', fontsize=14, fontweight='bold')
    axes[0, 1].legend(fontsize=11, loc='best')
    axes[0, 1].grid(True, alpha=0.3, linestyle='--')
    
    # Plot 3: AUC-ROC
    axes[1, 0].plot(epochs, history['train_auc'], 'b-', label='Training AUC', linewidth=2, marker='o', markersize=4)
    axes[1, 0].plot(epochs, history['val_auc'], 'r-', label='Validation AUC', linewidth=2, marker='s', markersize=4)
    axes[1, 0].set_xlabel('Epoch', fontsize=12, fontweight='bold')
    axes[1, 0].set_ylabel('AUC-ROC', fontsize=12, fontweight='bold')
    axes[1, 0].set_title('Training and Validation AUC-ROC', fontsize=14, fontweight='bold')
    axes[1, 0].legend(fontsize=11, loc='best')
    axes[1, 0].grid(True, alpha=0.3, linestyle='--')
    axes[1, 0].set_ylim([0, 1.05])
    
    # Plot 4: F1 Score
    axes[1, 1].plot(epochs, history['train_f1'], 'b-', label='Training F1', linewidth=2, marker='o', markersize=4)
    axes[1, 1].plot(epochs, history['val_f1'], 'r-', label='Validation F1', linewidth=2, marker='s', markersize=4)
    axes[1, 1].set_xlabel('Epoch', fontsize=12, fontweight='bold')
    axes[1, 1].set_ylabel('F1 Score', fontsize=12, fontweight='bold')
    axes[1, 1].set_title('Training and Validation F1 Score', fontsize=14, fontweight='bold')
    axes[1, 1].legend(fontsize=11, loc='best')
    axes[1, 1].grid(True, alpha=0.3, linestyle='--')
    axes[1, 1].set_ylim([0, 1.05])
    
    plt.tight_layout(pad=3.0)
    plt.savefig('plots/training_curves.png', dpi=150, bbox_inches='tight')
    print("   📊 Training curves saved: plots/training_curves.png")
    plt.close()
    
    # Also create individual plots for closer inspection
    plot_individual_metrics(history)


def plot_comparison(train_metrics, val_metrics, test_metrics):
    """Plot comparison of train/val/test metrics"""
    metrics_names = ['Accuracy', 'Precision', 'Recall', 'F1 Score', 'AUC-ROC']
    train_values = [train_metrics['accuracy'], train_metrics['precision'], 
                   train_metrics['recall'], train_metrics['f1'], train_metrics['auc']]
    val_values = [val_metrics['accuracy'], val_metrics['precision'], 
                 val_metrics['recall'], val_metrics['f1'], val_metrics['auc']]
    test_values = [test_metrics['accuracy'], test_metrics['precision'], 
                  test_metrics['recall'], test_metrics['f1'], test_metrics['auc']]
    
    x = np.arange(len(metrics_names))
    width = 0.25
    
    fig, ax = plt.subplots(figsize=(12, 6))
    ax.bar(x - width, train_values, width, label='Training', alpha=0.8)
    ax.bar(x, val_values, width, label='Validation', alpha=0.8)
    ax.bar(x + width, test_values, width, label='Test', alpha=0.8)
    
    ax.set_xlabel('Metrics', fontsize=12)
    ax.set_ylabel('Score', fontsize=12)
    ax.set_title('Model Performance: Train vs Validation vs Test', fontsize=14, fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels(metrics_names, fontsize=10)
    ax.legend(fontsize=10)
    ax.grid(True, axis='y', alpha=0.3)
    ax.set_ylim([0, 1.1])
    
    plt.tight_layout()
    plt.savefig('plots/metrics_comparison.png', dpi=150, bbox_inches='tight')
    print("   📊 Metrics comparison saved: plots/metrics_comparison.png")
    plt.close()


# ============================================================================
# MAIN EXECUTION
# ============================================================================
def main():
    parser = argparse.ArgumentParser(description='GAN-based Deepfake Detector')
    parser.add_argument('--mode', type=str, default='train', choices=['train', 'evaluate'],
                       help='Mode: train (full training) or evaluate (load weights and test only)')
    parser.add_argument('--data_dir', type=str, default='data',
                       help='Root directory containing train/val/test folders')
    parser.add_argument('--epochs', type=int, default=20,
                       help='Number of training epochs')
    parser.add_argument('--batch_size', type=int, default=32,
                       help='Batch size')
    parser.add_argument('--model_path', type=str, default='models/best_detector.pth',
                       help='Path to saved model for evaluation')
    
    args = parser.parse_args()
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Setup paths
    train_dir = os.path.join(args.data_dir, 'train')
    val_dir = os.path.join(args.data_dir, 'val')
    test_dir = os.path.join(args.data_dir, 'test')
    
    if args.mode == 'train':
        print("="*60)
        print("MODE: FULL TRAINING")
        print("="*60)
        
        # Train model
        model, history = train_detector(train_dir, val_dir, 
                                       num_epochs=args.epochs,
                                       batch_size=args.batch_size)
        
        # Evaluate on all splits
        test_transform = transforms.Compose([
            transforms.Resize((64, 64)),
            transforms.ToTensor(),
            transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
        ])
        
        print("\n" + "="*60)
        print("FINAL EVALUATION ON ALL SPLITS")
        print("="*60)
        
        # Load best model
        checkpoint = torch.load('models/best_detector.pth')
        model.load_state_dict(checkpoint['model_state_dict'])
        
        train_dataset = DeepfakeDataset(train_dir, test_transform)
        val_dataset = DeepfakeDataset(val_dir, test_transform)
        test_dataset = DeepfakeDataset(test_dir, test_transform)
        
        train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=False)
        val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False)
        test_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False)
        
        train_metrics = evaluate_model(model, train_loader, device, "Training")
        val_metrics = evaluate_model(model, val_loader, device, "Validation")
        test_metrics = evaluate_model(model, test_loader, device, "Test")
        
        # Save all metrics
        all_metrics = {
            'train': train_metrics,
            'val': val_metrics,
            'test': test_metrics
        }
        
        with open('models/evaluation_metrics.json', 'w') as f:
            json.dump(all_metrics, f, indent=2)
        
        print("\n✅ All metrics saved: models/evaluation_metrics.json")
        
        # Create comparison plot
        plot_comparison(train_metrics, val_metrics, test_metrics)
        
    else:  # evaluate mode
        print("="*60)
        print("MODE: EVALUATION ONLY (For Teacher)")
        print("="*60)
        print("⚡ This mode only loads pre-trained weights and evaluates")
        print("   No training required - runs in seconds!\n")
        
        # Load model
        model = Discriminator().to(device)
        checkpoint = torch.load(args.model_path, map_location=device)
        
        if 'model_state_dict' in checkpoint:
            model.load_state_dict(checkpoint['model_state_dict'])
            print(f"✓ Loaded model from epoch {checkpoint.get('epoch', 'unknown')}")
            print(f"  Training accuracy: {checkpoint.get('train_acc', 'N/A'):.2f}%")
            print(f"  Validation accuracy: {checkpoint.get('val_acc', 'N/A'):.2f}%")
        else:
            model.load_state_dict(checkpoint)
            print(f"✓ Loaded model weights from {args.model_path}")
        
        # Evaluate on test set only
        test_transform = transforms.Compose([
            transforms.Resize((64, 64)),
            transforms.ToTensor(),
            transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
        ])
        
        test_dataset = DeepfakeDataset(test_dir, test_transform)
        test_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False)
        
        test_metrics = evaluate_model(model, test_loader, device, "Test")
        
        # Save test metrics
        with open('test_results.json', 'w') as f:
            json.dump({'test': test_metrics}, f, indent=2)
        
        print("\n✅ Test results saved: test_results.json")
        print("\n" + "="*60)
        print("EVALUATION COMPLETE")
        print("="*60)


if __name__ == "__main__":
    # If running without arguments (e.g., in Jupyter), use default training mode
    import sys
    if len(sys.argv) == 1:
        print("💡 No arguments provided - running in TRAINING mode")
        print("   For teacher evaluation mode, run: python script.py --mode evaluate\n")
        sys.argv.append('--mode')
        sys.argv.append('train')
    
    main()