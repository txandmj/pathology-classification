import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as transforms
import torchvision.models as models
from datasets import load_dataset
from PIL import Image
import numpy as np
import os
from tqdm import tqdm
import warnings

warnings.filterwarnings('ignore')


class ColorNormalizer:
    """H&E Stain Color Normalizer"""

    def __init__(self):
        # Reference color matrix for standard H&E staining
        self.target_stains = np.array([
            [0.5626, 0.2159],  # Hematoxylin
            [0.7201, 0.8012],  # Eosin
            [0.4062, 0.5581]  # Background
        ])

    def normalize_he_color(self, image):
        """H&E Color Normalization"""
        try:
            image = image.astype(np.float32)
            od = -np.log((image + 1) / 240)
            od_hat = od[~np.any(od < 0.15, axis=2)]

            if len(od_hat) == 0:
                return image.astype(np.uint8)

            eigvals, eigvecs = np.linalg.eigh(np.cov(od_hat.T))

            if len(eigvals) >= 2:
                projection = np.dot(od.reshape(-1, 3), eigvecs[:, -2:])
                normalized = np.dot(projection, self.target_stains[:2, :].T)
                normalized = np.exp(-normalized) * 240 - 1
                normalized = normalized.reshape(image.shape)
                normalized = np.clip(normalized, 0, 255)
                return normalized.astype(np.uint8)
            else:
                return image.astype(np.uint8)
        except Exception:
            return image.astype(np.uint8)


class NCTCRCPretrainDataset(Dataset):
    """NCTCRC Pre-training Dataset"""

    def __init__(self, hf_dataset, transform=None, color_normalize=True):
        self.dataset = hf_dataset
        self.transform = transform
        self.color_normalize = color_normalize
        self.color_normalizer = ColorNormalizer() if color_normalize else None
        self.class_names = ['ADI', 'BACK', 'DEB', 'LYM', 'MUC', 'MUS', 'NORM', 'STR', 'TUM']
        print(f"NCTCRC Pre-training Dataset - 9 classes:")
        for i, name in enumerate(self.class_names):
            print(f"  {i}: {name}")

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        sample = self.dataset[idx]
        image = sample['image']
        label_str = sample['label']

        if image.mode != 'RGB':
            image = image.convert('RGB')

        if self.color_normalize and self.color_normalizer:
            image_array = np.array(image)
            image_array = self.color_normalizer.normalize_he_color(image_array)
            image = Image.fromarray(image_array)

        if self.transform:
            image = self.transform(image)

        label_idx = self.class_names.index(label_str)
        return image, torch.tensor(label_idx, dtype=torch.long)


class PathologyFeatureExtractor(nn.Module):
    """Pathology Feature Extractor - Pre-trained model"""

    def __init__(self, backbone='resnet50', num_pretrain_classes=9):
        super(PathologyFeatureExtractor, self).__init__()
        self.backbone_name = backbone
        if backbone == 'resnet50':
            self.backbone = models.resnet50(pretrained=True)
            self.feature_dim = self.backbone.fc.in_features
            self.backbone.fc = nn.Identity()
        # This classifier head is necessary to load the state_dict correctly
        self.pretrain_classifier = nn.Sequential(
            nn.Dropout(0.5),
            nn.Linear(self.feature_dim, 512),
            nn.ReLU(inplace=True),
            nn.BatchNorm1d(512),
            nn.Dropout(0.3),
            nn.Linear(512, num_pretrain_classes)
        )
        print(f"Created PathologyFeatureExtractor with Backbone: {backbone}")

    def forward(self, x):
        # This model is used for both training and feature extraction later
        features = self.backbone(x)
        return self.pretrain_classifier(features)


def get_pretrain_transforms():
    """Data transforms for the pre-training phase"""
    train_transform = transforms.Compose([
        transforms.Resize((256, 256)),
        transforms.RandomCrop(224),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomVerticalFlip(p=0.5),
        transforms.RandomRotation(degrees=90),
        transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        transforms.RandomErasing(p=0.1)
    ])
    val_transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    return train_transform, val_transform


def pretrain_on_nctcrc(model, num_epochs=15, batch_size=32, lr=1e-4, output_dir='../models'):
    """Pre-train the feature extractor on the NCTCRC dataset"""
    print("🔄 Step 1: NCTCRC-HE-100K Pre-training...")
    print("=" * 50)

    # Load dataset
    try:
        print("Downloading NCTCRC dataset...")
        dataset = load_dataset("DykeF/NCTCRCHE100K", trust_remote_code=True)
        train_dataset_hf = dataset['train']
        if len(train_dataset_hf) > 30000:
            print("Using a 30K subset for pre-training...")
            train_dataset_hf = train_dataset_hf.shuffle(seed=42).select(range(30000))
        print(f"Pre-training data size: {len(train_dataset_hf)} images")
    except Exception as e:
        print(f"❌ Dataset loading failed: {e}")
        return None, None

    # Data transforms and datasets
    train_transform, val_transform = get_pretrain_transforms()
    full_dataset = NCTCRCPretrainDataset(train_dataset_hf, transform=train_transform)
    train_size = int(0.85 * len(full_dataset))
    val_size = len(full_dataset) - train_size
    train_dataset, val_dataset = torch.utils.data.random_split(
        full_dataset, [train_size, val_size], generator=torch.Generator().manual_seed(42)
    )
    val_dataset.dataset.transform = val_transform

    # Data loaders
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=4, pin_memory=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=4, pin_memory=True)

    # Setup for training
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = model.to(device)
    optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=num_epochs)
    criterion = nn.CrossEntropyLoss()
    history = {'train_loss': [], 'train_acc': [], 'val_loss': [], 'val_acc': []}
    best_val_acc = 0

    print(f"Starting pre-training on {device} for {num_epochs} epochs...")

    for epoch in range(num_epochs):
        model.train()
        train_loss, train_correct, train_total = 0, 0, 0
        train_pbar = tqdm(train_loader, desc=f'Pretrain Epoch {epoch + 1}/{num_epochs}')
        for data, target in train_pbar:
            data, target = data.to(device), target.to(device)
            optimizer.zero_grad()
            output = model(data)
            loss = criterion(output, target)
            loss.backward()
            optimizer.step()
            train_loss += loss.item()
            _, predicted = torch.max(output, 1)
            train_total += target.size(0)
            train_correct += predicted.eq(target).sum().item()
            train_pbar.set_postfix({'Loss': f'{loss.item():.4f}', 'Acc': f'{100. * train_correct / train_total:.2f}%'})

        model.eval()
        val_loss, val_correct, val_total = 0, 0, 0
        with torch.no_grad():
            for data, target in val_loader:
                data, target = data.to(device), target.to(device)
                output = model(data)
                val_loss += criterion(output, target).item()
                _, predicted = torch.max(output, 1)
                val_total += target.size(0)
                val_correct += predicted.eq(target).sum().item()

        # Log metrics
        train_acc = 100. * train_correct / train_total
        val_acc = 100. * val_correct / val_total
        history['train_loss'].append(train_loss / len(train_loader))
        history['train_acc'].append(train_acc)
        history['val_loss'].append(val_loss / len(val_loader))
        history['val_acc'].append(val_acc)
        print(f'Epoch {epoch + 1}/{num_epochs}: Train Acc: {train_acc:.2f}%, Val Acc: {val_acc:.2f}%')

        # Save the best model
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            os.makedirs(output_dir, exist_ok=True)
            model_path = os.path.join(output_dir, 'pretrained_nctcrc_model.pth')
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'val_acc': val_acc,
                'history': history
            }, model_path)
            print(f'  ✅ Best model saved to {model_path} (Val Acc: {best_val_acc:.2f}%)')
        scheduler.step()

    print(f"🎯 Pre-training complete! Best validation accuracy: {best_val_acc:.2f}%")
    return model, history


def main():
    """Main function to run the pre-training workflow"""
    print("🎯 Pathology Image Classification: Pre-training Pipeline")
    print("=" * 60)
    print("📋 This script will pre-train a feature extractor on NCT-CRC-HE-100K.")
    print("📋 The output ('pretrained_nctcrc_model.pth') is used by the feature classification script.")

    # Define output directory for the model
    model_dir = '../models'
    model_path = os.path.join(model_dir, 'pretrained_nctcrc_model.pth')

    try:
        model = PathologyFeatureExtractor(backbone='resnet50')

        # Check if a pre-trained model already exists
        if os.path.exists(model_path):
            print(f"✅ Found an existing pre-trained model at '{model_path}'.")
            response = input("Enter 'y' to retrain, or press Enter to skip: ")
            if response.lower() != 'y':
                print("Skipping pre-training. The existing model will be used by the next script.")
                return

        print("\n🚀 Starting the pre-training workflow...")
        pretrain_on_nctcrc(model, num_epochs=15, batch_size=32, output_dir=model_dir)

        print(f"\n🎉 Pre-training workflow executed successfully!")
        print(f"📊 The model is saved at: {model_path}")
        print("➡️ Next step: Run 'corrected_feature_classifier.py' to use this model.")

    except Exception as e:
        print(f"\n❌ An error occurred during execution: {e}")
        import traceback
        print("Detailed error information:")
        print(traceback.format_exc())


if __name__ == '__main__':
    print("📦 Ensure the following dependencies are installed:")
    print("pip install torch torchvision datasets huggingface-hub tqdm scikit-learn opencv-python pillow numpy")
    print("-" * 60)
    main()
