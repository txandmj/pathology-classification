import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as transforms
import torchvision.models as models
from datasets import load_dataset
from PIL import Image
import numpy as np
import pandas as pd
import cv2
import os
from sklearn.metrics import roc_auc_score, classification_report, confusion_matrix
from sklearn.model_selection import StratifiedKFold, train_test_split
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm
import warnings
import json
from datetime import datetime

warnings.filterwarnings('ignore')


class ColorNormalizer:
    """H&E Stain Color Normalizer"""

    def __init__(self):
        # Reference color matrix for standard H&E staining
        # Based on the method by Macenko et al.
        self.target_stains = np.array([
            [0.5626, 0.2159],  # Hematoxylin
            [0.7201, 0.8012],  # Eosin
            [0.4062, 0.5581]  # Background
        ])

    def normalize_he_color(self, image):
        """H&E Color Normalization"""
        try:
            # Convert to float
            image = image.astype(np.float32)

            # Convert to Optical Density (OD) space
            od = -np.log((image + 1) / 240)

            # Remove background
            od_hat = od[~np.any(od < 0.15, axis=2)]

            if len(od_hat) == 0:
                return image.astype(np.uint8)

            # Calculate eigenvectors of the covariance matrix
            eigvals, eigvecs = np.linalg.eigh(np.cov(od_hat.T))

            # Get the two main stain directions
            if len(eigvals) >= 2:
                # Project into the standard stain space
                projection = np.dot(od.reshape(-1, 3), eigvecs[:, -2:])

                # Normalize
                normalized = np.dot(projection, self.target_stains[:2, :].T)
                normalized = np.exp(-normalized) * 240 - 1

                # Reshape back to the original shape
                normalized = normalized.reshape(image.shape)
                normalized = np.clip(normalized, 0, 255)

                return normalized.astype(np.uint8)
            else:
                return image.astype(np.uint8)

        except Exception:
            # If normalization fails, return the original image
            return image.astype(np.uint8)


class NCTCRCPretrainDataset(Dataset):
    """NCTCRC Pre-training Dataset"""

    def __init__(self, hf_dataset, transform=None, color_normalize=True):
        self.dataset = hf_dataset
        self.transform = transform
        self.color_normalize = color_normalize
        self.color_normalizer = ColorNormalizer() if color_normalize else None

        # 9 classes of the NCTCRC-HE-100K dataset
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

        # Ensure RGB format
        if image.mode != 'RGB':
            image = image.convert('RGB')

        # Color normalization
        if self.color_normalize and self.color_normalizer:
            image_array = np.array(image)
            image_array = self.color_normalizer.normalize_he_color(image_array)
            image = Image.fromarray(image_array)

        if self.transform:
            image = self.transform(image)

        # Find the integer index corresponding to the string label
        label_idx = self.class_names.index(label_str)

        return image, torch.tensor(label_idx, dtype=torch.long)


class LNMFinetuneDataset(Dataset):
    """LNM Fine-tuning Dataset - includes preprocessing"""

    def __init__(self, image_paths, labels, transform=None, color_normalize=True):
        self.image_paths = image_paths
        self.labels = labels
        self.transform = transform
        self.color_normalize = color_normalize
        self.color_normalizer = ColorNormalizer() if color_normalize else None

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        image_path = self.image_paths[idx]
        label = self.labels[idx]

        # Read image
        image = cv2.imread(image_path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        # Color normalization
        if self.color_normalize and self.color_normalizer:
            image = self.color_normalizer.normalize_he_color(image)

        image = Image.fromarray(image)

        if self.transform:
            image = self.transform(image)

        return image, torch.tensor(label, dtype=torch.long), image_path


class PathologyFeatureExtractor(nn.Module):
    """Pathology Feature Extractor - Pre-trained model"""

    def __init__(self, backbone='resnet50', num_pretrain_classes=9):
        super(PathologyFeatureExtractor, self).__init__()

        self.backbone_name = backbone

        if backbone == 'resnet50':
            self.backbone = models.resnet50(pretrained=True)
            self.feature_dim = self.backbone.fc.in_features
            self.backbone.fc = nn.Identity()  # Remove classification layer

        elif backbone == 'resnet18':
            self.backbone = models.resnet18(pretrained=True)
            self.feature_dim = self.backbone.fc.in_features
            self.backbone.fc = nn.Identity()

        # Pre-training classification head (for NCTCRC pre-training)
        self.pretrain_classifier = nn.Sequential(
            nn.Dropout(0.5),
            nn.Linear(self.feature_dim, 512),
            nn.ReLU(inplace=True),
            nn.BatchNorm1d(512),
            nn.Dropout(0.3),
            nn.Linear(512, num_pretrain_classes)
        )

        # Fine-tuning classification head (for LNM classification)
        self.finetune_classifier = nn.Sequential(
            nn.Dropout(0.3),  # Lighter regularization due to small data size
            nn.Linear(self.feature_dim, 128),
            nn.ReLU(inplace=True),
            nn.Dropout(0.2),
            nn.Linear(128, 2)  # LNM vs non-LNM
        )

        print(f"Created PathologyFeatureExtractor:")
        print(f"  Backbone: {backbone}")
        print(f"  Feature dimension: {self.feature_dim}")
        print(f"  Pretrain classes: {num_pretrain_classes}")
        print(f"  Finetune classes: 2")

    def forward(self, x, mode='pretrain'):
        # Feature extraction
        features = self.backbone(x)

        if mode == 'pretrain':
            return self.pretrain_classifier(features)
        elif mode == 'finetune':
            return self.finetune_classifier(features)
        elif mode == 'features':
            return features
        else:
            raise ValueError(f"Unknown mode: {mode}")


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


def get_finetune_transforms():
    """Data transforms for the fine-tuning phase - more conservative"""

    train_transform = transforms.Compose([
        transforms.Resize((256, 256)),
        transforms.RandomCrop(224),
        transforms.RandomHorizontalFlip(p=0.3),  # Reduce randomness
        transforms.RandomVerticalFlip(p=0.3),
        transforms.RandomRotation(degrees=15),  # Smaller rotation
        transforms.ColorJitter(brightness=0.1, contrast=0.1, saturation=0.1),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    val_transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    return train_transform, val_transform


def pretrain_on_nctcrc(model, num_epochs=15, batch_size=32, lr=1e-4):
    """Step 1: Pre-train on the NCTCRC dataset"""

    print("🔄 Step 1: NCTCRC-HE-100K Pre-training...")
    print("=" * 50)

    # Load dataset
    try:
        print("Downloading NCTCRC dataset...")
        dataset = load_dataset("DykeF/NCTCRCHE100K", trust_remote_code=True)
        train_dataset_hf = dataset['train']

        # Use a subset to save time (adjustable)
        if len(train_dataset_hf) > 30000:
            print("Using a 30K subset for pre-training...")
            train_dataset_hf = train_dataset_hf.shuffle(seed=42).select(range(30000))

        print(f"Pre-training data size: {len(train_dataset_hf)} images")

    except Exception as e:
        print(f"❌ Dataset loading failed: {e}")
        return None, None

    # Data transforms
    train_transform, val_transform = get_pretrain_transforms()

    # Create dataset
    full_dataset = NCTCRCPretrainDataset(
        train_dataset_hf,
        transform=train_transform,
        color_normalize=True
    )

    # Split into training and validation sets
    train_size = int(0.85 * len(full_dataset))
    val_size = len(full_dataset) - train_size

    train_dataset, val_dataset = torch.utils.data.random_split(
        full_dataset, [train_size, val_size],
        generator=torch.Generator().manual_seed(42)
    )

    # Apply different transform to the validation set
    val_dataset.dataset.transform = val_transform

    # Data loaders
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True,
                              num_workers=4, pin_memory=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False,
                            num_workers=4, pin_memory=True)

    # Set device and optimizer
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = model.to(device)

    optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=num_epochs)
    criterion = nn.CrossEntropyLoss()

    # Training history
    history = {'train_loss': [], 'train_acc': [], 'val_loss': [], 'val_acc': []}

    best_val_acc = 0

    print(f"Starting pre-training - {num_epochs} epochs...")

    for epoch in range(num_epochs):
        # Training phase
        model.train()
        train_loss = 0
        train_correct = 0
        train_total = 0

        train_pbar = tqdm(train_loader, desc=f'Pretrain Epoch {epoch + 1}/{num_epochs}')
        for batch_idx, (data, target) in enumerate(train_pbar):
            data, target = data.to(device), target.to(device)

            optimizer.zero_grad()
            output = model(data, mode='pretrain')
            loss = criterion(output, target)
            loss.backward()
            optimizer.step()

            train_loss += loss.item()
            _, predicted = torch.max(output, 1)
            train_total += target.size(0)
            train_correct += predicted.eq(target).sum().item()

            train_pbar.set_postfix({
                'Loss': f'{loss.item():.4f}',
                'Acc': f'{100. * train_correct / train_total:.2f}%'
            })

        # Validation phase
        model.eval()
        val_loss = 0
        val_correct = 0
        val_total = 0

        with torch.no_grad():
            for data, target in val_loader:
                data, target = data.to(device), target.to(device)
                output = model(data, mode='pretrain')
                val_loss += criterion(output, target).item()

                _, predicted = torch.max(output, 1)
                val_total += target.size(0)
                val_correct += predicted.eq(target).sum().item()

        # Calculate metrics
        train_loss /= len(train_loader)
        val_loss /= len(val_loader)
        train_acc = 100. * train_correct / train_total
        val_acc = 100. * val_correct / val_total

        history['train_loss'].append(train_loss)
        history['train_acc'].append(train_acc)
        history['val_loss'].append(val_loss)
        history['val_acc'].append(val_acc)

        print(f'Epoch {epoch + 1}/{num_epochs}:')
        print(f'  Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.2f}%')
        print(f'  Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.2f}%')

        # Save the best model
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'val_acc': val_acc,
                'history': history
            }, 'pretrained_nctcrc_model.pth')
            print(f'  ✅ Best model saved (Val Acc: {best_val_acc:.2f}%)')

        scheduler.step()
        print('-' * 60)

    print(f"🎯 Pre-training complete! Best validation accuracy: {best_val_acc:.2f}%")
    return model, history


def finetune_on_lnm(pretrained_model_path, lnm_image_paths, lnm_labels,
                    num_epochs=20, lr=1e-5, test_size=0.2):
    """Step 2: Fine-tune on the LNM data"""

    print("\n🎯 Step 2: LNM Data Fine-tuning...")
    print("=" * 50)

    # Load pre-trained model
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = PathologyFeatureExtractor(backbone='resnet50', num_pretrain_classes=9)

    checkpoint = torch.load(pretrained_model_path, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    model = model.to(device)

    print("✅ Pre-trained model loaded successfully")
    print(f"Pre-trained best validation accuracy: {checkpoint['val_acc']:.2f}%")

    # Split data: train/validation vs. test
    X_trainval, X_test, y_trainval, y_test = train_test_split(
        lnm_image_paths, lnm_labels,
        test_size=test_size,
        stratify=lnm_labels,
        random_state=42
    )

    print(f"Data split:")
    print(f"  Train+Validation: {len(X_trainval)} images")
    print(f"  Test: {len(X_test)} images")

    # 5-fold cross-validation for fine-tuning
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

    cv_results = []
    fold_models = []

    train_transform, val_transform = get_finetune_transforms()

    for fold, (train_idx, val_idx) in enumerate(skf.split(X_trainval, y_trainval)):
        print(f"\n--- Fine-tuning Fold {fold + 1}/5 ---")

        # Prepare fold data
        fold_train_paths = [X_trainval[i] for i in train_idx]
        fold_train_labels = [y_trainval[i] for i in train_idx]
        fold_val_paths = [X_trainval[i] for i in val_idx]
        fold_val_labels = [y_trainval[i] for i in val_idx]

        print(f"Training: {len(fold_train_paths)}, Validation: {len(fold_val_paths)}")

        # Create datasets
        train_dataset = LNMFinetuneDataset(fold_train_paths, fold_train_labels,
                                           transform=train_transform, color_normalize=True)
        val_dataset = LNMFinetuneDataset(fold_val_paths, fold_val_labels,
                                         transform=val_transform, color_normalize=True)

        train_loader = DataLoader(train_dataset, batch_size=8, shuffle=True, num_workers=0)
        val_loader = DataLoader(val_dataset, batch_size=8, shuffle=False, num_workers=0)

        # Reload pre-trained weights
        model_fold = PathologyFeatureExtractor(backbone='resnet50', num_pretrain_classes=9)
        model_fold.load_state_dict(checkpoint['model_state_dict'])
        model_fold = model_fold.to(device)

        # Freeze early layers of the backbone
        for name, param in model_fold.backbone.named_parameters():
            if 'layer4' not in name and 'fc' not in name:  # Only fine-tune the final layer4
                param.requires_grad = False

        # Optimizer - very small learning rate
        optimizer = optim.AdamW(filter(lambda p: p.requires_grad, model_fold.parameters()),
                                lr=lr, weight_decay=1e-3)
        scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.5)
        criterion = nn.CrossEntropyLoss()

        # Train the fine-tuning model
        best_val_auc = 0
        patience = 5
        patience_counter = 0

        for epoch in range(num_epochs):
            # Training
            model_fold.train()
            train_loss = 0

            for data, target, _ in train_loader:
                data, target = data.to(device), target.to(device)

                optimizer.zero_grad()
                output = model_fold(data, mode='finetune')
                loss = criterion(output, target)
                loss.backward()
                optimizer.step()

                train_loss += loss.item()

            # Validation
            model_fold.eval()
            val_predictions = []
            val_probabilities = []
            val_targets = []

            with torch.no_grad():
                for data, target, _ in val_loader:
                    data, target = data.to(device), target.to(device)
                    output = model_fold(data, mode='finetune')

                    probs = torch.softmax(output, dim=1)
                    preds = torch.argmax(output, dim=1)

                    val_predictions.extend(preds.cpu().numpy())
                    val_probabilities.extend(probs[:, 1].cpu().numpy())
                    val_targets.extend(target.cpu().numpy())

            # Calculate AUC
            if len(set(val_targets)) > 1:
                val_auc = roc_auc_score(val_targets, val_probabilities)
            else:
                val_auc = 0.5

            print(f'  Epoch {epoch + 1}: Train Loss: {train_loss / len(train_loader):.4f}, Val AUC: {val_auc:.4f}')

            # Early stopping
            if val_auc > best_val_auc:
                best_val_auc = val_auc
                patience_counter = 0
                # Save the best model for this fold
                torch.save(model_fold.state_dict(), f'finetune_fold_{fold}_model.pth')
            else:
                patience_counter += 1

            if patience_counter >= patience:
                print(f'  Early stopping at epoch {epoch + 1}')
                break

            scheduler.step()

        cv_results.append(best_val_auc)
        fold_models.append(f'finetune_fold_{fold}_model.pth')
        print(f'Fold {fold + 1} Best AUC: {best_val_auc:.4f}')

    # Cross-validation results
    mean_cv_auc = np.mean(cv_results)
    std_cv_auc = np.std(cv_results)

    print(f"\n📊 5-Fold Cross-Validation Results:")
    for i, auc in enumerate(cv_results):
        print(f"  Fold {i + 1}: AUC = {auc:.4f}")
    print(f"  Mean AUC: {mean_cv_auc:.4f} ± {std_cv_auc:.4f}")

    return fold_models, X_test, y_test, cv_results


def evaluate_on_test_set(fold_models, X_test, y_test):
    """Step 3: Evaluate on the hold-out test set"""

    print(f"\n🧪 Step 3: Hold-out Test Set Evaluation...")
    print("=" * 50)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    _, test_transform = get_finetune_transforms()

    # Test dataset
    test_dataset = LNMFinetuneDataset(X_test, y_test, transform=test_transform, color_normalize=True)
    test_loader = DataLoader(test_dataset, batch_size=8, shuffle=False, num_workers=0)

    # Ensemble predictions
    all_fold_predictions = []
    all_fold_probabilities = []

    for i, model_path in enumerate(fold_models):
        print(f"Loading Fold {i + 1} model...")

        model = PathologyFeatureExtractor(backbone='resnet50', num_pretrain_classes=9)
        model.load_state_dict(torch.load(model_path, map_location=device))
        model = model.to(device)
        model.eval()

        fold_predictions = []
        fold_probabilities = []

        with torch.no_grad():
            for data, _, paths in test_loader:
                data = data.to(device)
                output = model(data, mode='finetune')

                probs = torch.softmax(output, dim=1)
                preds = torch.argmax(output, dim=1)

                fold_predictions.extend(preds.cpu().numpy())
                fold_probabilities.extend(probs[:, 1].cpu().numpy())

        all_fold_predictions.append(fold_predictions)
        all_fold_probabilities.append(fold_probabilities)

    # Ensemble predictions - average probabilities
    ensemble_probabilities = np.mean(all_fold_probabilities, axis=0)
    ensemble_predictions = (ensemble_probabilities > 0.5).astype(int)

    # Calculate metrics
    from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

    accuracy = accuracy_score(y_test, ensemble_predictions)
    precision = precision_score(y_test, ensemble_predictions, zero_division=0)
    recall = recall_score(y_test, ensemble_predictions, zero_division=0)
    f1 = f1_score(y_test, ensemble_predictions, zero_division=0)

    if len(set(y_test)) > 1:
        auc = roc_auc_score(y_test, ensemble_probabilities)
    else:
        auc = 0.5

    print(f"\n🏆 Final Test Results (Hold-out + Ensemble):")
    print(f"  Accuracy: {accuracy:.4f} ({accuracy * 100:.1f}%)")
    print(f"  Precision: {precision:.4f}")
    print(f"  Recall: {recall:.4f}")
    print(f"  F1-Score: {f1:.4f}")
    print(f"  AUC: {auc:.4f}")

    # Detailed results
    print(f"\n📋 Detailed Test Results:")
    for i, (path, true_label, pred_label, prob) in enumerate(
            zip(X_test, y_test, ensemble_predictions, ensemble_probabilities)):
        filename = os.path.basename(path)
        true_class = "LNM" if true_label == 1 else "non-LNM"
        pred_class = "LNM" if pred_label == 1 else "non-LNM"
        status = "✅" if true_label == pred_label else "❌"

        print(f"  {status} {filename}: {pred_class} (conf:{prob:.3f}) vs {true_class}")

    return accuracy, auc, ensemble_predictions, ensemble_probabilities


def create_final_report(pretrain_history, cv_results, test_accuracy, test_auc):
    """Generate the final report"""

    report = {
        'timestamp': datetime.now().isoformat(),
        'methodology': 'Pretrain + Finetune',
        'pretrain_dataset': 'NCT-CRC-HE-100K',
        'pretrain_performance': {
            'best_val_acc': max(pretrain_history['val_acc']) if pretrain_history else 'N/A'
        },
        'finetune_cv_results': {
            'fold_aucs': cv_results,
            'mean_auc': np.mean(cv_results),
            'std_auc': np.std(cv_results),
            'cv_percent': (np.std(cv_results) / np.mean(cv_results)) * 100
        },
        'final_test_performance': {
            'accuracy': test_accuracy,
            'auc': test_auc
        },
        'comparison_with_baseline': {
            'small_dataset_dl': '0.57 ± 0.11',
            'traditional_ml': '0.68 ± 0.37',
            'pretrain_finetune': f'{test_accuracy:.2f} (single test)'
        }
    }

    # Save report
    with open('pathology_model_report.json', 'w') as f:
        json.dump(report, f, indent=2)

    print(f"\n" + "=" * 80)
    print("📋 Final Performance Report")
    print("=" * 80)

    print(f"🔬 Method: Pre-train + Fine-tune")
    print(f"📊 Pre-training Dataset: NCT-CRC-HE-100K")
    if pretrain_history:
        print(f"📈 Pre-training Best Validation Accuracy: {max(pretrain_history['val_acc']):.2f}%")

    print(f"\n🎯 Fine-tuning Cross-Validation:")
    for i, auc in enumerate(cv_results):
        print(f"  Fold {i + 1}: AUC = {auc:.4f}")
    print(f"  Mean CV AUC: {np.mean(cv_results):.4f} ± {np.std(cv_results):.4f}")
    print(f"  CV Stability: {(np.std(cv_results) / np.mean(cv_results)) * 100:.1f}%")

    print(f"\n🏆 Final Test Performance:")
    print(f"  Hold-out Test Accuracy: {test_accuracy:.4f} ({test_accuracy * 100:.1f}%)")
    print(f"  Hold-out Test AUC: {test_auc:.4f}")

    print(f"\n📊 Comparison with Baseline Methods:")
    print(f"  Small Dataset Deep Learning: 0.57 ± 0.11 (CV: 20%)")
    print(f"  Traditional Machine Learning:  0.68 ± 0.37 (CV: 54%)")
    print(f"  Pre-train + Fine-tune:       {test_accuracy:.2f} (Hold-out)")

    # Performance improvement analysis
    baseline_performance = 0.57  # Conservative deep learning baseline
    improvement = ((test_accuracy - baseline_performance) / baseline_performance) * 100

    if test_accuracy > 0.85:
        print(f"\n🎉 Excellent! The pre-train + fine-tune strategy significantly improved performance!")
        print(f"   Relative Improvement: {improvement:+.1f}%")
    elif test_accuracy > 0.75:
        print(f"\n✅ Good! The pre-training strategy brought a clear improvement!")
        print(f"   Relative Improvement: {improvement:+.1f}%")
    elif test_accuracy > 0.65:
        print(f"\n⚠️ Moderate improvement, suggest further optimization of the pre-training strategy")
        print(f"   Relative Improvement: {improvement:+.1f}%")
    else:
        print(f"\n❌ Strategy needs adjustment, may require longer pre-training or different fine-tuning methods")

    print(f"\n💡 Next Step Recommendations:")
    if test_auc > 0.8:
        print(f"1. ✅ Model has reached a clinically useful level")
        print(f"2. 🔄 Consider collecting more LNM data for further fine-tuning")
        print(f"3. 📊 Validate on a larger, independent test set")
    else:
        print(f"1. 🎯 Increase pre-training epochs or use a larger NCTCRC subset")
        print(f"2. 🔧 Try different fine-tuning learning rates or architectures")
        print(f"3. 📈 Consider using more data augmentation techniques")

    print(f"\n📄 Detailed report saved to: pathology_model_report.json")


def visualize_training_results(pretrain_history, cv_results):
    """Visualize the training results"""

    fig, axes = plt.subplots(2, 3, figsize=(18, 10))

    # Pre-training learning curves
    if pretrain_history:
        axes[0, 0].plot(pretrain_history['train_loss'], 'b-', label='Train Loss', alpha=0.8)
        axes[0, 0].plot(pretrain_history['val_loss'], 'r-', label='Val Loss', alpha=0.8)
        axes[0, 0].set_title('Pretraining Loss Curves')
        axes[0, 0].set_xlabel('Epoch')
        axes[0, 0].set_ylabel('Loss')
        axes[0, 0].legend()
        axes[0, 0].grid(True, alpha=0.3)

        axes[0, 1].plot(pretrain_history['train_acc'], 'b-', label='Train Acc', alpha=0.8)
        axes[0, 1].plot(pretrain_history['val_acc'], 'r-', label='Val Acc', alpha=0.8)
        axes[0, 1].set_title('Pretraining Accuracy Curves')
        axes[0, 1].set_xlabel('Epoch')
        axes[0, 1].set_ylabel('Accuracy (%)')
        axes[0, 1].legend()
        axes[0, 1].grid(True, alpha=0.3)

    # Cross-validation results
    axes[0, 2].bar(range(1, len(cv_results) + 1), cv_results, alpha=0.8, color='skyblue')
    axes[0, 2].axhline(y=np.mean(cv_results), color='red', linestyle='--',
                       label=f'Mean: {np.mean(cv_results):.3f}')
    axes[0, 2].axhline(y=0.8, color='green', linestyle='--', alpha=0.7, label='Clinical Threshold')
    axes[0, 2].set_title('Cross-Validation AUC Results')
    axes[0, 2].set_xlabel('Fold')
    axes[0, 2].set_ylabel('AUC')
    axes[0, 2].legend()
    axes[0, 2].grid(True, alpha=0.3)

    # Method comparison
    methods = ['Small Dataset\nDeep Learning', 'Traditional\nMachine Learning', 'Pretrain +\nFinetune']
    performances = [0.57, 0.68, np.mean(cv_results)]
    stds = [0.11, 0.37, np.std(cv_results)]
    colors = ['lightcoral', 'lightblue', 'lightgreen']

    axes[1, 0].bar(methods, performances, yerr=stds, capsize=5, alpha=0.8, color=colors)
    axes[1, 0].axhline(y=0.8, color='red', linestyle='--', alpha=0.7, label='Clinical Threshold')
    axes[1, 0].set_title('Method Comparison')
    axes[1, 0].set_ylabel('AUC')
    axes[1, 0].legend()
    axes[1, 0].grid(True, alpha=0.3)
    plt.setp(axes[1, 0].xaxis.get_majorticklabels(), rotation=45)

    # Stability analysis
    cv_coefficients = [
        (0.11 / 0.57) * 100,  # Small dataset DL
        (0.37 / 0.68) * 100,  # Traditional ML
        (np.std(cv_results) / np.mean(cv_results)) * 100  # Pretrain + Finetune
    ]

    colors_stability = ['green' if cv < 15 else 'orange' if cv < 30 else 'red' for cv in cv_coefficients]

    axes[1, 1].bar(methods, cv_coefficients, alpha=0.8, color=colors_stability)
    axes[1, 1].axhline(y=15, color='green', linestyle='--', alpha=0.7, label='Excellent (<15%)')
    axes[1, 1].axhline(y=30, color='orange', linestyle='--', alpha=0.7, label='Good (<30%)')
    axes[1, 1].set_title('Stability Comparison (CV %)')
    axes[1, 1].set_ylabel('Coefficient of Variation (%)')
    axes[1, 1].legend()
    axes[1, 1].grid(True, alpha=0.3)
    plt.setp(axes[1, 1].xaxis.get_majorticklabels(), rotation=45)

    # Learning process summary
    axes[1, 2].axis('off')
    summary_text = f"""
    🎯 Pre-train + Fine-tune Strategy Summary

    📚 Pre-training Phase:
    • Dataset: NCT-CRC-HE-100K
    • Goal: Learn pathology feature representations
    • Result: {"Success" if pretrain_history else "In Progress"}

    🎯 Fine-tuning Phase:
    • Data: 34 LNM images
    • Strategy: 5-Fold Cross-Validation
    • Mean AUC: {np.mean(cv_results):.3f}
    • Stability: {(np.std(cv_results) / np.mean(cv_results)) * 100:.1f}%

    📊 Performance Improvement:
    • Baseline: 0.57 ± 0.11
    • Current: {np.mean(cv_results):.2f} ± {np.std(cv_results):.2f}
    • Improvement: {((np.mean(cv_results) - 0.57) / 0.57) * 100:+.1f}%

    🏥 Clinical Assessment:
    {'✅ Reached clinical standard' if np.mean(cv_results) > 0.8 else '⚠️ Approaching clinical standard' if np.mean(cv_results) > 0.7 else '❌ Further optimization needed'}
    """

    axes[1, 2].text(0.05, 0.95, summary_text, transform=axes[1, 2].transAxes,
                    fontsize=10, verticalalignment='top', fontfamily='monospace',
                    bbox=dict(boxstyle="round,pad=0.5", facecolor="lightblue", alpha=0.8))

    plt.tight_layout()
    plt.savefig('pretrain_finetune_results.png', dpi=150, bbox_inches='tight')
    plt.show()


def main():
    """Main function: Complete pre-training + fine-tuning workflow"""

    print("🎯 Pathology Image Classification: Full Pre-train + Fine-tune Workflow")
    print("=" * 60)
    print("📋 Workflow Overview:")
    print("  1. Pre-train a feature extractor on NCT-CRC-HE-100K")
    print("  2. Fine-tune on your 34 LNM images")
    print("  3. Evaluate the final performance on a hold-out test set")
    print("  4. Generate a complete performance report")

    # Check and prepare LNM data
    print(f"\n📁 Preparing LNM data...")

    # Automatically find the data directory
    data_dirs = ['./Data', './data', 'Data', 'data']
    lnm_dir = None
    non_lnm_dir = None

    for data_dir in data_dirs:
        if os.path.exists(data_dir):
            possible_lnm = ['LNM', 'lnm']
            possible_non_lnm = ['NOT-LNM', 'NON-LNM', 'non-LNM']

            for lnm_name in possible_lnm:
                test_path = os.path.join(data_dir, lnm_name)
                if os.path.exists(test_path):
                    lnm_dir = test_path
                    break

            for non_lnm_name in possible_non_lnm:
                test_path = os.path.join(data_dir, non_lnm_name)
                if os.path.exists(test_path):
                    non_lnm_dir = test_path
                    break

            if lnm_dir and non_lnm_dir:
                break

    if not lnm_dir or not non_lnm_dir:
        print("❌ LNM data directory not found!")
        print("Please ensure the data structure is as follows:")
        print("Data/")
        print("├── LNM/        # LNM images")
        print("└── NOT-LNM/    # non-LNM images")
        return

    # Collect image paths and labels
    image_paths = []
    labels = []

    for img_file in os.listdir(lnm_dir):
        if img_file.lower().endswith(('.jpg', '.jpeg', '.png', '.tiff', '.tif')):
            image_paths.append(os.path.join(lnm_dir, img_file))
            labels.append(1)

    for img_file in os.listdir(non_lnm_dir):
        if img_file.lower().endswith(('.jpg', '.jpeg', '.png', '.tiff', '.tif')):
            image_paths.append(os.path.join(non_lnm_dir, img_file))
            labels.append(0)

    print(f"✅ LNM data preparation complete:")
    print(f"  Total images: {len(image_paths)}")
    print(f"  LNM: {sum(labels)}")
    print(f"  non-LNM: {len(labels) - sum(labels)}")

    if len(image_paths) < 10:
        print("⚠️ Warning: The number of images is very small, which may affect result stability")

    try:
        # Step 1: Pre-training
        print(f"\n🚀 Starting the full workflow...")

        model = PathologyFeatureExtractor(backbone='resnet50', num_pretrain_classes=9)

        # Check if a pre-trained model already exists
        if os.path.exists('pretrained_nctcrc_model.pth'):
            print("✅ Found an existing pre-trained model. Do you want to retrain?")
            response = input("Enter 'y' to retrain, or press Enter to use the existing model: ")
            if response.lower() == 'y':
                model, pretrain_history = pretrain_on_nctcrc(model, num_epochs=15, batch_size=32)
            else:
                print("Using the existing pre-trained model...")
                checkpoint = torch.load('pretrained_nctcrc_model.pth')
                pretrain_history = checkpoint.get('history', None)
        else:
            model, pretrain_history = pretrain_on_nctcrc(model, num_epochs=15, batch_size=32)

        if not os.path.exists('pretrained_nctcrc_model.pth'):
            print("❌ Pre-training failed, exiting the program")
            return

        # Step 2: Fine-tuning
        fold_models, X_test, y_test, cv_results = finetune_on_lnm(
            'pretrained_nctcrc_model.pth', image_paths, labels,
            num_epochs=20, lr=1e-5, test_size=0.2
        )

        # Step 3: Final Testing
        test_accuracy, test_auc, test_predictions, test_probabilities = evaluate_on_test_set(
            fold_models, X_test, y_test
        )

        # Step 4: Generate report and visualizations
        create_final_report(pretrain_history, cv_results, test_accuracy, test_auc)
        visualize_training_results(pretrain_history, cv_results)

        print(f"\n🎉 Full workflow executed successfully!")
        print(f"📊 Main result files:")
        print(f"  • pathology_model_report.json - Detailed performance report")
        print(f"  • pretrain_finetune_results.png - Results visualization")
        print(f"  • pretrained_nctcrc_model.pth - Pre-trained model")
        print(f"  • finetune_fold_*_model.pth - Fine-tuned models")

    except KeyboardInterrupt:
        print(f"\n❌ Program interrupted by user")
    except Exception as e:
        print(f"\n❌ An error occurred during execution: {e}")
        import traceback
        print("Detailed error information:")
        print(traceback.format_exc())

        print(f"\n🔧 Troubleshooting suggestions:")
        print(f"1. Check your internet connection (for downloading the NCTCRC dataset)")
        print(f"2. Ensure you have enough disk space (>5GB)")
        print(f"3. Check your CUDA/GPU settings")
        print(f"4. Confirm that all required packages are installed correctly")


if __name__ == '__main__':
    # Required package installation instructions
    print("📦 Ensure the following dependencies are installed:")
    print("pip install torch torchvision datasets huggingface-hub")
    print("pip install scikit-learn matplotlib seaborn tqdm")
    print("pip install opencv-python pillow numpy pandas")
    print("-" * 60)

    main()