import torch
import torch.nn as nn
import torchvision.transforms as transforms
import torchvision.models as models
from PIL import Image
import numpy as np
import cv2
import os
from tqdm import tqdm
import warnings
import random

warnings.filterwarnings('ignore')

# Valid image extensions
VALID_EXTENSIONS = ('.jpg', '.jpeg', '.png', '.tiff', '.tif')


def set_seeds(seed=42):
    """Set all random seeds for reproducibility"""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def load_feature_extractor(model_path=None):
    """Load feature extractor with improved robustness"""
    print("Loading feature extractor...")
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Use modern torchvision API
    try:
        weights = models.ResNet50_Weights.IMAGENET1K_V2
        backbone = models.resnet50(weights=weights)
    except AttributeError:
        # Fallback for older torchvision versions
        backbone = models.resnet50(pretrained=True)

    feature_dim = backbone.fc.in_features
    backbone.fc = nn.Identity()

    # Create simple feature extractor
    model = nn.Sequential(backbone)
    model.feature_dim = feature_dim
    model = model.to(device)
    model.eval()

    # Optionally load custom pre-trained weights
    if model_path and os.path.exists(model_path):
        print(f"Attempting to load custom weights: {model_path}")
        try:
            checkpoint = torch.load(model_path, map_location=device)
            state_dict = checkpoint.get('model_state_dict', checkpoint)

            # Filter to only backbone weights if needed
            backbone_state = {}
            for key, value in state_dict.items():
                if key.startswith('backbone.') or not ('classifier' in key):
                    # Remove 'backbone.' prefix if present
                    new_key = key.replace('backbone.', '0.')
                    backbone_state[new_key] = value

            # Load with strict=False to handle key mismatches
            missing, unexpected = model.load_state_dict(backbone_state, strict=False)
            print(f"Loaded custom weights - Missing: {len(missing)}, Unexpected: {len(unexpected)}")

            if missing:
                print("Missing keys (using ImageNet weights for these):", missing[:5])
            if unexpected:
                print("Unexpected keys (ignored):", unexpected[:5])

        except Exception as e:
            print(f"Failed to load custom weights: {e}")
            print("Continuing with ImageNet weights only")

    print(f"Feature extractor ready - Dim: {feature_dim}, Device: {device}")
    return model, device


def extract_features_from_images(model, image_paths, labels, device):
    """Extract features with improved error handling"""
    print(f"Extracting features from {len(image_paths)} images...")

    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    features_list = []
    valid_labels = []
    valid_paths = []

    model.eval()
    with torch.no_grad():
        for i, (image_path, label) in enumerate(tqdm(zip(image_paths, labels),
                                                     desc="Extracting features", total=len(image_paths))):
            try:
                # Read and validate image
                image = cv2.imread(image_path)
                if image is None:
                    print(f"Warning: Cannot read image {image_path}")
                    continue

                if image.shape[0] < 32 or image.shape[1] < 32:
                    print(f"Warning: Image too small {image_path}: {image.shape}")
                    continue

                image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
                image = Image.fromarray(image)

                # Apply transforms and extract features
                image_tensor = transform(image).unsqueeze(0).to(device)
                features = model(image_tensor)
                features = features.cpu().numpy().flatten()

                # Validate features
                if np.any(np.isnan(features)) or np.any(np.isinf(features)):
                    print(f"Warning: Invalid features from {image_path}")
                    continue

                features_list.append(features)
                valid_labels.append(label)
                valid_paths.append(image_path)

            except Exception as e:
                print(f"Error processing {image_path}: {e}")
                continue

    if not features_list:
        raise ValueError("No features extracted successfully!")

    X = np.array(features_list)
    y = np.array(valid_labels)

    print(f"Feature extraction complete:")
    print(f"  Successfully processed: {len(valid_paths)} images")
    print(f"  Feature dimensions: {X.shape[1]}")
    print(f"  Class distribution: {sum(y)} LNM, {len(y) - sum(y)} non-LNM")

    return X, y, valid_paths


