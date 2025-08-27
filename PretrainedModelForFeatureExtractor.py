import torch
import torch.nn as nn
import torchvision.transforms as transforms
import torchvision.models as models
from PIL import Image
import numpy as np
import pandas as pd
import cv2
import os
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold, cross_val_score, GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import roc_auc_score, classification_report, confusion_matrix, accuracy_score
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm
import warnings

warnings.filterwarnings('ignore')


class PathologyFeatureExtractor(nn.Module):
    """Load pre-trained model for feature extraction"""

    def __init__(self, backbone='resnet50', num_pretrain_classes=9):
        super(PathologyFeatureExtractor, self).__init__()

        if backbone == 'resnet50':
            self.backbone = models.resnet50(pretrained=True)
            self.feature_dim = self.backbone.fc.in_features
            self.backbone.fc = nn.Identity()

        # Pre-training classification head (for loading weights)
        self.pretrain_classifier = nn.Sequential(
            nn.Dropout(0.5),
            nn.Linear(self.feature_dim, 512),
            nn.ReLU(inplace=True),
            nn.BatchNorm1d(512),
            nn.Dropout(0.3),
            nn.Linear(512, num_pretrain_classes)
        )

        # Fine-tuning classification head (for loading weights)
        self.finetune_classifier = nn.Sequential(
            nn.Dropout(0.3),
            nn.Linear(self.feature_dim, 128),
            nn.ReLU(inplace=True),
            nn.Dropout(0.2),
            nn.Linear(128, 2)
        )

    def forward(self, x, mode='features'):
        features = self.backbone(x)

        if mode == 'features':
            return features  # Return the 2048-dimension feature vector
        elif mode == 'pretrain':
            return self.pretrain_classifier(features)
        elif mode == 'finetune':
            return self.finetune_classifier(features)


def load_pretrained_model(model_path):
    """Load the pre-trained model"""

    print(f"Loading pre-trained model: {model_path}")

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = PathologyFeatureExtractor(backbone='resnet50', num_pretrain_classes=9)

    # Load pre-trained weights
    checkpoint = torch.load(model_path, map_location=device)
    if 'model_state_dict' in checkpoint:
        model.load_state_dict(checkpoint['model_state_dict'])
        pretrain_acc = checkpoint.get('val_acc', 'N/A')
    else:
        model.load_state_dict(checkpoint)
        pretrain_acc = 'N/A'

    model = model.to(device)
    model.eval()

    print(f"✅ Pre-trained model loaded successfully")
    print(f"   Pre-training validation accuracy: {pretrain_acc}")
    print(f"   Feature dimension: {model.feature_dim}")
    print(f"   Device: {device}")

    return model, device


def get_feature_extraction_transform():
    """Data transform for feature extraction"""

    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    return transform


def extract_features_from_images(model, image_paths, labels, device):
    """Extract deep features from images"""

    print(f"Extracting features from {len(image_paths)} images...")

    transform = get_feature_extraction_transform()
    features_list = []
    valid_labels = []
    valid_paths = []

    model.eval()
    with torch.no_grad():
        for i, (image_path, label) in enumerate(tqdm(zip(image_paths, labels),
                                                     desc="Extracting Features", total=len(image_paths))):
            try:
                # Read image
                image = cv2.imread(image_path)
                if image is None:
                    print(f"⚠️ Could not read image: {image_path}")
                    continue

                image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
                image = Image.fromarray(image)

                # Apply transform
                image_tensor = transform(image).unsqueeze(0).to(device)

                # Extract features
                features = model(image_tensor, mode='features')
                features = features.cpu().numpy().flatten()  # Convert to 1D array

                features_list.append(features)
                valid_labels.append(label)
                valid_paths.append(image_path)

            except Exception as e:
                print(f"❌ Error processing image {image_path}: {e}")
                continue

    if not features_list:
        raise ValueError("Failed to extract any features!")

    X = np.array(features_list)
    y = np.array(valid_labels)

    print(f"✅ Feature extraction complete:")
    print(f"   Successfully processed: {len(valid_paths)} images")
    print(f"   Feature dimension: {X.shape[1]}")
    print(f"   LNM: {sum(y)}, non-LNM: {len(y) - sum(y)}")

    return X, y, valid_paths


def evaluate_classifiers_with_cv(X, y, image_paths):
    """Evaluate different classifiers using cross-validation"""

    print(f"\n🔍 Evaluating classifiers using 5-fold cross-validation...")
    print("=" * 60)

    # Standardize features
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    # Define classifiers
    classifiers = {
        'Random Forest': RandomForestClassifier(
            n_estimators=100,
            max_depth=10,
            min_samples_split=5,
            random_state=42,
            class_weight='balanced'
        ),
        'SVM (RBF)': SVC(
            kernel='rbf',
            probability=True,
            random_state=42,
            class_weight='balanced',
            C=1.0
        ),
        'Gradient Boosting': GradientBoostingClassifier(
            n_estimators=100,
            max_depth=5,
            learning_rate=0.1,
            random_state=42
        ),
        'Logistic Regression': LogisticRegression(
            random_state=42,
            class_weight='balanced',
            max_iter=1000
        )
    }

    # 5-fold cross-validation
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    results = {}

    for name, classifier in classifiers.items():
        print(f"\n📊 Evaluating {name}...")

        # AUC cross-validation
        cv_aucs = cross_val_score(classifier, X_scaled, y, cv=cv, scoring='roc_auc')

        # Accuracy cross-validation
        cv_accs = cross_val_score(classifier, X_scaled, y, cv=cv, scoring='accuracy')

        # Detailed fold results
        fold_results = []
        for fold, (train_idx, val_idx) in enumerate(cv.split(X_scaled, y)):
            X_train, X_val = X_scaled[train_idx], X_scaled[val_idx]
            y_train, y_val = y[train_idx], y[val_idx]

            # Train
            clf = type(classifier)(**classifier.get_params())
            clf.fit(X_train, y_train)

            # Predict
            y_pred = clf.predict(X_val)
            y_pred_proba = clf.predict_proba(X_val)[:, 1]

            # Metrics
            acc = accuracy_score(y_val, y_pred)
            auc = roc_auc_score(y_val, y_pred_proba) if len(set(y_val)) > 1 else 0.5

            fold_results.append({
                'fold': fold + 1,
                'accuracy': acc,
                'auc': auc,
                'val_size': len(y_val),
                'val_lnm': sum(y_val),
                'correct': sum(y_pred == y_val)
            })

            print(f"  Fold {fold + 1}: Acc={acc:.4f}, AUC={auc:.4f}, "
                  f"Val=({sum(y_val)} LNM, {len(y_val) - sum(y_val)} non-LNM)")

        # Statistical results
        mean_auc = np.mean(cv_aucs)
        std_auc = np.std(cv_aucs)
        mean_acc = np.mean(cv_accs)
        std_acc = np.std(cv_accs)

        results[name] = {
            'fold_results': fold_results,
            'mean_auc': mean_auc,
            'std_auc': std_auc,
            'mean_acc': mean_acc,
            'std_acc': std_acc,
            'cv_coeff_auc': (std_auc / mean_auc) * 100,
            'cv_coeff_acc': (std_acc / mean_acc) * 100
        }

        print(f"  Mean AUC: {mean_auc:.4f} ± {std_auc:.4f}")
        print(f"  Mean Acc: {mean_acc:.4f} ± {std_acc:.4f}")
        print(f"  AUC Stability: {results[name]['cv_coeff_auc']:.1f}% CV")

        # Stability assessment
        if results[name]['cv_coeff_auc'] < 10:
            stability = "Excellent"
        elif results[name]['cv_coeff_auc'] < 20:
            stability = "Good"
        else:
            stability = "Fair"

        print(f"  Stability: {stability}")

    return results, scaler


def perform_hyperparameter_tuning(X, y, best_classifier_name):
    """Perform hyperparameter tuning for the best classifier"""

    print(f"\n🔧 Performing hyperparameter tuning for {best_classifier_name}...")

    # Standardization
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    if best_classifier_name == 'Random Forest':
        param_grid = {
            'n_estimators': [50, 100, 200],
            'max_depth': [5, 10, 15, None],
            'min_samples_split': [2, 5, 10],
            'class_weight': ['balanced', None]
        }
        base_classifier = RandomForestClassifier(random_state=42)

    elif best_classifier_name == 'SVM (RBF)':
        param_grid = {
            'C': [0.1, 1, 10],
            'gamma': ['scale', 'auto', 0.01, 0.1],
            'class_weight': ['balanced', None]
        }
        base_classifier = SVC(kernel='rbf', probability=True, random_state=42)

    elif best_classifier_name == 'Logistic Regression':
        param_grid = {
            'C': [0.01, 0.1, 1, 10],
            'class_weight': ['balanced', None]
        }
        base_classifier = LogisticRegression(random_state=42, max_iter=1000)

    else:
        print("Skipping hyperparameter tuning")
        return None, scaler

    # Grid search
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    grid_search = GridSearchCV(
        base_classifier,
        param_grid,
        cv=cv,
        scoring='roc_auc',
        n_jobs=-1
    )

    grid_search.fit(X_scaled, y)

    print(f"Best parameters: {grid_search.best_params_}")
    print(f"Best cross-validation AUC: {grid_search.best_score_:.4f}")

    return grid_search.best_estimator_, scaler


def create_final_comparison_report(results):
    """Generate the final comparison report"""

    print(f"\n" + "=" * 80)
    print("📋 Final Performance Comparison Report")
    print("=" * 80)

    # Find the best method
    best_method = max(results.keys(), key=lambda x: results[x]['mean_auc'])
    best_result = results[best_method]

    print(f"🏆 Best Method: {best_method}")
    print(f"   AUC: {best_result['mean_auc']:.4f} ± {best_result['std_auc']:.4f}")
    print(f"   Accuracy: {best_result['mean_acc']:.4f} ± {best_result['std_acc']:.4f}")
    print(f"   AUC Stability: {best_result['cv_coeff_auc']:.1f}% CV")

    print(f"\n📊 Comparison of All Methods:")
    print(f"{'Method':<20} {'AUC':<15} {'Accuracy':<15} {'AUC-CV%':<10} {'Stability'}")
    print("-" * 75)

    for method, result in results.items():
        stability = "Excellent" if result['cv_coeff_auc'] < 10 else "Good" if result['cv_coeff_auc'] < 20 else "Fair"
        print(f"{method:<20} "
              f"{result['mean_auc']:.3f}±{result['std_auc']:.3f}   "
              f"{result['mean_acc']:.3f}±{result['std_acc']:.3f}   "
              f"{result['cv_coeff_auc']:>6.1f}      "
              f"{stability}")

    print(f"\n📈 Comparison with Previous Methods:")
    print(f"Small Dataset Deep Learning: 0.570 ± 0.110 (CV: 20.0%)")
    print(f"Traditional Machine Learning:  0.683 ± 0.367 (CV: 53.7%)")
    print(f"Deep Learning Fine-tuning:   0.625 ± 0.221 (CV: 35.3%)")
    print(f"Pre-trained Features + ML:   {best_result['mean_auc']:.3f} ± {best_result['std_auc']:.3f} "
          f"(CV: {best_result['cv_coeff_auc']:.1f}%)")

    # Improvement assessment
    baseline_auc = 0.570
    improvement = ((best_result['mean_auc'] - baseline_auc) / baseline_auc) * 100

    print(f"\n💡 Improvement Assessment:")
    if best_result['mean_auc'] > 0.8 and best_result['cv_coeff_auc'] < 15:
        print(f"🎉 Excellent! The feature extraction strategy significantly improved performance and stability!")
    elif best_result['mean_auc'] > 0.75 and best_result['cv_coeff_auc'] < 25:
        print(f"✅ Good! The strategy brought a clear improvement!")
    elif best_result['mean_auc'] > baseline_auc:
        print(f"⚠️ Moderate improvement, relative increase of {improvement:+.1f}%")
    else:
        print(f"❌ Further optimization of the feature extraction or classification strategy is needed")

    print(f"\n🎯 Clinical Usability Assessment:")
    clinical_folds = sum(1 for fold in best_result['fold_results'] if fold['auc'] >= 0.8)
    print(f"Folds reaching clinical threshold (AUC≥0.8): {clinical_folds}/5")

    if clinical_folds >= 4:
        print(f"✅ Clinically considerable for use - highly consistent performance")
    elif clinical_folds >= 3:
        print(f"⚠️ Potentially clinically useful - good performance in most cases")
    elif best_result['mean_auc'] >= 0.75:
        print(f"📊 Research-grade performance - requires more validation")
    else:
        print(f"❌ Further improvement needed")


def visualize_results(results):
    """Visualize the results"""

    fig, axes = plt.subplots(2, 3, figsize=(18, 12))

    # 1. AUC Comparison
    methods = list(results.keys())
    mean_aucs = [results[method]['mean_auc'] for method in methods]
    std_aucs = [results[method]['std_auc'] for method in methods]

    axes[0, 0].bar(methods, mean_aucs, yerr=std_aucs, capsize=5, alpha=0.8, color='skyblue')
    axes[0, 0].axhline(y=0.8, color='red', linestyle='--', alpha=0.7, label='Clinical Threshold')
    axes[0, 0].set_ylabel('AUC')
    axes[0, 0].set_title('Classifier AUC Comparison')
    axes[0, 0].legend()
    axes[0, 0].tick_params(axis='x', rotation=45)

    # 2. Stability Comparison
    cv_coeffs = [results[method]['cv_coeff_auc'] for method in methods]
    colors = ['green' if cv < 15 else 'orange' if cv < 25 else 'red' for cv in cv_coeffs]

    axes[0, 1].bar(methods, cv_coeffs, alpha=0.8, color=colors)
    axes[0, 1].axhline(y=15, color='green', linestyle='--', alpha=0.7, label='Excellent (<15%)')
    axes[0, 1].axhline(y=25, color='orange', linestyle='--', alpha=0.7, label='Good (<25%)')
    axes[0, 1].set_ylabel('Coefficient of Variation (%)')
    axes[0, 1].set_title('AUC Stability Comparison')
    axes[0, 1].legend()
    axes[0, 1].tick_params(axis='x', rotation=45)

    # 3. Method Evolution Comparison
    evolution_methods = ['Small Data DL', 'Traditional ML', 'DL Fine-tune', 'Pre-trained Feat+ML']
    evolution_aucs = [0.570, 0.683, 0.625, max(mean_aucs)]
    evolution_cvs = [20.0, 53.7, 35.3, min(cv_coeffs)]

    axes[0, 2].plot(evolution_methods, evolution_aucs, 'o-', linewidth=2, markersize=8, label='AUC')
    axes[0, 2].set_ylabel('AUC', color='blue')
    axes[0, 2].tick_params(axis='y', labelcolor='blue')
    axes[0, 2].tick_params(axis='x', rotation=45)

    ax2 = axes[0, 2].twinx()
    ax2.plot(evolution_methods, evolution_cvs, 's--', color='red', linewidth=2, markersize=8, label='CV%')
    ax2.set_ylabel('Coefficient of Variation (%)', color='red')
    ax2.tick_params(axis='y', labelcolor='red')

    axes[0, 2].set_title('Method Evolution Trend')

    # 4-6. Fold details for each classifier
    for i, (method, result) in enumerate(list(results.items())[:3]):
        if i < 3:
            fold_aucs = [fold['auc'] for fold in result['fold_results']]

            axes[1, i].bar(range(1, 6), fold_aucs, alpha=0.8, color='lightcoral')
            axes[1, i].axhline(y=np.mean(fold_aucs), color='blue', linestyle='--',
                               label=f'Mean: {np.mean(fold_aucs):.3f}')
            axes[1, i].axhline(y=0.8, color='red', linestyle='--', alpha=0.7)
            axes[1, i].set_xlabel('Fold')
            axes[1, i].set_ylabel('AUC')
            axes[1, i].set_title(f'{method} - Fold Details')
            axes[1, i].legend()
            axes[1, i].set_ylim(0.3, 1.05)

    plt.tight_layout()
    plt.savefig('feature_extraction_results.png', dpi=150, bbox_inches='tight')
    plt.show()


def main():
    """Main function"""

    print("🎯 Pre-trained Model Feature Extraction + Traditional Classifiers")
    print("=" * 60)
    print("Strategy: Use a trained deep model as a feature extractor, combined with traditional ML classifiers")

    # Check for pre-trained model
    pretrained_model_path = 'pretrained_nctcrc_model.pth'
    if not os.path.exists(pretrained_model_path):
        print(f"❌ Pre-trained model not found: {pretrained_model_path}")
        print("Please run the pre-training code first to generate the model file")
        return

    # Load pre-trained model
    try:
        model, device = load_pretrained_model(pretrained_model_path)
    except Exception as e:
        print(f"❌ Model loading failed: {e}")
        return

    # Prepare LNM data
    print(f"\n📁 Preparing LNM data...")

    # Find data directory
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
        return

    # Collect images
    image_paths = []
    labels = []

    for img_file in os.listdir(lnm_dir):
        if img_file.lower().endswith(('.jpg', '.jpeg', 'png', '.tiff', '.tif')):
            image_paths.append(os.path.join(lnm_dir, img_file))
            labels.append(1)

    for img_file in os.listdir(non_lnm_dir):
        if img_file.lower().endswith(('.jpg', '.jpeg', 'png', '.tiff', '.tif')):
            image_paths.append(os.path.join(non_lnm_dir, img_file))
            labels.append(0)

    print(f"✅ Data collection complete:")
    print(f"   Total images: {len(image_paths)}")
    print(f"   LNM: {sum(labels)}")
    print(f"   non-LNM: {len(labels) - sum(labels)}")

    # Extract features
    try:
        X, y, valid_paths = extract_features_from_images(model, image_paths, labels, device)
    except Exception as e:
        print(f"❌ Feature extraction failed: {e}")
        return

    # Evaluate classifiers
    results, scaler = evaluate_classifiers_with_cv(X, y, valid_paths)

    # Hyperparameter tuning
    best_method = max(results.keys(), key=lambda x: results[x]['mean_auc'])
    print(f"\n🔧 Performing hyperparameter tuning for the best method: {best_method}...")

    try:
        tuned_classifier, _ = perform_hyperparameter_tuning(X, y, best_method)
        if tuned_classifier:
            # Re-evaluate the tuned classifier
            cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
            X_scaled = scaler.transform(X)
            tuned_aucs = cross_val_score(tuned_classifier, X_scaled, y, cv=cv, scoring='roc_auc')

            print(f"Performance after tuning: {np.mean(tuned_aucs):.4f} ± {np.std(tuned_aucs):.4f}")

            # If tuning improved results, update the results
            if np.mean(tuned_aucs) > results[best_method]['mean_auc']:
                results[f'{best_method} (Tuned)'] = {
                    'mean_auc': np.mean(tuned_aucs),
                    'std_auc': np.std(tuned_aucs),
                    'cv_coeff_auc': (np.std(tuned_aucs) / np.mean(tuned_aucs)) * 100,
                    'fold_results': []  # Simplified
                }
    except Exception as e:
        print(f"Hyperparameter tuning failed: {e}")

    # Generate report and visualizations
    create_final_comparison_report(results)
    visualize_results(results)

    print(f"\n🎉 Analysis complete!")
    print(f"Charts saved to: feature_extraction_results.png")


if __name__ == '__main__':
    main()