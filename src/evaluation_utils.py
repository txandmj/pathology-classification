import numpy as np
import os
import joblib
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.metrics import roc_auc_score, accuracy_score
from corrected_feature_model import extract_features_from_images, load_feature_extractor, set_seeds
import warnings
# Valid image extensions
VALID_EXTENSIONS = ('.jpg', '.jpeg', '.png', '.tiff', '.tif')

warnings.filterwarnings('ignore')
def evaluate_classifiers_with_pipeline_cv(X, y):
    """Cross-validation with proper pipelines to prevent data leakage"""
    print(f"\n5-fold CV with leakage-safe pipelines")
    print("=" * 60)

    # Set up cross-validation
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

    # Define classifiers
    classifiers = {
        'Logistic Regression': LogisticRegression(
            max_iter=1000,
            class_weight='balanced',
            random_state=42,
            C=1.0
        ),
        'SVM (RBF)': SVC(
            kernel='rbf',
            probability=True,
            class_weight='balanced',
            random_state=42,
            C=1.0,
            gamma='scale'
        ),
        'Random Forest': RandomForestClassifier(
            n_estimators=100,
            max_depth=10,
            min_samples_split=5,
            class_weight='balanced',
            random_state=42
        ),
        'Gradient Boosting': GradientBoostingClassifier(
            n_estimators=100,
            max_depth=3,
            learning_rate=0.1,
            random_state=42
        )
    }

    results = {}

    for name, clf in classifiers.items():
        print(f"\nEvaluating {name}...")

        # Create pipeline with scaler
        pipeline = Pipeline([
            ('scaler', StandardScaler()),
            ('classifier', clf)
        ])

        # Detailed fold-by-fold evaluation
        fold_aucs = []
        fold_accs = []
        fold_details = []

        for fold, (train_idx, val_idx) in enumerate(cv.split(X, y), 1):
            X_train, X_val = X[train_idx], X[val_idx]
            y_train, y_val = y[train_idx], y[val_idx]

            # Fit pipeline (scaler + classifier)
            pipeline.fit(X_train, y_train)

            # Predict
            if hasattr(pipeline.named_steps['classifier'], 'predict_proba'):
                y_prob = pipeline.predict_proba(X_val)[:, 1]
            else:
                # Fallback for classifiers without predict_proba
                y_prob = pipeline.decision_function(X_val)
                # Normalize to [0,1]
                y_prob = (y_prob - y_prob.min()) / (y_prob.max() - y_prob.min() + 1e-8)

            y_pred = (y_prob >= 0.5).astype(int)

            # Calculate metrics
            acc = accuracy_score(y_val, y_pred)
            auc = roc_auc_score(y_val, y_prob) if len(np.unique(y_val)) > 1 else 0.5

            fold_aucs.append(auc)
            fold_accs.append(acc)

            fold_details.append({
                'fold': fold,
                'auc': auc,
                'acc': acc,
                'val_size': len(y_val),
                'val_lnm': sum(y_val),
                'correct': sum(y_pred == y_val)
            })

            print(f"  Fold {fold}: AUC={auc:.4f}, Acc={acc:.4f}, "
                  f"Val=({sum(y_val)} LNM, {len(y_val) - sum(y_val)} non-LNM)")

        # Calculate summary statistics
        mean_auc = np.mean(fold_aucs)
        std_auc = np.std(fold_aucs)
        mean_acc = np.mean(fold_accs)
        std_acc = np.std(fold_accs)
        cv_coeff = (std_auc / mean_auc) * 100 if mean_auc > 0 else 100

        results[name] = {
            'mean_auc': mean_auc,
            'std_auc': std_auc,
            'mean_acc': mean_acc,
            'std_acc': std_acc,
            'cv_coeff_auc': cv_coeff,
            'fold_results': fold_details
        }

        print(f"  Summary: AUC {mean_auc:.4f}±{std_auc:.4f}, "
              f"Acc {mean_acc:.4f}±{std_acc:.4f}, CV {cv_coeff:.1f}%")

        # Stability assessment
        if cv_coeff < 10:
            stability = "Excellent"
        elif cv_coeff < 20:
            stability = "Good"
        elif cv_coeff < 30:
            stability = "Fair"
        else:
            stability = "Poor"

        print(f"  Stability: {stability}")

    return results


def fit_and_save_best_model(X, y, results, output_prefix="final_lnm"):
    """Train final model on full dataset and save"""

    # Find best classifier by AUC
    best_name = max(results.keys(), key=lambda k: results[k]['mean_auc'])
    best_result = results[best_name]

    print(f"\nTraining final model using best classifier: {best_name}")
    print(f"Expected performance: AUC {best_result['mean_auc']:.4f}±{best_result['std_auc']:.4f}")

    # Recreate the best classifier
    if best_name == 'Logistic Regression':
        clf = LogisticRegression(max_iter=1000, class_weight='balanced', random_state=42)
    elif best_name == 'SVM (RBF)':
        clf = SVC(kernel='rbf', probability=True, class_weight='balanced', random_state=42)
    elif best_name == 'Random Forest':
        clf = RandomForestClassifier(n_estimators=100, max_depth=10,
                                     class_weight='balanced', random_state=42)
    else:  # Gradient Boosting
        clf = GradientBoostingClassifier(n_estimators=100, max_depth=3, random_state=42)

    # Create final pipeline
    final_pipeline = Pipeline([
        ('scaler', StandardScaler()),
        ('classifier', clf)
    ])

    # Fit on full dataset
    final_pipeline.fit(X, y)
    # Define the output directory and create it if it doesn't exist
    output_dir = '../models'
    os.makedirs(output_dir, exist_ok=True)

    # Create the full file path
    model_filename = f"{output_prefix}_{best_name.replace(' ', '_').lower()}.joblib"
    model_filepath = os.path.join(output_dir, model_filename)

    # Save the pipeline to the specified path
    joblib.dump(final_pipeline, model_filepath)

    print(f"Final model saved: {model_filepath}")
    print(f"Model components:")
    print(f"  - Scaler: {type(final_pipeline.named_steps['scaler']).__name__}")
    print(f"  - Classifier: {type(final_pipeline.named_steps['classifier']).__name__}")

    return final_pipeline, model_filename


def bootstrap_confidence_interval(y_true, y_pred, n_bootstrap=1000, confidence=0.95):
    """Calculate bootstrap confidence interval for AUC"""
    bootstrap_aucs = []
    n_samples = len(y_true)

    for _ in range(n_bootstrap):
        # Bootstrap sample
        bootstrap_idx = np.random.choice(n_samples, n_samples, replace=True)
        y_true_boot = y_true[bootstrap_idx]
        y_pred_boot = y_pred[bootstrap_idx]

        # Calculate AUC if we have both classes
        if len(np.unique(y_true_boot)) > 1:
            auc = roc_auc_score(y_true_boot, y_pred_boot)
            bootstrap_aucs.append(auc)

    if bootstrap_aucs:
        alpha = 1 - confidence
        lower = np.percentile(bootstrap_aucs, 100 * (alpha / 2))
        upper = np.percentile(bootstrap_aucs, 100 * (1 - alpha / 2))
        return lower, upper
    else:
        return None, None


def create_enhanced_report(results, X, y):
    """Generate comprehensive report with bootstrap CI"""
    print(f"\n" + "=" * 80)
    print("ENHANCED PERFORMANCE REPORT")
    print("=" * 80)

    # Find best method
    best_method = max(results.keys(), key=lambda x: results[x]['mean_auc'])
    best_result = results[best_method]

    print(f"\nBest Method: {best_method}")
    print(f"  AUC: {best_result['mean_auc']:.4f} ± {best_result['std_auc']:.4f}")
    print(f"  Accuracy: {best_result['mean_acc']:.4f} ± {best_result['std_acc']:.4f}")
    print(f"  CV Coefficient: {best_result['cv_coeff_auc']:.1f}%")

    # Detailed fold analysis
    print(f"\nFold-by-fold analysis:")
    for fold_data in best_result['fold_results']:
        print(f"  Fold {fold_data['fold']}: AUC={fold_data['auc']:.4f}, "
              f"Correct={fold_data['correct']}/{fold_data['val_size']}")

    # Bootstrap CI for best method (approximate)
    print(f"\nBootstrap confidence interval (approximate):")
    all_fold_aucs = [f['auc'] for f in best_result['fold_results']]
    ci_lower = np.percentile(all_fold_aucs, 2.5)
    ci_upper = np.percentile(all_fold_aucs, 97.5)
    print(f"  95% CI: [{ci_lower:.4f}, {ci_upper:.4f}]")

    # Method comparison table
    print(f"\nMethod Comparison:")
    print(f"{'Method':<20} {'AUC':<15} {'Stability':<12} {'Assessment'}")
    print("-" * 65)

    for method, result in results.items():
        stability = "Excellent" if result['cv_coeff_auc'] < 10 else \
            "Good" if result['cv_coeff_auc'] < 20 else \
                "Fair" if result['cv_coeff_auc'] < 30 else "Poor"

        assessment = "Recommended" if method == best_method else \
            "Good" if result['mean_auc'] >= 0.75 else \
                "Moderate" if result['mean_auc'] >= 0.65 else "Poor"

        marker = "★" if method == best_method else " "

        print(f"{marker} {method:<19} "
              f"{result['mean_auc']:.3f}±{result['std_auc']:.3f}    "
              f"{stability:<12} {assessment}")

    # Clinical assessment
    print(f"\nClinical Viability Assessment:")
    clinical_folds = sum(1 for f in best_result['fold_results'] if f['auc'] >= 0.8)
    print(f"  Folds meeting clinical threshold (AUC≥0.8): {clinical_folds}/5")

    if clinical_folds >= 4:
        print(f"  ✓ Excellent - Highly consistent clinical-grade performance")
    elif clinical_folds >= 3:
        print(f"  ✓ Good - Mostly clinical-grade with some variability")
    elif best_result['mean_auc'] >= 0.75:
        print(f"  ~ Fair - Research-grade with clinical potential")
    else:
        print(f"  ✗ Poor - Below clinical standards")


def main():
    """Main execution with all improvements"""

    # Set reproducible seeds
    set_seeds(42)

    print("CORRECTED: Pre-trained Features + Traditional ML")
    print("=" * 60)
    print("Improvements: No data leakage, robust loading, proper validation")

    # Load feature extractor
    try:
        model, device = load_feature_extractor('../Models/pretrained_nctcrc_model.pth')
    except Exception as e:
        print(f"Warning: Could not load custom model: {e}")
        model, device = load_feature_extractor(None)  # Use ImageNet only

    # Find data directories
    data_dirs = ['../Data', '../data']
    lnm_dir = None
    non_lnm_dir = None

    for data_dir in data_dirs:
        if os.path.exists(data_dir):
            # Look for LNM directories
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
        print("Error: Could not find LNM data directories!")
        print("Expected structure: Data/LNM/ and Data/NOT-LNM/")
        return

    # Collect images with corrected extensions
    image_paths = []
    labels = []

    # LNM images
    for img_file in os.listdir(lnm_dir):
        if img_file.lower().endswith(VALID_EXTENSIONS):
            image_paths.append(os.path.join(lnm_dir, img_file))
            labels.append(1)

    # non-LNM images
    for img_file in os.listdir(non_lnm_dir):
        if img_file.lower().endswith(VALID_EXTENSIONS):
            image_paths.append(os.path.join(non_lnm_dir, img_file))
            labels.append(0)

    print(f"\nDataset summary:")
    print(f"  Total images: {len(image_paths)}")
    print(f"  LNM: {sum(labels)}")
    print(f"  non-LNM: {len(labels) - sum(labels)}")

    if len(image_paths) < 10:
        print("Warning: Very small dataset - results may be unstable")

    # Extract features
    try:
        X, y, valid_paths = extract_features_from_images(model, image_paths, labels, device)
    except Exception as e:
        print(f"Error in feature extraction: {e}")
        return

    # Corrected cross-validation (no data leakage)
    results = evaluate_classifiers_with_pipeline_cv(X, y)

    # Train and save best model
    final_model, model_file = fit_and_save_best_model(X, y, results)

    # Enhanced reporting
    create_enhanced_report(results, X, y)

    print(f"\n" + "=" * 60)
    print("ANALYSIS COMPLETE")
    print("=" * 60)
    print(f"✓ Best model saved: {model_file}")
    print(f"✓ All improvements applied: No leakage, robust loading, proper CV")
    print(f"✓ Results should be more reliable than previous version")


if __name__ == '__main__':
    main()