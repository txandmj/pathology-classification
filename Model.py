import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.svm import SVC
from sklearn.model_selection import StratifiedKFold, cross_val_score, GridSearchCV
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import roc_auc_score, classification_report, confusion_matrix
from sklearn.feature_selection import SelectKBest, f_classif
import matplotlib.pyplot as plt
import warnings

warnings.filterwarnings('ignore')


def traditional_ml_pipeline(X, y, feature_names):
    """traditional machine learning pipeline"""

    print(f"\n🤖 traditional machine learning pipeline")
    print("=" * 50)

    # data preprocessing
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    # feature selection
    selector = SelectKBest(score_func=f_classif, k=min(50, X.shape[1]))  # 选择最佳50个特征
    X_selected = selector.fit_transform(X_scaled, y)

    selected_indices = selector.get_support(indices=True)
    selected_features = [feature_names[i] for i in selected_indices]

    print(f"✅ feature selection: {X.shape[1]} → {X_selected.shape[1]} features")
    print(f"top 10 selected features: {selected_features[:10]}")

    # define classifiers
    classifiers = {
        'Random Forest': RandomForestClassifier(
            n_estimators=100, max_depth=5, min_samples_split=5,
            random_state=42, class_weight='balanced'
        ),
        'SVM (RBF)': SVC(
            kernel='rbf', probability=True, random_state=42,
            class_weight='balanced', C=1.0, gamma='scale'
        ),
        'Gradient Boosting': GradientBoostingClassifier(
            n_estimators=100, max_depth=3, learning_rate=0.1,
            random_state=42
        )
    }

    # Cross-validation setup
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

    results = {}

    for name, classifier in classifiers.items():
        print(f"\n🔄 evaluationg {name}...")

        # cross-validation scores
        cv_scores = cross_val_score(classifier, X_selected, y, cv=cv, scoring='roc_auc')

        # pre-fold analysis
        fold_results = []
        for fold, (train_idx, val_idx) in enumerate(cv.split(X_selected, y)):
            X_train, X_val = X_selected[train_idx], X_selected[val_idx]
            y_train, y_val = y[train_idx], y[val_idx]

            # train and predict
            clf_copy = type(classifier)(**classifier.get_params())
            clf_copy.fit(X_train, y_train)

            y_pred_proba = clf_copy.predict_proba(X_val)[:, 1]
            auc = roc_auc_score(y_val, y_pred_proba)
            fold_results.append(auc)

            print(f"  Fold {fold + 1}: AUC = {auc:.4f}")

        mean_auc = np.mean(fold_results)
        std_auc = np.std(fold_results)

        results[name] = {
            'fold_aucs': fold_results,
            'mean_auc': mean_auc,
            'std_auc': std_auc,
            'cv_coeff': (std_auc / mean_auc) * 100
        }

        print(f"  Mean AUC: {mean_auc:.4f} ± {std_auc:.4f}")
        print(f"  CV: {results[name]['cv_coeff']:.1f}%")

        # 稳定性评估
        if std_auc < 0.05:
            stability = "优秀"
        elif std_auc < 0.1:
            stability = "良好"
        elif std_auc < 0.15:
            stability = "一般"
        else:
            stability = "较差"

        print(f"  稳定性: {stability}")

    return results, selected_features, scaler, selector


def feature_importance_analysis(X, y, feature_names):
    """Feature importance analysis"""

    print(f"\n🔍 Feature importance analysis")
    print("=" * 30)

    # Random forest for feature importance
    rf = RandomForestClassifier(n_estimators=100, random_state=42)
    rf.fit(X, y)

    importance_df = pd.DataFrame({
        'feature': feature_names,
        'importance': rf.feature_importances_
    }).sort_values('importance', ascending=False)

    print("top 20 most important features:")
    print(importance_df.head(20).to_string(index=False))

    # Visualization of top 15 features
    plt.figure(figsize=(10, 8))
    top_features = importance_df.head(15)
    plt.barh(range(len(top_features)), top_features['importance'], alpha=0.8)
    plt.yticks(range(len(top_features)), top_features['feature'])
    plt.xlabel('Feature Importance')
    plt.title('Top 15 Most Important Features')
    plt.gca().invert_yaxis()
    plt.tight_layout()
    plt.savefig('feature_importance.png', dpi=150)
    plt.show()

    return importance_df


def visualize_results(results):
    """Visualization of cross-validation results"""

    plt.figure(figsize=(15, 5))

    # Subplot 1: AUC comparison
    plt.subplot(1, 3, 1)
    methods = list(results.keys())
    mean_aucs = [results[method]['mean_auc'] for method in methods]
    std_aucs = [results[method]['std_auc'] for method in methods]

    bars = plt.bar(methods, mean_aucs, yerr=std_aucs, capsize=5, alpha=0.8)
    plt.axhline(y=0.8, color='red', linestyle='--', alpha=0.7, label='Clinical Threshold')
    plt.axhline(y=0.7, color='orange', linestyle='--', alpha=0.7, label='Good Performance')
    plt.ylabel('AUC')
    plt.title('Model Performance Comparison')
    plt.xticks(rotation=45)
    plt.legend()
    plt.ylim(0.4, 1.0)

    # Subplot 2: Stability comparison
    plt.subplot(1, 3, 2)
    cv_coeffs = [results[method]['cv_coeff'] for method in methods]
    colors = ['green' if cv < 10 else 'orange' if cv < 20 else 'red' for cv in cv_coeffs]

    plt.bar(methods, cv_coeffs, color=colors, alpha=0.8)
    plt.axhline(y=10, color='green', linestyle='--', alpha=0.7, label='Excellent (<10%)')
    plt.axhline(y=20, color='orange', linestyle='--', alpha=0.7, label='Good (<20%)')
    plt.ylabel('Coefficient of Variation (%)')
    plt.title('Stability Comparison')
    plt.xticks(rotation=45)
    plt.legend()

    # subplot3: fold-level AUC distributions
    plt.subplot(1, 3, 3)

    for i, method in enumerate(methods):
        fold_aucs = results[method]['fold_aucs']
        x_positions = [i + 1 + np.random.normal(0, 0.1) for _ in fold_aucs]
        plt.scatter(x_positions, fold_aucs, alpha=0.7, s=60, label=method)

    plt.axhline(y=0.8, color='red', linestyle='--', alpha=0.7)
    plt.axhline(y=0.7, color='orange', linestyle='--', alpha=0.7)
    plt.xlabel('Method')
    plt.ylabel('Fold AUC')
    plt.title('AUC Distribution Across Folds')
    plt.xticks(range(1, len(methods) + 1), methods, rotation=45)
    plt.legend()
    plt.ylim(0.4, 1.0)

    plt.tight_layout()
    plt.savefig('traditional_ml_results.png', dpi=150, bbox_inches='tight')
    plt.show()

