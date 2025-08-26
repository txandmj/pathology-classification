import torch
from torch.utils.data import DataLoader
from sklearn.model_selection import StratifiedKFold
from TrainModel import get_conservative_transforms, train_conservative_model
from PathologyDataset import DiagnosticPathologyDataset
from Model import ConservativeModel
import numpy as np
import cv2
import os
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings('ignore')


def data_leakage_check(image_paths, labels, fold_splits):
    """检查数据泄漏"""

    print("\n🔍 Data Leakage Check:")
    print("=" * 40)

    for fold_idx, (train_idx, val_idx) in enumerate(fold_splits):
        train_paths = [image_paths[i] for i in train_idx]
        val_paths = [image_paths[i] for i in val_idx]

        # 检查文件名重叠
        train_names = set([os.path.basename(p) for p in train_paths])
        val_names = set([os.path.basename(p) for p in val_paths])
        overlap = train_names.intersection(val_names)

        print(f"Fold {fold_idx + 1}:")
        print(f"  Train images: {len(train_paths)}")
        print(f"  Val images: {len(val_paths)}")
        print(f"  Filename overlap: {len(overlap)} {'❌' if overlap else '✅'}")

        if overlap:
            print(f"    Overlapping files: {list(overlap)[:5]}...")  # 显示前5个

        # 检查类别分布
        train_labels = [labels[i] for i in train_idx]
        val_labels = [labels[i] for i in val_idx]

        print(f"  Train LNM/non-LNM: {sum(train_labels)}/{len(train_labels) - sum(train_labels)}")
        print(f"  Val LNM/non-LNM: {sum(val_labels)}/{len(val_labels) - sum(val_labels)}")

        # 检查极端不平衡
        if sum(val_labels) == 0 or sum(val_labels) == len(val_labels):
            print(f"  ⚠️  WARNING: Validation set has only one class!")


def diagnostic_cross_validate(image_paths, labels, n_splits=5):
    """诊断性交叉验证"""

    # 首先进行数据泄漏检查
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)
    fold_splits = list(skf.split(image_paths, labels))
    data_leakage_check(image_paths, labels, fold_splits)

    cv_results = []
    detailed_results = []

    train_transform, val_transform = get_conservative_transforms()

    for fold, (train_idx, val_idx) in enumerate(fold_splits):
        print(f'\n{"=" * 25} Fold {fold + 1}/{n_splits} {"=" * 25}')

        # 划分数据
        train_paths = [image_paths[i] for i in train_idx]
        train_labels = [labels[i] for i in train_idx]
        val_paths = [image_paths[i] for i in val_idx]
        val_labels = [labels[i] for i in val_idx]

        # 检查类别分布
        train_lnm = sum(train_labels)
        val_lnm = sum(val_labels)
        print(f'Train: {len(train_paths)} images ({train_lnm} LNM, {len(train_labels) - train_lnm} non-LNM)')
        print(f'Val: {len(val_paths)} images ({val_lnm} LNM, {len(val_labels) - val_lnm} non-LNM)')

        # 如果验证集类别不平衡，跳过此fold
        if val_lnm == 0 or val_lnm == len(val_labels):
            print("❌ Skipping fold due to single-class validation set")
            continue

        # 创建数据集（减少patch数量）
        train_dataset = DiagnosticPathologyDataset(
            train_paths, train_labels, train_transform,
            patch_size=224, patches_per_image=2, seed=42 + fold  # 每个fold不同的种子
        )
        val_dataset = DiagnosticPathologyDataset(
            val_paths, val_labels, val_transform,
            patch_size=224, patches_per_image=1, seed=42 + fold  # 验证时只用1个patch
        )

        # 小batch size
        train_loader = DataLoader(train_dataset, batch_size=4, shuffle=True,
                                  num_workers=0, drop_last=True)  # 避免多进程问题
        val_loader = DataLoader(val_dataset, batch_size=4, shuffle=False,
                                num_workers=0)

        # 创建保守模型
        model = ConservativeModel(num_classes=2, dropout_rate=0.7)

        # 训练
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        print(f'Using device: {device}')

        model, history = train_conservative_model(model, train_loader, val_loader,
                                                  num_epochs=15, device=device)

        # 评估
        best_auc = max(history['val_auc']) if history['val_auc'] else 0.5
        cv_results.append(best_auc)

        # 详细结果
        fold_result = {
            'fold': fold + 1,
            'best_auc': best_auc,
            'final_train_loss': history['train_loss'][-1] if history['train_loss'] else 0,
            'final_val_loss': history['val_loss'][-1] if history['val_loss'] else 0,
            'overfitting_gap': history['val_loss'][-1] - history['train_loss'][-1] if history['val_loss'] and history[
                'train_loss'] else 0,
            'train_size': len(train_dataset),
            'val_size': len(val_dataset)
        }
        detailed_results.append(fold_result)

        print(f'Fold {fold + 1} Results:')
        print(f'  Best AUC: {best_auc:.4f}')
        print(f'  Overfitting gap: {fold_result["overfitting_gap"]:.4f}')

        # 绘制此fold的训练曲线
        if history['val_auc']:
            plot_fold_analysis(history, fold_result, fold)

    return cv_results, detailed_results


def plot_fold_analysis(history, fold_result, fold_num):
    """绘制单个fold的详细分析"""

    fig, axes = plt.subplots(2, 3, figsize=(15, 8))

    # 损失曲线
    axes[0, 0].plot(history['train_loss'], 'b-', label='Train', linewidth=2)
    axes[0, 0].plot(history['val_loss'], 'r-', label='Validation', linewidth=2)
    axes[0, 0].set_title('Loss Curves')
    axes[0, 0].set_xlabel('Epoch')
    axes[0, 0].set_ylabel('Loss')
    axes[0, 0].legend()
    axes[0, 0].grid(True, alpha=0.3)

    # AUC曲线
    axes[0, 1].plot(history['val_auc'], 'g-', linewidth=2)
    axes[0, 1].axhline(y=1.0, color='r', linestyle='--', alpha=0.7, label='Perfect AUC')
    axes[0, 1].axhline(y=0.8, color='orange', linestyle='--', alpha=0.7, label='Clinical Threshold')
    axes[0, 1].set_title('Validation AUC')
    axes[0, 1].set_xlabel('Epoch')
    axes[0, 1].set_ylabel('AUC')
    axes[0, 1].legend()
    axes[0, 1].grid(True, alpha=0.3)
    axes[0, 1].set_ylim(0.4, 1.05)

    # 过拟合分析
    overfitting_gap = np.array(history['val_loss']) - np.array(history['train_loss'])
    axes[0, 2].plot(overfitting_gap, 'purple', linewidth=2)
    axes[0, 2].axhline(y=0, color='k', linestyle='-', alpha=0.3)
    axes[0, 2].axhline(y=0.1, color='orange', linestyle='--', alpha=0.7, label='Warning Threshold')
    axes[0, 2].set_title('Overfitting Gap')
    axes[0, 2].set_xlabel('Epoch')
    axes[0, 2].set_ylabel('Val Loss - Train Loss')
    axes[0, 2].legend()
    axes[0, 2].grid(True, alpha=0.3)

    # 准确率对比
    axes[1, 0].plot(history['train_acc'], 'b-', label='Train', linewidth=2)
    axes[1, 0].plot(history['val_acc'], 'r-', label='Validation', linewidth=2)
    axes[1, 0].set_title('Accuracy Curves')
    axes[1, 0].set_xlabel('Epoch')
    axes[1, 0].set_ylabel('Accuracy (%)')
    axes[1, 0].legend()
    axes[1, 0].grid(True, alpha=0.3)

    # 最终指标总结
    axes[1, 1].axis('off')
    summary_text = f"""
Fold {fold_num + 1} Summary:

Best AUC: {fold_result['best_auc']:.4f}
Final Train Loss: {fold_result['final_train_loss']:.4f}
Final Val Loss: {fold_result['final_val_loss']:.4f}
Overfitting Gap: {fold_result['overfitting_gap']:.4f}

Dataset Sizes:
Train Patches: {fold_result['train_size']}
Val Patches: {fold_result['val_size']}

Health Check:
{'✅ Healthy' if fold_result['overfitting_gap'] < 0.1 and fold_result['best_auc'] < 0.98 else '⚠️  Needs Attention'}
    """
    axes[1, 1].text(0.1, 0.5, summary_text, fontsize=10, verticalalignment='center',
                    bbox=dict(boxstyle="round,pad=0.3", facecolor="lightgray", alpha=0.5))

    # 稳定性分析
    if len(history['val_auc']) > 5:
        last_5_auc = history['val_auc'][-5:]
        auc_std = np.std(last_5_auc)
        axes[1, 2].plot(range(len(history['val_auc'])), history['val_auc'], 'g-', alpha=0.7)
        axes[1, 2].plot(range(len(history['val_auc']) - 5, len(history['val_auc'])),
                        last_5_auc, 'r-', linewidth=3, label=f'Last 5 epochs (std={auc_std:.3f})')
        axes[1, 2].set_title('AUC Stability')
        axes[1, 2].set_xlabel('Epoch')
        axes[1, 2].set_ylabel('AUC')
        axes[1, 2].legend()
        axes[1, 2].grid(True, alpha=0.3)

    plt.suptitle(f'Diagnostic Analysis - Fold {fold_num + 1}', fontsize=14)
    plt.tight_layout()
    plt.savefig(f'diagnostic_fold_{fold_num + 1}.png', dpi=150, bbox_inches='tight')
    plt.show()


def final_diagnostic_report(cv_results, detailed_results):
    """生成最终诊断报告"""

    print('\n' + '=' * 70)
    print('📋 FINAL DIAGNOSTIC REPORT')
    print('=' * 70)

    if not cv_results:
        print("❌ No valid results obtained!")
        return

    mean_auc = np.mean(cv_results)
    std_auc = np.std(cv_results)

    print(f'\n📊 Performance Summary:')
    print(f'  Folds completed: {len(cv_results)}/5')
    print(f'  Mean AUC: {mean_auc:.4f} ± {std_auc:.4f}')
    print(f'  Min AUC: {min(cv_results):.4f}')
    print(f'  Max AUC: {max(cv_results):.4f}')

    # 稳定性分析
    auc_range = max(cv_results) - min(cv_results)
    print(f'\n🎯 Stability Analysis:')
    print(f'  AUC Range: {auc_range:.4f}')

    if auc_range > 0.2:
        print(f'  ⚠️  HIGH INSTABILITY: Results vary significantly across folds')
    elif auc_range > 0.1:
        print(f'  ⚠️  MODERATE INSTABILITY: Some variation across folds')
    else:
        print(f'  ✅ STABLE: Consistent performance across folds')

    # 过拟合检查
    perfect_count = sum(1 for auc in cv_results if auc >= 0.99)
    print(f'\n🔍 Overfitting Check:')
    print(f'  Folds with AUC ≥ 0.99: {perfect_count}/{len(cv_results)}')

    if perfect_count > 0:
        print(f'  ⚠️  WARNING: {perfect_count} fold(s) show signs of severe overfitting')
        print(f'     Consider: smaller model, more regularization, more data')
    else:
        print(f'  ✅ No signs of severe overfitting')

    # 临床可用性评估
    print(f'\n🏥 Clinical Viability:')
    clinical_threshold = 0.8
    clinical_folds = sum(1 for auc in cv_results if auc >= clinical_threshold)

    if mean_auc >= clinical_threshold and std_auc < 0.1:
        print(f'  ✅ CLINICALLY VIABLE: High performance with low variance')
    elif mean_auc >= clinical_threshold:
        print(f'  ⚠️  PROMISING BUT UNSTABLE: Good average but high variance')
    else:
        print(f'  ❌ NOT READY: Performance below clinical threshold')

    print(f'  Folds meeting clinical threshold (≥{clinical_threshold}): {clinical_folds}/{len(cv_results)}')

    # 建议
    print(f'\n💡 Recommendations:')
    if perfect_count > 0:
        print(f'  1. 🔧 Reduce model complexity (smaller network, more dropout)')
        print(f'  2. 📊 Check for data leakage between train/val sets')
        print(f'  3. 🎯 Increase regularization (weight decay, early stopping)')

    if std_auc > 0.15:
        print(f'  4. 📈 Collect more data to improve stability')
        print(f'  5. 🔄 Use ensemble methods to reduce variance')

    if mean_auc < clinical_threshold:
        print(f'  6. 🎨 Try different architectures or pre-trained models')
        print(f'  7. 🔍 Analyze misclassified cases for insights')

    # 绘制综合分析图
    plot_comprehensive_analysis(cv_results, detailed_results)


def plot_comprehensive_analysis(cv_results, detailed_results):
    """绘制综合分析图表"""

    if not cv_results:
        return

    fig, axes = plt.subplots(2, 3, figsize=(18, 10))

    # 1. 各fold AUC对比
    axes[0, 0].bar(range(1, len(cv_results) + 1), cv_results,
                   color=['red' if auc >= 0.99 else 'orange' if auc >= 0.8 else 'lightcoral'
                          for auc in cv_results], alpha=0.7)
    axes[0, 0].axhline(y=0.8, color='green', linestyle='--', alpha=0.7, label='Clinical Threshold')
    axes[0, 0].axhline(y=0.99, color='red', linestyle='--', alpha=0.7, label='Overfitting Alert')
    axes[0, 0].set_xlabel('Fold')
    axes[0, 0].set_ylabel('AUC')
    axes[0, 0].set_title('AUC by Fold')
    axes[0, 0].legend()
    axes[0, 0].grid(True, alpha=0.3)

    # 2. AUC分布直方图
    axes[0, 1].hist(cv_results, bins=min(10, len(cv_results)),
                    color='skyblue', alpha=0.7, edgecolor='black')
    axes[0, 1].axvline(x=np.mean(cv_results), color='red', linestyle='--',
                       label=f'Mean: {np.mean(cv_results):.3f}')
    axes[0, 1].axvline(x=0.8, color='green', linestyle='--', alpha=0.7, label='Clinical')
    axes[0, 1].set_xlabel('AUC')
    axes[0, 1].set_ylabel('Frequency')
    axes[0, 1].set_title('AUC Distribution')
    axes[0, 1].legend()

    # 3. 过拟合分析
    if detailed_results:
        overfitting_gaps = [r['overfitting_gap'] for r in detailed_results]
        axes[0, 2].bar(range(1, len(overfitting_gaps) + 1), overfitting_gaps,
                       color=['red' if gap > 0.2 else 'orange' if gap > 0.1 else 'green'
                              for gap in overfitting_gaps], alpha=0.7)
        axes[0, 2].axhline(y=0.1, color='orange', linestyle='--', alpha=0.7, label='Warning')
        axes[0, 2].axhline(y=0.2, color='red', linestyle='--', alpha=0.7, label='Danger')
        axes[0, 2].set_xlabel('Fold')
        axes[0, 2].set_ylabel('Overfitting Gap')
        axes[0, 2].set_title('Overfitting Analysis')
        axes[0, 2].legend()
        axes[0, 2].grid(True, alpha=0.3)

    # 4. 稳定性分析
    axes[1, 0].boxplot(cv_results, labels=['AUC'])
    axes[1, 0].axhline(y=0.8, color='green', linestyle='--', alpha=0.7)
    axes[1, 0].set_ylabel('AUC')
    axes[1, 0].set_title('AUC Stability (Boxplot)')
    axes[1, 0].grid(True, alpha=0.3)

    # 5. 数据集大小vs性能
    if detailed_results:
        train_sizes = [r['train_size'] for r in detailed_results]
        axes[1, 1].scatter(train_sizes, cv_results,
                           c=['red' if auc >= 0.99 else 'blue' for auc in cv_results],
                           alpha=0.7, s=100)
        axes[1, 1].set_xlabel('Training Set Size (patches)')
        axes[1, 1].set_ylabel('AUC')
        axes[1, 1].set_title('Dataset Size vs Performance')
        axes[1, 1].grid(True, alpha=0.3)

    # 6. 总结统计
    axes[1, 2].axis('off')

    mean_auc = np.mean(cv_results)
    std_auc = np.std(cv_results)
    perfect_count = sum(1 for auc in cv_results if auc >= 0.99)
    clinical_count = sum(1 for auc in cv_results if auc >= 0.8)

    summary_text = f"""
📊 SUMMARY STATISTICS

Performance:
• Mean AUC: {mean_auc:.4f}
• Std AUC: {std_auc:.4f}
• Range: {min(cv_results):.4f} - {max(cv_results):.4f}

Stability:
• CV: {(std_auc / mean_auc) * 100:.1f}%
• {'✅ Stable' if std_auc < 0.1 else '⚠️ Unstable'}

Clinical Viability:
• Folds ≥ 0.80: {clinical_count}/{len(cv_results)}
• {'✅ Ready' if clinical_count >= 4 else '❌ Not Ready'}

Overfitting:
• Perfect AUC: {perfect_count}/{len(cv_results)}
• {'⚠️ Suspected' if perfect_count > 0 else '✅ None detected'}
    """

    axes[1, 2].text(0.05, 0.5, summary_text, fontsize=12,
                    verticalalignment='center', fontfamily='monospace',
                    bbox=dict(boxstyle="round,pad=0.5", facecolor="lightblue", alpha=0.8))

    plt.suptitle('Comprehensive Model Diagnostic Report', fontsize=16, fontweight='bold')
    plt.tight_layout()
    plt.savefig('comprehensive_diagnostic_report.png', dpi=150, bbox_inches='tight')
    plt.show()


def main():
    """主函数 - 诊断版本"""

    print("🔍 启动病理图像分类诊断系统")
    print("=" * 60)

    # 数据路径设置
    data_dir = './Data'  # 请修改为实际路径

    print("📁 加载数据中...")

    # 收集图像路径和标签
    lnm_dir = os.path.join(data_dir, 'LNM')
    non_lnm_dir = os.path.join(data_dir, 'NOT-LNM')

    image_paths = []
    labels = []

    # LNM图像
    if os.path.exists(lnm_dir):
        for img_file in sorted(os.listdir(lnm_dir)):  # 排序确保一致性
            if img_file.lower().endswith(('.jpg', '.jpeg', '.png', '.tiff', '.tif')):
                image_paths.append(os.path.join(lnm_dir, img_file))
                labels.append(1)

    # non-LNM图像
    if os.path.exists(non_lnm_dir):
        for img_file in sorted(os.listdir(non_lnm_dir)):  # 排序确保一致性
            if img_file.lower().endswith(('.jpg', '.jpeg', '.png', '.tiff', '.tif')):
                image_paths.append(os.path.join(non_lnm_dir, img_file))
                labels.append(0)

    print(f'📊 数据集概览:')
    print(f'  总图像数: {len(image_paths)}')
    print(f'  LNM: {sum(labels)}')
    print(f'  non-LNM: {len(labels) - sum(labels)}')
    print(f'  类别比例: {sum(labels) / len(labels):.3f} (LNM)')

    if len(image_paths) == 0:
        print("❌ 未找到图像文件!")
        print("   请检查数据目录路径设置")
        return

    if len(set(labels)) < 2:
        print("❌ 只找到一个类别的数据!")
        print("   请确保LNM和non-LNM文件夹都包含图像")
        return

    # 数据质量检查
    print(f'\n🔍 数据质量检查:')

    # 检查图像文件完整性
    valid_paths = []
    valid_labels = []

    for path, label in zip(image_paths, labels):
        try:
            img = cv2.imread(path)
            if img is not None:
                valid_paths.append(path)
                valid_labels.append(label)
            else:
                print(f"  ⚠️  无法读取: {os.path.basename(path)}")
        except Exception as e:
            print(f"  ❌ 错误文件: {os.path.basename(path)} - {e}")

    image_paths = valid_paths
    labels = valid_labels

    print(f'  有效图像: {len(image_paths)}/{len(image_paths)}')

    if len(image_paths) < 10:
        print("⚠️  警告: 图像数量太少，可能影响交叉验证结果")

    # 执行诊断性交叉验证
    print(f'\n🔄 启动诊断性5折交叉验证...')
    print(f'⚙️  配置: 保守模型 + 强正则化 + 数据泄漏检测')

    cv_results, detailed_results = diagnostic_cross_validate(image_paths, labels, n_splits=5)

    # 生成最终诊断报告
    final_diagnostic_report(cv_results, detailed_results)

    print(f'\n✅ 诊断完成! 请查看生成的图表和报告。')

    # 给出具体建议
    if cv_results:
        mean_auc = np.mean(cv_results)
        std_auc = np.std(cv_results)
        perfect_count = sum(1 for auc in cv_results if auc >= 0.99)

        print(f'\n💡 针对您当前结果的具体建议:')

        if perfect_count > 0:
            print(f'  🚨 紧急: {perfect_count}个fold出现完美AUC，强烈建议:')
            print(f'     - 检查是否有数据泄漏')
            print(f'     - 减少model复杂度')
            print(f'     - 增强正则化')

        if std_auc > 0.15:
            print(f'  📊 稳定性差，建议:')
            print(f'     - 收集更多数据')
            print(f'     - 使用ensemble方法')

        if mean_auc < 0.8:
            print(f'  📈 性能待提升，建议:')
            print(f'     - 尝试不同的预训练模型')
            print(f'     - 调整数据增强策略')
            print(f'     - 分析错误分类案例')


if __name__ == '__main__':
    main()