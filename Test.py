import os
import cv2
from PathologyDataset import load_and_extract_features
from Model import feature_importance_analysis, traditional_ml_pipeline, visualize_results
import warnings

warnings.filterwarnings('ignore')


def main():
    """Main function - enhanced for debugging"""

    print("🔬 Traditional Machine Learning for Pathology Image Classification")
    print("=" * 50)
    print("Use Case: Small datasets (< 100 images)")

    # Data path - auto detection
    possible_data_dirs = [
        './Data',
        './data',
        'Data',
        'data',
        'path/to/your/data'
    ]

    data_dir = None
    for dir_path in possible_data_dirs:
        if os.path.exists(dir_path):
            data_dir = dir_path
            print(f"✅ Found data directory: {data_dir}")
            break

    if data_dir is None:
        print("❌ Data directory not found!")
        print("Please ensure the data folder exists, with possible names:")
        for d in possible_data_dirs:
            print(f"  - {d}")
        print("\nOr modify the data_dir variable in the code")
        return

    # Try different subdirectory names
    lnm_dirs = ['LNM', 'lnm', 'LNM_images', '1']
    non_lnm_dirs = ['NOT-LNM', 'NON-LNM', 'non-LNM', 'non_lnm', 'normal', '0']

    lnm_dir = None
    non_lnm_dir = None

    for subdir in lnm_dirs:
        test_path = os.path.join(data_dir, subdir)
        if os.path.exists(test_path):
            lnm_dir = test_path
            print(f"✅ Found LNM directory: {lnm_dir}")
            break

    for subdir in non_lnm_dirs:
        test_path = os.path.join(data_dir, subdir)
        if os.path.exists(test_path):
            non_lnm_dir = test_path
            print(f"✅ Found non-LNM directory: {non_lnm_dir}")
            break

    if lnm_dir is None or non_lnm_dir is None:
        print("\n❌ Image subdirectories not found!")
        print(f"Searching in {data_dir}:")
        print(f"LNM class image directories: {lnm_dirs}")
        print(f"non-LNM class image directories: {non_lnm_dirs}")

        # List the subdirectories that actually exist
        if os.path.exists(data_dir):
            subdirs = [d for d in os.listdir(data_dir) if os.path.isdir(os.path.join(data_dir, d))]
            print(f"Actual existing subdirectories: {subdirs}")
        return

    # Collect image paths and labels
    image_paths = []
    labels = []

    # Supported image formats
    supported_formats = ('.jpg', '.jpeg', '.png', '.tiff', '.tif', '.bmp')

    # LNM images
    print(f"\n🔍 Scanning LNM images...")
    lnm_count = 0
    for img_file in sorted(os.listdir(lnm_dir)):
        if img_file.lower().endswith(supported_formats):
            full_path = os.path.join(lnm_dir, img_file)
            image_paths.append(full_path)
            labels.append(1)
            lnm_count += 1
            print(f"  Found: {img_file}")

    # non-LNM images
    print(f"\n🔍 Scanning non-LNM images...")
    non_lnm_count = 0
    for img_file in sorted(os.listdir(non_lnm_dir)):
        if img_file.lower().endswith(supported_formats):
            full_path = os.path.join(non_lnm_dir, img_file)
            image_paths.append(full_path)
            labels.append(0)
            non_lnm_count += 1
            print(f"  Found: {img_file}")

    print(f"\n📊 Dataset Information:")
    print(f"  Total Images: {len(image_paths)}")
    print(f"  LNM: {lnm_count}")
    print(f"  non-LNM: {non_lnm_count}")

    if len(image_paths) == 0:
        print("❌ No image files found!")
        print(f"Supported formats: {supported_formats}")
        return

    if len(image_paths) < 10:
        print("⚠️ Low number of images, results may be unstable")

    # Test reading the first image
    print(f"\n🧪 Testing image reading...")
    test_image = cv2.imread(image_paths[0])
    if test_image is None:
        print(f"❌ Could not read test image: {image_paths[0]}")
        return
    else:
        print(f"✅ Successfully read test image, dimensions: {test_image.shape}")

    try:
        # Feature extraction
        X, y, feature_names, valid_paths = load_and_extract_features(image_paths, labels)

        # Feature importance analysis
        importance_df = feature_importance_analysis(X, y, feature_names)

        # Traditional machine learning pipeline
        results, selected_features, scaler, selector = traditional_ml_pipeline(X, y, feature_names)

        # Visualize results
        visualize_results(results)

        # Final report
        print(f"\n" + "=" * 60)
        print("📋 Final Report - Traditional Machine Learning Method")
        print("=" * 60)

        best_method = max(results.keys(), key=lambda x: results[x]['mean_auc'])
        best_result = results[best_method]

        print(f"🏆 Best Method: {best_method}")
        print(f"   Mean AUC: {best_result['mean_auc']:.4f} ± {best_result['std_auc']:.4f}")
        print(f"   CV: {best_result['cv_coeff']:.1f}%")

        # Comparison with deep learning results
        print(f"\n📊 Method Comparison:")
        print(f"Deep Learning (Conservative):  0.57 ± 0.11 (CV: 20%)")
        print(f"Deep Learning (Original):  0.79 ± 0.18 (CV: 23%, unstable)")
        print(
            f"Traditional ML ({best_method}): {best_result['mean_auc']:.2f} ± {best_result['std_auc']:.2f} (CV: {best_result['cv_coeff']:.1f}%)")

        if best_result['cv_coeff'] < 20:
            print(f"✅ Traditional machine learning shows better stability!")

        # Clinical usability assessment
        clinical_folds = sum(1 for auc in best_result['fold_aucs'] if auc >= 0.8)
        print(f"\n🏥 Clinical Usability:")
        print(f"   Folds reaching clinical threshold (≥0.8): {clinical_folds}/5")

        if clinical_folds >= 3:
            print(f"   ✅ Clinically considerable for use")
        elif best_result['mean_auc'] >= 0.7:
            print(f"   ⚠️ Has potential, requires validation with more data")
        else:
            print(f"   ❌ Insufficient performance, more data collection needed")

        print(f"\n💡 Recommendations:")
        print(f"1. Traditional machine learning is more suitable for your small dataset")
        print(f"2. Focus on feature engineering and data quality")
        print(f"3. Consider collecting more data to improve performance")
        print(f"4. You could try semi-supervised learning methods")

    except Exception as e:
        print(f"\n❌ An error occurred during execution: {e}")
        import traceback
        print("Detailed error information:")
        print(traceback.format_exc())

        print("\n🔧 Troubleshooting Suggestions:")
        print("1. Ensure all dependencies are installed:")
        print("   pip install scikit-image scikit-learn opencv-python matplotlib seaborn pandas")
        print("2. Check the integrity of the image files")
        print("3. Confirm the data directory structure is correct")


if __name__ == '__main__':
    main()