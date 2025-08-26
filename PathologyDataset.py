import numpy as np
import cv2
import os
from skimage import feature, measure
from skimage.filters import threshold_otsu
from skimage.color import rgb2gray, rgb2hsv
from skimage.util import img_as_ubyte
import warnings

warnings.filterwarnings('ignore')


class PathologyFeatureExtractor:
    """Pathology Image Feature Extractor - specialized for H&E staining"""

    def __init__(self):
        self.feature_names = []

    def extract_color_features(self, image):
        """Extract color features - H&E specific"""
        features = []
        names = []

        # RGB channel statistics
        for i, channel in enumerate(['R', 'G', 'B']):
            ch = image[:, :, i]
            features.extend([
                np.mean(ch), np.std(ch), np.median(ch),
                np.percentile(ch, 25), np.percentile(ch, 75),
                np.min(ch), np.max(ch)
            ])
            names.extend([f'{channel}_mean', f'{channel}_std', f'{channel}_median',
                          f'{channel}_p25', f'{channel}_p75', f'{channel}_min', f'{channel}_max'])

        # HSV color space
        hsv = rgb2hsv(image)
        for i, channel in enumerate(['H', 'S', 'V']):
            ch = hsv[:, :, i]
            features.extend([np.mean(ch), np.std(ch)])
            names.extend([f'{channel}_mean', f'{channel}_std'])

        # H&E-specific color ratios
        # Hematoxylin (blue-purple) vs Eosin (pink) approximate separation
        gray = rgb2gray(image)
        blue_ratio = np.mean(image[:, :, 2]) / (np.mean(gray) + 1e-6)  # blue-purple
        red_ratio = np.mean(image[:, :, 0]) / (np.mean(gray) + 1e-6)  # red
        he_contrast = abs(blue_ratio - red_ratio)  # H&E对比度

        features.extend([blue_ratio, red_ratio, he_contrast])
        names.extend(['hematoxylin_ratio', 'eosin_ratio', 'he_contrast'])

        return features, names

    def extract_texture_features(self, image):
        """Extract texture features"""
        features = []
        names = []

        gray = rgb2gray(image)
        gray_uint = img_as_ubyte(gray)

        # LBP (Local Binary Patterns)
        radius = 3
        n_points = 8 * radius
        lbp = feature.local_binary_pattern(gray, n_points, radius, method='uniform')

        # LBP histogram
        lbp_hist, _ = np.histogram(lbp.ravel(), bins=n_points + 2, range=(0, n_points + 2))
        lbp_hist = lbp_hist.astype(float)
        lbp_hist /= (lbp_hist.sum() + 1e-7)   # normalize

        for i in range(min(10, len(lbp_hist))):  # take only first 10 LBP features
            features.append(lbp_hist[i])
            names.append(f'lbp_bin_{i}')

        # GLCM (Gray Level Co-occurrence Matrix)
        distances = [1, 2]
        angles = [0, 45, 90, 135]

        for distance in distances:
            for angle in angles:
                glcm = feature.graycomatrix(gray_uint, [distance], [np.radians(angle)],
                                            symmetric=True, normed=True)

                # GLCM properties
                contrast = feature.graycoprops(glcm, 'contrast')[0, 0]
                dissimilarity = feature.graycoprops(glcm, 'dissimilarity')[0, 0]
                homogeneity = feature.graycoprops(glcm, 'homogeneity')[0, 0]
                energy = feature.graycoprops(glcm, 'energy')[0, 0]

                features.extend([contrast, dissimilarity, homogeneity, energy])
                names.extend([f'glcm_contrast_d{distance}_a{angle}',
                              f'glcm_dissim_d{distance}_a{angle}',
                              f'glcm_homog_d{distance}_a{angle}',
                              f'glcm_energy_d{distance}_a{angle}'])

        # Gabor filters - fallback for compatibility
        try:
            from skimage.filters import gabor
            frequencies = [0.1, 0.3]
            angles_gabor = [0, 45, 90, 135]

            for freq in frequencies:
                for angle in angles_gabor:
                    real, _ = gabor(gray, frequency=freq, theta=np.radians(angle))
                    features.extend([np.mean(real), np.std(real)])
                    names.extend([f'gabor_mean_f{freq}_a{angle}', f'gabor_std_f{freq}_a{angle}'])
        except ImportError:
            # fallback: gradient features if Gabor not available
            from skimage.filters import sobel_h, sobel_v

            grad_h = sobel_h(gray)
            grad_v = sobel_v(gray)
            grad_mag = np.sqrt(grad_h ** 2 + grad_v ** 2)

            features.extend([
                np.mean(grad_h), np.std(grad_h),
                np.mean(grad_v), np.std(grad_v),
                np.mean(grad_mag), np.std(grad_mag)
            ])
            names.extend([
                'gradient_h_mean', 'gradient_h_std',
                'gradient_v_mean', 'gradient_v_std',
                'gradient_mag_mean', 'gradient_mag_std'
            ])

        return features, names

    def extract_morphological_features(self, image):
        """Extract morphological features"""
        features = []
        names = []

        gray = rgb2gray(image)

        # Adaptive threshold segmentation
        try:
            thresh = threshold_otsu(gray)
            binary = gray > thresh
        except:
            binary = gray > 0.5

        # Region properties
        labeled = measure.label(binary)
        regions = measure.regionprops(labeled)

        if len(regions) > 0:
            # Number of regions
            features.append(len(regions))
            names.append('num_regions')

            # Region area statistics
            areas = [region.area for region in regions]
            features.extend([np.mean(areas), np.std(areas), np.max(areas)])
            names.extend(['area_mean', 'area_std', 'area_max'])

            # Shape features
            eccentricities = [region.eccentricity for region in regions if region.eccentricity > 0]
            solidity = [region.solidity for region in regions if region.solidity > 0]

            if eccentricities:
                features.extend([np.mean(eccentricities), np.std(eccentricities)])
                names.extend(['eccentricity_mean', 'eccentricity_std'])
            else:
                features.extend([0, 0])
                names.extend(['eccentricity_mean', 'eccentricity_std'])

            if solidity:
                features.extend([np.mean(solidity), np.std(solidity)])
                names.extend(['solidity_mean', 'solidity_std'])
            else:
                features.extend([0, 0])
                names.extend(['solidity_mean', 'solidity_std'])
        else:
            # default values when no regions detected
            features.extend([0, 0, 0, 0, 0, 0, 0, 0])
            names.extend(['num_regions', 'area_mean', 'area_std', 'area_max',
                          'eccentricity_mean', 'eccentricity_std', 'solidity_mean', 'solidity_std'])

        # Edge density
        edges = feature.canny(gray, sigma=1)
        edge_density = np.sum(edges) / edges.size
        features.append(edge_density)
        names.append('edge_density')

        return features, names

    def extract_statistical_features(self, image):
        """Extract statistical features"""
        features = []
        names = []

        gray = rgb2gray(image)

        # Basic statistics
        features.extend([
            np.mean(gray), np.std(gray), np.var(gray),
            np.median(gray), np.min(gray), np.max(gray),
            np.percentile(gray, 10), np.percentile(gray, 90),
            np.percentile(gray, 25), np.percentile(gray, 75)
        ])
        names.extend([
            'intensity_mean', 'intensity_std', 'intensity_var',
            'intensity_median', 'intensity_min', 'intensity_max',
            'intensity_p10', 'intensity_p90', 'intensity_p25', 'intensity_p75'
        ])

        # Higher order moments
        from scipy import stats
        features.extend([
            stats.skew(gray.flatten()),
            stats.kurtosis(gray.flatten())
        ])
        names.extend(['skewness', 'kurtosis'])

        return features, names

    def extract_all_features(self, image):
        """Extract all feature categories"""
        all_features = []
        all_names = []

        # resize for consistency
        if image.shape[:2] != (224, 224):
            image = cv2.resize(image, (224, 224))

        # extract features
        color_feat, color_names = self.extract_color_features(image)
        texture_feat, texture_names = self.extract_texture_features(image)
        morph_feat, morph_names = self.extract_morphological_features(image)
        stat_feat, stat_names = self.extract_statistical_features(image)

        all_features.extend(color_feat + texture_feat + morph_feat + stat_feat)
        all_names.extend(color_names + texture_names + morph_names + stat_names)

        self.feature_names = all_names
        return np.array(all_features)


def load_and_extract_features(image_paths, labels):
    """Load images and extract features - with enhanced error handling"""

    extractor = PathologyFeatureExtractor()
    features_list = []
    valid_labels = []
    valid_paths = []

    print("🔍 Extracting traditional machine learning features...")

    for i, (path, label) in enumerate(zip(image_paths, labels)):
        try:
            print(f"Processing image {i + 1}/{len(image_paths)}: {os.path.basename(path)}")

            # Read image
            image = cv2.imread(path)
            if image is None:
                print(f"⚠️ Cannot read image: {path}")
                continue

            # Check image size
            if image.shape[0] < 50 or image.shape[1] < 50:
                print(f"⚠️ Image too small: {path}, size: {image.shape}")
                continue

            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

            # Extract features
            features = extractor.extract_all_features(image)

            # Check feature validity
            if len(features) == 0:
                print(f"⚠️ No features extracted: {path}")
                continue

            if np.any(np.isnan(features)) or np.any(np.isinf(features)):
                print(f"⚠️ Image contains invalid features: {path}")
                # Replace invalid values with 0
                features = np.nan_to_num(features, nan=0.0, posinf=0.0, neginf=0.0)

            features_list.append(features)
            valid_labels.append(label)
            valid_paths.append(path)

            print(f"✅ Successfully extracted {len(features)} features")

        except Exception as e:
            print(f"❌ Error processing image {path}: {e}")
            import traceback
            print(f"Detailed error: {traceback.format_exc()}")
            continue

    if not features_list:
        print("❌ No features were successfully extracted!")
        print("Possible reasons:")
        print("1. Image files are corrupted or unsupported format")
        print("2. Wrong file paths")
        print("3. Missing required Python packages")
        print("\nPlease check:")
        print("- Whether the images can be opened normally")
        print("- Install: pip install scikit-image opencv-python")
        raise ValueError("No features were successfully extracted!")

    X = np.array(features_list)
    y = np.array(valid_labels)

    print(f"✅ Feature extraction completed:")
    print(f"   Number of valid images: {len(valid_paths)}")
    print(f"   Feature dimension: {X.shape[1]}")
    print(f"   Class distribution: {np.sum(y)} LNM, {len(y) - np.sum(y)} non-LNM")

    return X, y, extractor.feature_names, valid_paths