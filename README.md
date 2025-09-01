# Lymph Node Metastasis Classification using Deep Learning

This project explores automated lymph node metastasis (LNM) detection in H&E stained pathological images using a hybrid deep learning + machine learning approach.

Challenge: Only 38 labeled clinical images available (22 LNM, 16 non-LNM)

Solution: Pre-train ResNet-50 on 100k colorectal cancer patches (NCT-CRC-HE-100K), then use it as a frozen feature extractor

Classifier: Logistic Regression on extracted features

Result: 0.832 AUC, with 4/5 folds reaching clinical-grade performance (AUC ≥ 0.8)

Key Contributions

Overcomes small dataset limitation with pre-trained features

Rigorous 5-fold stratified cross-validation with leakage prevention

Deployable ML pipeline (joblib + pretrained model)

Clinical significance: demonstrates feasibility of AI-assisted pathology with minimal data

📂 Code, models, and usage instructions provided for reproducibility.

⚠️ Research only — not for clinical use.

## Usage Instructions

### Environment Setup
```bash
pip install torch torchvision scikit-learn matplotlib seaborn tqdm joblib
pip install datasets huggingface-hub  # For NCT-CRC dataset
```

### Running the Complete Pipeline
```bash
# Step 1: Pre-training (optional - can use provided weights)
python src/pretrain_pipeline.py

# Step 2: Feature extraction and classification
python src/corrected_feature_classifier.py

# Step 3: Model will be saved as final_lnm_logistic_regression.joblib to Models folder
```

### Inference on New Images
```python
import joblib
from src.feature_extraction import extract_features

# Load trained pipeline
pipeline = joblib.load('final_lnm_logistic_regression.joblib')

# Extract features from new image
features = extract_features(new_image_path)

# Predict
prediction = pipeline.predict_proba(features.reshape(1, -1))
lnm_probability = prediction[0, 1]
```

## Research Contributions

### Methodological Innovation
- Demonstrated effectiveness of feature extraction over fine-tuning for small medical datasets
- Developed rigorous validation framework preventing data leakage
- Created reproducible pipeline for small-sample medical AI

### Clinical Impact
- Achieved clinical-grade performance with minimal training data
- Provided foundation for AI-assisted pathological diagnosis
- Established benchmark for small-sample medical image classification

### Technical Excellence
- Implemented state-of-the-art safeguards against common ML pitfalls
- Created interpretable and deployable model pipeline
- Demonstrated proper statistical validation in medical AI context

## Limitations and Future Work

### Current Limitations
- Small validation dataset (38 images) limits generalizability assessment
- Single-institution data source may not represent population diversity
- Feature extraction approach may miss task-specific fine details

### Recommended Extensions
1. **Scale Validation**: Test on larger, multi-institutional datasets
2. **Comparison Studies**: Benchmark against practicing pathologists
3. **Integration Development**: Build clinical workflow integration tools
4. **Explainability**: Develop feature visualization for clinical interpretation

## Research Ethics and Disclaimers

**Research Purpose Only**: This system is developed for research and educational purposes. Not intended for clinical diagnosis or medical decision-making.

**Data Privacy**: All medical images processed with appropriate privacy safeguards. No patient data stored or transmitted.

**Clinical Oversight**: Any clinical application requires appropriate medical supervision and validation studies.


## Contact and Collaboration

For research collaboration, questions, or access to additional materials:
- Email: txandmj@outlook.com
- GitHub: https://github.com/txandmj/pathology-classification 
- Portfolio: https://txandmj.github.io/pathology-classification-showcase/

## Acknowledgments

- Medical collaborators for providing labeled pathological images
- HuggingFace community for NCT-CRC-HE-100K dataset access
- Open-source scientific computing community for tools and frameworks

---

*This project demonstrates the potential for AI-assisted medical diagnosis while emphasizing the importance of rigorous methodology, clinical validation, and ethical implementation in medical AI research.*
