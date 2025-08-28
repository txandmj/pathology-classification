import streamlit as st
import torch
import torch.nn as nn
import torchvision.models as models
import torchvision.transforms as transforms
from PIL import Image
import joblib

# --- 1. Define Model Architecture and Loading Functions ---
# (This part of the code needs to be consistent with your training script)

class PathologyFeatureExtractor(nn.Module):
    """Load the pre-trained model for feature extraction"""

    def __init__(self, backbone='resnet50'):
        super(PathologyFeatureExtractor, self).__init__()
        if backbone == 'resnet50':
            self.backbone = models.resnet50(pretrained=False)  # Set to False here, as we will be loading our own weights
            self.feature_dim = self.backbone.fc.in_features
            self.backbone.fc = nn.Identity()

        # Define an empty classifier head to match the saved model's structure
        self.pretrain_classifier = nn.Sequential(
            nn.Dropout(0.5), nn.Linear(self.feature_dim, 512), nn.ReLU(inplace=True),
            nn.BatchNorm1d(512), nn.Dropout(0.3), nn.Linear(512, 9)
        )
        self.finetune_classifier = nn.Sequential(
            nn.Dropout(0.3), nn.Linear(self.feature_dim, 128), nn.ReLU(inplace=True),
            nn.Dropout(0.2), nn.Linear(128, 2)
        )

    def forward(self, x):
        return self.backbone(x)


# Use Streamlit's caching feature to avoid reloading the model every time
@st.cache_resource
def load_models():
    """Load all required models and preprocessors"""
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Load the feature extractor
    feature_extractor = PathologyFeatureExtractor(backbone='resnet50')
    checkpoint = torch.load('pretrained_nctcrc_model.pth', map_location=device)
    feature_extractor.load_state_dict(checkpoint['model_state_dict'])
    feature_extractor = feature_extractor.to(device)
    feature_extractor.eval()

    # Load the final classifier and scaler
    classifier = joblib.load('Models/final_lnm_classifier.joblib')
    scaler = joblib.load('Models/final_scaler.joblib')

    return feature_extractor, classifier, scaler, device


# --- 2. Define Image Processing and Prediction Functions ---

def process_image(image_bytes):
    """Convert the uploaded image file into the Tensor required by the model"""
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    image = Image.open(image_bytes).convert("RGB")
    return transform(image).unsqueeze(0)


def predict(image_tensor, feature_extractor, classifier, scaler, device):
    """Execute the complete prediction pipeline"""
    with torch.no_grad():
        # 1. Extract deep features
        image_tensor = image_tensor.to(device)
        features = feature_extractor(image_tensor)
        features = features.cpu().numpy()

        # 2. Standardize features
        features_scaled = scaler.transform(features)

        # 3. Predict probabilities
        probability = classifier.predict_proba(features_scaled)[0]

        return probability


# --- 3. Build the Streamlit User Interface ---

st.set_page_config(page_title="Lymph Node Metastasis Diagnosis", layout="wide")

st.title("🔬 Intelligent Assistant for Lymph Node Metastasis Diagnosis")
st.write("Upload an H&E stained pathology image, and the model will predict the presence of Lymph Node Metastasis (LNM).")

# Load models
with st.spinner('Loading models, please wait...'):
    feature_extractor, classifier, scaler, device = load_models()
st.success('Models loaded successfully!')

# Create file uploader component
uploaded_file = st.file_uploader("Please upload your pathology image here...", type=["jpg", "jpeg", "png", "tif", "tiff"])

if uploaded_file is not None:
    # Display the uploaded image
    st.image(uploaded_file, caption='Uploaded Image', use_column_width=True)
    st.write("")

    # Perform prediction
    with st.spinner('Model is analyzing the image...'):
        image_tensor = process_image(uploaded_file)
        probabilities = predict(image_tensor, feature_extractor, classifier, scaler, device)

    st.subheader("📈 Diagnosis Results")

    # Parse the results
    prob_non_lnm = probabilities[0]
    prob_lnm = probabilities[1]
    prediction = "LNM (Lymph Node Metastasis)" if prob_lnm > prob_non_lnm else "non-LNM (No Lymph Node Metastasis)"
    confidence = prob_lnm if prediction == "LNM (Lymph Node Metastasis)" else prob_non_lnm

    # Display results in different colors based on the prediction
    if prediction == "LNM (Lymph Node Metastasis)":
        st.error(f"**Prediction: {prediction}**")
    else:
        st.success(f"**Prediction: {prediction}**")

    st.metric(label="Confidence", value=f"{confidence:.2%}")

    # Display probability bar
    st.progress(prob_lnm)
    st.write(f"Probability of LNM: {prob_lnm:.2%}, Probability of non-LNM: {prob_non_lnm:.2%}")