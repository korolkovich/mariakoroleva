# app.py
import streamlit as st
import torch
import numpy as np
from PIL import Image
from torchvision import transforms
from torchvision.models import (
    efficientnet_b0, efficientnet_b3, efficientnet_b2,
    convnext_small,
    densenet121,
    resnet50
)
from pytorch_grad_cam import GradCAM
from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget
from pytorch_grad_cam.utils.image import show_cam_on_image
import openai
from openai import OpenAI, AuthenticationError, APIError
from collections import OrderedDict
import torch.nn as nn
import pandas as pd
import os

# === Page styling (Dark Theme) ===
st.set_page_config(page_title="Skin Lesion Classifier", layout="wide", page_icon="🧪")
st.markdown("""
    <style>
        /* Target root and body elements */
        :root, body, .stApp { background-color: #0E1117 !important; color: #FAFAFA !important; }
        .main, .block-container, section[data-testid="stSidebar"] > div:first-child { background-color: #0E1117 !important; color: #FAFAFA !important; font-family: 'Segoe UI', 'Roboto', sans-serif; font-size: 16px; }
        header[data-testid="stHeader"] { background-color: #0E1117 !important; }
        p, div, span, li, label, .stMarkdown, .stText, th, td { color: #FAFAFA !important; }
        h1, h2, h3, h4 { color: #33C4FF !important; }
        h1 { font-size: 36px; text-align: center;} h2 { font-size: 28px; } h3 { font-size: 22px; } h4 { font-size: 18px; }
        section[data-testid="stSidebar"] > div:first-child { background-color: #1A1F2A !important; }
        section[data-testid="stSidebar"] h1, section[data-testid="stSidebar"] h2, section[data-testid="stSidebar"] h3, section[data-testid="stSidebar"] h4, section[data-testid="stSidebar"] label, section[data-testid="stSidebar"] p { color: #E1E1E1 !important; }
        .stButton>button { font-size: 16px !important; border-radius: 5px; background-color: #007bff !important; color: white !important; border: none !important; padding: 8px 16px; }
        .stButton>button:hover { background-color: #0056b3 !important; }
        .stSelectbox label { color: #E1E1E1 !important; }
        div[data-baseweb="select"] > div, .st-br, .st-bq, .st-cb, .st-ca, .st-bu, .st-bv, .st-bw, .st-bx { background-color: #262730 !important; color: #FAFAFA !important; border: 1px solid #444 !important; }
        li[role="option"] { color: #FAFAFA !important; }
        .stFileUploader label { color: #E1E1E1 !important; }
        .stFileUploader > div > div { background-color: #262730; border: 1px solid #444; color: #FAFAFA; }
        .stImage > img { display: block; margin-left: auto; margin-right: auto; border-radius: 10px; }
        .stTable th, .stDataFrame th { background-color: #262730; color: #A0A0A0; }
        .stDataFrame { background-color: #1A1F2A; }
        .stAlert { border-radius: 5px; border: 1px solid #444; }
        .stAlert[data-baseweb="alert"] { background-color: #262730 !important; color: #FAFAFA !important; }
        .stAlert[data-baseweb="alert"] a { color: #33C4FF !important; }
        .block-container { padding: 2rem 3rem; }
        .center-text { text-align: center; }
        .small-text { font-size: small; }
    </style>
""", unsafe_allow_html=True)

st.markdown("<h1>🧪 Skin Lesion Classifier with Grad-CAM</h1>", unsafe_allow_html=True)
st.markdown("<p class='center-text' style='font-size:18px;'>Upload a skin lesion image and select a model to get predictions and explanations.</p>", unsafe_allow_html=True)
st.markdown("---")

# === Model config ===
model_paths = {
    "efficientnet_b0": "trained_model_weights/efficientnet_b0_finetuned.pth",
    "efficientnet_b3": "trained_model_weights/efficientnet_b3_finetuned.pth",
    "convnext_small": "trained_model_weights/convnext_small_finetuned.pth",
    "efficientnet_b2": "trained_model_weights/efficientnet_b2_finetuned.pth",
    "densenet121": "trained_model_weights/densenet121_finetuned.pth",
    "resnet50": "trained_model_weights/resnet50_finetuned.pth",
}
# Filter available models based on existing weight files
available_models = {name: path for name, path in model_paths.items() if os.path.exists(path)}
if len(available_models) < len(model_paths):
    missing = set(model_paths.keys()) - set(available_models.keys())
    st.sidebar.warning(f"Weights missing for: {', '.join(missing)}.")

class_names = ['Basal_cell_carcinoma', 'Benign_keratosislike_lesions', 'Melanoma', 'Melanocytic_nevi']
num_classes = len(class_names)

# === Diagnosis messages ===
diagnosis_messages = {
    "Melanoma": "⚠️ **Potential Melanoma:** Features consistent with melanoma detected. *Urgent consultation with a dermatologist is strongly recommended.*",
    "Basal_cell_carcinoma": "⚠️ **Possible Basal Cell Carcinoma:** Features suggestive of BCC detected. *A professional dermatological examination is advised.*",
    "Benign_keratosislike_lesions": "✅ **Likely Benign Keratosis-like Lesion:** Characteristics appear benign. *Monitor for changes and consider routine skin checks.*",
    "Melanocytic_nevi": "✅ **Likely Melanocytic Nevus (Mole):** Appears to be a common mole. *Monitor for changes (ABCDE rule) and consult a doctor if concerned.*"
}

# === Model Loader ===
@st.cache_resource
def load_model(model_name):
    model = None
    try:
        # Load model architecture using torchvision
        if model_name == "efficientnet_b0": model = efficientnet_b0(weights=None); num_ftrs = model.classifier[1].in_features; model.classifier[1] = nn.Linear(num_ftrs, num_classes)
        elif model_name == "efficientnet_b2": model = efficientnet_b2(weights=None); num_ftrs = model.classifier[1].in_features; model.classifier[1] = nn.Linear(num_ftrs, num_classes)
        elif model_name == "efficientnet_b3": model = efficientnet_b3(weights=None); num_ftrs = model.classifier[1].in_features; model.classifier[1] = nn.Linear(num_ftrs, num_classes)
        elif model_name == "convnext_small": model = convnext_small(weights=None); num_ftrs = model.classifier[2].in_features; model.classifier[2] = nn.Linear(num_ftrs, num_classes)
        elif model_name == "densenet121": model = densenet121(weights=None); num_ftrs = model.classifier.in_features; model.classifier = nn.Linear(num_ftrs, num_classes)
        elif model_name == "resnet50": model = resnet50(weights=None); num_ftrs = model.fc.in_features; model.fc = nn.Linear(num_ftrs, num_classes)
        else: raise ValueError(f"Unknown model selected: {model_name}")

        # Load the saved state dictionary
        state_dict_path = available_models[model_name]
        state_dict = torch.load(state_dict_path, map_location="cpu")

        # Remove 'module.' prefix if saved with DataParallel
        if list(state_dict.keys())[0].startswith('module.'):
            state_dict = OrderedDict((k[7:], v) for k, v in state_dict.items())

        # Load weights strictly
        model.load_state_dict(state_dict, strict=True)
        model.eval()
        return model
    except FileNotFoundError:
        st.error(f"Weight file not found at: {state_dict_path}.")
        return None
    except RuntimeError as e:
        st.error(f"RuntimeError loading {model_name} weights: {e}")
        return None
    except Exception as e:
        st.error(f"Unexpected error loading model {model_name}: {e}")
        return None

# === Get Transform ===
@st.cache_data
def get_transform():
    # ImageNet normalization stats (must match training)
    NORM_MEAN = [0.485, 0.456, 0.406]
    NORM_STD = [0.229, 0.224, 0.225]
    normalize = transforms.Normalize(mean=NORM_MEAN, std=NORM_STD)
    return transforms.Compose([
        transforms.Resize((224, 224)), # Size used during training
        transforms.ToTensor(),
        normalize # Apply ImageNet normalization
    ])

# === Grad-CAM Prediction ===
def get_prediction_and_gradcam(model, image_tensor, model_name):
    input_tensor = image_tensor.unsqueeze(0).clone().detach().requires_grad_(True)
    targets = None
    cam_image_pil = None

    try:
        # Select appropriate target layer based on torchvision model architecture
        if model_name.startswith("efficientnet"): target_layer = model.features[-1][0]
        elif model_name.startswith("convnext"): target_layer = model.features[-1][0].block[-1]
        elif model_name.startswith("densenet"): target_layer = model.features.denseblock4
        elif model_name.startswith("resnet"): target_layer = model.layer4
        else:
             # Generic fallback for unknown architectures
             target_layer = None
             for name, module in model.named_modules():
                 if isinstance(module, nn.Conv2d): target_layer = module
             if target_layer is None: raise ValueError("No Conv2d layer found.")

        if target_layer is None: raise ValueError("Target layer not determined.")
        target_layers = [target_layer]

        # Run Grad-CAM
        model.eval()
        cam = GradCAM(model=model, target_layers=target_layers)

        # Explicitly target the predicted class
        with torch.no_grad():
           initial_logits = model(input_tensor.clone().detach())
           predicted_class_index = torch.argmax(initial_logits, dim=1).item()
        targets = [ClassifierOutputTarget(predicted_class_index)]

        grayscale_cam = cam(input_tensor=input_tensor, targets=targets)[0, :]

        # De-normalize image for correct CAM visualization
        rgb_img_normalized = image_tensor.permute(1, 2, 0).cpu().numpy()
        transform_details = get_transform()
        mean = np.array(transform_details.transforms[-1].mean)
        std = np.array(transform_details.transforms[-1].std)
        rgb_img = std * rgb_img_normalized + mean
        rgb_img = np.clip(rgb_img, 0, 1)

        # Overlay CAM and convert to PIL
        cam_image = show_cam_on_image(rgb_img, grayscale_cam, use_rgb=True)
        cam_image_pil = Image.fromarray((cam_image * 255).astype(np.uint8))

    except Exception as e:
        st.error(f"Error during Grad-CAM generation for {model_name}: {e}")
        st.warning("Displaying prediction without Grad-CAM.")

    # Get final prediction probabilities
    with torch.no_grad():
        model.eval()
        logits = model(input_tensor.clone().detach())
        probabilities = torch.softmax(logits, dim=1).squeeze().cpu().numpy()
        prediction_idx = int(np.argmax(probabilities))

    return cam_image_pil, prediction_idx, probabilities

# === ChatGPT Query Function ===
def query_gpt(label):
    """Queries OpenAI GPT for information about the predicted label."""
    try:
        client = OpenAI(api_key=st.secrets["openai_api_key"])
        prompt = f"""Explain the skin condition '{label}' to a patient in simple terms (3-5 sentences). Mention general level of concern and emphasize consulting a dermatologist."""
        chat_completion = client.chat.completions.create(
            messages=[{"role": "user", "content": prompt}],
            model="gpt-3.5-turbo",
        )
        return chat_completion.choices[0].message.content.strip()
    except AuthenticationError as e:
        st.error(f"OpenAI Authentication Error: {e}")
        return "Auth error."
    except APIError as e:
        st.error(f"OpenAI API Error: {e}")
        return f"API error: {e.status_code}"
    except KeyError:
         st.error("OpenAI API key missing from Streamlit Secrets.")
         return "API key config missing."
    except Exception as e:
        st.error(f"Unexpected OpenAI error: {e}")
        return "Unexpected error."

# === Streamlit UI ===
CONFIDENCE_THRESHOLD = 0.50 # Confidence threshold (e.g., 65%)

st.sidebar.header("⚙️ Controls")
model_choice = st.sidebar.selectbox("1. Choose a model", list(available_models.keys()), index=0)
uploaded_file = st.sidebar.file_uploader("2. Upload an image", type=["jpg", "jpeg", "png"])
use_chatgpt = st.sidebar.checkbox("Get Explanation from ChatGPT", value=True)

# Main layout columns
col_main_1, col_main_2 = st.columns(2)

if uploaded_file is None:
    st.info("👈 Please upload an image using the sidebar control to begin analysis.")

if uploaded_file is not None:
    # --- Left Column: Image Upload and Prediction/Probs (or low confidence message) ---
    with col_main_1:
        try:
            image = Image.open(uploaded_file).convert("RGB")
            st.markdown("<h3>Uploaded Image</h3>", unsafe_allow_html=True)
            st.image(image, caption="Original Image", use_container_width=True)
        except Exception as e:
            st.error(f"Error opening image file: {e}")
            st.stop()

        model = load_model(model_choice)

        if model:
            transform = get_transform()
            try:
                tensor_img = transform(image)
            except Exception as e:
                st.error(f"Error transforming image: {e}")
                st.stop()

            # Get prediction and CAM results
            with st.spinner(f"Analyzing with {model_choice}..."):
                cam_image_pil, pred_class_idx, probs = get_prediction_and_gradcam(model, tensor_img, model_choice)
                predicted_label = class_names[pred_class_idx]
                max_prob = np.max(probs)

            # Check if confidence meets threshold
            if max_prob >= CONFIDENCE_THRESHOLD:
                # Display diagnosis and probabilities
                st.markdown(f"<h2>Predicted Diagnosis: {predicted_label}</h2>", unsafe_allow_html=True)
                alert_type = st.warning if "Melanoma" in predicted_label or "Carcinoma" in predicted_label else st.info
                alert_type(diagnosis_messages[predicted_label])

                st.markdown("<h4>📊 Prediction Probabilities:</h4>", unsafe_allow_html=True)
                prob_list = sorted(zip(class_names, probs), key=lambda x: x[1], reverse=True)
                prob_data = {"Class": [name for name, prob in prob_list],
                             "Probability": [f"{prob*100:.2f}%" for name, prob in prob_list]}
                prob_df = pd.DataFrame(prob_data)
                st.dataframe(prob_df, use_container_width=True, hide_index=True)
            else:
                # Display low confidence message
                st.markdown("<h2>Prediction Uncertainty</h2>", unsafe_allow_html=True)
                st.error(f"Model confidence below threshold ({CONFIDENCE_THRESHOLD*100:.0f}%). "
                         f"Highest probability: {max_prob*100:.1f}% for '{predicted_label}'.")
                st.warning("💡 **Suggestion:** Please try uploading a clearer image with better lighting or focus.")

        else:
            st.error(f"Model '{model_choice}' failed to load.")
            st.stop()

    # --- Right Column: CAM and Explanation (Explanation only if high confidence) ---
    if 'cam_image_pil' in locals():
        with col_main_2:
            st.markdown("<h3>🧠 Model Analysis</h3>", unsafe_allow_html=True)
            # Display Grad-CAM regardless of confidence (if available)
            if cam_image_pil:
                st.image(cam_image_pil, caption=f"Grad-CAM ({model_choice})", use_container_width=True)
            else:
                st.warning("Grad-CAM visualization failed.")

            # Display ChatGPT Explanation only if confidence is high
            if max_prob >= CONFIDENCE_THRESHOLD and use_chatgpt:
                 st.markdown("---")
                 st.markdown("<h4>💬 Explanation (General Info):</h4>", unsafe_allow_html=True)
                 with st.spinner("Asking ChatGPT..."):
                     explanation = query_gpt(predicted_label)
                     st.info(explanation)
            elif max_prob < CONFIDENCE_THRESHOLD and use_chatgpt:
                 st.info("ChatGPT explanation skipped due to low model confidence.")

# Footer
if uploaded_file is not None:
    st.markdown("---")
    st.markdown("<p class='center-text small-text'>Disclaimer: This tool is for informational purposes only and does not substitute professional medical advice. Always consult a qualified dermatologist for diagnosis.</p>", unsafe_allow_html=True)