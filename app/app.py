from matplotlib import pyplot as plt
import streamlit as st
import torch
import torch.nn as nn
import numpy as np
import cv2
import tensorflow as tf
from PIL import Image
import zipfile
import io
import os
import pickle
import base64

import pandas as pd
import json


# ==========================
# Class Labels
# ==========================

# UNIVERSAL
# CLASS_LABELS = {
#     0: "Background",
#     1: "Urban",
#     2: "Agriculture",
#     3: "Forest",
#     4: "Water",
#     5: "Barren",
#     6: "Rangeland",
#     7: "Others"
# }

# # Define fixed colors for legend (so they don't change randomly each run)
# CLASS_COLORS = {
#     0: (0, 0, 0),         # Black - Background
#     1: (255, 0, 0),       # Red - Urban
#     2: (0, 255, 0),       # Green - Agriculture
#     3: (34, 139, 34),     # Dark Green - Forest
#     4: (0, 0, 255),       # Blue - Water
#     5: (210, 180, 140),   # Tan - Barren
#     6: (255, 255, 0),     # Yellow - Rangeland
#     7: (128, 0, 128),     # Purple - Others
# }

# ==========================
# Class Labels (EuroSAT style)
# ==========================
CLASS_LABELS = {
    0: "Annual Crop",
    1: "Forest",
    2: "Herbaceous Vegetation",
    3: "Highway",
    4: "Residential",
    5: "Industrial",
    6: "Pasture",
    7: "River",
    8: "Sea/Lake",
    9: "Other"
}

CLASS_COLORS = {
    0: (0, 255, 0),        # Bright Green
    1: (34, 139, 34),      # Dark Green
    2: (50, 205, 50),      # Lime Green
    3: (128, 128, 128),    # Gray
    4: (220, 20, 60),      # Crimson
    5: (178, 34, 34),      # Firebrick
    6: (218, 165, 32),     # Goldenrod
    7: (0, 0, 255),        # Blue
    8: (65, 105, 225),     # Royal Blue
    9: (128, 0, 128)       # Purple
}

def colorize_mask(mask, n_classes=8):
    """Map predicted mask values to fixed colors."""
    color_mask = np.zeros((mask.shape[0], mask.shape[1], 3), dtype=np.uint8)
    for cls_id, color in CLASS_COLORS.items():
        color_mask[mask == cls_id] = color
    return color_mask





# ==========================
# Define UNet Model (same as Sprint 4/5)
# ==========================

class DoubleConv(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(DoubleConv, self).__init__()
        self.net = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, 3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )
    def forward(self, x):
        return self.net(x)

class UNet(nn.Module):
    def __init__(self, n_channels, n_classes):
        super(UNet, self).__init__()
        self.down1 = DoubleConv(n_channels, 64)
        self.pool1 = nn.MaxPool2d(2)
        self.down2 = DoubleConv(64, 128)
        self.pool2 = nn.MaxPool2d(2)
        self.down3 = DoubleConv(128, 256)
        self.pool3 = nn.MaxPool2d(2)
        self.down4 =DoubleConv(256, 512)
        self.pool4 = nn.MaxPool2d(2)
        
        self.bottleneck = DoubleConv(512, 1024)
        
        self.up4 = nn.ConvTranspose2d(1024, 512, 2, stride=2)
        self.conv4 = DoubleConv(1024, 512)
        self.up3 = nn.ConvTranspose2d(512, 256, 2, stride=2)
        self.conv3 = DoubleConv(512, 256)
        self.up2 = nn.ConvTranspose2d(256, 128, 2, stride=2)
        self.conv2 = DoubleConv(256, 128)
        self.up1 = nn.ConvTranspose2d(128, 64, 2, stride=2)
        self.conv1 = DoubleConv(128, 64)
        
        self.outc = nn.Conv2d(64, n_classes, 1)
        
    def forward(self, x):
        d1 = self.down1(x)
        d2 = self.down2(self.pool1(d1))
        d3 = self.down3(self.pool2(d2))
        d4 = self.down4(self.pool3(d3))
        
        b = self.bottleneck(self.pool4(d4))
        
        u4 = self.conv4(torch.cat([self.up4(b), d4], dim=1))
        u3 = self.conv3(torch.cat([self.up3(u4), d3], dim=1))
        u2 = self.conv2(torch.cat([self.up2(u3), d2], dim=1))
        u1 = self.conv1(torch.cat([self.up1(u2), d1], dim=1))
        
        return self.outc(u1)
    
# ==========================
# Utility Functions
# ==========================

# -------------------
# Load Model (.pth)
# -------------------
def load_model(model_path, n_classes=10):
    model = UNet(n_channels=3, n_classes=n_classes)
    model.load_state_dict(torch.load(model_path, map_location="cpu"))
    model.eval()
    return model

# -------------------
# Load Model (.h5)
# -------------------
@st.cache_resource
def load_model(path="app/outputs/multiclass_unet_best.h5"):
    model = tf.keras.models.load_model(path, compile=False)
    return model
# -------------------
# Inference Function
# -------------------
def predict_image(img, model, sized_img, n_classes=5, colormap="JET"):
    # Preprocess
    print(f'model: {model}')
    img_resized = img.resize(sized_img)
    img_array = np.array(img_resized) / 255.0
    tensor = np.expand_dims(img_array, axis=0)
    
    # Predict
    preds = model.predict(tensor)
    if n_classes > 2:
        mask = np.argmax(preds, axis=-1)[0]
    else:
        mask = (preds[0, :, :, 0] > 0.5).astype(np.uint8)

    # Apply colormap
    cmap_attr = getattr(cv2, f'COLORMAP_{colormap.upper()}')
    mask_colored = cv2.applyColorMap((mask * 40).astype(np.uint8), cmap_attr)
    mask_colored = cv2.cvtColor(mask_colored, cv2.COLOR_BGR2RGB)
    
    # Overlay
    overlay = cv2.addWeighted(np.array(img_resized), 0.6, mask_colored, 0.4, 0)
    return img_resized, mask_colored, overlay

def get_download_link(images_dict):
     """Pack multiple images into a ZIP and return download link"""
     buf = io.BytesIO()
     with zipfile.ZipFile(buf, "w") as z:
         for name, img in images_dict.items():
             img_pil = Image.fromarray(img)
             img_bytes = io.BytesIO()
             img_pil.save(img_bytes, format="PNG")
             z.writestr(f"{name}.png", img_bytes.getvalue())
     b64 = base64.b64encode(buf.getvalue()).decode()
     return f'<a href="data:application/zip;base64,{b64}" download="segmentation_results.zip"> Downlaod Results </a>'

# For .pth train model
def preprocess_image(uploaded_img, size = 128):
    img = Image.open(uploaded_img).convert("RGB")
    img_resized = img.resize((size, size))
    img_arr = np.array(img_resized) / 255.0
    img_tensor = torch.from_numpy(img_arr.transpose(2, 0, 1)).unsqueeze(0).float()
    return img, img_tensor

def predict_mask(model, img_tensor):
    with torch.no_grad():
        logits = model(img_tensor)
        probs = torch.softmax(logits, dim=1)
        mask = torch.argmax(probs, dim=1).squeeze(0).cpu().numpy()
    return mask

def colorize_mask(mask, n_classes=10):
    colors = np.random.randint(0, 255, size=(n_classes, 3), dtype=np.uint8)
    color_mask = colors[mask]
    return color_mask

def create_overlay(original, mask_color, alpha=0.5):
    original_resized = cv2.resize(np.array(original), (mask_color.shape[1], mask_color.shape[0]))
    overlay = cv2.addWeighted(original_resized, 1-alpha, mask_color, alpha, 0)
    return overlay

# ==========================
# Streamlit UI
# ==========================
st.set_page_config(page_title="Satellite Land Cover Segmentation", layout="wide")
st.title("🌍  Land Cover Segmentation (UNet)")



tab1, tab2, tab3 = st.tabs(["🔍 Segmentation", "📊 Metrics", "ℹ️ About"])

# -------------------
# Tab 1: Segmentation
# -------------------
with tab1:
    with st.sidebar:
        st.header("⚙️ Controls")
        uploaded_file = st.sidebar.file_uploader("Upload Satellite Image", type=["jpg","png","tif"])
        model_type = st.selectbox("Choose Model", ["Multiclass UNet", "Binary UNet"])
        colormap = st.selectbox("Choose Colormap", ["JET", "VIRIDIS", "PLASMA", "INFERNO", "TURBO"])
        
        path = st.text_input("Model Path (.h5)", "app/outputs/multiclass_unet_best.h5")
        alpha = st.slider("Overlay Transparency", 0.0, 1.0, 0.4, 0.1)
        
        run_button = st.button("Run Segmentation")
        
        
    
    if uploaded_file and run_button:
        img = Image.open(uploaded_file).convert("RGB")
        
        n_classes = 5 if model_type == "Multiclass UNet" else 2
        model_path = "app/outputs/multiclass_unet_best.h5" if n_classes == 5 else "app/outputs/unet_best.h5"
        model = load_model(path=model_path)
        sized_img = (256, 256) if n_classes == 5 else (128, 128)
        
        original, mask, overlay = predict_image(img, model, sized_img,  n_classes=n_classes, colormap=colormap)
        st.success("Segmentation Complete")
        col1 , col2, col3 = st.columns(3)
        with col1:
            st.image(original, caption="Original", use_container_width=True)
        with col2:
            st.image(mask,caption="Predicted Mask", use_container_width=True)
        with col3:
            st.image(overlay, caption="Overlay", use_container_width=True)
            
        st.markdown(get_download_link({
            "original": np.array(original),
            "mask": mask,
            "overlay": overlay
        }), unsafe_allow_html=True)
    
    else:
        st.info("Upload an image and click **Run Segmentation** to start.")
      
      
# -------------------
# Tab 2: Metrics
# -------------------
with tab2:
    st.header("📊 Training Metrics")
    try:
        # with open("outputs/history.pkl", "rb") as f:
        #     history = pickle.load(f)
        
        # metrics = ["loss", "accuracy", "iou", "dice"]
        # for m in metrics:
        #     if m in history:
        #         fig, ax = plt.subplots()
        #         ax.plot(history[m], label=f"Train {m}")
        #         if f"val_{m}" in history:
        #             ax.plot(history[f'val_{m}'], label=f"Val_{m}")
        #         ax.set_title(f"{m.upper()} Curve")
        #         ax.set_xlabel("Epoch")
        #         ax.legend()
        #         st.pyplot(fig)
  


        # st.subheader("📊 Training History")

        metrics = None

        # --- Load training metrics ---
        if os.path.exists("app/outputs/training_log.csv"):
            metrics = pd.read_csv("app/outputs/training_log.csv")
            st.success("Loaded training metrics from metrics.csv ✅")

        elif os.path.exists("app/outputs/test_metrics.json"):
            with open("app/outputs/test_metrics.json", "r") as f:
                data = json.load(f)
                metrics = pd.DataFrame(data)
            st.success("Loaded training metrics from metrics.json ✅")

        else:
            st.warning("⚠️ No training metrics found (expected outputs/metrics.csv or outputs/metrics.json).")

        # --- Plot combined charts ---
        if metrics is not None:
            st.write("### 📈 Training Curves")

            if "epoch" in metrics.columns:
                metrics.set_index("epoch", inplace=True)

            # Helper to plot pairs
            def plot_pair(cols, title):
                available = [c for c in cols if c in metrics.columns]
                if available:
                    st.line_chart(metrics[available], height=300, use_container_width=True)
                    st.caption(f"**{title}**: " + ", ".join(available))

            # Loss curves
            plot_pair(["loss", "val_loss"], "Loss Curves")

            # Accuracy curves
            plot_pair(["accuracy", "val_accuracy"], "Accuracy Curves")

            # IoU curves
            plot_pair(["iou", "val_iou"], "IoU Curves")

            # Dice curves (if available)
            plot_pair(["dice_coef", "val_dice_coef"], "Dice Coefficient Curves")

    except Exception as e:
        st.error("Could not load training history. Ensure history.pkl is available in `outputs/`.")         
   
with tab3:
    st.header("About this app")
    st.markdown("""
        This application demonstrates **Land Cover Segmentation** using a **UNet model** trained on satellite imagery.  
    It was developed in multiple **sprints**:
   

    **Features:**
    - Upload your own satellite image
    - Choose between Binary & Multiclass UNet
    - View original, mask, and overlay side by side
    - 📥 Download results as ZIP
    - 📊 Inspect training curves

    Built with **TensorFlow/Keras, Streamlit, OpenCV, Matplotlib**. 
                """)
    
    
    
    
# ✅ Add Legend to Sidebar
with st.sidebar:
    st.header("🗺️ Legend")
    for cls_id, label in CLASS_LABELS.items():
        color = CLASS_COLORS[cls_id]
        st.markdown(
            f"<div style='display:flex;align-items:center;'>"
            f"<div style='width:20px;height:20px;background-color:rgb{color};margin-right:8px;'></div>"
            f"{label}</div>",
            unsafe_allow_html=True
        )
    
    
# Sidebar
# st.sidebar.header("Upload & Settings")
# uploaded_file = st.sidebar.file_uploader("Upload Satellite Image", type=["jpg","png","tif"])
# model_choice = st.sidebar.selectbox("Select Model", ["Multiclass UNet"])
# model_path = "app/outputs/multiclass_unet_best.h5" # put trained model here

# if uploaded_file is not None:
#     # Preprocess 
#     original_img, img_tensor = preprocess_image(uploaded_file, size=128)
    
#     # Load model
#     model = load_model(model_path, n_classes=10)
    
#      # Predict
#     mask = predict_mask(model, img_tensor)
#     mask_color = colorize_mask(mask, n_classes=10)
#     overlay = create_overlay(original_img, mask_color)
    
#     # Display
#     st.subheader("Results")
#     col1, col2, col3 = st.columns(3)
#     col1.image(original_img, caption="Original", use_container_width=True)
#     col2.image(mask_color, caption="Predicted Mask", use_container_width=True)
#     col3.image(overlay, caption="Overlay", use_container_width=True)
    
#     # Download Results
#     if st.button("Download Results as ZIP"):
#         buf = io.BytesIO()
#         with zipfile.ZipFile(buf, "w") as zipf:
#             mask_pil = Image.fromarray(mask_color)
#             overlay_pil = Image.fromarray(overlay)
            
#             mask_pil.save("mask.png")
#             overlay_pil.save("overlay.png")
            
#             zipf.write("mask.png")
#             zipf.write("overlay.png")
            
#         st.download_button(
#             label="Download ZIP",
#             data=buf.getvalue(),
#             file_name="segmentation_results.zip",
#             mine="application/zip"
#         )
#         os.remove("mask.png")
#         os.remove("overlay.png")

    
# ✅ Add Legend to Sidebar
# with st.sidebar:
#     st.header("🗺️ Legend")
#     for cls_id, label in CLASS_LABELS.items():
#         color = CLASS_COLORS[cls_id]
#         st.markdown(
#             f"<div style='display:flex;align-items:center;'>"
#             f"<div style='width:20px;height:20px;background-color:rgb{color};margin-right:8px;'></div>"
#             f"{label}</div>",
#             unsafe_allow_html=True
#         )
    
    
# Sidebar
# st.sidebar.header("Upload & Settings")
# uploaded_file = st.sidebar.file_uploader("Upload Satellite Image", type=["jpg","png","tif"])
# model_choice = st.sidebar.selectbox("Select Model", ["Multiclass UNet"])
# model_path = "app/outputs/multiclass_unet_best.h5" # put trained model here

# if uploaded_file is not None:
#     # Preprocess 
#     original_img, img_tensor = preprocess_image(uploaded_file, size=128)
    
#     # Load model
#     model = load_model(model_path, n_classes=10)
    
#      # Predict
#     mask = predict_mask(model, img_tensor)
#     mask_color = colorize_mask(mask, n_classes=10)
#     overlay = create_overlay(original_img, mask_color)
    
#     # Display
#     st.subheader("Results")
#     col1, col2, col3 = st.columns(3)
#     col1.image(original_img, caption="Original", use_container_width=True)
#     col2.image(mask_color, caption="Predicted Mask", use_container_width=True)
#     col3.image(overlay, caption="Overlay", use_container_width=True)
    
#     # Download Results
#     if st.button("Download Results as ZIP"):
#         buf = io.BytesIO()
#         with zipfile.ZipFile(buf, "w") as zipf:
#             mask_pil = Image.fromarray(mask_color)
#             overlay_pil = Image.fromarray(overlay)
            
#             mask_pil.save("mask.png")
#             overlay_pil.save("overlay.png")
            
#             zipf.write("mask.png")
#             zipf.write("overlay.png")
            
#         st.download_button(
#             label="Download ZIP",
#             data=buf.getvalue(),
#             file_name="segmentation_results.zip",
#             mine="application/zip"
#         )
#         os.remove("mask.png")
#         os.remove("overlay.png")