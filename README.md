# 🌱 Smart Vegetation Detection (EuroSAT Segmentation)

This project is a **Streamlit web app** for detecting and segmenting land cover classes from satellite images using a trained deep learning model on the **EuroSAT dataset**.

## 🚀 Features
- Upload any image → automatic vegetation & land cover segmentation
- EuroSAT official color-coded legend
- Training history charts (Loss & Accuracy from CSV logs)
- Interactive, responsive Streamlit UI

## 📂 Project Structure
smart_vegetation_detection/
│
├── app/
│ ├── app.py # Main Streamlit app
│ ├── model.h5 # Trained segmentation model
│ ├── history.csv # Training metrics log
│ ├── requirements.txt # Dependencies
│ └── README.md # Documentation

## 🛠️ Installation (Local)
1. Clone repo  
   ```bash
   git clone https://github.com/rehmanghani2/smart-vegetation-detection.git
   cd smart-vegetation-detection/app

2. Install dependencies

pip install -r requirements.txt


3. Run the app

streamlit run app.py