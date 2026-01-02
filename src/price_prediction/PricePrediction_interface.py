import json
import streamlit as st
import numpy as np
import pandas as pd
import joblib
from feature_extraction import *
from pathlib import Path
import re

#Project Root
BASE_DIR = Path(__file__).resolve().parent.parent.parent

#Filepaths
metrics_filename = "PricePrediction_metrics.json"
model_filename = "model.obj"
METRICS_PATH = BASE_DIR / "metrics" / metrics_filename
MODEL_PATH = Path(__file__).resolve().parent / model_filename


#Max scores for slider scales
MAX_CPU_SCORE = 2000
MAX_GPU_SCORE = 5000

#Load the resources and cache them for usage
@st.cache_resource
def loadModel():
    try:
        model = joblib.load(MODEL_PATH)
        return model
    except FileNotFoundError:
        st.error("Model file not found at: {}".format(MODEL_PATH))
        st.stop()

@st.cache_resource
def loadMAE():
    #Find metrics file and load it
    if METRICS_PATH.exists():
        with open(METRICS_PATH) as f:
            metrics = json.load(f)
    elif Path(metrics_filename).exists():
        with open(metrics_filename) as f:
            metrics = json.load(f)
    else:
        st.error("Metrics file not found in {}.\
                 Please train the model again and make sure the metrics file in the correct folder.".format(METRICS_PATH))
        st.stop()

    #Load model with best mae
    for model in metrics:
        if model["isUsed"]:
            print("Using {} model...".format(model["model"]))
            return model["mae"]

model = loadModel()
mae = loadMAE()

#Check if prediction was made or error occured and set the variables
if 'prediction_made' not in st.session_state:
    st.session_state.prediction_made = False
if 'error' not in st.session_state:
    st.session_state.error = None 
    
#Error handling if values are not set
if 'lower_bound' not in st.session_state:
    st.session_state.lower_bound = 0.0
if 'upper_bound' not in st.session_state:
    st.session_state.upper_bound = 0.0
if 'cpu_score' not in st.session_state:
    st.session_state.cpu_score = 0.0
if 'gpu_score' not in st.session_state:
    st.session_state.gpu_score = 0.0


st.markdown("<h1 style='color: #A0FFFF;'>Computer Price Predictor</h1>", unsafe_allow_html=True)
st.markdown("<p style='color: #A0FFFF;'>Enter the computer specifications to estimate its price.</p>", unsafe_allow_html=True)

#Device type, brand and other features
device_type = st.selectbox("Device Type", ["Laptop", "Desktop"])
brand = st.selectbox("Brand", [
    "ASUS", "Apple", "Acer", "Dell", "HP", "Lenovo",
    "Gigabyte", "MSI", "Razer", "Samsung"
])
release_year = st.number_input("Release Year", 2005, 2025, 2023)
os = st.selectbox("Operating System", ["Windows", "macOS", "Linux", "ChromeOS"])

#CPU & GPU
cpu_model = st.text_input("CPU Model", "Intel i5-12400")
gpu_model = st.text_input("GPU Model", "RTX 3050")
vram_gb = st.number_input("VRAM (GB)", 0, 64, 8)

#Ram & Storage
ram_gb = st.number_input("RAM (GB)", 1, 256, 16)
storage_type = st.selectbox("Storage Type", ["SSD", "HDD", "Hybrid", "NVMe"])
storage_gb = st.number_input("Storage (GB)", 64, 16000, 512)

#Screen properties
display_type = st.selectbox("Display Type", ["IPS", "OLED", "LED", "VA", "QLED", "Mini-LED"])
screen_width = st.number_input("Screen Width (px)", 600, 7680, 1920)
screen_height = st.number_input("Screen Height (px)", 400, 4320, 1080)
refresh_hz = st.number_input("Refresh Rate (Hz)", 30, 480, 144)
display_size_in = st.number_input("Display Size (inches)", 10.0, 20.0, 15.6)

#Get CPU and GPU brands by model
cpu_brand = getModelBrandCPU(cpu_model)
gpu_brand = getModelBrandGPU(gpu_model)

#Apply one hot encoding to categorical datas
def one_hot(column, value, categories):
    vec = {}
    for c in categories:
        vec[f"{column}_{c}"] = 1 if value == c else 0
    return vec

#Define the categorical values
categorical_data = {}
categorical_data.update(one_hot("brand", brand, [
    "Apple", "Acer", "ASUS", "Dell", "HP", "Lenovo",
    "Gigabyte", "MSI", "Razer", "Samsung"
]))

categorical_data.update(one_hot("device_type", device_type, ["Laptop", "Desktop"]))
categorical_data.update(one_hot("os", os, ["Windows", "macOS", "Linux", "ChromeOS"]))
categorical_data.update(one_hot("cpu_brand", cpu_brand, ["Intel", "AMD", "Apple"]))
categorical_data.update(one_hot("gpu_brand", gpu_brand, ["NVIDIA", "AMD", "Intel", "Apple"]))
categorical_data.update(one_hot("display_type", display_type, ["IPS", "OLED", "LED", "VA", "QLED", "Mini-LED"]))
categorical_data.update(one_hot("storage_type", storage_type, ["SSD", "HDD", "Hybrid", "NVMe"]))

#CPU and GPU model patterns for feature extraction
cpu_pattern = r"(i[3579]|Ryzen\s[3579]|Apple M)"
gpu_pattern = r"(RTX|RX|Arc\s[AB]|Apple Integrated)"

#Extract CPU and GPU features from model names
try:
    cpu_gen = extract_cpu_gen(cpu_model)
except:
    cpu_gen = 0
try:
    cpu_family = re.search(cpu_pattern, cpu_model).group(0) if re.search(cpu_pattern, cpu_model) else ""
except:
    cpu_family = ""
try:
    cpu_tier = extract_cpu_tier(cpu_model)
except:
    cpu_tier = 0
try:
    gpu_gen = extract_gpu_gen(gpu_model)
except:
    gpu_gen = 0
try:
    gpu_tier = extract_gpu_tier(gpu_model)
except:
    gpu_tier = 0

#Calculate final CPU and GPU score from extracted features
cpu_score = calculate_cpu_performance(cpu_model, cpu_family, cpu_gen, cpu_tier)
gpu_score = calculate_gpu_performance(gpu_model, gpu_gen, gpu_tier, cpu_gen)

#Calculate screen ppi(pixels per inch) from display features
ppi = np.sqrt(screen_width**2 + screen_height**2) / display_size_in

#Define numerical dictionary for model input
numeric_data = {
    "ram_gb": ram_gb,
    "storage_gb": storage_gb,
    "vram_gb": vram_gb,
    "cpu_score": cpu_score,
    "gpu_score": gpu_score,
    "cpu_gen": cpu_gen,
    "gpu_gen": gpu_gen,
    "release_year": release_year,
    "screen_width": screen_width,
    "screen_height": screen_height,
    "ppi": ppi,
    "refresh_hz": refresh_hz,
    "display_size_in": display_size_in,
}

#At last, make a dataframe for model input
input_dict = {**numeric_data, **categorical_data}
input_df = pd.DataFrame([input_dict])

#Code that runs after predict button has pressed.
if st.button("Predict Price"):
    #Check for invalid values otherwise make a prediction
    if cpu_brand == "Unknown" or gpu_brand == "Unknown":
        st.session_state.error = ":red[Incompatible CPU or GPU model!]"
        st.session_state.prediction_made = False
    else:
        st.session_state.error = None 
        
        prediction = np.expm1(model.predict(input_df)[0])
        
        st.session_state.lower_bound = prediction - mae
        st.session_state.upper_bound = prediction + mae
        st.session_state.cpu_score = cpu_score
        st.session_state.gpu_score = gpu_score
        st.session_state.prediction_made = True

st.write("---")

#Show the UI depending on session state
if st.session_state.error:
    st.subheader(st.session_state.error)

elif st.session_state.prediction_made:
    
    st.header("Prediction Results")
    
    st.markdown(
        """
        <h3 style='display: flex; gap: 10px;'> 
            <span style='color: #A0FFFF;'>Predicted Price Range:</span> 
            <span style='color: #00FF00;'>${:.2f} - ${:.2f}</span>
        </h3>
        """.format(st.session_state.lower_bound, st.session_state.upper_bound),
        unsafe_allow_html=True
    )
    
    st.markdown(
        """
        <h3 style='display: flex; gap: 10px; font-size: 24px;'> 
            <span style='color: #A0FFFF;'>Estimated CPU Score:</span> 
            <span style='color: #00FF00;'>{:.4f}</span>
        </h3>
        """.format(st.session_state.cpu_score),
        unsafe_allow_html=True
    )
    #Progress bar for CPU score
    st.progress(st.session_state.cpu_score / MAX_CPU_SCORE)
    
    st.markdown(
        """
        <h3 style='display: flex; gap: 10px; font-size: 24px;'> 
            <span style='color: #A0FFFF;'>Estimated GPU Score:</span> 
            <span style='color: #00FF00;'>{:.4f}</span>
        </h3>
        """.format(st.session_state.gpu_score),
        unsafe_allow_html=True
    )
    #Progress bar for GPU score
    st.progress(st.session_state.gpu_score / MAX_GPU_SCORE)