import os
import pandas as pd
import numpy as np
import streamlit as st
import tensorflow as tf
import matplotlib.pyplot as plt
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout
from tensorflow.keras.optimizers import RMSprop
from tensorflow.keras.preprocessing.image import ImageDataGenerator, img_to_array
from PIL import Image
from datetime import datetime

# --- 1. NEW: 7-CLASS DIAGNOSIS MODEL (Local Dataset) ---
def train_multi_disease_model():
    # Points to the folder you just placed
    base_dir = './Freshwater_Fish_Disease' 
    train_dir = os.path.join(base_dir, 'Train') 
    test_dir = os.path.join(base_dir, 'Test')

    # Data Augmentation for the 7 classes
    train_gen = ImageDataGenerator(rescale=1./255, rotation_range=20, horizontal_flip=True).flow_from_directory(
        train_dir, target_size=(150, 150), batch_size=32, class_mode='categorical')
    
    val_gen = ImageDataGenerator(rescale=1./255).flow_from_directory(
        test_dir, target_size=(150, 150), batch_size=32, class_mode='categorical')

    class_names = sorted(os.listdir(train_dir))
    
    model = Sequential([
        Conv2D(32, (3,3), activation='relu', input_shape=(150, 150, 3)),
        MaxPooling2D(2,2),
        Conv2D(64, (3,3), activation='relu'),
        MaxPooling2D(2,2),
        Flatten(),
        Dense(512, activation='relu'),
        Dense(len(class_names), activation='softmax') # 7 output neurons
    ])

    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
    st.info("Training 7-Class Diagnosis Model from Local Folder...")
    model.fit(train_gen, epochs=15, validation_data=val_gen, verbose=2)
    model.save('disease_multi_model.h5')
    
    with open('disease_labels.txt', 'w') as f:
        for name in class_names: f.write(name + '\n')
    return model, class_names

# --- 2. YOUR ORIGINAL TRAINING FUNCTIONS (Keeping Unchanged) ---
def train_model():
    # [Original Binary Logic Code remains here]
    pass

def train_species_model():
    # [Original Species Logic Code remains here]
    pass

# --- 3. UPDATED LOADING UTILITIES ---
def load_all_models():
    # Load Binary Model
    if not os.path.exists('model_trained.h5'): d_model = train_model()
    else: d_model = tf.keras.models.load_model('model_trained.h5')
    
    # Load Species Model
    if not os.path.exists('species_model.h5'): s_model, s_labels = train_species_model()
    else:
        s_model = tf.keras.models.load_model('species_model.h5')
        with open('labels.txt', 'r') as f: s_labels = [line.strip() for line in f.readlines()]

    # Load NEW 7-Class Diagnosis Model
    if not os.path.exists('disease_multi_model.h5'): m_model, m_labels = train_multi_disease_model()
    else:
        m_model = tf.keras.models.load_model('disease_multi_model.h5')
        with open('disease_labels.txt', 'r') as f: m_labels = [line.strip() for line in f.readlines()]

    # Handshake Fix: Initialize all three models
    dummy = np.zeros((1, 150, 150, 3))
    _ = d_model(dummy)
    _ = s_model(dummy)
    _ = m_model(dummy)
    
    return d_model, s_model, m_model, s_labels, m_labels

def preprocess_image(image_file): 
    img = Image.open(image_file).convert('RGB').resize((150, 150))
    return np.expand_dims(img_to_array(img) / 255.0, axis=0)

def calculate_optimal_threshold(model, h_path, i_path):
    if not os.path.exists(h_path) or not os.path.exists(i_path): return 0.5 
    h = model.predict(preprocess_image(h_path), verbose=0)[0][0]
    i = model.predict(preprocess_image(i_path), verbose=0)[0][0]
    return (h + i) / 2

# --- 4. UPDATED MAIN INTERFACE ---
def main():
    st.set_page_config(page_title="FishML Analytics", layout="wide")
    st.title("🐟 Fish Health, Species & Diagnostic Analytics")

    # Session State trackers for all 3 models
    for key in ['d_correct', 'd_wrong', 's_correct', 's_wrong', 'm_correct', 'm_wrong']:
        if key not in st.session_state: st.session_state[key] = 0
    if 'wrong_log' not in st.session_state: st.session_state.wrong_log = []

    d_model, s_model, m_model, s_labels, m_labels = load_all_models()
    threshold = calculate_optimal_threshold(d_model, "./healthy.png", "./infected.png")

    # Sidebar Stats
    with st.sidebar:
        st.header("📊 Session Metrics")
        st.write(f"Health Matches: {st.session_state.d_correct}")
        st.write(f"Species Matches: {st.session_state.s_correct}")
        st.write(f"Diagnosis Matches: {st.session_state.m_correct}")

    image_file = st.file_uploader("Upload Fish Photo", type=["jpg", "png", "jpeg"])

    if image_file:
        img_array = preprocess_image(image_file)
        
        # Run All 3 Predictions
        d_score = d_model.predict(img_array, verbose=0)[0][0]
        health = "INFECTED" if d_score > threshold else "HEALTHY"
        
        s_pred = s_model.predict(img_array, verbose=0)
        species = s_labels[np.argmax(s_pred)]
        
        m_pred = m_model.predict(img_array, verbose=0)
        diagnosis = m_labels[np.argmax(m_pred)]

        if "Healthy" in diagnosis:
            health = "HEALTHY"
        else:
            # If diagnosis is a disease, we mark as infected 
            # OR we rely on the binary threshold as a backup
            health = "INFECTED"

        col1, col2 = st.columns(2)
        with col1:
            st.image(image_file, use_container_width=True)
            
        with col2:
            st.subheader("Results")
            st.info(f"🧬 Detected Species: **{species}**")
            st.warning(f"🩺 Status: **{health}**")
            st.error(f"🔬 Diagnosis: **{diagnosis}**")
            
            # Validation for the new Diagnosis model
            st.divider()
            st.write("**Verify Diagnostic Result:**")
            mc, mw = st.columns(2)
            if mc.button("✅ Diagnosis Correct"):
                st.session_state.m_correct += 1
                st.rerun()
            if mw.button("❌ Diagnosis Wrong"):
                st.session_state.m_wrong += 1
                st.session_state.wrong_log.append({"Time": datetime.now().strftime("%H:%M"), "Type": "Diagnosis", "Species": species, "Prediction": diagnosis})
                st.rerun()
            if health == "HEALTHY":
                st.success(f"🩺 Status: **{health}**")
                st.balloons() # Nice touch for healthy fish
            else:
                st.error(f"🩺 Status: **{health}**")
                st.warning(f"🔬 Diagnosis: **{diagnosis}**")

if __name__ == "__main__":
    main()