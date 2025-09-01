import time
import streamlit as st

import tensorflow as tf
import numpy as np
   # <-- you forgot this
class_name = [
            'Apple___Apple_scab','Apple___Black_rot','Apple___Cedar_apple_rust','Apple___healthy',
            'Background_without_leaves','Blueberry___healthy','Cherry___Powdery_mildew','Cherry___healthy',
            'Corn___Cercospora_leaf_spot Gray_leaf_spot','Corn___Common_rust','Corn___Northern_Leaf_Blight','Corn___healthy',
            'Grape___Black_rot','Grape___Esca_(Black_Measles)','Grape___Leaf_blight_(Isariopsis_Leaf_Spot)','Grape___healthy',
            'Orange___Haunglongbing_(Citrus_greening)','Peach___Bacterial_spot','Peach___healthy','Pepper,_bell___Bacterial_spot',
            'Pepper,_bell___healthy','Potato___Early_blight','Potato___Late_blight','Potato___healthy','Raspberry___healthy',
            'Soybean___healthy','Squash___Powdery_mildew','Strawberry___Leaf_scorch','Strawberry___healthy',
            'Tomato___Bacterial_spot','Tomato___Early_blight','Tomato___Late_blight','Tomato___Leaf_Mold',
            'Tomato___Septoria_leaf_spot','Tomato___Spider_mites Two-spotted_spider_mite','Tomato___Target_Spot',
            'Tomato___Tomato_Yellow_Leaf_Curl_Virus','Tomato___Tomato_mosaic_virus','Tomato___healthy'
]
# tensorflow models prediction
def cnnmodel_prediction(test_image):
    model = tf.keras.models.load_model('mod_1 (1).keras')
    image = tf.keras.preprocessing.image.load_img(test_image, target_size=(128,128))
    input_arr = tf.keras.preprocessing.image.img_to_array(image)
    input_arr = np.array([input_arr])
    prediction = model.predict(input_arr)
    result_index = np.argmax(prediction)
    return result_index
def mobmodel_prediction(test_image):
    model = tf.keras.models.load_model('MobileNetV2 (3).keras')
    image = tf.keras.preprocessing.image.load_img(test_image, target_size=(128,128))
    input_arr = tf.keras.preprocessing.image.img_to_array(image)
    input_arr = np.array([input_arr])
    prediction = model.predict(input_arr)
    result_index = np.argmax(prediction)
    return result_index
def resmodel_prediction(test_image):
    model = tf.keras.models.load_model('ResNet50.keras')
    image = tf.keras.preprocessing.image.load_img(test_image, target_size=(128,128))
    input_arr = tf.keras.preprocessing.image.img_to_array(image)
    input_arr = np.array([input_arr])
    prediction = model.predict(input_arr)
    result_index = np.argmax(prediction)
    return result_index
# sidebar
st.sidebar.title('DashBoard')
app_mode = st.sidebar.selectbox("select Page", ["Home","About","BaseLine CNN","MobileNetV2","ResNet50"])

# Homepage
if app_mode == "Home":
    st.header("PLANT DISEASE RECOGNITION")
    st.write("""
#  Plant Disease Recognition

This app helps you *identify plant leaf diseases* using images. It compares three deep-learning models and shows a quick prediction with the class name. It’s designed to be *simple, fast, and farmer-friendly*.

### How to use
1.⁠ ⁠Choose a model from the left sidebar (BaseLine CNN, MobileNetV2, or ResNet50).
2.⁠ ⁠Upload a *leaf photo* (clear, single leaf if possible).
3.⁠ ⁠Click *Show Image* (optional) and then *Predict*.
4.⁠ ⁠See the *predicted disease* and *inference time*.

### Models available
•⁠  ⁠*BaseLine CNN* – small custom model to establish a baseline.
•⁠  ⁠*MobileNetV2* – *lightweight* and fast; ideal for web/mobile deployment.
•⁠  ⁠*ResNet50* – strong large model for comparison.

### What to expect
•⁠  ⁠Trained on a public, licensed dataset of *~55k images across 39 classes* (healthy + multiple diseases). :contentReference[oaicite:0]{index=0}
•⁠  ⁠Typical validation accuracy (10 epochs, 128×128 images):
  - BaseLine CNN: *~91%*
  - MobileNetV2: *~95.7%*
  - ResNet50: *~95.7%*  :contentReference[oaicite:1]{index=1}

	⁠*Tip:* For everyday use, *MobileNetV2* gives the best balance of *speed + accuracy* and is the recommended default for deployment. :contentReference[oaicite:2]{index=2}

### Notes
•⁠  ⁠This app runs *locally*; uploaded images are used only for the prediction session.
•⁠  ⁠Works best with clear images of a single leaf on a plain background.
•⁠  ⁠Results are a *decision-support aid* and not a replacement for expert agronomy advice.
""")



# About
elif app_mode == "About":
    st.header("About")
    st.write("""
# About this project

### What is this?
A MSc Data Analytics dissertation project that builds a *complete pipeline* for crop disease classification—from *ethical data sourcing* and *model training* to a *Streamlit web app* that anyone can use. :contentReference[oaicite:3]{index=3}

### Why did we build it?
Farmers often lack fast, affordable tools to identify diseases early. Heavy models can be accurate but *hard to deploy* on modest devices. This project asks: *Can a lightweight model match large-model accuracy while staying easy to deploy?* :contentReference[oaicite:4]{index=4}

### Data (high-level)
•⁠  ⁠Source: *Mendeley Data* (public, licensed)  
•⁠  ⁠Content: ~55,000 *RGB leaf images, **39 classes* (healthy + diseases)  
•⁠  ⁠Split: *Stratified 80/20* train/validation; additional random test images for demos. :contentReference[oaicite:5]{index=5}

### Methods (in brief)
•⁠  ⁠Image size *128×128*, RGB; normalization and efficient data pipelines.
•⁠  ⁠Three models trained/evaluated for 10 epochs:
  - *BaseLine CNN* (custom, ~1.8M params).
  - *MobileNetV2* (transfer learning; frozen backbone + GAP head).
  - *ResNet50* (transfer learning; frozen backbone + GAP head). :contentReference[oaicite:6]{index=6}

### Key findings (summary)
•⁠  ⁠*MobileNetV2 ≈ ResNet50* in accuracy (*~95.7%* validation) while being *much lighter* and faster—best suited for web/mobile use.
•⁠  ⁠BaseLine CNN delivers a solid *~91%* baseline but shows a larger generalization gap on some minority classes. :contentReference[oaicite:7]{index=7}

### Ethics & compliance
•⁠  ⁠*No personal or sensitive data*; all images are public and licensed.
•⁠  ⁠Data stored on *DKIT GitLab*; usage follows DKIT policies.
•⁠  ⁠This app is for *educational & decision support*; confirm critical decisions with experts. :contentReference[oaicite:8]{index=8}

### Limitations & future work
•⁠  ⁠Some *class imbalance* can affect minority classes.
•⁠  ⁠Next steps: targeted augmentation, light *fine-tuning* of MobileNetV2, *Grad-CAM* explainability, and *TensorFlow Lite* for on-device use. :contentReference[oaicite:9]{index=9}

*Author:* NEELI SAI NIKHIL (D00274644)  
*Supervisor:* Dr. Zohaib Ijaz  
*Repository:* See GitLab link in the report. :contentReference[oaicite:10]{index=10}
""")

# Disease Recognition
elif app_mode == "BaseLine CNN":
    st.header("Disease Recognition")
    test_image = st.file_uploader("Choose an image:", key="file_cnn")
    if st.button('Show Image', key='show_cnn'):
        st.image(test_image, use_column_width=True)
    if st.button("predict", key="pred_cnn"): 
        if test_image is None:
            st.warning("Please upload an image first.")
            st.stop()
        if hasattr(test_image, "seek"):  # <--- add this
            test_image.seek(0)    
        st.write("CNN Prediction")

        # measure time
        start = time.time()
        result_index = cnnmodel_prediction(test_image)
        end = time.time()
        elapsed = end - start
        st.caption(f"Inference time: {elapsed*1000:.1f} ms")
        st.success("Model is predicting it's a {}".format(class_name[result_index]))
        # define classes


        
elif app_mode == "MobileNetV2":
    st.header("Disease Recognition")
    test_image = st.file_uploader("Choose an image:", key="file_mob")
    if st.button('Show Image', key='show_mob'):
        st.image(test_image, use_column_width=True)
    if st.button("predict", key="pred_mob"): 
        if test_image is None:
            st.warning("Please upload an image first.")
            st.stop()
        if hasattr(test_image, "seek"):  # <--- add this
            test_image.seek(0)
        st.write("CNN Prediction")

        # measure time
        start = time.time()
        result_index = mobmodel_prediction(test_image)
        end = time.time()
        elapsed = end - start
        st.caption(f"Inference time: {elapsed*1000:.1f} ms")
        st.success("Model is predicting it's a {}".format(class_name[result_index]))
        # define classes


        
elif app_mode == "ResNet50":
    st.header("Disease Recognition")
    test_image = st.file_uploader("Choose an image:", key="file_res")
    if st.button("Show Image", key="show_res"):
        st.image(test_image, use_column_width=True)     
    if st.button("predict", key="pred_res"):
        if test_image is None:
            st.warning("Please upload an image first.")
            st.stop()
        if hasattr(test_image, "seek"):  # <--- add this
            test_image.seek(0)
        st.write("CNN Prediction")

        # measure time
        start = time.time()
        result_index = resmodel_prediction(test_image)
        end = time.time()
        elapsed = end - start
        st.caption(f"Inference time: {elapsed*1000:.1f} ms")
        st.success("Model is predicting it's a {}".format(class_name[result_index]))
        # define classes


        
