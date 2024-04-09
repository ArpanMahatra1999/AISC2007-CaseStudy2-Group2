import streamlit as st
import tensorflow as tf
import numpy as np

# Load the trained model for prediction
@st.cache_resource
def load_predict_model(model_name):
    if model_name == 'EfficientNet':
        model = tf.keras.models.load_model('models/en_model.h5')
    elif model_name == 'EfficientNet with 20 Unfreezed layers':
        model = tf.keras.models.load_model('models/enu_model.h5')
    elif model_name == 'Convolutional Neural Network':
        model = tf.keras.models.load_model('models/cnn_model.h5')
    else:
        model = None
    return model

# Load the trained model for generation
@st.cache_resource
def load_generate_model(model_name):
    if model_name == 'DCGAN':
        model = tf.keras.models.load_model(
            'models/dcgan_generator.h5')
    elif model_name == 'VAE':
        model = tf.keras.models.load_model(
            'models/vae_decoder.h5')
    else:
        model = None
    return model

# Function to preprocess the image
def preprocess_image(image):
    image = tf.keras.preprocessing.image.load_img(image, target_size=(224, 224))
    img_array = tf.keras.preprocessing.image.img_to_array(image)
    img_array = np.expand_dims(img_array, axis=0)
    img_array /= 255.
    return img_array

# Function to make predictions
def predict(image, model):
    processed_image = preprocess_image(image)
    prediction = model.predict(processed_image)
    prediction_idx = np.argmax(prediction)
    expressions = ['Ahegao', 'Angry', 'Happy', 'Neutral', 'Sad', 'Surprise']
    return expressions[prediction_idx]

# Function to make generations
def generate(model, model_name):
    if model_name == "DCGAN":
        generator = model
        noise = tf.random.normal([4,100])
        generated_images = np.asarray(generator(noise,training=False))
    else:
        decoder = model
        latent_points = np.random.normal(size=(4, 100))
        generated_images = decoder.predict(latent_points)
    return generated_images

# Streamlit app
st.title("Face Prediction & Generation App")

# Dropdown menu to choose the model
predict_model_name = st.selectbox("Choose Model for Predicting Expression", ['EfficientNet', 'EfficientNet with 20 Unfreezed layers', 'Convolutional Neural Network'])

# Upload image
uploaded_file = st.file_uploader("Upload Image", type=["jpg", "jpeg", "png"])
if uploaded_file is not None:
    if st.button('Predict'):
        predict_model = load_predict_model(predict_model_name)
        if predict_model is not None:
            prediction = predict(uploaded_file, predict_model)
            st.write(f"Prediction: {prediction}")
        else:
            st.write("Please select a model.")

# Dropdown menu to choose the model
generate_model_name = st.selectbox("Choose Model for Generating Faces", ['DCGAN', 'VAE'])

if st.button('Generate'):
    generate_model = load_generate_model(generate_model_name)
    if generate_model is not None:
        generation = generate(generate_model, generate_model_name)
        if generate_model_name == "DCGAN":
            for i in range(4):
                st.image((generation[i,:,:,:]*127.5+127.5).astype("int"))
        else:
            for i in range(4):
                st.image(generation[i])
    else:
        st.write("Please select a model")