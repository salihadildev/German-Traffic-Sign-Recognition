import streamlit as st
from tensorflow.keras.models import load_model
from PIL import Image
import numpy as np

model = load_model("gtsrb_cnn_model.h5")

def process_image(img):
    img = img.resize((32, 32))
    img = np.array(img)
    img = img / 255.0
    img = np.expand_dims(img, axis=0)
    return img

st.title("German Traffic Sign :vertical_traffic_light:")
st.write("Resim seç veya kamera ile resim çek, model hangi isaret olduğunu tahmin etsin")

label_names = [ 'Speed limit (20km/h)',
'Speed limit (30km/h)',
'Speed limit (50km/h)',
'Speed limit (60km/h)',
'Speed limit (70km/h)',
'Speed limit (80km/h)',
'End of speed limit (80km/h)',
'Speed limit (100km/h)',
'Speed limit (120km/h)',
'No passing',
'No passing veh over 3.5 tons',
'Right-of-way at intersection',
'Priority road',
'Yield',
'Stop',
'No vehicles',
'Veh > 3.5 tons prohibited',
'No entry',
'General caution',
'Dangerous curve left',
'Dangerous curve right',
'Double curve',
'Bumpy road',
'Slippery road',
'Road narrows on the right',
'Road work',
'Traffic signals',
'Pedestrians',
'Children crossing',
'Bicycles crossing',
'Beware of ice/snow',
'Wild animals crossing',
'End speed + passing limits',
'Turn right ahead',
'Turn left ahead',
'Ahead only',
'Go straight or right',
'Go straight or left',
'Keep right',
'Keep left',
'Roundabout mandatory',
'End of no passing',
'End no passing veh > 3.5 tons']

file = st.file_uploader('Bir resim seç', type=['jpg', 'jpeg', 'png'])

if file is not None:
    img = Image.open(file)
    st.image(img, caption='Yüklenen resim')
    image = process_image(img)
    prediction = model.predict(image)
    predicted_class = np.argmax(prediction)
    st.write(f"Tahmin edilen isaret: {label_names[predicted_class]}")

