import tensorflow as tf
from tensorflow.keras.preprocessing.image import load_img, img_to_array
import numpy as np
import matplotlib.pyplot as plt

model = tf.keras.models.load_model('cnn_fibroid_model.h5')

def preprocess_image(image_path, img_height=128, img_width=128):
    img = load_img(image_path, target_size=(img_height, img_width)) 
    img_array = img_to_array(img)  
    img_array = np.expand_dims(img_array, axis=0)  
    img_array /= 255.0 
    return img_array

def predict_image(model, image_path):
    img_array = preprocess_image(image_path)
    prediction = model.predict(img_array)
    predicted_class = np.round(prediction).astype(int)[0][0] 
    return prediction[0][0], predicted_class

def plot_prediction(prediction, predicted_class):
    categories = ['Non Uterine fibroid', 'Uterine fibroid']
    
    fig, ax = plt.subplots()
    ax.bar(categories, [1 - prediction, prediction], color=['blue', 'red'])
    ax.set_ylim([0, 1])
    plt.title(f'Prediction: {"Fibroid" if predicted_class == 1 else "Non-Fibroid"}')
    plt.ylabel('Probability')
    plt.xlabel('Class')
    plt.show()

image_path = 

prediction, predicted_class = predict_image(model, image_path)

plot_prediction(prediction, predicted_class)
