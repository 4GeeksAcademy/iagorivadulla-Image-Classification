
# your code here
import numpy as np
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image

model = load_model('../models/model.h5')

def load_image(path):
    
    #loads and formates the image
    
    img = image.load_img(path, target_size=(200, 200))
    img_array = image.img_to_array(img)
    img_array = img_array / 255.0
    img_array = np.expand_dims(img_array, axis= 0)
    return img_array

def predict(path):
    
    #predicts the previous image formated
    
    names = ['Cat', 'Dog'] #the model cat == 0 and dog == 1
    
    img = load_image(path)
    pred = model.predict(img)
    idx = np.argmax(pred)
    prob = np.max(pred)
    return print(f'This is a {names[idx]} with a {prob} probability')


predict('../data/lapili.jpg')
