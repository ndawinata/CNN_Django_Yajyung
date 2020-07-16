import matplotlib.pyplot as plt
import cv2
import numpy as np
#%matplotlib inline

from keras.preprocessing import image
from keras.models import load_model
import pickle

files = 'model.pkl'
f = open(files, 'rb')

model = pickle.load(f)
f.close()
# 6,148,72,35,0,33.6,0.627,50,1
# print(model.predict([[0]]))

def classifier(path):
    raw_img = image.load_img(path, target_size=(64, 64))
    raw_img = image.img_to_array(raw_img)
    raw_img = np.expand_dims(raw_img, axis=0)
    raw_img = raw_img/255
    prediction = model.predict_classes(raw_img)[0][0]
    accuracy = model.predict(raw_img)[0][0]
    plt.imshow(cv2.imread(path))
    print('Accuracy', accuracy)
    if (prediction):
        print("it's a Cumulus")
    else:
        print("it's a Cumulonimbus")


classifier("cb_01.jpg")