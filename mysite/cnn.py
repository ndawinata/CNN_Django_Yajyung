# -*- coding: utf-8 -*-


# Commented out IPython magic to ensure Python compatibility.
import matplotlib.pyplot as plt
import cv2
import warnings
import numpy as np
#%matplotlib inline

from keras.optimizers import adam
from keras.models import Model, Sequential
from keras.preprocessing import image
from keras.preprocessing.image import ImageDataGenerator
from keras.models import Sequential
from keras.layers import Activation, Dropout, Flatten, Dense, Conv2D, MaxPooling2D
from keras.models import load_model
from keras.preprocessing import image
from keras.utils.vis_utils import plot_model
import pickle
image_gen = ImageDataGenerator(rotation_range=30,
                                width_shift_range=0.1,
                               height_shift_range=0.1,
                               rescale=1/255,
                               shear_range=0.2,
                               zoom_range=0.2,
                               horizontal_flip=True,
                               fill_mode='nearest'
                               )


image_shape = (64, 64, 3)

"""**MODEL**"""

model = Sequential()

model.add(Conv2D(filters=32, kernel_size=(5, 5), input_shape=(
    64, 64, 3), activation='relu', padding='same',))
model.add(MaxPooling2D(pool_size=(2, 2)))

model.add(Conv2D(filters=64, kernel_size=(5, 5), input_shape=(
    64, 64, 3), activation='relu', padding='same',))
model.add(MaxPooling2D(pool_size=(2, 2)))

model.add(Flatten())

model.add(Dense(512))
model.add(Activation("relu"))

model.add(Dropout(0.5))

model.add(Dense(1))
model.add(Activation('sigmoid'))  # sigmoid

model.summary()

adam = adam(lr=0.001)
model.compile(loss='binary_crossentropy',
              optimizer='adam', metrics=['accuracy'])

"""TRAINING **MODEL**"""

batch_size = 30
train_image_gen = image_gen.flow_from_directory("./Image recognition/dataset/train",
                                                target_size=image_shape[:2],
                                                batch_size=batch_size,
                                                class_mode='binary')

test_image_gen = image_gen.flow_from_directory("./Image recognition/dataset/test",
                                               target_size=image_shape[:2],
                                               batch_size=batch_size,
                                               class_mode='binary')

train_image_gen.class_indices

warnings.filterwarnings('ignore')

result = model.fit(train_image_gen, epochs=10,
                   steps_per_epoch=250,
                   validation_data=test_image_gen,
                   validation_steps=12)

model.save('Cb_Cu_classifier.h5')

# EVALUATING THE MODEL

#plt.plot(result.history['accuracy'])
#plt.plot(result.history['loss'])
#plt.ylabel('Persentase Accuracy')
#plt.xlabel('Epochs')
#plt.title('TRAINING')
#plt.show()

#plt.plot(result.history['val_accuracy'])
#plt.plot(result.history['val_loss'])
#plt.ylabel('Persentase Accuracy')
#plt.xlabel('Epochs')
#plt.title('VALIDATION')
#plt.show()

# PREDICTING ON NEW IMAGES

#train_image_gen.class_indices

#cu_file = "cb_04.jpg"

#cu_img = image.load_img(cu_file, target_size=(64, 64))

#cu_img = image.img_to_array(cu_img)

#cu_img = np.expand_dims(cu_img, axis=0)
#cu_img = cu_img/255

#prediction_prob = model.predict(cu_img)

# output prediction
#print(f'probability that image is a cu is: {prediction_prob}')

model = load_model('Cb_Cu_classifier.h5')
pickle.dump(model, open('model.pkl','wb'))

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


classifier("cb_04.jpg")
