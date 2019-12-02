from tensorflow import keras
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator,img_to_array,load_img
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Flatten,Dense,Conv2D,MaxPooling2D,Dropout,Activation,InputLayer
import numpy as np


# load image from directory use image dataGenerator

datagen = ImageDataGenerator(
    rescale=1. / 255,
    shear_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True
)


train_image_data=datagen.flow_from_directory(
    'D:\AI&ML\Pandas\image_classification_with_keras_CNN\imadeData',
        target_size=(150, 150),
        batch_size=32,
        class_mode='categorical'
       )

print(train_image_data)
#  create model to train our image dataset cnn model

model = Sequential()

# model.add(InputLayer(input_shape=(150,150,3)))

model.add(Conv2D(32,(3,3),input_shape=(150, 150, 3)))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2,2)))

model.add(Conv2D(32,(3,3)))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2,2)))

model.add(Conv2D(64,(3,3)))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2,2)))

model.add(Flatten())
model.add(Dense(64))
model.add(Activation('relu'))
model.add(Dropout(0.5))
model.add(Dense((5)))
model.add(Activation('softmax'))

print(model.summary())
print(model.output_shape)

#  compile model
model.compile(
    loss='binary_crossentropy',
    optimizer='adam',
    metrices=['accuracy']
)


# fit the model
model.fit_generator(
    train_image_data,
    steps_per_epoch=2000,
        epochs=50
)



# save the model

model.save('myImgeModel.h5')


