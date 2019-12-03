from tensorflow.keras.preprocessing.image import ImageDataGenerator,img_to_array,load_img
from tensorflow.keras.models import load_model
import numpy as np

model = load_model('myImgeModel.h5')

img = ('C:\\Users\\Ruba\\Desktop\\Ruba.jpg')

img = load_img(img,target_size=(150,150,3))
img_array = img_to_array(img)/255
print(img_array)
img_array=np.expand_dims(img_array,axis=0)
result = model.predict(img_array)
result = np.argmax(result)
print(result)

#  this get wich category of my unknown value
datagen = ImageDataGenerator(rescale=1./255)
classes = datagen.flow_from_directory('D:\AI&ML\Pandas\image_classification_with_keras_CNN\imadeData',target_size=(150,150,3))
class_indexes = classes.class_indices    #it return dictionary value as key : name of files & values: index
classes1=np.array(list(classes.class_indices.keys()))  # it reurn numpy array for class_indexes dictionary keys
print(classes1[result])
