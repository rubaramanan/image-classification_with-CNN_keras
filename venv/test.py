from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import img_to_array,load_img,ImageDataGenerator
import numpy as np

model = load_model('myImgeModel.h5')

img_url = 'D:\AI&ML\Pandas\image_classification_with_keras_CNN\imadeData\cat\cat_0000.jpg'

img = load_img(img_url,target_size=(150,150,3))

img = img_to_array(img)/255

img = np.expand_dims(img,axis=0)

image_genrator = ImageDataGenerator(rescale=1/255)

image_data = image_genrator.flow_from_directory('D:\AI&ML\Pandas\image_classification_with_keras_CNN\imadeData',target_size=(150,150,3))

class_names = image_data.class_indices.items()
class_names = np.array([key.title() for key, value in class_names])

result = model.predict(img)
print(result)
# ans = np.argmax(result)

print("Given Image ",img_url.split('/')[-1])
print("Predicated value ",class_names[ans])