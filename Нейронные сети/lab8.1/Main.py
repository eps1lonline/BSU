from keras.applications.vgg16 import VGG16
from keras.preprocessing import image
from keras.applications.vgg16 import preprocess_input, decode_predictions
from PIL import Image
import matplotlib.pyplot as plt
import numpy as np

model = VGG16(weights='imagenet')


img = image.load_img('phone.jpg', target_size=(224, 224))
print(img)
x = image.img_to_array(img)
x = np.expand_dims(x, axis=0)
x = preprocess_input(x)

preds = model.predict(x)

print('Результаты распознавания:', decode_predictions(preds, top=3)[0])


plt.imshow(img)
plt.axis('off')
plt.show()