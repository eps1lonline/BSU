from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import load_img, img_to_array
import numpy as np
import matplotlib.pyplot as plt

model = load_model('model2_vgg16_cats_dogs.h5')

img = load_img('bear.jpg', target_size=(150, 150))  
img_array = img_to_array(img) / 255.0  
img_array = np.expand_dims(img_array, axis=0) 

predictions = model.predict(img_array)
print(predictions)

classes = ['bird', 'cat','dog']
predicted_class = classes[np.argmax(predictions)]
confidence = np.max(predictions) * 100

plt.imshow(img)
plt.axis('off') 
plt.title(f"Predicted: {predicted_class} ({confidence:.2f}%)")
plt.show()
