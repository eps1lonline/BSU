from keras.preprocessing import image
from keras.models import Model
from keras.layers import Dense, GlobalAveragePooling2D
from keras import applications
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from keras import optimizers
from keras.models import Sequential
from keras.layers import Dropout, Flatten, Dense
from keras.models import Model
 
# Размер изображений
img_width, img_height = 150, 150
# Путь к каталогу с изображениями для обучения
train_data_dir = 'data/train'
# Путь к каталогу с изображениями для валидации
validation_data_dir = 'data/validation'
# Количество изображений для обучения
nb_train_samples = 1001
# Количество изображений для валидации
nb_validation_samples = 250
# Количество эпох
epochs = 5
# Размер выборки
batch_size = 16
 
# Загружаем сеть VGG16 без части, которая отвечает за классификацию
base_model = applications.VGG16(weights='imagenet', include_top=False)
 
# Добавляем слои классификации
x = base_model.output
x = GlobalAveragePooling2D()(x)
x = Dense(1024, activation='relu')(x)
# Выходной слой с двумя нейронами для классов "кот", "собака" "птица"
predictions = Dense(3, activation='softmax')(x)
 
# Составляем сеть из двух частей
model = Model(inputs=base_model.input, outputs=predictions)
 
# "Замораживаем" сверточные уровни сети VGG16
# Обучаем только вновь добавленные слои
for layer in base_model.layers:
    layer.trainable = False
 
# Компилируем модель
model.compile(optimizer='rmsprop', loss='categorical_crossentropy', metrics=['accuracy'])
 
# Создаем генератор данных для обучения
datagen = ImageDataGenerator(rescale=1. / 255)
train_generator = datagen.flow_from_directory(
    train_data_dir,
    target_size=(img_width, img_height),
    batch_size=batch_size,
    class_mode= 'categorical')
 
# Создаем генератор данных для валидации
validation_generator = datagen.flow_from_directory(
    validation_data_dir,
    target_size=(img_width, img_height),
    batch_size=batch_size,
    class_mode= 'categorical')
 
# Обучаем модель с помощью генератора
model.fit(
    train_generator,
    steps_per_epoch = nb_train_samples // batch_size,
    epochs=epochs,
    validation_data=validation_generator,
    validation_steps = nb_validation_samples // batch_size)
 
# Сохраняем нейронную сеть 
model.save('model2_vgg16_cats_dogs.h5')