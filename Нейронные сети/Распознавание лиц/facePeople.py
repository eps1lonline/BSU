import face_recognition
import cv2

# Загрузка изображений
image1 = face_recognition.load_image_file("one.jpg")  # Ваше первое изображение
image2 = face_recognition.load_image_file("one2.png")  # Второе изображение

# Получение векторов лиц
face_encodings1 = face_recognition.face_encodings(image1)
face_encodings2 = face_recognition.face_encodings(image2)

# Проверка, есть ли лица на изображениях
if len(face_encodings1) == 0:
    print("На первом изображении лица не обнаружены.")
elif len(face_encodings2) == 0:
    print("На втором изображении лица не обнаружены.")
else:
    # Сравнение лиц
    results = face_recognition.compare_faces([face_encodings1[0]], face_encodings2[0])

    if results[0]:
        print("Это одно и то же лицо.")
    else:
        print("Это разные лица.")

# Отображение изображений (опционально)
cv2.imshow("Image 1", cv2.cvtColor(image1, cv2.COLOR_RGB2BGR))
cv2.imshow("Image 2", cv2.cvtColor(image2, cv2.COLOR_RGB2BGR))
cv2.waitKey(0)
cv2.destroyAllWindows() 