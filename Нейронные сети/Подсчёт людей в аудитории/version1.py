import cv2
import numpy as np

names = [
    "Alice", "Bob", "Charlie", "David", "Eve", 
    "Frank", "Grace", "Hannah", "Ivy", "Jack", 
    "Kathy", "Leo", "Mia", "Noah", "Olivia", 
    "Paul", "Quincy", "Rachel", "Steve", "Tracy"
]

prototxt_path = "deploy.prototxt"
model_path = "res10_300x300_ssd_iter_140000.caffemodel"
net = cv2.dnn.readNetFromCaffe(prototxt_path, model_path)

image_path = "audit1.jpg"
image = cv2.imread(image_path)

(h, w) = image.shape[:2]
blob = cv2.dnn.blobFromImage(cv2.resize(image, (300, 300)), 1.0, (300, 300), (104.0, 177.0, 123.0))

net.setInput(blob)
detections = net.forward()

zones = []

people_count = 0

def draw_zone(event, x, y, flags, param):
    global zones, image_copy
    if event == cv2.EVENT_LBUTTONDOWN:
        zones.append((x, y))
    if event == cv2.EVENT_LBUTTONUP:
        zones[-1] = (zones[-1][0], zones[-1][1], x, y)
        cv2.rectangle(image_copy, (zones[-1][0], zones[-1][1]), (x, y), (255, 0, 0), 2)

image_copy = image.copy()

cv2.namedWindow("Image")
cv2.setMouseCallback("Image", draw_zone)

print("Click and drag to define the zones on the image.")
while True:
    cv2.imshow("Image", image_copy)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

for i in range(detections.shape[2]):
    confidence = detections[0, 0, i, 2]

    if confidence > 0.15:
        box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
        (startX, startY, endX, endY) = box.astype("int")

        for zone in zones:
            zone_startX, zone_startY, zone_endX, zone_endY = zone
            if zone_startX < startX < zone_endX and zone_startY < startY < zone_endY:
                people_count += 1
                break

        cv2.rectangle(image, (startX, startY), (endX, endY), (0, 255, 0), 2)
        cv2.putText(image, '', (startX, startY - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 255, 0), 2)

time = '12:32'
with open("result1.txt", "w") as file:
    file.write(f'{"id":<8}{"name":<12}{"time_check":<10}\n')
    for i in range(people_count):
        file.write(f'{i:<5}\t{names[i]:<10}\t{time:<10}\n')
    file.write(f'\nPeople_count: {people_count}\n')

cv2.imwrite('outputImg1.jpg', image)
cv2.destroyAllWindows()
