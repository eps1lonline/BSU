from ultralytics import YOLO
import cv2

names = [
    "Alice", "Bob", "Charlie", "David", "Eve", 
    "Frank", "Grace", "Hannah", "Ivy", "Jack", 
    "Kathy", "Leo", "Mia", "Noah", "Olivia", 
    "Paul", "Quincy", "Rachel", "Steve", "Tracy"
]

model = YOLO('yolov8n.pt', verbose=False)

image = cv2.imread('audit1.jpg')

roi = cv2.selectROI("Select ROI", image, fromCenter=False, showCrosshair=True)

x1, y1, w, h = roi
x2, y2 = x1 + w, y1 + h

roi_image = image[y1:y2, x1:x2]

results = model(roi_image)
people_count = 0

for result in results[0].boxes:
    x1, y1, x2, y2 = result.xyxy[0]
    conf = result.conf[0]
    cls = result.cls[0]

    if cls == 0:
        people_count += 1
        cv2.rectangle(roi_image, (int(x1), int(y1)), (int(x2), int(y2)), (0, 255, 0), 2)
        cv2.putText(roi_image, '', (int(x1), int(y1) - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 0, 0), 2)

time = '12:32'
with open("result2.txt", "w") as file:
    file.write(f'{"id":<8}{"name":<12}{"time_check":<10}\n')
    for i in range(0, people_count):
        file.write(f'{i:<5}\t{names[i]:<10}\t{time:<10}\n')
    file.write(f'\nPeople_count: {people_count}\n')

cv2.imwrite('outputImg2.jpg', roi_image)

print(f'end')
