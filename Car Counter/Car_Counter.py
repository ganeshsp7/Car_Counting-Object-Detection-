import math
import numpy as np
from ultralytics import YOLO
import cv2
import cvzone
from sort import *

cap = cv2.VideoCapture("../Videos/cars.mp4")

model = YOLO("../Yolo-Weights/yolov8l.pt")

mask = cv2.imread("mask.png")

# Tracking
tracker = Sort(20,3, 0.3)

limit = [300,350,673,350]
totalCount = []


while True:
    success, img=cap.read()
    imgRegion = cv2.bitwise_and(img,mask)
    imgGraphics = cv2.imread("graphics.png", cv2.IMREAD_UNCHANGED)
    img = cvzone.overlayPNG(img, imgGraphics, (0, 0))
    results = model(imgRegion, stream=True)

    detections = np.empty((0, 5))

    # Names of all classes
    names = model.names
    print(names)

    for r in results:
        boxes = r.boxes
        for box in boxes:
            # Bounding  Box
            x1, y1, x2, y2 = box.xyxy[0]
            x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
            # cv2.rectangle(img, (x1, y1), (x2, y2), (255, 0, 255), 3)
            w, h = x2-x1, y2-y1

            # Confidence Score
            conf = math.ceil((box.conf[0]*100))/100
            print("Score: ",conf)

            # class Name
            class_name = names[int(box.cls)]
            print("class: ",class_name)

            if class_name == "car" or class_name == "bus" or class_name == "truck" or class_name == "motorcycle" and conf > 0.3:
                # cvzone.cornerRect(img, (x1, y1, w, h), l=10, rt=5)
                # cvzone.putTextRect(img, f'{class_name}  {conf}', (max(0, x1), max(35, y1)),0.6, 1,5, offset=5)
                currentArray = np.array([x1,y1,x2,y2,conf])
                detections = np.vstack((detections,currentArray))

    resultsTracker = tracker.update(detections)

    cv2.line(img,(limit[0],limit[1]),(limit[2],limit[3]),(0,0,255),5)

    for result in resultsTracker:
        x1,y1,x2,y2,id = result
        x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
        print(result)
        w, h = x2 - x1, y2 - y1
        cvzone.cornerRect(img, (x1, y1, w, h), l=10, rt=2, colorR=(255,0,255))
        cvzone.putTextRect(img, f'{id}', (max(0, x1), max(35, y1)), 2, 3, 5, offset=10)

        cx, cy = x1+w//2, y1+h//2
        cv2.circle(img,(cx,cy),5,(255,0,255), cv2.FILLED)

        if limit[0] < cx < limit[2] and limit[1]-15 < cy < limit[3]+15:
            if totalCount.count(id) == 0:
                totalCount.append(id)
                cv2.line(img, (limit[0], limit[1]), (limit[2], limit[3]), (0, 255, 0), 5)

    # cvzone.putTextRect(img, f'Count: {len(totalCount)}', (50, 50))
    cv2.putText(img, str(len(totalCount)), (255,  100), cv2.FONT_HERSHEY_PLAIN, 5, (50, 50, 255), 8)

    cv2.imshow("Image",img)
    # cv2.imshow("ImageRegion", imgRegion)
    cv2.waitKey(1)


