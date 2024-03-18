import cv2
!pip install easyocr
!wget https://github.com/AlexeyAB/darknet/releases/download/darknet_yolo_v4_pre/yolov4.weights
!wget https://raw.githubusercontent.com/AlexeyAB/darknet/master/cfg/yolov4.cfg
import easyocr
import numpy as np
from matplotlib import pyplot as plt
from google.colab import drive
import re
drive.mount('/content/drive')
# 필요한 라이브러리 설치
import os

# EasyOCR 리더 초기화
reader = easyocr.Reader([ 'ko'], gpu=False)

# 이미지 파일 경로 설정
image_path = '/content/drive/My Drive/hackerton/4.jpg' #1번이 기본 전기차 2번이 비전기차 3번이 빈상태
image = cv2.imread(image_path)

# YOLO 모델 설정 파일 및 가중치 파일 로드
yolo_config_path = 'yolov4.cfg'
yolo_weights_path = 'yolov4.weights'
net = cv2.dnn.readNet(yolo_weights_path, yolo_config_path)

height, width, _ = image.shape
blob = cv2.dnn.blobFromImage(image, 0.00392, (416, 416), (0, 0, 0), True, crop=False)
net.setInput(blob)
outs = net.forward(net.getUnconnectedOutLayersNames())

boxes = []
confidences = []
for out in outs:
    for detection in out:
        scores = detection[5:]
        class_id = np.argmax(scores)
        confidence = scores[class_id]
        if confidence > 0.5 and class_id == 2:  # 자동차에 해당하는 class_id
            center_x = int(detection[0] * width)
            center_y = int(detection[1] * height)
            w = int(detection[2] * width)
            h = int(detection[3] * height)
            x = int(center_x - w / 2)
            y = int(center_y - h / 2)
            boxes.append([x, y, w, h])
            confidences.append(float(confidence))

if len(boxes) > 0:
    x, y, w, h = boxes[np.argmax(confidences)]
    cv2.rectangle(image, (x, y), (x + w, y + h), (0, 255, 0), 2)
    !git clone https://github.com/ultralytics/yolov5  # YOLOv5 저장소 복제
    %cd yolov5
    !pip install -r requirements.txt  # 요구 사항 설치

    # 이미지에서 객체 감지 실행
    !python detect.py --weights yolov5s.pt --img 640 --conf 0.25 --source "{image_path}"
    # 감지 결과 표시
    from PIL import Image
    detect_path = f'runs/detect/exp/{os.path.basename(image_path)}'
    detected_image = Image.open(detect_path)  
    plt.imshow(detected_image)
    plt.title("detected_image")
    plt.show()
    zoom_image = image[y:y+h, x:x+w]
    plt.imshow(cv2.cvtColor(zoom_image, cv2.COLOR_BGR2RGB))
    plt.title("Zoomed Image")
    plt.show()
    

    # OCR 처리 및 번호판 값 출력
    ocr_result = reader.readtext(zoom_image, paragraph=False)
    plate_pattern = re.compile(r'\d{1,3}[가-힣]\d{4}')
    for result in ocr_result:
        text = result[1]
        if plate_pattern.match(text.replace(" ", "")):
            print("Detected Text:", text)
            break

    avg_color = np.mean(zoom_image, axis=(0, 1))
 ##   print(f"Detected Area Average RGB: {avg_color}")

    # 전기차 판별 조건: R-G > 10 or R-B > 10
    if abs(avg_color[0] - avg_color[1]) > 10 or abs(avg_color[0] - avg_color[2]) > 10:
        print("Possible Electric Car Detected")
    else:
        print("Non-Electric Car Detected")
else:
    print("No car detected.")
    plt.imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
    plt.title("Original Image")
    plt.show()
    !git clone https://github.com/ultralytics/yolov5  # YOLOv5 저장소 복제
    %cd yolov5
    !pip install -r requirements.txt  # 요구 사항 설치

    # 이미지에서 객체 감지 실행
    !python detect.py --weights yolov5s.pt --img 640 --conf 0.25 --source "{image_path}"
    # 감지 결과 표시
    from PIL import Image
    detect_path = f'runs/detect/exp/{os.path.basename(image_path)}'
    detected_image = Image.open(detect_path)
    plt.imshow(detected_image)
    plt.title("Original Image")
    plt.show()  