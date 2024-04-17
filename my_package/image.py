
from PIL import Image
from io import BytesIO
import logging
from fastapi import HTTPException

from easyocr import Reader

from yolov5 import YOLOv5

import cv2
import numpy as np
import re
import torch

# easyocr에서 ssl 비활성화시켜줘서 로컬에서 작동함(로컬환경:mac os)
import ssl
ssl._create_default_https_context = ssl._create_unverified_context


logging.basicConfig(level=logging.INFO)
# model = YOLOv5(weights="yolov5s.pt", device="cpu")
model = YOLOv5("yolov5s.pt", device="cpu")

async def process_image(image):
    logging.info('Start image processing')
    contents = await image.read()
    image = Image.open(BytesIO(contents))

    logging.info(image)
    image_info = {
        'width': image.width,
        'height': image.height,
        'file': image.format,
    }

    return {
        "filename": 'temp',
        "format": image.format,
        "size": image.size,
        "mode": image.mode
    }

async def process_image(image):
    contents = await image.read()
    result_text=''
    result_type=''

    reader = Reader(['ko'], gpu=False)
    logging.info('before')
    fiveModel = torch.hub.load('ultralytics/yolov5', 'yolov5s')


    tmpImg = Image.open(BytesIO(contents))
    temp = fiveModel(tmpImg)
    detected_objects = temp.pandas().xyxy[0]

    #이미지 read
    nparr = np.fromstring(contents, np.uint8)
    image = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

    for _, obj in detected_objects.iterrows():
                x1, y1, x2, y2 = int(obj['xmin']), int(obj['ymin']), int(obj['xmax']), int(obj['ymax'])
                # 감지된 객체의 영역을 추출
                crop_img = image[y1:y2, x1:x2]
                class_name = obj['name']
                
                # 차량 객체를 확인
                if class_name == "car":  # 'car'는 모델에서 차량 객체에 해당하는 클래스명이어야 함
                    zoom_image = image[y1:y2, x1:x2]
                  
                
                # OCR 처리
                ocr_result = reader.readtext(crop_img, paragraph=False)
                plate_pattern = re.compile(r'\d{1,3}[가-힣]\d{4}')

                for result in ocr_result:
                    text = result[1]
                    logging.info('detected_text',text)
                    if plate_pattern.match(text.replace(" ", "")):
                            logging.info('detected_text',text)
                            result_text=text
                            avg_color = np.mean(zoom_image, axis=(0, 1))
                            logging.info("후!")

                            if abs(avg_color[0] - avg_color[1]) > 10 or abs(avg_color[0] - avg_color[2]) > 10:
                                logging.info("Possible Electric Car Detected")
                                result_type='Electric'
                            else:
                                logging.info("Non-Electric Car Detected")
                                result_type='Non-electric'
                            break
                    
                
    return {
            "resultText": result_text,
            "resultType": result_type,
        }




async def sample_process_image(image):
    try:
        logging.info('Start image processing')
        contents = await image.read()


        reader = Reader(['ko'], gpu=False)
        # 욜로 모델 생성

        #이미지 read
        nparr = np.fromstring(contents, np.uint8)
        image = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)


        # YOLO5 모델로 read
        fiveModel = torch.hub.load('ultralytics/yolov5', 'yolov5s', pretrained=True)
        tmpImg = Image.open(BytesIO(contents))
        temp = fiveModel(tmpImg)
        detected_objects = temp.pandas().xyxy[0]
        logging.info("Sample",detected_objects)

        # YOLO4 모델 설정 파일 및 가중치 파일 로드
        yolo_config_path = 'yolov4.cfg'
        yolo_weights_path = 'yolov4.weights'
        net = cv2.dnn.readNet(yolo_weights_path, yolo_config_path)

        blob = cv2.dnn.blobFromImage(image, 0.00392, (416, 416), (0, 0, 0), True, crop=False)
        net.setInput(blob)
        outs = net.forward(net.getUnconnectedOutLayersNames())

        sample_image = Image.open(BytesIO(contents))
        width,height = sample_image.size

        
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


        logging.info('boxes',boxes)
        logging.info('confidences',confidences)
        if len(boxes) > 0:
            x, y, w, h = boxes[np.argmax(confidences)]
            zoom_image = image[y:y+h, x:x+w]
            cv2.rectangle(image, (x, y), (x + w, y + h), (0, 255, 0), 2)
            #Arguments: (TypeError("YOLOv5.predict() got an unexpected keyword argument 'conf'"),) 이 에러로 인해 임계치 conf가 빠짐
            results = model.predict(image_rgb, size=640)

            results.render()
            detected_objs = results.pandas().xyxy[0]  # 감지된 객체들의 정보

            result_text=''
            result_type='None'

            for _, obj in detected_objs.iterrows():
                x1, y1, x2, y2 = int(obj['xmin']), int(obj['ymin']), int(obj['xmax']), int(obj['ymax'])
                # 감지된 객체의 영역을 추출
                crop_img = image[y1:y2, x1:x2]
                
                # OCR 처리
                ocr_result = reader.readtext(crop_img, paragraph=False)
                plate_pattern = re.compile(r'\d{1,3}[가-힣]\d{4}')

                for result in ocr_result:
                    text = result[1]
                    logging.info('detected_text',text)
                    if plate_pattern.match(text.replace(" ", "")):
                            logging.info('detected_text',text)
                            result_text=text
                            break
                    
                logging.info("전!")
                avg_color = np.mean(zoom_image, axis=(0, 1))
                logging.info("후!")

                if abs(avg_color[0] - avg_color[1]) > 10 or abs(avg_color[0] - avg_color[2]) > 10:
                    logging.info("Possible Electric Car Detected")
                    result_type='Electric'
                else:
                    logging.info("Non-Electric Car Detected")
                    result_type='Non-electric'
        else:
            logging.info("No car detected.")
            result_type='None'

        logging.info("result",result_text,result_type)
        return {
            "resultText": result_text,
            "resultType": result_type,
        }
    except Exception as e:
        logging.info('error',e)
    finally:
        logging.info('f')

def validate_image_file(filetype):

    if filetype not in ["image/jpeg", "image/png", "image/gif"]:
        raise HTTPException(status_code=400, detail="Unsupported file type.")
