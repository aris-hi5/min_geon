import cv2
import base64
import requests
import numpy as np

# Roboflow API Key 설정
api_key = "jiF92U977ApPPypiK3XQ"
base_url = "https://detect.roboflow.com/robot_arm-lgox/1"

# 웹캠 초기화
cap = cv2.VideoCapture(0)

if not cap.isOpened():
    print("Error: Could not open video capture device.")
    exit()

def get_segmentation_predictions(frame):
    # 프레임을 JPEG로 인코딩
    _, buffer = cv2.imencode('.jpg', frame)
    jpg_as_text = base64.b64encode(buffer).decode('utf-8')

    # API 요청 페이로드 구성
    infer_payload = {
        "image": {
            "type": "base64",
            "value": jpg_as_text,
        },
        "image_id": "example_image_id",
    }

    # API 요청
    res = requests.post(
        f"{base_url}?api_key={api_key}",
        json=infer_payload,
    )

    # 응답에서 세그멘테이션 결과 추출
    if res.status_code == 200:
        return res.json()['predictions']
    else:
        print(f"Error: {res.status_code}, {res.text}")
        return None

while True:
    # 웹캠에서 프레임 읽기
    ret, frame = cap.read()
    
    if not ret:
        print("Error: Could not read frame from video capture device.")
        break

    # 세그멘테이션 결과 얻기
    predictions = get_segmentation_predictions(frame)
    
    # 세그멘테이션 결과를 프레임에 그리기 (필요에 따라 조정)
    if predictions:
        for prediction in predictions:
            # 예: 사각형 그리기
            x, y, w, h = prediction['x'], prediction['y'], prediction['width'], prediction['height']
            x0, y0, x1, y1 = int(x - w / 2), int(y - h / 2), int(x + w / 2), int(y + h / 2)
            cv2.rectangle(frame, (x0, y0), (x1, y1), (0, 255, 0), 2)
            cv2.putText(frame, f"{prediction['class']} ({prediction['confidence']:.2f})", (x0, y0 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)
    
    # 결과 프레임 표시
    cv2.imshow('Live Segmentation Predictions', frame)
    
    # 'q' 키를 누르면 종료
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# 자원 해제
cap


























# import cv2
# import os
# import time

# def check_camera_indices(max_indices=30):
#     available_indices = []
#     for index in range(max_indices):
#         cap = cv2.VideoCapture(index)
#         if cap.isOpened():
#             print(f"Camera found at index {index}")
#             available_indices.append(index)
#             cap.release()
#         else:
#             print(f"No camera found at index {index}")
#     return available_indices

# # Ensure the required environment variable is set
# os.environ['QT_QPA_PLATFORM'] = 'xcb'

# while True:
#     print("Checking for available camera indices...")
#     available_cameras = check_camera_indices()
#     print(f"Available camera indices: {available_cameras}")

#     if available_cameras:
#         camera_index = available_cameras[0]
#         print(f"Using camera at index {camera_index}")

#         cap = cv2.VideoCapture(camera_index)

#         if not cap.isOpened():
#             print("Error: Could not open video device.")
#             exit()

#         while True:
#             ret, frame = cap.read()
#             if not ret:
#                 print("Failed to grab frame")
#                 break

#             cv2.imshow('Camera Feed', frame)

#             if cv2.waitKey(1) & 0xFF == ord('q'):
#                 break

#         cap.release()
#         cv2.destroyAllWindows()

#     # Wait for a few seconds before checking again
#     time.sleep(5)

