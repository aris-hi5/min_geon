import cv2

import time

import os


 
# 캡처한 이미지를 저장할 경로 설정

save_path = "captured_images"

if not os.path.exists(save_path):

    os.makedirs(save_path)


# Function to find the correct camera index
def find_camera_index():
    index = 0
    arr = []
    i = 10
    while i > 0:
        cap = cv2.VideoCapture(index)
        if cap.read()[0]:
            arr.append(index)
            cap.release()
        index += 1
        i -= 1
    return arr

# Print available camera indices
print("Available camera indices: ", find_camera_index())






# 웹캠 캡처 시작

cap = cv2.VideoCapture(0)

if not cap.isOpened():

    print("웹캠을 열 수 없습니다.")

    exit()



capture_interval = 2  # 캡처 간격 (초)

start_time = time.time()

capture_count = 0



while cap.isOpened():

    ret, frame = cap.read()

    if not ret:

        break



    current_time = time.time()

    elapsed_time = current_time - start_time



    if elapsed_time >= capture_interval:

        # 캡처한 이미지 저장

        img_name = os.path.join(save_path, f"captured_image_{capture_count}.jpg")

        cv2.imwrite(img_name, frame)

        print(f"{img_name} 저장됨.")

        capture_count += 1

        start_time = current_time  # 타이머 리셋



    # 실시간 영상 표시

    cv2.imshow('Live Feed', frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):

        break



cap.release()

cv2.destroyAllWindows()