# import cv2
# import numpy as np
# import time
# import torch
# import cv2.aruco as aruco
# import mediapipe as mp
# from ultralytics import YOLO

# def detect_star_shape(image, canny_thresh1, canny_thresh2):
#     gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
#     blurred = cv2.GaussianBlur(gray, (5, 5), 0)
#     edged = cv2.Canny(blurred, canny_thresh1, canny_thresh2)
#     contours, _ = cv2.findContours(edged.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

#     if len(contours) == 1 or len(contours) == 2:
#         contour = contours[0]
#         epsilon = 0.02 * cv2.arcLength(contour, True)
#         approx = cv2.approxPolyDP(contour, epsilon, True)
#         if len(approx) >= 10:
#             return True, approx, edged, len(contours)
#     return False, None, edged, len(contours)

# def nothing(x):
#     pass

# def draw_rounded_rectangle(image, top_left, width, height, radius, color, thickness):
#     x, y = top_left
#     top_right = (x + width, y)
#     bottom_right = (x + width, y + height)
#     bottom_left = (x, y + height)
#     cv2.line(image, (x + radius, y), (x + width - radius, y), color, thickness)
#     cv2.line(image, (x, y + radius), (x, y + height), color, thickness)
#     cv2.line(image, (x + width, y + radius), (x + width, y + height), color, thickness)
#     cv2.line(image, (x, y + height), (x + width, y + height), color, thickness)
#     cv2.ellipse(image, (x + radius, y + radius), (radius, radius), 180, 0, 90, color, thickness)
#     cv2.ellipse(image, (x + width - radius, y + radius), (radius, radius), 270, 0, 90, color, thickness)

# # Camera parameters for 3D to 2D conversion
# camera_matrix = np.array([[474.51901407, 0, 302.47811758],
#                           [0, 474.18970657, 250.66191453],
#                           [0, 0, 1]])
# dist_coeffs = np.array([[-0.06544764, -0.07654065, -0.00761827, -0.00279316, 0.08062307]])
# pixel_to_mm_ratio = 1.55145
# robot_origin_x = 295
# robot_origin_y = 184

# def convert_to_robot_coordinates(bbox, label):
#     x1, y1, x2, y2 = map(int, bbox)
#     center_x = (x1 + x2) / 2
#     center_y = (y1 + y2) / 2
#     robot_coords_mm_x = (center_x - robot_origin_x) * pixel_to_mm_ratio * -1
#     robot_coords_mm_y = (center_y - robot_origin_y) * pixel_to_mm_ratio
#     print(f"{label} detected at robot coordinates: ({robot_coords_mm_x:.2f} mm, {robot_coords_mm_y:.2f} mm)")

#     return robot_coords_mm_x, robot_coords_mm_y

# def send_to_robot(x, y, action="pick"):
#     print(f"Sending robot to coordinates: x={x:.2f} mm, y={y:.2f} mm for action: {action}")
#     # 로봇 제어 API 호출 예제 (이 부분을 실제 로봇 제어 코드로 대체하세요)
#     # robot.move_to(x, y)
#     # if action == "pick":
#     #     robot.pick()
#     # elif action == "sweep":
#     #     robot.sweep()

# def get_robot_current_position():
#     # 실제 로봇 제어 API를 사용하여 로봇 팔의 현재 위치를 반환하도록 구현하세요
#     # 예시: (x, y) 좌표를 반환한다고 가정합니다.
#     current_x = 10  # 여기에 실제 값으로 대체
#     current_y = 10  # 여기에 실제 값으로 대체
#     return current_x, current_y

# def robot_returned_to_origin():
#     # 로봇 팔의 원래 위치 좌표
#     origin_x = 0  # 여기에 실제 원래 위치 값으로 대체
#     origin_y = 0  # 여기에 실제 원래 위치 값으로 대체
    
#     # 로봇 팔의 현재 위치를 얻음
#     current_x, current_y = get_robot_current_position()
    
#     # 현재 위치와 원래 위치를 비교하여 일정 오차 범위 내에 있으면 True 반환
#     if abs(current_x - origin_x) < 5 and abs(current_y - origin_y) < 5:  # 오차 범위 5mm로 설정
#         return True
#     else:
#         return False

# # 비디오 캡처 및 YOLO 모델 초기화
# cap = cv2.VideoCapture(0)
# model = YOLO('/home/jinjuuk/dev_ws/pt_files/newjeans.pt')
# device = 'cuda' if torch.cuda.is_available() else 'cpu'

# if torch.cuda.is_available():
#     model.to(device)
# print(f'Using device: {device}')

# class_names = ['fb_cup', 'fb_star', 'side_cup', 'side_star', 'trash']

# mp_hands = mp.solutions.hands
# hands = mp_hands.Hands(max_num_hands=2, min_detection_confidence=0.7)
# aruco_dict = aruco.getPredefinedDictionary(aruco.DICT_6X6_250)
# parameters = aruco.DetectorParameters()
# last_seen = {}

# cv2.namedWindow('Frame')
# cv2.createTrackbar('Threshold', 'Frame', 0, 20000, nothing)
# cv2.createTrackbar('Canny Thresh1', 'Frame', 50, 255, nothing)
# cv2.createTrackbar('Canny Thresh2', 'Frame', 112, 255, nothing)
# cv2.createTrackbar('Brightness', 'Frame', 50, 100, nothing)
# cv2.createTrackbar('Confidence', 'Frame', 50, 100, nothing)
# cv2.createTrackbar('Diff_Thresh', 'Frame', 45, 255, nothing)
# cv2.createTrackbar('Hue Min', 'Frame', 0, 179, nothing)
# cv2.createTrackbar('Hue Max', 'Frame', 179, 179, nothing)
# cv2.createTrackbar('Sat Min', 'Frame', 0, 255, nothing)
# cv2.createTrackbar('Sat Max', 'Frame', 255, 255, nothing)
# cv2.createTrackbar('Val Min', 'Frame', 0, 255, nothing)
# cv2.createTrackbar('Val Max', 'Frame', 255, 255, nothing)

# roi_x_large = 55
# roi_y_large = 0
# roi_width_large = 485
# roi_height_large = 315
# roi_x_medium = 270
# roi_y_medium = 0
# roi_width_medium = 270
# roi_height_medium = 60
# roi_x_small = 464
# roi_y_small = 118
# roi_width_small = 35
# roi_height_small = 35

# fgbg = cv2.createBackgroundSubtractorMOG2()
# initial_gray = None
# post_cleanup_gray = None
# detection_enabled = False
# cleanup_detection_enabled = False
# yolo_detection_enabled = False

# robot_arm_returned = False  # 로봇 팔 상태를 추적하는 변수

# while True:
#     ret, frame = cap.read()
#     if not ret:
#         print("Failed to capture frame from camera.")
#         break

#     threshold = cv2.getTrackbarPos('Threshold', 'Frame')
#     canny_thresh1 = cv2.getTrackbarPos('Canny Thresh1', 'Frame')
#     canny_thresh2 = cv2.getTrackbarPos('Canny Thresh2', 'Frame')
#     brightness = cv2.getTrackbarPos('Brightness', 'Frame')
#     confidence_threshold = cv2.getTrackbarPos('Confidence', 'Frame') / 100.0

#     diff_thresh = cv2.getTrackbarPos('Diff_Thresh', 'Frame')
#     h_min = cv2.getTrackbarPos('Hue Min', 'Frame')
#     h_max = cv2.getTrackbarPos('Hue Max', 'Frame')
#     s_min = cv2.getTrackbarPos('Sat Min', 'Frame')
#     s_max = cv2.getTrackbarPos('Sat Max', 'Frame')
#     v_min = cv2.getTrackbarPos('Val Min', 'Frame')
#     v_max = cv2.getTrackbarPos('Val Max', 'Frame')

#     frame = cv2.convertScaleAbs(frame, alpha=1, beta=(brightness - 50) * 2)
#     fgmask = fgbg.apply(frame)

#     roi_large = fgmask[roi_y_large:roi_y_large + roi_height_large, roi_x_large:roi_x_large + roi_width_large]
#     intrusion_detected = np.sum(roi_large) > threshold

#     warning_detected = False
#     trash_detected = False

#     frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
#     result = hands.process(frame_rgb)
#     if result.multi_hand_landmarks:
#         for hand_landmarks in result.multi_hand_landmarks:
#             for landmark in hand_landmarks.landmark:
#                 x = int(landmark.x * frame.shape[1])
#                 y = int(landmark.y * frame.shape[0])
#                 if roi_x_large < x < roi_x_large + roi_width_large and roi_y_large < y < roi_y_large + roi_height_large and intrusion_detected:
#                     warning_detected = True
#                     break
#             mp.solutions.drawing_utils.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)

#     roi_medium = frame[roi_y_medium:roi_y_medium + roi_height_medium, roi_x_medium:roi_x_medium + roi_width_medium]
#     corners, ids, rejectedImgPoints = aruco.detectMarkers(roi_medium, aruco_dict, parameters=parameters)
#     current_time = time.time()

#     if ids is not None and len(ids) > 0:
#         aruco.drawDetectedMarkers(roi_medium, corners, ids)
#         for id in ids.flatten():
#             last_seen[id] = current_time
#         for id, last_time in list(last_seen.items()):
#             if current_time - last_time > 3:
#                 cv2.putText(frame, f"Action executed for marker ID {id}", (50, 100), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
#                 del last_seen[id]
#                 break

#     small_roi = frame[roi_y_small:roi_y_small + roi_height_small, roi_x_small:roi_x_small + roi_width_small]
#     star_detected, star_contour, edged, contour_count = detect_star_shape(small_roi, canny_thresh1, canny_thresh2)
#     if star_detected:
#         star_contour += [roi_x_small, roi_y_small]
#         cv2.drawContours(frame, [star_contour], -1, (0, 255, 0), 3)
#         cv2.putText(frame, "Star detected", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 2)
#     else:
#         cv2.putText(frame, "Star not detected", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 2)

#     frame_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
#     frame_gray = cv2.GaussianBlur(frame_gray, (21, 21), 0)
#     hsv_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

#     key = cv2.waitKey(1) & 0xFF
#     if key == ord('s'):
#         initial_gray = frame_gray
#         detection_enabled = True
#         print("Initial background set, detection enabled.")
        
#         # 아이스크림 만들기 동작 시작 후 3초 대기
#         time.sleep(3)

#     if detection_enabled and initial_gray is not None:
#         frame_delta = cv2.absdiff(initial_gray, frame_gray)
#         thresh = cv2.threshold(frame_delta, diff_thresh, 255, cv2.THRESH_BINARY)[1]
#         thresh = cv2.dilate(thresh, None, iterations=2)
#         contours, _ = cv2.findContours(thresh.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
#         for contour in contours:
#             if cv2.contourArea(contour) < 500:
#                 continue
#             (x, y, w, h) = cv2.boundingRect(contour)
#             cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
#             cv2.putText(frame, "Other trash detected", (10, frame.shape[0] - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
#             # Other trash detected 위치로 로봇을 보내기
#             other_trash_coords_mm_x, other_trash_coords_mm_y = convert_to_robot_coordinates((x, y, x + w, y + h), "Other Trash")
#             send_to_robot(other_trash_coords_mm_x, other_trash_coords_mm_y, action="sweep")
#             trash_detected = True

#     # 로봇이 원래 위치로 돌아왔는지 확인하고 YOLO 감지 활성화
#     if not robot_arm_returned and robot_returned_to_origin():
#         robot_arm_returned = True
#         yolo_detection_enabled = True
#         print("Robot returned to origin, YOLO detection enabled.")

#     if yolo_detection_enabled:
#         yolo_roi = frame[roi_y_large:roi_y_large + roi_height_large, roi_x_large:roi_x_large + roi_width_large]
#         results = model(yolo_roi)
#         objects_to_pick = []
#         objects_to_sweep = []
        
#         for result in results:
#             for box in result.boxes:
#                 x1, y1, x2, y2 = box.xyxy[0]
#                 class_id = int(box.cls[0])
#                 class_name = class_names[class_id]
#                 if box.conf.item() >= confidence_threshold:
#                     x1, y1, x2, y2 = map(int, (x1 + roi_x_large, y1 + roi_y_large, x2 + roi_x_large, y2 + roi_y_large))
#                     robot_coords_mm_x, robot_coords_mm_y = convert_to_robot_coordinates((x1, y1, x2, y2), class_name)
#                     if class_name in ['side_cup', 'side_star']:
#                         objects_to_pick.append((robot_coords_mm_x, robot_coords_mm_y))
#                     else:
#                         objects_to_sweep.append((robot_coords_mm_x, robot_coords_mm_y))
#                     # 바운딩 박스 그리기
#                     cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
#                     text = f'{class_name}'
#                     text_x = x1
#                     text_y = y1 - 10 if y1 - 10 > 10 else y1 + 10
#                     cv2.putText(frame, text, (text_x, text_y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

#         # 먼저 side_cup과 side_star를 처리
#         for coords in objects_to_pick:
#             send_to_robot(coords[0], coords[1], action="pick")

#         # fb_cup, fb_star, trash를 처리
#         for coords in objects_to_sweep:
#             send_to_robot(coords[0], coords[1], action="sweep")

#     draw_rounded_rectangle(frame, (roi_x_large, roi_y_large), roi_width_large, roi_height_large, 20, (255, 0, 0), 2)
#     cv2.rectangle(frame, (roi_x_medium, roi_y_medium), (roi_x_medium + roi_width_medium, roi_y_medium + roi_height_medium), (0, 255, 0), 2)
#     cv2.rectangle(frame, (roi_x_small, roi_y_small), (roi_x_small + roi_width_small, roi_y_small + roi_height_small), (0, 0, 255), 2)
#     cv2.putText(frame, f"Contours: {contour_count}", (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 255), 2)
#     frame[roi_y_small:roi_y_small + roi_height_small, roi_x_small:roi_x_small + roi_width_small] = cv2.cvtColor(edged, cv2.COLOR_GRAY2BGR)

#     if warning_detected:
#         cv2.putText(frame, 'WARNING DETECTED!', (10, frame.shape[0] - 60), cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0, 0, 255), 3)

#     if trash_detected:
#         cv2.putText(frame, 'TRASH DETECTED', (10, frame.shape[0] - 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 0), 2)

#     cv2.imshow('Frame', frame)
#     if key == ord('q'):
#         break

# cap.release()
# cv2.destroyAllWindows()







# import cv2
# import numpy as np
# import time
# import torch
# import cv2.aruco as aruco
# import mediapipe as mp
# from ultralytics import YOLO

# def detect_star_shape(image, canny_thresh1, canny_thresh2):
#     gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
#     blurred = cv2.GaussianBlur(gray, (5, 5), 0)
#     edged = cv2.Canny(blurred, canny_thresh1, canny_thresh2)
#     contours, _ = cv2.findContours(edged.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

#     if len(contours) == 1 or len(contours) == 2:
#         contour = contours[0]
#         epsilon = 0.02 * cv2.arcLength(contour, True)
#         approx = cv2.approxPolyDP(contour, epsilon, True)
#         if len(approx) >= 10:
#             return True, approx, edged, len(contours)
#     return False, None, edged, len(contours)

# def nothing(x):
#     pass

# def draw_rounded_rectangle(image, top_left, width, height, radius, color, thickness):
#     x, y = top_left
#     top_right = (x + width, y)
#     bottom_right = (x + width, y + height)
#     bottom_left = (x, y + height)
#     cv2.line(image, (x + radius, y), (x + width - radius, y), color, thickness)
#     cv2.line(image, (x, y + radius), (x, y + height), color, thickness)
#     cv2.line(image, (x + width, y + radius), (x + width, y + height), color, thickness)
#     cv2.line(image, (x, y + height), (x + width, y + height), color, thickness)
#     cv2.ellipse(image, (x + radius, y + radius), (radius, radius), 180, 0, 90, color, thickness)
#     cv2.ellipse(image, (x + width - radius, y + radius), (radius, radius), 270, 0, 90, color, thickness)

# # Camera parameters for 3D to 2D conversion
# camera_matrix = np.array([[474.51901407, 0, 302.47811758],
#                           [0, 474.18970657, 250.66191453],
#                           [0, 0, 1]])
# dist_coeffs = np.array([[-0.06544764, -0.07654065, -0.00761827, -0.00279316, 0.08062307]])
# pixel_to_mm_ratio = 1.55145
# robot_origin_x = 295
# robot_origin_y = 184

# def convert_to_robot_coordinates(bbox, label):
#     x1, y1, x2, y2 = map(int, bbox)
#     center_x = (x1 + x2) / 2
#     center_y = (y1 + y2) / 2
#     robot_coords_mm_x = (center_x - robot_origin_x) * pixel_to_mm_ratio * -1
#     robot_coords_mm_y = (center_y - robot_origin_y) * pixel_to_mm_ratio
#     print(f"{label} detected at robot coordinates: ({robot_coords_mm_x:.2f} mm, {robot_coords_mm_y:.2f} mm)")

#     return robot_coords_mm_x, robot_coords_mm_y, (center_x, center_y)

# def send_to_robot(x, y, action="pick"):
#     print(f"Sending robot to coordinates: x={x:.2f} mm, y={y:.2f} mm for action: {action}")
#     # 로봇 제어 API 호출 예제 (이 부분을 실제 로봇 제어 코드로 대체하세요)
#     # robot.move_to(x, y)
#     # if action == "pick":
#     #     robot.pick()
#     # elif action == "sweep":
#     #     robot.sweep()

# def get_robot_current_position():
#     # 실제 로봇 제어 API를 사용하여 로봇 팔의 현재 위치를 반환하도록 구현하세요
#     # 예시: (x, y) 좌표를 반환한다고 가정합니다.
#     current_x = 0  # 여기에 실제 값으로 대체
#     current_y = 0  # 여기에 실제 값으로 대체
#     return current_x, current_y

# def robot_returned_to_origin():
#     # 로봇 팔의 원래 위치 좌표
#     origin_x = 0  # 여기에 실제 원래 위치 값으로 대체
#     origin_y = 0  # 여기에 실제 원래 위치 값으로 대체
    
#     # 로봇 팔의 현재 위치를 얻음
#     current_x, current_y = get_robot_current_position()
    
#     # 현재 위치와 원래 위치를 비교하여 일정 오차 범위 내에 있으면 True 반환
#     if abs(current_x - origin_x) < 5 and abs(current_y - origin_y) < 5:  # 오차 범위 5mm로 설정
#         return True
#     else:
#         return False

# # 비디오 캡처 및 YOLO 모델 초기화
# cap = cv2.VideoCapture(0)
# model = YOLO('/home/jinjuuk/dev_ws/pt_files/newjeans.pt')
# device = 'cuda' if torch.cuda.is_available() else 'cpu'

# if torch.cuda.is_available():
#     model.to(device)
# print(f'Using device: {device}')

# class_names = ['fb_cup', 'fb_star', 'side_cup', 'side_star', 'trash']

# mp_hands = mp.solutions.hands
# hands = mp_hands.Hands(max_num_hands=2, min_detection_confidence=0.7)
# aruco_dict = aruco.getPredefinedDictionary(aruco.DICT_6X6_250)
# parameters = aruco.DetectorParameters()
# last_seen = {}

# cv2.namedWindow('Frame')
# cv2.createTrackbar('Threshold', 'Frame', 0, 20000, nothing)
# cv2.createTrackbar('Canny Thresh1', 'Frame', 50, 255, nothing)
# cv2.createTrackbar('Canny Thresh2', 'Frame', 112, 255, nothing)
# cv2.createTrackbar('Brightness', 'Frame', 50, 100, nothing)
# cv2.createTrackbar('Confidence', 'Frame', 50, 100, nothing)
# cv2.createTrackbar('Diff_Thresh', 'Frame', 45, 255, nothing)
# cv2.createTrackbar('Hue Min', 'Frame', 0, 179, nothing)
# cv2.createTrackbar('Hue Max', 'Frame', 179, 179, nothing)
# cv2.createTrackbar('Sat Min', 'Frame', 0, 255, nothing)
# cv2.createTrackbar('Sat Max', 'Frame', 255, 255, nothing)
# cv2.createTrackbar('Val Min', 'Frame', 0, 255, nothing)
# cv2.createTrackbar('Val Max', 'Frame', 255, 255, nothing)

# roi_x_large = 55
# roi_y_large = 0
# roi_width_large = 485
# roi_height_large = 315
# roi_x_medium = 270
# roi_y_medium = 0
# roi_width_medium = 270
# roi_height_medium = 60
# roi_x_small = 464
# roi_y_small = 118
# roi_width_small = 35
# roi_height_small = 35

# fgbg = cv2.createBackgroundSubtractorMOG2()
# initial_gray = None
# post_cleanup_gray = None
# detection_enabled = False
# cleanup_detection_enabled = False
# yolo_detection_enabled = False

# robot_arm_returned = False  # 로봇 팔 상태를 추적하는 변수

# while True:
#     ret, frame = cap.read()
#     if not ret:
#         print("Failed to capture frame from camera.")
#         break

#     threshold = cv2.getTrackbarPos('Threshold', 'Frame')
#     canny_thresh1 = cv2.getTrackbarPos('Canny Thresh1', 'Frame')
#     canny_thresh2 = cv2.getTrackbarPos('Canny Thresh2', 'Frame')
#     brightness = cv2.getTrackbarPos('Brightness', 'Frame')
#     confidence_threshold = cv2.getTrackbarPos('Confidence', 'Frame') / 100.0

#     diff_thresh = cv2.getTrackbarPos('Diff_Thresh', 'Frame')
#     h_min = cv2.getTrackbarPos('Hue Min', 'Frame')
#     h_max = cv2.getTrackbarPos('Hue Max', 'Frame')
#     s_min = cv2.getTrackbarPos('Sat Min', 'Frame')
#     s_max = cv2.getTrackbarPos('Sat Max', 'Frame')
#     v_min = cv2.getTrackbarPos('Val Min', 'Frame')
#     v_max = cv2.getTrackbarPos('Val Max', 'Frame')

#     frame = cv2.convertScaleAbs(frame, alpha=1, beta=(brightness - 50) * 2)
#     fgmask = fgbg.apply(frame)

#     roi_large = fgmask[roi_y_large:roi_y_large + roi_height_large, roi_x_large:roi_x_large + roi_width_large]
#     intrusion_detected = np.sum(roi_large) > threshold

#     warning_detected = False
#     trash_detected = False

#     frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
#     result = hands.process(frame_rgb)
#     if result.multi_hand_landmarks:
#         for hand_landmarks in result.multi_hand_landmarks:
#             for landmark in hand_landmarks.landmark:
#                 x = int(landmark.x * frame.shape[1])
#                 y = int(landmark.y * frame.shape[0])
#                 if roi_x_large < x < roi_x_large + roi_width_large and roi_y_large < y < roi_y_large + roi_height_large and intrusion_detected:
#                     warning_detected = True
#                     break
#             mp.solutions.drawing_utils.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)

#     roi_medium = frame[roi_y_medium:roi_y_medium + roi_height_medium, roi_x_medium:roi_x_medium + roi_width_medium]
#     corners, ids, rejectedImgPoints = aruco.detectMarkers(roi_medium, aruco_dict, parameters=parameters)
#     current_time = time.time()

#     if ids is not None and len(ids) > 0:
#         aruco.drawDetectedMarkers(roi_medium, corners, ids)
#         for id in ids.flatten():
#             last_seen[id] = current_time
#         for id, last_time in list(last_seen.items()):
#             if current_time - last_time > 3:
#                 cv2.putText(frame, f"Action executed for marker ID {id}", (50, 100), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
#                 del last_seen[id]
#                 break

#     small_roi = frame[roi_y_small:roi_y_small + roi_height_small, roi_x_small:roi_x_small + roi_width_small]
#     star_detected, star_contour, edged, contour_count = detect_star_shape(small_roi, canny_thresh1, canny_thresh2)
#     if star_detected:
#         star_contour += [roi_x_small, roi_y_small]
#         cv2.drawContours(frame, [star_contour], -1, (0, 255, 0), 3)
#         cv2.putText(frame, "Star detected", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 2)
#     else:
#         cv2.putText(frame, "Star not detected", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 2)

#     frame_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
#     frame_gray = cv2.GaussianBlur(frame_gray, (21, 21), 0)
#     hsv_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

#     key = cv2.waitKey(1) & 0xFF
#     if key == ord('s'):
#         initial_gray = frame_gray
#         detection_enabled = True
#         print("Initial background set, detection enabled.")
        
#         # 아이스크림 만들기 동작 시작 후 3초 대기
#         time.sleep(3)

#     if detection_enabled and initial_gray is not None:
#         frame_delta = cv2.absdiff(initial_gray, frame_gray)
#         thresh = cv2.threshold(frame_delta, diff_thresh, 255, cv2.THRESH_BINARY)[1]
#         thresh = cv2.dilate(thresh, None, iterations=2)
#         contours, _ = cv2.findContours(thresh.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
#         for contour in contours:
#             if cv2.contourArea(contour) < 500:
#                 continue
#             (x, y, w, h) = cv2.boundingRect(contour)
#             cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
#             cv2.putText(frame, "Motion detected", (10, frame.shape[0] - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
#             # Motion detected 위치로 로봇을 보내기
#             motion_coords_mm_x, motion_coords_mm_y, _ = convert_to_robot_coordinates((x, y, x + w, y + h), "motion")
#             send_to_robot(motion_coords_mm_x, motion_coords_mm_y, action="sweep")
#             trash_detected = True

#     # 로봇이 원래 위치로 돌아왔는지 확인하고 YOLO 감지 활성화
#     if not robot_arm_returned and robot_returned_to_origin():
#         robot_arm_returned = True
#         yolo_detection_enabled = True
#         print("Robot returned to origin, YOLO detection enabled.")

#     if yolo_detection_enabled:
#         yolo_roi = frame[roi_y_large:roi_y_large + roi_height_large, roi_x_large:roi_x_large + roi_width_large]
#         results = model(yolo_roi)
#         objects_to_pick = []
#         objects_to_sweep = []
        
#         for result in results:
#             for box in result.boxes:
#                 x1, y1, x2, y2 = box.xyxy[0]
#                 class_id = int(box.cls[0])
#                 class_name = class_names[class_id]
#                 if box.conf.item() >= confidence_threshold:
#                     x1, y1, x2, y2 = map(int, (x1 + roi_x_large, y1 + roi_y_large, x2 + roi_x_large, y2 + roi_y_large))
#                     robot_coords_mm_x, robot_coords_mm_y, center = convert_to_robot_coordinates((x1, y1, x2, y2), class_name)
#                     if class_name in ['side_cup', 'side_star']:
#                         objects_to_pick.append((robot_coords_mm_x, robot_coords_mm_y, center))
#                     else:
#                         objects_to_sweep.append((robot_coords_mm_x, robot_coords_mm_y, center))
#                     # 바운딩 박스 그리기
#                     cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
#                     text = f'{class_name}'
#                     text_x = x1
#                     text_y = y1 - 10 if y1 - 10 > 10 else y1 + 10
#                     cv2.putText(frame, text, (text_x, text_y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

#         # 카메라 중앙 좌표
#         camera_center = (frame.shape[1] / 2, frame.shape[0] / 2)

#         # 먼 거리 순서대로 정렬
#         objects_to_pick.sort(key=lambda obj: np.linalg.norm(np.array(obj[2]) - np.array(camera_center)), reverse=True)
#         objects_to_sweep.sort(key=lambda obj: np.linalg.norm(np.array(obj[2]) - np.array(camera_center)), reverse=True)

#         # 먼저 side_cup과 side_star를 처리
#         for coords in objects_to_pick:
#             send_to_robot(coords[0], coords[1], action="pick")

#         # fb_cup, fb_star, trash를 처리
#         for coords in objects_to_sweep:
#             send_to_robot(coords[0], coords[1], action="sweep")

#     draw_rounded_rectangle(frame, (roi_x_large, roi_y_large), roi_width_large, roi_height_large, 20, (255, 0, 0), 2)
#     cv2.rectangle(frame, (roi_x_medium, roi_y_medium), (roi_x_medium + roi_width_medium, roi_y_medium + roi_height_medium), (0, 255, 0), 2)
#     cv2.rectangle(frame, (roi_x_small, roi_y_small), (roi_x_small + roi_width_small, roi_y_small + roi_height_small), (0, 0, 255), 2)
#     cv2.putText(frame, f"Contours: {contour_count}", (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 255), 2)
#     frame[roi_y_small:roi_y_small + roi_height_small, roi_x_small:roi_x_small + roi_width_small] = cv2.cvtColor(edged, cv2.COLOR_GRAY2BGR)

#     if warning_detected:
#         cv2.putText(frame, 'WARNING DETECTED!', (10, frame.shape[0] - 60), cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0, 0, 255), 3)

#     if trash_detected:
#         cv2.putText(frame, 'TRASH DETECTED', (10, frame.shape[0] - 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 0), 2)

#     cv2.imshow('Frame', frame)
#     if key == ord('q'):
#         break

# cap.release()
# cv2.destroyAllWindows()










# import cv2
# import numpy as np
# import time
# import torch
# import cv2.aruco as aruco
# import mediapipe as mp
# from ultralytics import YOLO

# def detect_star_shape(image, canny_thresh1, canny_thresh2):
#     gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
#     blurred = cv2.GaussianBlur(gray, (5, 5), 0)
#     edged = cv2.Canny(blurred, canny_thresh1, canny_thresh2)
#     contours, _ = cv2.findContours(edged.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

#     if len(contours) == 1 or len(contours) == 2:
#         contour = contours[0]
#         epsilon = 0.02 * cv2.arcLength(contour, True)
#         approx = cv2.approxPolyDP(contour, epsilon, True)
#         if len(approx) >= 10:
#             return True, approx, edged, len(contours)
#     return False, None, edged, len(contours)

# def nothing(x):
#     pass

# def draw_rounded_rectangle(image, top_left, width, height, radius, color, thickness):
#     x, y = top_left
#     top_right = (x + width, y)
#     bottom_right = (x + width, y + height)
#     bottom_left = (x, y + height)
#     cv2.line(image, (x + radius, y), (x + width - radius, y), color, thickness)
#     cv2.line(image, (x, y + radius), (x, y + height), color, thickness)
#     cv2.line(image, (x + width, y + radius), (x + width, y + height), color, thickness)
#     cv2.line(image, (x, y + height), (x + width, y + height), color, thickness)
#     cv2.ellipse(image, (x + radius, y + radius), (radius, radius), 180, 0, 90, color, thickness)
#     cv2.ellipse(image, (x + width - radius, y + radius), (radius, radius), 270, 0, 90, color, thickness)

# # Camera parameters for 3D to 2D conversion
# camera_matrix = np.array([[474.51901407, 0, 302.47811758],
#                           [0, 474.18970657, 250.66191453],
#                           [0, 0, 1]])
# dist_coeffs = np.array([[-0.06544764, -0.07654065, -0.00761827, -0.00279316, 0.08062307]])
# pixel_to_mm_ratio = 1.55145
# robot_origin_x = 295
# robot_origin_y = 184

# def convert_to_robot_coordinates(bbox, label):
#     x1, y1, x2, y2 = map(int, bbox)
#     center_x = (x1 + x2) / 2
#     center_y = (y1 + y2) / 2
#     robot_coords_mm_x = (center_x - robot_origin_x) * pixel_to_mm_ratio * -1
#     robot_coords_mm_y = (center_y - robot_origin_y) * pixel_to_mm_ratio
#     print(f"{label} detected at robot coordinates: ({robot_coords_mm_x:.2f} mm, {robot_coords_mm_y:.2f} mm)")

#     return robot_coords_mm_x, robot_coords_mm_y, (center_x, center_y)

# def send_to_robot(x, y, action="pick"):
#     print(f"Sending robot to coordinates: x={x:.2f} mm, y={y:.2f} mm for action: {action}")
#     # 로봇 제어 API 호출 예제 (이 부분을 실제 로봇 제어 코드로 대체하세요)
#     # robot.move_to(x, y)
#     # if action == "pick":
#     #     robot.pick()
#     # elif action == "sweep":
#     #     robot.sweep()

# def get_robot_current_position():
#     # 실제 로봇 제어 API를 사용하여 로봇 팔의 현재 위치를 반환하도록 구현하세요
#     # 예시: (x, y) 좌표를 반환한다고 가정합니다.
#     current_x = 0  # 여기에 실제 값으로 대체
#     current_y = 0  # 여기에 실제 값으로 대체
#     return current_x, current_y

# def robot_returned_to_origin():
#     # 로봇 팔의 원래 위치 좌표
#     origin_x = 0  # 여기에 실제 원래 위치 값으로 대체
#     origin_y = 0  # 여기에 실제 원래 위치 값으로 대체
    
#     # 로봇 팔의 현재 위치를 얻음
#     current_x, current_y = get_robot_current_position()
    
#     # 현재 위치와 원래 위치를 비교하여 일정 오차 범위 내에 있으면 True 반환
#     if abs(current_x - origin_x) < 5 and abs(current_y - origin_y) < 5:  # 오차 범위 5mm로 설정
#         return True
#     else:
#         return False

# # 비디오 캡처 및 YOLO 모델 초기화
# cap = cv2.VideoCapture(0)
# model = YOLO('/home/jinjuuk/dev_ws/pt_files/newjeans.pt')
# device = 'cuda' if torch.cuda.is_available() else 'cpu'

# if torch.cuda.is_available():
#     model.to(device)
# print(f'Using device: {device}')

# class_names = ['fb_cup', 'fb_star', 'side_cup', 'side_star', 'trash']

# mp_hands = mp.solutions.hands
# hands = mp_hands.Hands(max_num_hands=2, min_detection_confidence=0.7)
# aruco_dict = aruco.getPredefinedDictionary(aruco.DICT_6X6_250)
# parameters = aruco.DetectorParameters()
# last_seen = {}

# cv2.namedWindow('Frame')
# cv2.createTrackbar('Threshold', 'Frame', 0, 20000, nothing)
# cv2.createTrackbar('Canny Thresh1', 'Frame', 50, 255, nothing)
# cv2.createTrackbar('Canny Thresh2', 'Frame', 112, 255, nothing)
# cv2.createTrackbar('Brightness', 'Frame', 50, 100, nothing)
# cv2.createTrackbar('Confidence', 'Frame', 50, 100, nothing)
# cv2.createTrackbar('Diff_Thresh', 'Frame', 45, 255, nothing)
# cv2.createTrackbar('Hue Min', 'Frame', 0, 179, nothing)
# cv2.createTrackbar('Hue Max', 'Frame', 179, 179, nothing)
# cv2.createTrackbar('Sat Min', 'Frame', 0, 255, nothing)
# cv2.createTrackbar('Sat Max', 'Frame', 255, 255, nothing)
# cv2.createTrackbar('Val Min', 'Frame', 0, 255, nothing)
# cv2.createTrackbar('Val Max', 'Frame', 255, 255, nothing)

# roi_x_large = 55
# roi_y_large = 0
# roi_width_large = 485
# roi_height_large = 315
# roi_x_medium = 270
# roi_y_medium = 0
# roi_width_medium = 270
# roi_height_medium = 60
# roi_x_small = 464
# roi_y_small = 118
# roi_width_small = 35
# roi_height_small = 35

# fgbg = cv2.createBackgroundSubtractorMOG2()
# initial_gray = None
# post_cleanup_gray = None
# detection_enabled = False
# cleanup_detection_enabled = False
# yolo_detection_enabled = False

# robot_arm_returned = False  # 로봇 팔 상태를 추적하는 변수

# while True:
#     ret, frame = cap.read()
#     if not ret:
#         print("Failed to capture frame from camera.")
#         break

#     threshold = cv2.getTrackbarPos('Threshold', 'Frame')
#     canny_thresh1 = cv2.getTrackbarPos('Canny Thresh1', 'Frame')
#     canny_thresh2 = cv2.getTrackbarPos('Canny Thresh2', 'Frame')
#     brightness = cv2.getTrackbarPos('Brightness', 'Frame')
#     confidence_threshold = cv2.getTrackbarPos('Confidence', 'Frame') / 100.0

#     diff_thresh = cv2.getTrackbarPos('Diff_Thresh', 'Frame')
#     h_min = cv2.getTrackbarPos('Hue Min', 'Frame')
#     h_max = cv2.getTrackbarPos('Hue Max', 'Frame')
#     s_min = cv2.getTrackbarPos('Sat Min', 'Frame')
#     s_max = cv2.getTrackbarPos('Sat Max', 'Frame')
#     v_min = cv2.getTrackbarPos('Val Min', 'Frame')
#     v_max = cv2.getTrackbarPos('Val Max', 'Frame')

#     frame = cv2.convertScaleAbs(frame, alpha=1, beta=(brightness - 50) * 2)
#     fgmask = fgbg.apply(frame)

#     roi_large = fgmask[roi_y_large:roi_y_large + roi_height_large, roi_x_large:roi_x_large + roi_width_large]
#     intrusion_detected = np.sum(roi_large) > threshold

#     warning_detected = False
#     trash_detected = False

#     frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
#     result = hands.process(frame_rgb)
#     if result.multi_hand_landmarks:
#         for hand_landmarks in result.multi_hand_landmarks:
#             for landmark in hand_landmarks.landmark:
#                 x = int(landmark.x * frame.shape[1])
#                 y = int(landmark.y * frame.shape[0])
#                 if roi_x_large < x < roi_x_large + roi_width_large and roi_y_large < y < roi_y_large + roi_height_large and intrusion_detected:
#                     warning_detected = True
#                     break
#             mp.solutions.drawing_utils.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)

#     roi_medium = frame[roi_y_medium:roi_y_medium + roi_height_medium, roi_x_medium:roi_x_medium + roi_width_medium]
#     corners, ids, rejectedImgPoints = aruco.detectMarkers(roi_medium, aruco_dict, parameters=parameters)
#     current_time = time.time()

#     if ids is not None and len(ids) > 0:
#         aruco.drawDetectedMarkers(roi_medium, corners, ids)
#         for id in ids.flatten():
#             last_seen[id] = current_time
#         for id, last_time in list(last_seen.items()):
#             if current_time - last_time > 3:
#                 cv2.putText(frame, f"Action executed for marker ID {id}", (50, 100), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
#                 del last_seen[id]
#                 break

#     small_roi = frame[roi_y_small:roi_y_small + roi_height_small, roi_x_small:roi_x_small + roi_width_small]
#     star_detected, star_contour, edged, contour_count = detect_star_shape(small_roi, canny_thresh1, canny_thresh2)
#     if star_detected:
#         star_contour += [roi_x_small, roi_y_small]
#         cv2.drawContours(frame, [star_contour], -1, (0, 255, 0), 3)
#         cv2.putText(frame, "Star detected", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 2)
#     else:
#         cv2.putText(frame, "Star not detected", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 2)

#     frame_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
#     frame_gray = cv2.GaussianBlur(frame_gray, (21, 21), 0)
#     hsv_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

#     key = cv2.waitKey(1) & 0xFF
#     if key == ord('s'):
#         initial_gray = frame_gray
#         detection_enabled = True
#         print("Initial background set, detection enabled.")
        
#         # 아이스크림 만들기 동작 시작 후 3초 대기
#         time.sleep(3)

#     if detection_enabled and initial_gray is not None:
#         frame_delta = cv2.absdiff(initial_gray, frame_gray)
#         thresh = cv2.threshold(frame_delta, diff_thresh, 255, cv2.THRESH_BINARY)[1]
#         thresh = cv2.dilate(thresh, None, iterations=2)
#         contours, _ = cv2.findContours(thresh.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
#         for contour in contours:
#             if cv2.contourArea(contour) < 500:
#                 continue
#             (x, y, w, h) = cv2.boundingRect(contour)
#             cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
#             cv2.putText(frame, "Motion detected", (10, frame.shape[0] - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
#             # Motion detected 위치로 로봇을 보내기
#             motion_coords_mm_x, motion_coords_mm_y, _ = convert_to_robot_coordinates((x, y, x + w, y + h), "motion")
#             send_to_robot(motion_coords_mm_x, motion_coords_mm_y, action="sweep")
#             trash_detected = True

#     # 로봇이 원래 위치로 돌아왔는지 확인하고 YOLO 감지 활성화
#     if not robot_arm_returned and robot_returned_to_origin():
#         robot_arm_returned = True
#         yolo_detection_enabled = True
#         print("Robot returned to origin, YOLO detection enabled.")

#     if yolo_detection_enabled:
#         yolo_roi = frame[roi_y_large:roi_y_large + roi_height_large, roi_x_large:roi_x_large + roi_width_large]
#         results = model(yolo_roi)
#         objects_to_pick = []
#         objects_to_sweep = []
        
#         for result in results:
#             for box in result.boxes:
#                 x1, y1, x2, y2 = box.xyxy[0]
#                 class_id = int(box.cls[0])
#                 class_name = class_names[class_id]
#                 if box.conf.item() >= confidence_threshold:
#                     x1, y1, x2, y2 = map(int, (x1 + roi_x_large, y1 + roi_y_large, x2 + roi_x_large, y2 + roi_y_large))
#                     robot_coords_mm_x, robot_coords_mm_y, center = convert_to_robot_coordinates((x1, y1, x2, y2), class_name)
#                     if class_name in ['side_cup', 'side_star']:
#                         objects_to_pick.append((robot_coords_mm_x, robot_coords_mm_y, center, class_name))
#                     else:
#                         objects_to_sweep.append((robot_coords_mm_x, robot_coords_mm_y, center, class_name))
#                     # 바운딩 박스 그리기
#                     cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
#                     text = f'{class_name}'
#                     text_x = x1
#                     text_y = y1 - 10 if y1 - 10 > 10 else y1 + 10
#                     cv2.putText(frame, text, (text_x, text_y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

#         # 카메라 중앙 좌표
#         camera_center = (frame.shape[1] / 2, frame.shape[0] / 2)

#         # 먼 거리 순서대로 정렬
#         objects_to_pick.sort(key=lambda obj: np.linalg.norm(np.array(obj[2]) - np.array(camera_center)), reverse=True)
#         objects_to_sweep.sort(key=lambda obj: np.linalg.norm(np.array(obj[2]) - np.array(camera_center)), reverse=True)

#         # 정렬된 순서 출력
#         print("Objects to pick (sorted by distance from camera center):")
#         for obj in objects_to_pick:
#             distance = np.linalg.norm(np.array(obj[2]) - np.array(camera_center))
#             print(f"Class: {obj[3]}, Distance: {distance:.2f}, Center: {obj[2]}")

#         # 먼저 side_cup과 side_star를 처리
#         for coords in objects_to_pick:
#             send_to_robot(coords[0], coords[1], action="pick")

#         # fb_cup, fb_star, trash를 처리
#         for coords in objects_to_sweep:
#             send_to_robot(coords[0], coords[1], action="sweep")

#     draw_rounded_rectangle(frame, (roi_x_large, roi_y_large), roi_width_large, roi_height_large, 20, (255, 0, 0), 2)
#     cv2.rectangle(frame, (roi_x_medium, roi_y_medium), (roi_x_medium + roi_width_medium, roi_y_medium + roi_height_medium), (0, 255, 0), 2)
#     cv2.rectangle(frame, (roi_x_small, roi_y_small), (roi_x_small + roi_width_small, roi_y_small + roi_height_small), (0, 0, 255), 2)
#     cv2.putText(frame, f"Contours: {contour_count}", (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 255), 2)
#     frame[roi_y_small:roi_y_small + roi_height_small, roi_x_small:roi_x_small + roi_width_small] = cv2.cvtColor(edged, cv2.COLOR_GRAY2BGR)

#     if warning_detected:
#         cv2.putText(frame, 'WARNING DETECTED!', (10, frame.shape[0] - 60), cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0, 0, 255), 3)

#     if trash_detected:
#         cv2.putText(frame, 'TRASH DETECTED', (10, frame.shape[0] - 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 0), 2)

#     cv2.imshow('Frame', frame)
#     if key == ord('q'):
#         break

# cap.release()
# cv2.destroyAllWindows()




import cv2
import numpy as np
import time
import torch
import cv2.aruco as aruco
import mediapipe as mp
from ultralytics import YOLO

def detect_star_shape(image, canny_thresh1, canny_thresh2):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    edged = cv2.Canny(blurred, canny_thresh1, canny_thresh2)
    contours, _ = cv2.findContours(edged.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    if len(contours) == 1 or len(contours) == 2:
        contour = contours[0]
        epsilon = 0.02 * cv2.arcLength(contour, True)
        approx = cv2.approxPolyDP(contour, epsilon, True)
        if len(approx) >= 10:
            return True, approx, edged, len(contours)
    return False, None, edged, len(contours)

def nothing(x):
    pass

def draw_rounded_rectangle(image, top_left, width, height, radius, color, thickness):
    x, y = top_left
    top_right = (x + width, y)
    bottom_right = (x + width, y + height)
    bottom_left = (x, y + height)
    cv2.line(image, (x + radius, y), (x + width - radius, y), color, thickness)
    cv2.line(image, (x, y + radius), (x, y + height), color, thickness)
    cv2.line(image, (x + width, y + radius), (x + width, y + height), color, thickness)
    cv2.line(image, (x, y + height), (x + width, y + height), color, thickness)
    cv2.ellipse(image, (x + radius, y + radius), (radius, radius), 180, 0, 90, color, thickness)
    cv2.ellipse(image, (x + width - radius, y + radius), (radius, radius), 270, 0, 90, color, thickness)

# Camera parameters for 3D to 2D conversion
camera_matrix = np.array([[474.51901407, 0, 302.47811758],
                          [0, 474.18970657, 250.66191453],
                          [0, 0, 1]])
dist_coeffs = np.array([[-0.06544764, -0.07654065, -0.00761827, -0.00279316, 0.08062307]])
pixel_to_mm_ratio = 1.55145
robot_origin_x = 295
robot_origin_y = 184

def convert_to_robot_coordinates(bbox, label):
    x1, y1, x2, y2 = map(int, bbox)
    center_x = (x1 + x2) / 2
    center_y = (y1 + y2) / 2
    robot_coords_mm_x = (center_x - robot_origin_x) * pixel_to_mm_ratio * -1
    robot_coords_mm_y = (center_y - robot_origin_y) * pixel_to_mm_ratio
    print(f"{label} detected at robot coordinates: ({robot_coords_mm_x:.2f} mm, {robot_coords_mm_y:.2f} mm)")

    return robot_coords_mm_x, robot_coords_mm_y, (center_x, center_y)

def send_to_robot(x, y, action="pick"):
    print(f"Sending robot to coordinates: x={x:.2f} mm, y={y:.2f} mm for action: {action}")
    # 로봇 제어 API 호출 예제 (이 부분을 실제 로봇 제어 코드로 대체하세요)
    # robot.move_to(x, y)
    # if action == "pick":
    #     robot.pick()
    # elif action == "sweep":
    #     robot.sweep()

def get_robot_current_position():
    # 실제 로봇 제어 API를 사용하여 로봇 팔의 현재 위치를 반환하도록 구현하세요
    # 예시: (x, y) 좌표를 반환한다고 가정합니다.
    current_x = 0  # 여기에 실제 값으로 대체
    current_y = 0  # 여기에 실제 값으로 대체
    return current_x, current_y

def robot_returned_to_origin():
    # 로봇 팔의 원래 위치 좌표
    origin_x = 0  # 여기에 실제 원래 위치 값으로 대체
    origin_y = 0  # 여기에 실제 원래 위치 값으로 대체
    
    # 로봇 팔의 현재 위치를 얻음
    current_x, current_y = get_robot_current_position()
    
    # 현재 위치와 원래 위치를 비교하여 일정 오차 범위 내에 있으면 True 반환
    if abs(current_x - origin_x) < 5 and abs(current_y - origin_y) < 5:  # 오차 범위 5mm로 설정
        return True
    else:
        return False

# 비디오 캡처 및 YOLO 모델 초기화
cap = cv2.VideoCapture(0)
model = YOLO('/home/jinjuuk/dev_ws/pt_files/newjeans.pt')
device = 'cuda' if torch.cuda.is_available() else 'cpu'

if torch.cuda.is_available():
    model.to(device)
print(f'Using device: {device}')

class_names = ['fb_cup', 'fb_star', 'side_cup', 'side_star', 'trash']

mp_hands = mp.solutions.hands
hands = mp_hands.Hands(max_num_hands=2, min_detection_confidence=0.7)
aruco_dict = aruco.getPredefinedDictionary(aruco.DICT_6X6_250)
parameters = aruco.DetectorParameters()
last_seen = {}

cv2.namedWindow('Frame')
cv2.createTrackbar('Threshold', 'Frame', 0, 20000, nothing)
cv2.createTrackbar('Canny Thresh1', 'Frame', 50, 255, nothing)
cv2.createTrackbar('Canny Thresh2', 'Frame', 112, 255, nothing)
cv2.createTrackbar('Brightness', 'Frame', 50, 100, nothing)
cv2.createTrackbar('Confidence', 'Frame', 50, 100, nothing)
cv2.createTrackbar('Diff_Thresh', 'Frame', 45, 255, nothing)
cv2.createTrackbar('Hue Min', 'Frame', 0, 179, nothing)
cv2.createTrackbar('Hue Max', 'Frame', 179, 179, nothing)
cv2.createTrackbar('Sat Min', 'Frame', 0, 255, nothing)
cv2.createTrackbar('Sat Max', 'Frame', 255, 255, nothing)
cv2.createTrackbar('Val Min', 'Frame', 0, 255, nothing)
cv2.createTrackbar('Val Max', 'Frame', 255, 255, nothing)

roi_x_large = 55
roi_y_large = 0
roi_width_large = 485
roi_height_large = 315
roi_x_medium = 270
roi_y_medium = 0
roi_width_medium = 270
roi_height_medium = 60
roi_x_small = 464
roi_y_small = 118
roi_width_small = 35
roi_height_small = 35

fgbg = cv2.createBackgroundSubtractorMOG2()
initial_gray = None
post_cleanup_gray = None
detection_enabled = False
cleanup_detection_enabled = False
yolo_detection_enabled = False

robot_arm_returned = False  # 로봇 팔 상태를 추적하는 변수

while True:
    ret, frame = cap.read()
    if not ret:
        print("Failed to capture frame from camera.")
        break

    threshold = cv2.getTrackbarPos('Threshold', 'Frame')
    canny_thresh1 = cv2.getTrackbarPos('Canny Thresh1', 'Frame')
    canny_thresh2 = cv2.getTrackbarPos('Canny Thresh2', 'Frame')
    brightness = cv2.getTrackbarPos('Brightness', 'Frame')
    confidence_threshold = cv2.getTrackbarPos('Confidence', 'Frame') / 100.0

    diff_thresh = cv2.getTrackbarPos('Diff_Thresh', 'Frame')
    h_min = cv2.getTrackbarPos('Hue Min', 'Frame')
    h_max = cv2.getTrackbarPos('Hue Max', 'Frame')
    s_min = cv2.getTrackbarPos('Sat Min', 'Frame')
    s_max = cv2.getTrackbarPos('Sat Max', 'Frame')
    v_min = cv2.getTrackbarPos('Val Min', 'Frame')
    v_max = cv2.getTrackbarPos('Val Max', 'Frame')

    frame = cv2.convertScaleAbs(frame, alpha=1, beta=(brightness - 50) * 2)
    fgmask = fgbg.apply(frame)

    roi_large = fgmask[roi_y_large:roi_y_large + roi_height_large, roi_x_large:roi_x_large + roi_width_large]
    intrusion_detected = np.sum(roi_large) > threshold

    warning_detected = False
    trash_detected = False

    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    result = hands.process(frame_rgb)
    if result.multi_hand_landmarks:
        for hand_landmarks in result.multi_hand_landmarks:
            for landmark in hand_landmarks.landmark:
                x = int(landmark.x * frame.shape[1])
                y = int(landmark.y * frame.shape[0])
                if roi_x_large < x < roi_x_large + roi_width_large and roi_y_large < y < roi_y_large + roi_height_large and intrusion_detected:
                    warning_detected = True
                    break
            mp.solutions.drawing_utils.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)

    roi_medium = frame[roi_y_medium:roi_y_medium + roi_height_medium, roi_x_medium:roi_x_medium + roi_width_medium]
    corners, ids, rejectedImgPoints = aruco.detectMarkers(roi_medium, aruco_dict, parameters=parameters)
    current_time = time.time()

    if ids is not None and len(ids) > 0:
        aruco.drawDetectedMarkers(roi_medium, corners, ids)
        for id in ids.flatten():
            last_seen[id] = current_time
        for id, last_time in list(last_seen.items()):
            if current_time - last_time > 3:
                cv2.putText(frame, f"Action executed for marker ID {id}", (50, 100), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
                del last_seen[id]
                break

    small_roi = frame[roi_y_small:roi_y_small + roi_height_small, roi_x_small:roi_x_small + roi_width_small]
    star_detected, star_contour, edged, contour_count = detect_star_shape(small_roi, canny_thresh1, canny_thresh2)
    if star_detected:
        star_contour += [roi_x_small, roi_y_small]
        cv2.drawContours(frame, [star_contour], -1, (0, 255, 0), 3)
        cv2.putText(frame, "Star detected", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 2)
    else:
        cv2.putText(frame, "Star not detected", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 2)

    frame_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    frame_gray = cv2.GaussianBlur(frame_gray, (21, 21), 0)
    hsv_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

    key = cv2.waitKey(1) & 0xFF
    if key == ord('s'):
        initial_gray = frame_gray
        detection_enabled = True
        print("Initial background set, detection enabled.")
        
        # 아이스크림 만들기 동작 시작 후 3초 대기
        time.sleep(3)

    if detection_enabled and initial_gray is not None:
        frame_delta = cv2.absdiff(initial_gray, frame_gray)
        thresh = cv2.threshold(frame_delta, diff_thresh, 255, cv2.THRESH_BINARY)[1]
        thresh = cv2.dilate(thresh, None, iterations=2)
        contours, _ = cv2.findContours(thresh.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        for contour in contours:
            if cv2.contourArea(contour) < 500:
                continue
            (x, y, w, h) = cv2.boundingRect(contour)
            cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
            cv2.putText(frame, "Other trash detected", (10, frame.shape[0] - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
            # Motion detected 위치로 로봇을 보내기
            other_trash_coords_mm_x, other_trash_coords_mm_y, _ = convert_to_robot_coordinates((x, y, x + w, y + h), "Other trash")
            #print(f"Other trash detected at robot coordinates: ({other_trash_coords_mm_x:.2f} mm, {other_trash_coords_mm_y:.2f} mm)")
            send_to_robot(other_trash_coords_mm_x, other_trash_coords_mm_y, action="sweep")
            trash_detected = True

    # 로봇이 원래 위치로 돌아왔는지 확인하고 YOLO 감지 활성화
    if not robot_arm_returned and robot_returned_to_origin():
        robot_arm_returned = True
        yolo_detection_enabled = True
        print("Robot returned to origin, YOLO detection enabled.")

    if yolo_detection_enabled:
        yolo_roi = frame[roi_y_large:roi_y_large + roi_height_large, roi_x_large:roi_x_large + roi_width_large]
        results = model(yolo_roi)
        objects_to_pick = []
        objects_to_sweep = []
        
        for result in results:
            for box in result.boxes:
                x1, y1, x2, y2 = box.xyxy[0]
                class_id = int(box.cls[0])
                class_name = class_names[class_id]
                if box.conf.item() >= confidence_threshold:
                    x1, y1, x2, y2 = map(int, (x1 + roi_x_large, y1 + roi_y_large, x2 + roi_x_large, y2 + roi_y_large))
                    robot_coords_mm_x, robot_coords_mm_y, center = convert_to_robot_coordinates((x1, y1, x2, y2), class_name)
                    if class_name in ['side_cup', 'side_star']:
                        objects_to_pick.append((robot_coords_mm_x, robot_coords_mm_y, center, class_name))
                    else:
                        objects_to_sweep.append((robot_coords_mm_x, robot_coords_mm_y, center, class_name))
                    # 바운딩 박스 그리기
                    cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                    text = f'{class_name}'
                    text_x = x1
                    text_y = y1 - 10 if y1 - 10 > 10 else y1 + 10
                    cv2.putText(frame, text, (text_x, text_y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

        # 카메라 중앙 좌표
        camera_center = (frame.shape[1] / 2, frame.shape[0] / 2)

        # 먼 거리 순서대로 정렬
        objects_to_pick.sort(key=lambda obj: np.linalg.norm(np.array(obj[2]) - np.array(camera_center)), reverse=True)
        objects_to_sweep.sort(key=lambda obj: np.linalg.norm(np.array(obj[2]) - np.array(camera_center)), reverse=True)

        # 정렬된 순서 출력
        print("Objects to pick (sorted by distance from camera center):")
        for obj in objects_to_pick:
            distance = np.linalg.norm(np.array(obj[2]) - np.array(camera_center))
            print(f"Class: {obj[3]}, Distance: {distance:.2f}, Center: {obj[2]}")

        print("Objects to sweep (sorted by distance from camera center):")
        for obj in objects_to_sweep:
            distance = np.linalg.norm(np.array(obj[2]) - np.array(camera_center))
            print(f"Class: {obj[3]}, Distance: {distance:.2f}, Center: {obj[2]}")

        # 먼저 side_cup과 side_star를 처리
        for coords in objects_to_pick:
            send_to_robot(coords[0], coords[1], action="pick")

        # fb_cup, fb_star, trash를 처리
        for coords in objects_to_sweep:
            send_to_robot(coords[0], coords[1], action="sweep")

    draw_rounded_rectangle(frame, (roi_x_large, roi_y_large), roi_width_large, roi_height_large, 20, (255, 0, 0), 2)
    cv2.rectangle(frame, (roi_x_medium, roi_y_medium), (roi_x_medium + roi_width_medium, roi_y_medium + roi_height_medium), (0, 255, 0), 2)
    cv2.rectangle(frame, (roi_x_small, roi_y_small), (roi_x_small + roi_width_small, roi_y_small + roi_height_small), (0, 0, 255), 2)
    cv2.putText(frame, f"Contours: {contour_count}", (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 255), 2)
    frame[roi_y_small:roi_y_small + roi_height_small, roi_x_small:roi_x_small + roi_width_small] = cv2.cvtColor(edged, cv2.COLOR_GRAY2BGR)

    if warning_detected:
        cv2.putText(frame, 'WARNING DETECTED!', (10, frame.shape[0] - 60), cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0, 0, 255), 3)

    if trash_detected:
        cv2.putText(frame, 'TRASH DETECTED', (10, frame.shape[0] - 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 0), 2)

    cv2.imshow('Frame', frame)
    if key == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
