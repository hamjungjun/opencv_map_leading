import cv2
import numpy as np

# 1. 웹캠 연결
cap = cv2.VideoCapture(0)

# 2. ArUco 사전 설정 (DICT_4X4_50)
aruco_dict = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_4X4_50)

try:
    aruco_params = cv2.aruco.DetectorParameters()
    detector = cv2.aruco.ArucoDetector(aruco_dict, aruco_params)
    use_new_api = True
except AttributeError:
    aruco_params = cv2.aruco.DetectorParameters_create()
    use_new_api = False

print("ArUco 마커 인식 테스트 시작... ('q' 누르면 종료)")

while True:
    ret, frame = cap.read()
    if not ret:
        print("카메라 프레임을 불러올 수 없습니다.")
        break

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # 마커 검출 시도
    if use_new_api:
        corners, ids, rejected = detector.detectMarkers(gray)
    else:
        corners, ids, rejected = cv2.aruco.detectMarkers(gray, aruco_dict, parameters=aruco_params)

    # 마커가 검출되면 화면에 초록색 테두리와 ID 표시
    if ids is not None:
        cv2.aruco.drawDetectedMarkers(frame, corners, ids)
        for i, marker_id in enumerate(ids):
            print(f"✅ 마커 인식 성공! [ID: {marker_id[0]}]")
    else:
        # 인식 실패 시 원인 파악용 (마커 후보군 빨간선 표시)
        if rejected is not None and len(rejected) > 0:
            cv2.aruco.drawDetectedMarkers(frame, rejected, borderColor=(0, 0, 255))

    cv2.putText(frame, f"Detected Markers: {len(ids) if ids is not None else 0}", (20, 40),
                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)

    cv2.imshow("ArUco Detection Test", frame)

    if cv2.waitKey(30) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()