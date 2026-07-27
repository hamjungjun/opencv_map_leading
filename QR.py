'''import cv2
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
cv2.destroyAllWindows()'''

#====================   미로 생성   =======================================
import cv2
import numpy as np

# 1. 맵 전체 크기 축소 (600x600 px)
MAZE_SIZE = 600
img = np.ones((MAZE_SIZE, MAZE_SIZE, 3), dtype=np.uint8) * 255

# BGR 기준 파란색 & 벽 두께 축소 (12px)
BLUE_COLOR = (255, 0, 0)
THICKNESS = 12  

# ----------------------------------------------------
# 1. 외곽 테두리 벽 생성 & 입출구
# ----------------------------------------------------
cv2.rectangle(img, (15, 15), (585, 585), BLUE_COLOR, THICKNESS)

# 입구(좌상단) & 출구(우하단) 뚫어주기
cv2.rectangle(img, (15, 30), (15, 110), (255, 255, 255), THICKNESS + 4)
cv2.rectangle(img, (585, 490), (585, 570), (255, 255, 255), THICKNESS + 4)

# ----------------------------------------------------
# 2. 컴팩트 미로 벽 배치 (600x600 스케일 맞춤)
# ----------------------------------------------------
walls = [
    # (x1, y1, x2, y2)
    (130, 15, 130, 380),
    (250, 220, 250, 585),
    (370, 15, 370, 420),
    (490, 180, 490, 585),
    (130, 220, 250, 220),
    (370, 350, 490, 350),
    (250, 120, 370, 120),
]

for w in walls:
    cv2.line(img, (w[0], w[1]), (w[2], w[3]), BLUE_COLOR, THICKNESS)

# ----------------------------------------------------
# 3. 자동 스케일 인식용 기준 파란 사각형 (50px x 50px)
# ----------------------------------------------------
# 우측 상단 빈 공간에 배치
cv2.rectangle(img, (510, 30), (560, 80), BLUE_COLOR, -1)

# 이미지 저장 및 출력
cv2.imwrite("blue_maze_compact.png", img)
print("-> 작은 크기의 'blue_maze_compact.png' 이미지가 저장되었습니다!")

cv2.imshow("Compact Blue Maze", img)
cv2.waitKey(0)
cv2.destroyAllWindows()