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

#====================   미로 생성   =======================================\
import cv2
import numpy as np

# 1. 맵 전체 크기 (600x600 px)
MAZE_SIZE = 600
img = np.ones((MAZE_SIZE, MAZE_SIZE, 3), dtype=np.uint8) * 255

# BGR 색상 정의 & 벽 두께 (12px)
BLUE_COLOR = (255, 0, 0)
RED_COLOR = (0, 0, 255)
THICKNESS = 12  

# ----------------------------------------------------
# 1. 외곽 테두리 벽 생성 & 입출구 Open
# ----------------------------------------------------
cv2.rectangle(img, (15, 15), (585, 585), BLUE_COLOR, THICKNESS)

# 입구(좌상단) & 출구(우하단)
cv2.rectangle(img, (15, 30), (15, 110), (255, 255, 255), THICKNESS + 4)
cv2.rectangle(img, (585, 490), (585, 570), (255, 255, 255), THICKNESS + 4)

# ----------------------------------------------------
# 2. 극도로 단순화된 오픈형 미로 벽 배치 (완전히 더 뚫음)
# ----------------------------------------------------
# 복잡한 갈림길을 없애고 거대한 통로 공간 확보
walls = [
    # (x1, y1, x2, y2)
    # 주요 구획용 긴 세로 벽 2개만 남김 (상단/하단 갭 매우 크게 설정)
    (150, 15,  150, 480),   # 아래쪽에 거대한 통로(y: 480~585) 생성
    (350, 120, 350, 585),   # 위쪽에 거대한 통로(y: 15~120) 생성

    # 이동을 막는 모든 가로 벽 제거
]

for w in walls:
    cv2.line(img, (w[0], w[1]), (w[2], w[3]), BLUE_COLOR, THICKNESS)

# ----------------------------------------------------
# 3. 기준 마커/사각점 (우측 상단 빈 공간)
# ----------------------------------------------------
# 넓어진 공간을 고려하여 위치 살짝 조정
cv2.rectangle(img, (510, 30), (560, 80), RED_COLOR, -1)

# 이미지 저장 및 출력
cv2.imwrite("blue_maze_open.png", img)
print("-> 완전히 '더 뚫린' 오픈형 'blue_maze_open.png' 이미지가 저장되었습니다!")

cv2.imshow("Open Blue Maze", img)
cv2.waitKey(0)
cv2.destroyAllWindows()