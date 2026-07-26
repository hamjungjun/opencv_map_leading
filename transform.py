import cv2
import numpy as np

# ==========================================
# 1. 실시간 웹캠 연결
# ==========================================
cap = cv2.VideoCapture(1)  # 외장 카메라는 1 또는 2로 변경

if not cap.isOpened():
    print("카메라를 열 수 없습니다. 연결 상태를 확인해주세요.")
    exit()

# 카메라 해상도 설정
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

# 전역 변수 설정
drawing = False      # 마우스 드래그 중 여부
ix, iy = -1, -1      # 드래그 시작 좌표
fx, fy = -1, -1      # 드래그 종료 좌표
bbox_selected = False

# 변환 및 좌표 관련 변수
matrix = None
MAP_WIDTH = 1000
MAP_HEIGHT = 1000  # 드래그 후 비율에 따라 자동 재계산됨
mouse_x, mouse_y = 0, 0
goal_pt = None

# -------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
# Standard Scale 변수
# -------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
square_px = 0.0  # 카메라 인식 기준 사각형 한 변의 px 길이
real_cm = 0.0    # User가 입력한 기준 사각형의 실제 한 변 길이 (cm)
cm_scale = 0.0   # 스케일 계수 (cm / px)
# -------------------------------------------------------------------------------------------------------------------------------------------------------------------------------


# ==========================================
# 파란색 기준 사각형 검출 함수
# ==========================================
def detect_reference_square(warped_img):
    """
    HSV 영역에서 파란색 객체를 찾아 사각형의 평균 픽셀 변 길이를 반환
    """
    hsv = cv2.cvtColor(warped_img, cv2.COLOR_BGR2HSV)

    # 파란색 HSV 범주 설정 (환경 빛에 따라 필요시 조절)
    lower_blue = np.array([90, 80, 50])
    upper_blue = np.array([135, 255, 255])

    mask = cv2.inRange(hsv, lower_blue, upper_blue)

    # 노이즈 제거
    kernel = np.ones((3, 3), np.uint8)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)

    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    best_square_px = 0.0
    best_cnt = None

    for cnt in contours:
        area = cv2.contourArea(cnt)
        if area < 300:  # 너무 작은 노이즈 제외
            continue

        # 다각형 근사 (4개 꼭짓점 찾기)
        peri = cv2.arcLength(cnt, True)
        approx = cv2.approxPolyDP(cnt, 0.04 * peri, True)

        if len(approx) == 4:
            pts = approx.reshape(4, 2)
            # 변 길이계산
            side1 = np.linalg.norm(pts[0] - pts[1])
            side2 = np.linalg.norm(pts[1] - pts[2])
            side3 = np.linalg.norm(pts[2] - pts[3])
            side4 = np.linalg.norm(pts[3] - pts[0])

            # 4변의 평균 픽셀 길이
            avg_side = (side1 + side2 + side3 + side4) / 4.0

            if avg_side > best_square_px:
                best_square_px = avg_side
                best_cnt = cnt

    return best_square_px, best_cnt


# ==========================================
# 2. 마우스 콜백 함수 정의
# ==========================================
def draw_rectangle(event, x, y, flags, param):
    """ 1단계: 실시간 카메라 화면에서 드래그 영역 선택 """
    global ix, iy, fx, fy, drawing, bbox_selected

    if event == cv2.EVENT_LBUTTONDOWN:
        drawing = True
        ix, iy = x, y
        bbox_selected = False

    elif event == cv2.EVENT_MOUSEMOVE:
        if drawing:
            fx, fy = x, y

    elif event == cv2.EVENT_LBUTTONUP:
        drawing = False
        fx, fy = x, y
        bbox_selected = True
        print(f"\n[선택한 사각형 영역]: 시작({ix}, {iy}) -> 끝({fx}, {fy})")
        print("-> [Enter] 키를 누르면 스케일 변환 및 로봇 추적을 시작합니다.")


def track_coordinates(event, x, y, flags, param):
    """ 2단계: 펴진 맵에서 마우스 실시간 좌표 및 Goal 지점 선택 """
    global mouse_x, mouse_y, goal_pt
    if event == cv2.EVENT_MOUSEMOVE:
        mouse_x, mouse_y = x, y
    elif event == cv2.EVENT_LBUTTONDOWN:
        goal_pt = (x, y)
        cart_gx = x
        cart_gy = MAP_HEIGHT - y
        print(f"[Goal 지정 완료]: 화면 ({x}, {y}) | 카테시안 ({cart_gx}, {cart_gy}) px")


# ==========================================
# 3. ArUco 마커 디텍터 설정
# ==========================================
aruco_dict = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_4X4_50)
try:
    aruco_params = cv2.aruco.DetectorParameters()
    detector = cv2.aruco.ArucoDetector(aruco_dict, aruco_params)
    use_new_api = True
except AttributeError:
    aruco_params = cv2.aruco.DetectorParameters_create()
    use_new_api = False

# 메인 윈도우 생성
cv2.namedWindow("Drag Map Region", cv2.WINDOW_NORMAL)
cv2.resizeWindow("Drag Map Region", 1280, 720)
cv2.setMouseCallback("Drag Map Region", draw_rectangle)

print("=" * 60)
print(" 사용 방법:")
print(" 1. 카메라 화면에서 맵 영역을 마우스 클릭 후 드래그하세요.")
print(" 2. 박스가 잘 잡혔으면 키보드 [Enter] 또는 [Space]를 누르세요.")
print(" 3. 'r' 키: 영역 재선택 | 'q' 키: 프로그램 종료")
print("=" * 60)


# ==========================================
# 4. 실시간 메인 루프
# ==========================================
while True:
    ret, frame = cap.read()
    if not ret:
        print("카메라 프레임을 불러올 수 없습니다.")
        break

    # ----------------------------------------------------
    # [1단계] 실시간 영상 전송 & 드래그 스케일 영역 선택
    # ----------------------------------------------------
    if matrix is None:
        display_frame = frame.copy()

        # 드래그 박스 그려주기
        if ix != -1 and fx != -1:
            cv2.rectangle(display_frame, (ix, iy), (fx, fy), (0, 255, 0), 2)

        cv2.putText(display_frame, "Drag map area & Press [Enter]", (20, 40),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2, cv2.LINE_AA)

        cv2.imshow("Drag Map Region", display_frame)

    # ----------------------------------------------------
    # [2단계] 스케일 변환된 맵 + 기준 사각형 검출 + ArUco 로봇 위치 추적
    # ----------------------------------------------------
    else:
        # 실시간 카메라 프레임을 스케일 조정하여 Warp 변환
        warped_map = cv2.warpPerspective(frame, matrix, (MAP_WIDTH, MAP_HEIGHT))
        display_img = warped_map.copy()

        # ----------------------------------------------------
        # 파란색 기준 사각형 자동 검출 및 스케일 업데이트
        # ----------------------------------------------------
        detected_px, cnt = detect_reference_square(warped_map)
        if detected_px > 0:
            square_px = detected_px
            if real_cm > 0:
                cm_scale = real_cm / square_px

            # 인식된 기준 사각형 외각 시각화
            cv2.drawContours(display_img, [cnt], -1, (0, 255, 0), 2)
            cv2.putText(display_img, f"Ref Sq: {square_px:.1f}px", (cnt[0][0][0], cnt[0][0][1] - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 255, 0), 1)

        # ArUco 마커 검출 (로봇 위치 인식)
        gray = cv2.cvtColor(display_img, cv2.COLOR_BGR2GRAY)
        if use_new_api:
            corners, ids, _ = detector.detectMarkers(gray)
        else:
            corners, ids, _ = cv2.aruco.detectMarkers(gray, aruco_dict, parameters=aruco_params)

        robot_screen_pt = None
        robot_cartesian_pt = None
        robot_angle_deg = None

        if ids is not None and len(corners) > 0:
            marker_corners = corners[0][0]  # 검출된 첫 번째 마커

            # 1. 로봇 화면 좌표 (px)
            center_x = int(np.mean(marker_corners[:, 0]))
            center_y = int(np.mean(marker_corners[:, 1]))
            robot_screen_pt = (center_x, center_y)

            # 2. 기존 좌하단 (0,0) 원점 기준 카테시안 좌표 변환
            cart_rx = center_x
            cart_ry = MAP_HEIGHT - center_y
            robot_cartesian_pt = (cart_rx, cart_ry)

            # 3. 로봇 헤딩 각도(θ) 계산
            p0, p1 = marker_corners[0], marker_corners[1]
            dx, dy = p1[0] - p0[0], p1[1] - p0[1]
            angle_rad = np.arctan2(-dy, dx)
            robot_angle_deg = np.degrees(angle_rad) % 360

            # 4. 로봇 시각화 (파란 원 및 방향 화살표)
            cv2.circle(display_img, robot_screen_pt, 8, (255, 0, 0), -1)

            arrow_len = 35
            arrow_end_x = int(center_x + arrow_len * np.cos(np.radians(robot_angle_deg)))
            arrow_end_y = int(center_y - arrow_len * np.sin(np.radians(robot_angle_deg)))
            cv2.arrowedLine(display_img, robot_screen_pt, (arrow_end_x, arrow_end_y),
                            (255, 0, 0), 2, tipLength=0.3)

            # 로봇 위치 정보 표기
            info_text = f"ROBOT: ({cart_rx}, {cart_ry}) | {robot_angle_deg:.1f}deg"
            cv2.putText(display_img, info_text, (center_x - 70, center_y - 20),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 0, 0), 1, cv2.LINE_AA)

        # ----------------------------------------------------
        # 마우스 UI (십자 마커 & 좌표/스케일 출력)
        # ----------------------------------------------------
        cartesian_x = mouse_x
        cartesian_y = MAP_HEIGHT - mouse_y

        # 빨간색 십자 마커
        cv2.drawMarker(display_img, (mouse_x, mouse_y), (0, 0, 255),
                       markerType=cv2.MARKER_CROSS, markerSize=15, thickness=1)

        # 마우스 위치 좌표 텍스트
        text_x = mouse_x + 15 if mouse_x + 160 < MAP_WIDTH else mouse_x - 160
        text_y = mouse_y - 10 if mouse_y - 20 > 0 else mouse_y + 20
        cv2.putText(display_img, f"({cartesian_x}, {cartesian_y}) px", (text_x, text_y),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 1, cv2.LINE_AA)

        # 왼쪽 상단 스케일 정보 UI
        scale_info = f"Scale: {cm_scale:.4f} cm/px (1px = {cm_scale:.2f}cm)" if cm_scale > 0 else "Scale: N/A"
        cv2.putText(display_img, scale_info, (20, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 2, cv2.LINE_AA)

        # Goal 지점 및 로봇과의 거리 표시 (px 및 cm)
        if goal_pt is not None:
            cv2.circle(display_img, goal_pt, 8, (0, 0, 255), -1)
            cv2.putText(display_img, "GOAL", (goal_pt[0] + 10, goal_pt[1] - 5),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1, cv2.LINE_AA)

            if robot_screen_pt is not None:
                cv2.line(display_img, robot_screen_pt, goal_pt, (0, 255, 0), 2)
                dist_px = np.sqrt((goal_pt[0] - robot_screen_pt[0])**2 + (goal_pt[1] - robot_screen_pt[1])**2)
                mid_x = (robot_screen_pt[0] + goal_pt[0]) // 2
                mid_y = (robot_screen_pt[1] + goal_pt[1]) // 2

                # cm 변환 출력
                if cm_scale > 0:
                    dist_cm = dist_px * cm_scale
                    dist_text = f"Dist: {dist_px:.1f}px ({dist_cm:.1f}cm)"
                else:
                    dist_text = f"Dist: {dist_px:.1f}px"

                cv2.putText(display_img, dist_text, (mid_x, mid_y - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2, cv2.LINE_AA)

        cv2.imshow("Warped Top-View Map", display_img)

    # ----------------------------------------------------
    # 키보드 입력 처리
    # ----------------------------------------------------
    key = cv2.waitKey(20) & 0xFF

    # Enter(13) 또는 Space(32) 키: 원근 변환 행렬 생성 및 사용자 실측 cm 입력받기
    if key in [13, 32] and bbox_selected and matrix is None:
        x1, x2 = min(ix, fx), max(ix, fx)
        y1, y2 = min(iy, fy), max(iy, fy)

        box_w = x2 - x1
        box_h = y2 - y1

        if box_w > 10 and box_h > 10:
            # 비율에 맞춘 MAP_HEIGHT 스케일링 계산
            MAP_HEIGHT = int(MAP_WIDTH * (box_h / box_w))

            src_pts = np.float32([[x1, y1], [x2, y1], [x2, y2], [x1, y2]])
            dst_pts = np.float32([[0, 0], [MAP_WIDTH, 0], [MAP_WIDTH, MAP_HEIGHT], [0, MAP_HEIGHT]])

            matrix = cv2.getPerspectiveTransform(src_pts, dst_pts)

            # 사용자로부터 파란색 사각형의 실제 한 변 크기(cm) 입력받기
            print("\n" + "=" * 50)
            try:
                user_input = input("기준 파란색 사각형의 실제 한 변 길이(cm)를 입력하세요: ")
                real_cm = float(user_input)
                print(f"-> 입력된 실제 크기: {real_cm} cm")
            except ValueError:
                print("-> 잘못된 입력입니다. 스케일 계산 없이 픽셀 단위로만 표시합니다.")
                real_cm = 0.0
            print("=" * 50 + "\n")

            cv2.destroyWindow("Drag Map Region")
            cv2.namedWindow("Warped Top-View Map", cv2.WINDOW_NORMAL)
            cv2.resizeWindow("Warped Top-View Map", 1000, int(1000 * (MAP_HEIGHT / MAP_WIDTH)))
            cv2.setMouseCallback("Warped Top-View Map", track_coordinates)

    # 'r' 키: 영역 초기화 및 재드래그
    elif key == ord('r'):
        matrix = None
        bbox_selected = False
        ix, iy, fx, fy = -1, -1, -1, -1
        goal_pt = None
        real_cm = 0.0
        square_px = 0.0
        cm_scale = 0.0
        cv2.destroyAllWindows()
        cv2.namedWindow("Drag Map Region", cv2.WINDOW_NORMAL)
        cv2.resizeWindow("Drag Map Region", 1280, 720)
        cv2.setMouseCallback("Drag Map Region", draw_rectangle)

    # 'q' 키: 종료
    elif key == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()