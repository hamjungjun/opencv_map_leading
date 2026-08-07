import cv2
import numpy as np
import math
from collections import deque


# ==========================================
# 0. 포즈 스무더 (Pose Smoother) 클래스
# ==========================================
class PoseSmoother:
    def __init__(self, window_size=5):
        self.x_history = deque(maxlen=window_size)
        self.y_history = deque(maxlen=window_size)
        self.sin_history = deque(maxlen=window_size)
        self.cos_history = deque(maxlen=window_size)

    def update(self, x, y, angle_deg):
        self.x_history.append(x)
        self.y_history.append(y)

        rad = math.radians(angle_deg)
        self.sin_history.append(math.sin(rad))
        self.cos_history.append(math.cos(rad))

        avg_x = int(np.mean(self.x_history))
        avg_y = int(np.mean(self.y_history))

        avg_sin = np.mean(self.sin_history)
        avg_cos = np.mean(self.cos_history)
        avg_angle_deg = math.degrees(math.atan2(avg_sin, avg_cos)) % 360.0

        return (avg_x, avg_y), avg_angle_deg

    def reset(self):
        self.x_history.clear()
        self.y_history.clear()
        self.sin_history.clear()
        self.cos_history.clear()


# ==========================================
# 1. 콜백용 빈 함수 & 전역 변수
# ==========================================
def nothing(value):
    pass


latest_hsv_map = None  # 마우스 클릭으로 HSV 값 확인할 warped 마스크용 이미지

# 전역 상태 변수
drawing = False  # 마우스 드래그 상태
ix, iy = -1, -1  # 드래그 시작 좌표
fx, fy = -1, -1  # 드래그 종료 좌표
bbox_selected = False

matrix = None  # Perspective Transform 행렬
MAP_WIDTH = 1000
MAP_HEIGHT = 1000  # 드래그 영역 비율에 맞춰 자동 재계산
mouse_x, mouse_y = 0, 0
goal_pt = None

# Standard Scale 변수
square_px = 0.0  # 기준 사각형 한 변의 px
real_cm = 0.0  # 사용자 입력 실제 한 변의 길이 (cm)
cm_scale = 0.0  # 스케일 계수 (cm / px)


# ==========================================
# 2. 마우스 콜백 함수들
# ==========================================
def draw_rectangle(event, x, y, flags, param):
    """1단계: 원본 카메라 화면에서 맵 영역 드래그 선택"""
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
        print(f"\n[선택한 영역]: ({ix}, {iy}) -> ({fx}, {fy})")
        print("-> [Enter] 키를 누르면 스케일 입력 및 원근 변환을 진행합니다.")


def track_coordinates(event, x, y, flags, param):
    """2단계: Top-View 맵에서 마우스 실시간 좌표, Goal 지정 및 HSV 값 확인"""
    global mouse_x, mouse_y, goal_pt, latest_hsv_map

    if event == cv2.EVENT_MOUSEMOVE:
        mouse_x, mouse_y = x, y

    elif event == cv2.EVENT_LBUTTONDOWN:
        goal_pt = (x, y)
        cart_gx = x
        cart_gy = MAP_HEIGHT - y
        print(f"[Goal 지정]: 화면 ({x}, {y}) | 카테시안 ({cart_gx}, {cart_gy}) px")

        if latest_hsv_map is not None:
            h, w = latest_hsv_map.shape[:2]
            if 0 <= x < w and 0 <= y < h:
                hv, sv, vv = latest_hsv_map[y, x]
                print(
                    f" -> [클릭 지점 HSV]: H={int(hv)}, S={int(sv)}, V={int(vv)}"
                )


# ==========================================
# 3. 빨간색 기준 사각형 검출 함수 (강화 버전)
# ==========================================
def detect_reference_square(warped_img, red_mask):
    """
    HSV 마스크 영역에서 빨간색 기준 사각형을 유연하면서도 정확하게 검출
    """
    contours, _ = cv2.findContours(
        red_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
    )

    best_square_px = 0.0
    best_cnt = None

    for cnt in contours:
        area = cv2.contourArea(cnt)
        if area < 200:  # 노이즈 필터링
            continue

        # 1차 시도: 다각형 근사 (approxPolyDP)
        peri = cv2.arcLength(cnt, True)
        approx = cv2.approxPolyDP(cnt, 0.035 * peri, True)

        # 다각형 점이 4개 내외(4~6개)일 때 사각형 검증
        if 4 <= len(approx) <= 6:
            rect = cv2.minAreaRect(cnt)
            (w_rect, h_rect) = rect[1]

            if w_rect == 0 or h_rect == 0:
                continue

            aspect_ratio = float(w_rect) / h_rect
            # 정사각형에 가까운지 체크 (종횡비 0.7 ~ 1.4)
            if 0.7 <= aspect_ratio <= 1.4:
                avg_side = (w_rect + h_rect) / 2.0
                if avg_side > best_square_px:
                    best_square_px = avg_side
                    best_cnt = cnt

    return best_square_px, best_cnt


# ==========================================
# 4. 카메라 및 ArUco, UI 창 초기화
# ==========================================
CAMERA_INDEX = 0  # 환경에 맞춰 0, 1, 2 중 설정

cap = cv2.VideoCapture(CAMERA_INDEX)
if not cap.isOpened():
    print(
        f"{CAMERA_INDEX}번 카메라를 열 수 없습니다. 연결 상태를 확인해주세요."
    )
    exit()

cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

# ArUco 디텍터 초기화
if not hasattr(cv2, "aruco"):
    raise RuntimeError("cv2.aruco 모듈을 찾을 수 없습니다.")

aruco_dict = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_4X4_50)

try:
    aruco_params = cv2.aruco.DetectorParameters()
    aruco_params.cornerRefinementMethod = cv2.aruco.CORNER_REFINE_SUBPIX
    detector = cv2.aruco.ArucoDetector(aruco_dict, aruco_params)
    use_new_api = True
except AttributeError:
    aruco_params = cv2.aruco.DetectorParameters_create()
    aruco_params.cornerRefinementMethod = cv2.aruco.CORNER_REFINE_SUBPIX
    use_new_api = False

# Pose Smoother 생성
smoother = PoseSmoother(window_size=5)

# 트랙바 창 생성
cv2.namedWindow("HSV Controls")

# 파란색 장애물 트랙바 초기값 설정
cv2.createTrackbar("Blue H Min", "HSV Controls", 90, 179, nothing)
cv2.createTrackbar("Blue H Max", "HSV Controls", 135, 179, nothing)
cv2.createTrackbar("Blue S Min", "HSV Controls", 70, 255, nothing)
cv2.createTrackbar("Blue V Min", "HSV Controls", 40, 255, nothing)

# 빨간색 사각형 트랙바 초기값 설정 (2번째 코드와 동일)
cv2.createTrackbar("Red H1 Max", "HSV Controls", 10, 179, nothing)
cv2.createTrackbar("Red H2 Min", "HSV Controls", 170, 179, nothing)
cv2.createTrackbar("Red S Min", "HSV Controls", 100, 255, nothing)
cv2.createTrackbar("Red V Min", "HSV Controls", 100, 255, nothing)

# 드래그 영역 설정용 창 생성
cv2.namedWindow("Drag Map Region", cv2.WINDOW_NORMAL)
cv2.resizeWindow("Drag Map Region", 1280, 720)
cv2.setMouseCallback("Drag Map Region", draw_rectangle)

# 모폴로지 연산 커널
morph_kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))

print("=" * 60)
print(" [시작 설명]")
print(" 1. 카메라 화면에서 맵 전체 영역을 마우스로 드래그하세요.")
print(
    " 2. 박스 지정 후 [Enter] 또는 [Space]를 눌러 스케일(cm)을 입력하세요."
)
print(
    " 3. [HSV Controls] 창의 트랙바로 벽(파란색)과 사각형(빨간색)을 정확히 튜닝하세요."
)
print(
    " 4. 'p' 키: 현재 HSV 설정값 출력 | 'r' 키: 맵 영역 재선택 | 'q' 키: 종료"
)
print("=" * 60)


# ==========================================
# 5. 실시간 메인 루프
# ==========================================
while True:
    ret, frame = cap.read()
    if not ret:
        print("카메라 프레임을 불러올 수 없습니다.")
        break

    # ----------------------------------------------------
    # [1단계] 영역 드래그
    # ----------------------------------------------------
    if matrix is None:
        display_frame = frame.copy()

        if ix != -1 and fx != -1:
            cv2.rectangle(display_frame, (ix, iy), (fx, fy), (0, 255, 0), 2)

        cv2.putText(
            display_frame,
            "Drag map area & Press [Enter]",
            (20, 40),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (0, 255, 255),
            2,
            cv2.LINE_AA,
        )

        cv2.imshow("Drag Map Region", display_frame)

    # ----------------------------------------------------
    # [2단계] Top-View 변환 및 처리
    # ----------------------------------------------------
    else:
        warped_map = cv2.warpPerspective(
            frame, matrix, (MAP_WIDTH, MAP_HEIGHT)
        )
        display_img = warped_map.copy()

        hsv_map = cv2.cvtColor(warped_map, cv2.COLOR_BGR2HSV)
        latest_hsv_map = hsv_map.copy()

        # --- [A] 파란색 마스크 ---
        blue_h_min = cv2.getTrackbarPos("Blue H Min", "HSV Controls")
        blue_h_max = cv2.getTrackbarPos("Blue H Max", "HSV Controls")
        blue_s_min = cv2.getTrackbarPos("Blue S Min", "HSV Controls")
        blue_v_min = cv2.getTrackbarPos("Blue V Min", "HSV Controls")
        blue_h_min = min(blue_h_min, blue_h_max)

        lower_blue = np.array(
            [blue_h_min, blue_s_min, blue_v_min], dtype=np.uint8
        )
        upper_blue = np.array([blue_h_max, 255, 255], dtype=np.uint8)

        blue_mask = cv2.inRange(hsv_map, lower_blue, upper_blue)
        blue_mask = cv2.morphologyEx(
            blue_mask, cv2.MORPH_OPEN, morph_kernel, iterations=1
        )
        blue_mask = cv2.morphologyEx(
            blue_mask, cv2.MORPH_CLOSE, morph_kernel, iterations=2
        )

        # --- [B] 빨간색 마스크 (2번째 코드 조건 적용) ---
        red_h1_max = cv2.getTrackbarPos("Red H1 Max", "HSV Controls")
        red_h2_min = cv2.getTrackbarPos("Red H2 Min", "HSV Controls")
        red_s_min = cv2.getTrackbarPos("Red S Min", "HSV Controls")
        red_v_min = cv2.getTrackbarPos("Red V Min", "HSV Controls")

        lower_red1 = np.array([0, red_s_min, red_v_min], dtype=np.uint8)
        upper_red1 = np.array([red_h1_max, 255, 255], dtype=np.uint8)

        lower_red2 = np.array([red_h2_min, red_s_min, red_v_min], dtype=np.uint8)
        upper_red2 = np.array([179, 255, 255], dtype=np.uint8)

        mask1 = cv2.inRange(hsv_map, lower_red1, upper_red1)
        mask2 = cv2.inRange(hsv_map, lower_red2, upper_red2)
        red_mask = cv2.bitwise_or(mask1, mask2)

        # 노이즈 다듬기
        red_mask = cv2.morphologyEx(
            red_mask, cv2.MORPH_OPEN, morph_kernel, iterations=1
        )
        red_mask = cv2.morphologyEx(
            red_mask, cv2.MORPH_CLOSE, morph_kernel, iterations=2
        )

        # 장애물 오버레이
        overlay = display_img.copy()
        overlay[blue_mask > 0] = (0, 0, 255)
        display_img = cv2.addWeighted(display_img, 0.7, overlay, 0.3, 0)

        # ----------------------------------------------------
        # 빨간색 사각형 검출 및 스케일 업데이트
        # ----------------------------------------------------
        detected_px, cnt = detect_reference_square(warped_map, red_mask)
        if detected_px > 0:
            square_px = detected_px
            if real_cm > 0:
                cm_scale = real_cm / square_px

            cv2.drawContours(display_img, [cnt], -1, (0, 255, 0), 2)
            cv2.putText(
                display_img,
                f"Red Ref Sq: {square_px:.1f}px",
                (cnt[0][0][0], max(15, cnt[0][0][1] - 10)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.45,
                (0, 255, 0),
                1,
            )

        # ----------------------------------------------------
        # ArUco 마커 검출 + PoseSmoother
        # ----------------------------------------------------
        gray = cv2.cvtColor(warped_map, cv2.COLOR_BGR2GRAY)
        if use_new_api:
            corners, ids, _ = detector.detectMarkers(gray)
        else:
            corners, ids, _ = cv2.aruco.detectMarkers(
                gray, aruco_dict, parameters=aruco_params
            )

        robot_screen_pt = None
        if ids is not None and len(corners) > 0:
            marker_corners = corners[0][0]

            raw_cx = int(np.mean(marker_corners[:, 0]))
            raw_cy = int(np.mean(marker_corners[:, 1]))

            p0, p1 = marker_corners[0], marker_corners[1]
            dx, dy = p1[0] - p0[0], p1[1] - p0[1]
            angle_rad = np.arctan2(-dy, dx)
            raw_angle_deg = np.degrees(angle_rad) % 360

            # 스무딩 업데이트
            robot_screen_pt, robot_angle_deg = smoother.update(
                raw_cx, raw_cy, raw_angle_deg
            )

            center_x, center_y = robot_screen_pt
            cart_rx = center_x
            cart_ry = MAP_HEIGHT - center_y

            # 로봇 위치 및 각도 화살표 표시
            cv2.circle(display_img, robot_screen_pt, 8, (255, 0, 0), -1)
            arrow_len = 35
            arrow_end_x = int(
                center_x + arrow_len * np.cos(np.radians(robot_angle_deg))
            )
            arrow_end_y = int(
                center_y - arrow_len * np.sin(np.radians(robot_angle_deg))
            )
            cv2.arrowedLine(
                display_img,
                robot_screen_pt,
                (arrow_end_x, arrow_end_y),
                (255, 0, 0),
                2,
                tipLength=0.3,
            )

            info_text = (
                f"ROBOT: ({cart_rx}, {cart_ry}) | {robot_angle_deg:.1f}deg"
            )
            cv2.putText(
                display_img,
                info_text,
                (center_x - 70, center_y - 20),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.45,
                (255, 0, 0),
                1,
                cv2.LINE_AA,
            )

        # ----------------------------------------------------
        # UI 및 Goal 표시
        # ----------------------------------------------------
        cartesian_x = mouse_x
        cartesian_y = MAP_HEIGHT - mouse_y

        cv2.drawMarker(
            display_img,
            (mouse_x, mouse_y),
            (0, 0, 255),
            markerType=cv2.MARKER_CROSS,
            markerSize=15,
            thickness=1,
        )

        text_x = mouse_x + 15 if mouse_x + 160 < MAP_WIDTH else mouse_x - 160
        text_y = mouse_y - 10 if mouse_y - 20 > 0 else mouse_y + 20
        cv2.putText(
            display_img,
            f"({cartesian_x}, {cartesian_y}) px",
            (text_x, text_y),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            (0, 255, 255),
            1,
            cv2.LINE_AA,
        )

        scale_info = (
            f"Scale: {cm_scale:.4f} cm/px" if cm_scale > 0 else "Scale: N/A"
        )
        cv2.putText(
            display_img,
            scale_info,
            (20, 30),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (255, 255, 0),
            2,
            cv2.LINE_AA,
        )

        if goal_pt is not None:
            cv2.circle(display_img, goal_pt, 8, (0, 0, 255), -1)
            cv2.putText(
                display_img,
                "GOAL",
                (goal_pt[0] + 10, goal_pt[1] - 5),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                (0, 0, 255),
                1,
                cv2.LINE_AA,
            )

            if robot_screen_pt is not None:
                cv2.line(display_img, robot_screen_pt, goal_pt, (0, 255, 0), 2)
                dist_px = np.sqrt(
                    (goal_pt[0] - robot_screen_pt[0]) ** 2
                    + (goal_pt[1] - robot_screen_pt[1]) ** 2
                )
                mid_x = (robot_screen_pt[0] + goal_pt[0]) // 2
                mid_y = (robot_screen_pt[1] + goal_pt[1]) // 2

                dist_text = (
                    f"Dist: {dist_px * cm_scale:.1f}cm"
                    if cm_scale > 0
                    else f"Dist: {dist_px:.1f}px"
                )
                cv2.putText(
                    display_img,
                    dist_text,
                    (mid_x, mid_y - 10),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.5,
                    (0, 255, 0),
                    2,
                    cv2.LINE_AA,
                )

        cv2.imshow("Warped Top-View Map", display_img)
        cv2.imshow("Blue Maze Mask", blue_mask)
        cv2.imshow("Red Square Mask", red_mask)

    # ----------------------------------------------------
    # 키보드 입력 이벤트
    # ----------------------------------------------------
    key = cv2.waitKey(20) & 0xFF

    if key in [13, 32] and bbox_selected and matrix is None:
        x1, x2 = min(ix, fx), max(ix, fx)
        y1, y2 = min(iy, fy), max(iy, fy)

        box_w = x2 - x1
        box_h = y2 - y1

        if box_w > 10 and box_h > 10:
            MAP_HEIGHT = int(MAP_WIDTH * (box_h / box_w))

            src_pts = np.float32([[x1, y1], [x2, y1], [x2, y2], [x1, y2]])
            dst_pts = np.float32(
                [[0, 0], [MAP_WIDTH, 0], [MAP_WIDTH, MAP_HEIGHT], [0, MAP_HEIGHT]]
            )

            matrix = cv2.getPerspectiveTransform(src_pts, dst_pts)

            print("\n" + "=" * 50)
            try:
                user_input = input(
                    "기준 빨간색 사각형의 실제 한 변 길이(cm)를 입력하세요: "
                )
                real_cm = float(user_input)
                print(f"-> 설정 완료: {real_cm} cm")
            except ValueError:
                print("-> 잘못된 입력입니다. 픽셀 단위로 구동합니다.")
                real_cm = 0.0
            print("=" * 50 + "\n")

            cv2.destroyWindow("Drag Map Region")
            cv2.namedWindow("Warped Top-View Map", cv2.WINDOW_NORMAL)
            cv2.resizeWindow(
                "Warped Top-View Map",
                1000,
                int(1000 * (MAP_HEIGHT / MAP_WIDTH)),
            )
            cv2.setMouseCallback("Warped Top-View Map", track_coordinates)

    elif key == ord("p"):
        print("\n===== [현재 파란색 HSV 범위] =====")
        print(
            f"H: {blue_h_min} ~ {blue_h_max} | S: {blue_s_min} | V: {blue_v_min}"
        )
        print("===== [현재 빨간색 HSV 범위] =====")
        print(
            f"H1: 0 ~ {red_h1_max} | H2: {red_h2_min} ~ 179 | S: {red_s_min} | V:"
            f" {red_v_min}"
        )
        print("==================================\n")

    elif key == ord("r"):
        matrix = None
        bbox_selected = False
        ix, iy, fx, fy = -1, -1, -1, -1
        goal_pt = None
        real_cm, square_px, cm_scale = 0.0, 0.0, 0.0
        smoother.reset()
        cv2.destroyAllWindows()

        cv2.namedWindow("HSV Controls")
        cv2.createTrackbar("Blue H Min", "HSV Controls", 90, 179, nothing)
        cv2.createTrackbar("Blue H Max", "HSV Controls", 135, 179, nothing)
        cv2.createTrackbar("Blue S Min", "HSV Controls", 70, 255, nothing)
        cv2.createTrackbar("Blue V Min", "HSV Controls", 40, 255, nothing)

        cv2.createTrackbar("Red H1 Max", "HSV Controls", 10, 179, nothing)
        cv2.createTrackbar("Red H2 Min", "HSV Controls", 170, 179, nothing)
        cv2.createTrackbar("Red S Min", "HSV Controls", 100, 255, nothing)
        cv2.createTrackbar("Red V Min", "HSV Controls", 100, 255, nothing)

        cv2.namedWindow("Drag Map Region", cv2.WINDOW_NORMAL)
        cv2.resizeWindow("Drag Map Region", 1280, 720)
        cv2.setMouseCallback("Drag Map Region", draw_rectangle)

    elif key == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()