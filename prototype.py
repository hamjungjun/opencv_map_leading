from collections import deque
import math
import socket
import cv2
import numpy as np

# ==========================================
# 0. 네트워크(UDP) & PID 제어 설정
# ==========================================
# 라즈베리파이의 Wi-Fi IP 주소를 입력하세요.
RASPBERRY_PI_IP = ""
UDP_PORT = 8080

# UDP 소켓 생성
udp_sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)

# PID 게인 설정 (실제 로봇 주행 환경에 맞춰 튜닝이 필요합니다)
Kp_angle = 2.0  # 각도 비례 게인
Kd_angle = 0.1  # 각도 미분 게인
Kp_dist = 0.8  # 거리 비례 게인
Kd_dist = 0.05  # 거리 미분 게인

prev_angle_error = 0.0
prev_dist_error = 0.0


def normalize_angle(angle_rad):
    """각도 오차를 -pi ~ pi 범위로 정규화"""
    while angle_rad > math.pi:
        angle_rad -= 2 * math.pi
    while angle_rad < -math.pi:
        angle_rad += 2 * math.pi
    return angle_rad


def send_motor_command(v_left, v_right):
    """라즈베리파이로 속도 명령 전송 (-255 ~ 255)"""
    v_l = int(np.clip(v_left, -255, 255))
    v_r = int(np.clip(v_right, -255, 255))
    msg = f"{v_l},{v_r}".encode("utf-8")
    try:
        udp_sock.sendto(msg, (RASPBERRY_PI_IP, UDP_PORT))
    except Exception as e:
        print(f"[UDP 전송 오류]: {e}")


def compute_pid_control(robot_pt, robot_angle_deg, goal_pt, dt=0.05):
    """로봇 위치/각도와 목표 지점을 기반으로 모터 속도(v_l, v_r) 계산"""
    global prev_angle_error, prev_dist_error

    if robot_pt is None or goal_pt is None:
        return 0, 0

    rx, ry = robot_pt       #로봇의 2차원 좌표
    gx, gy = goal_pt        #목표지점 2차원 좌표

    dx = gx - rx
    dy = gy - ry

    distance_px = math.sqrt(dx**2 + dy**2)

    # Goal에 매우 가까워지면 정지 (도착 기준: 15px 이내)
    if distance_px < 15:
        return 0, 0

    # 목표 각도 계산 (기존 arctan2(-dy, dx) 방식 유지)
    target_angle_deg = math.degrees(math.atan2(-dy, dx)) % 360.0

    # 각도 오차 계산
    angle_error_rad = math.radians(target_angle_deg - robot_angle_deg)
    angle_error_rad = normalize_angle(angle_error_rad)

    # P-D 제어 계산
    angle_deriv = (angle_error_rad - prev_angle_error) / dt
    w = (Kp_angle * angle_error_rad) + (Kd_angle * angle_deriv)
    prev_angle_error = angle_error_rad

    dist_deriv = (distance_px - prev_dist_error) / dt
    v = (Kp_dist * distance_px) + (Kd_dist * dist_deriv)
    prev_dist_error = distance_px

    # 각도 오차가 25도 이상이면 제자리 회전 우선
    if abs(math.degrees(angle_error_rad)) > 25:
        v = 0

    # 차동 구동 모터 속도 분배
    v_left = v - (w * 50)
    v_right = v + (w * 50)

    return v_left, v_right


# ==========================================
# 1. 포즈 스무더 (Pose Smoother) 클래스
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
# 2. 콜백용 빈 함수 & 전역 변수
# ==========================================
def nothing(value):
    pass


latest_hsv_map = None

drawing = False
ix, iy = -1, -1
fx, fy = -1, -1
bbox_selected = False

matrix = None
MAP_WIDTH = 1000
MAP_HEIGHT = 1000
mouse_x, mouse_y = 0, 0
goal_pt = None

square_px = 0.0
real_cm = 0.0
cm_scale = 0.0


# ==========================================
# 3. 마우스 콜백 함수들
# ==========================================
def draw_rectangle(event, x, y, flags, param):
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
        print(
            "-> [Enter] 키를 누르면 스케일 입력 및 원근 변환을 진행합니다."
        )


def track_coordinates(event, x, y, flags, param):
    global mouse_x, mouse_y, goal_pt, latest_hsv_map

    if event == cv2.EVENT_MOUSEMOVE:
        mouse_x, mouse_y = x, y

    elif event == cv2.EVENT_LBUTTONDOWN:
        goal_pt = (x, y)
        cart_gx = x
        cart_gy = MAP_HEIGHT - y
        print(
            f"[Goal 지정]: 화면 ({x}, {y}) | 카테시안 ({cart_gx}, {cart_gy})"
            " px"
        )

        if latest_hsv_map is not None:
            h, w = latest_hsv_map.shape[:2]
            if 0 <= x < w and 0 <= y < h:
                hv, sv, vv = latest_hsv_map[y, x]
                print(
                    f" -> [클릭 지점 HSV]: H={int(hv)}, S={int(sv)},"
                    f" V={int(vv)}"
                )


# ==========================================
# 4. 빨간색 기준 사각형 검출 함수
# ==========================================
def detect_reference_square(warped_img, red_mask):
    contours, _ = cv2.findContours(
        red_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
    )

    best_square_px = 0.0
    best_cnt = None

    for cnt in contours:
        area = cv2.contourArea(cnt)
        if area < 200:
            continue

        peri = cv2.arcLength(cnt, True)
        approx = cv2.approxPolyDP(cnt, 0.035 * peri, True)

        if 4 <= len(approx) <= 6:
            rect = cv2.minAreaRect(cnt)
            (w_rect, h_rect) = rect[1]

            if w_rect == 0 or h_rect == 0:
                continue

            aspect_ratio = float(w_rect) / h_rect
            if 0.7 <= aspect_ratio <= 1.4:
                avg_side = (w_rect + h_rect) / 2.0
                if avg_side > best_square_px:
                    best_square_px = avg_side
                    best_cnt = cnt

    return best_square_px, best_cnt


# ==========================================
# 5. 카메라 및 ArUco, UI 창 초기화
# ==========================================
CAMERA_INDEX = 1

cap = cv2.VideoCapture(CAMERA_INDEX)
if not cap.isOpened():
    print(
        f"{CAMERA_INDEX}번 카메라를 열 수 없습니다. 연결 상태를 확인해주세요."
    )
    exit()

cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

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

smoother = PoseSmoother(window_size=5)

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

morph_kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))

print("=" * 60)
print(" [시작 설명]")
print(" 1. 카메라 화면에서 맵 전체 영역을 마우스로 드래그하세요.")
print(
    " 2. 박스 지정 후 [Enter] 또는 [Space]를 눌러 스케일(cm)을 입력하세요."
)
print(
    " 3. [HSV Controls] 창의 트랙바로 벽(파란색)과 사각형(빨간색)을 정확히"
    " 튜닝하세요."
)
print(
    " 4. 'p' 키: 현재 HSV 설정값 출력 | 'r' 키: 맵 영역 재선택 | 'q' 키: 종료"
)
print("=" * 60)

# ==========================================
# 6. 실시간 메인 루프
# ==========================================
while True:
    ret, frame = cap.read()
    if not ret:
        print("카메라 프레임을 불러올 수 없습니다.")
        break

    # [1단계] 영역 드래그
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

    # [2단계] Top-View 변환 및 처리
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

        # --- [B] 빨간색 마스크 ---
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

        # 빨간색 사각형 검출 및 스케일 업데이트
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

        # --- [C] ArUco 마커 검출 ---
        gray = cv2.cvtColor(warped_map, cv2.COLOR_BGR2GRAY)
        if use_new_api:
            corners, ids, _ = detector.detectMarkers(gray)
        else:
            corners, ids, _ = cv2.aruco.detectMarkers(
                gray, aruco_dict, parameters=aruco_params
            )

        robot_screen_pt = None
        robot_angle_deg = 0.0

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

        # --- [D] PID 제어 계산 및 라즈베리파이 명령 전송 ---
        if goal_pt is not None and robot_screen_pt is not None:
            v_l, v_r = compute_pid_control(
                robot_screen_pt, robot_angle_deg, goal_pt
            )
            send_motor_command(v_l, v_r)
        else:
            # 목표가 설정되지 않았거나 로봇을 놓치면 정지 명령 전송
            send_motor_command(0, 0)

        # --- [E] UI 및 Goal 표시 ---
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
                [
                    [0, 0],
                    [MAP_WIDTH, 0],
                    [MAP_WIDTH, MAP_HEIGHT],
                    [0, MAP_HEIGHT],
                ]
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
            f"H: {blue_h_min} ~ {blue_h_max} | S: {blue_s_min} | V:"
            f" {blue_v_min}"
        )
        print("===== [현재 빨간색 HSV 범위] =====")
        print(
            f"H1: 0 ~ {red_h1_max} | H2: {red_h2_min} ~ 179 | S: {red_s_min} |"
            f" V: {red_v_min}"
        )
        print("==================================\n")

    elif key == ord("r"):
        send_motor_command(0, 0)  # 리셋 시 모터 정지 명령
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
        send_motor_command(0, 0)  # 종료 시 모터 정지 명령
        break

cap.release()
cv2.destroyAllWindows()
