import cv2
import numpy as np
	
# 1. 이미지 로드
image_path = "C:\\Users\\jeong\\OneDrive\\Desktop\\erangel.png"  # 이미지 경로	
img = cv2.imread(image_path)

if img is None:
    print("이미지를 불러올 수 없습니다. 경로를 확인해주세요.")
    exit()

img_copy = img.copy()

# 드래그 전역 변수
drawing = False      # 마우스 드래그 중인지 여부
ix, iy = -1, -1      # 드래그 시작 좌표 (좌상단)
fx, fy = -1, -1      # 드래그 종료 좌표 (우하단)
bbox_selected = False

# 변환된 맵의 마우스 실시간 좌표 저장 변수 (OpenCV 픽셀 원본)
mouse_x, mouse_y = 0, 0

def draw_rectangle(event, x, y, flags, param):
    """ 1단계: 드래그 영역 선택 콜백 함수 """
    global ix, iy, fx, fy, drawing, img_copy, bbox_selected

    # 1. 마우스 누름 -> 드래그 시작
    if event == cv2.EVENT_LBUTTONDOWN:
        drawing = True
        ix, iy = x, y
        bbox_selected = False

    # 2. 마우스 이동 -> 드래그 실시간 시각화
    elif event == cv2.EVENT_MOUSEMOVE:
        if drawing:
            img_copy = img.copy()
            cv2.rectangle(img_copy, (ix, iy), (x, y), (0, 255, 0), 1)
            cv2.imshow("Drag Map Region", img_copy)

    # 3. 마우스 뗌 -> 드래그 완료
    elif event == cv2.EVENT_LBUTTONUP:
        drawing = False
        fx, fy = x, y
        bbox_selected = True
        
        img_copy = img.copy()
        cv2.rectangle(img_copy, (ix, iy), (fx, fy), (0, 255, 0), 1)
        cv2.imshow("Drag Map Region", img_copy)
        
        print(f"\n[선택한 사각형 영역]: 시작({ix}, {iy}) -> 끝({fx}, {fy})")
        print("-> 키보드 [Enter] 또는 [Space]를 누르면 원근 변환을 수행합니다.")
x
def track_coordinates(event, x, y, flags, param):
    """ 2단계: 펴진 맵에서 마우스 실시간 좌표 위치 업데이트 콜백 함수 """
    global mouse_x, mouse_y
    if event == cv2.EVENT_MOUSEMOVE:
        mouse_x, mouse_y = x, y

# ==========================================
# 창 설정 및 메인 실행 (1단계: 영역 드래그)
# ==========================================
cv2.namedWindow("Drag Map Region", cv2.WINDOW_NORMAL)
cv2.resizeWindow("Drag Map Region", 2592, 1944) # 창 크기 조절

cv2.setMouseCallback("Drag Map Region", draw_rectangle)

print("=" * 60)
print(" 사용 방법:")
print(" 1. 맵 영역의 [좌상단]에서 마우스 클릭 후 [우하단]으로 드래그하세요.")
print(" 2. 사각형이 잘 잡혔으면 키보드 [Enter]를 누르세요.")
print(" 3. 다시 그리고 싶으면 마우스로 다시 드래그하시면 됩니다.")
print("=" * 60)

cv2.imshow("Drag Map Region", img_copy)
cv2.waitKey(0)

# ==========================================
# 변환 수행 및 실시간 좌표 UI 출력 (2단계)
# ==========================================
if bbox_selected:
    x1, x2 = min(ix, fx), max(ix, fx)
    y1, y2 = min(iy, fy), max(iy, fy)

    box_w = x2 - x1
    box_h = y2 - y1

    MAP_WIDTH = 1000
    MAP_HEIGHT = int(1000 * (box_h / box_w)) if box_w > 0 else 1000

    src_pts = np.float32([
        [x1, y1],
        [x2, y1],
        [x2, y2],
        [x1, y2]
    ])

    dst_pts = np.float32([
        [0, 0],
        [MAP_WIDTH, 0],
        [MAP_WIDTH, MAP_HEIGHT],
        [0, MAP_HEIGHT]
    ])

    matrix = cv2.getPerspectiveTransform(src_pts, dst_pts)
    warped_map = cv2.warpPerspective(img, matrix, (MAP_WIDTH, MAP_HEIGHT))

    # 기존 드래그 창 닫고 새 결과 창 생성
    cv2.destroyAllWindows()

    cv2.namedWindow("Warped Top-View Map", cv2.WINDOW_NORMAL)
    cv2.resizeWindow("Warped Top-View Map", 1000, int(1000 * (MAP_HEIGHT / MAP_WIDTH)))
    
    # 펴진 맵 창에 좌표 추적 마우스 콜백 등록
    cv2.setMouseCallback("Warped Top-View Map", track_coordinates)

    print(f"\n[변환 완료] 맵 크기: {MAP_WIDTH} x {MAP_HEIGHT} px")
    print("-> 좌하단 원점(0, 0) 기준 좌표 표시 중...")
    print("-> 종료하려면 'q' 키를 누르세요.")

    # 실시간 좌표 출력을 위한 메인 루프
    while True:
        display_img = warped_map.copy()

        # [핵심] 좌하단을 (0,0) 원점으로 변환하는 수학 계산
        cartesian_x = mouse_x
        cartesian_y = MAP_HEIGHT - mouse_y  # Y축 뒤집기

        # 1. 마우스 위치에 빨간색 십자선 그리기 (그릴 때는 OpenCV 화면 좌표 사용)
        cv2.drawMarker(display_img, (mouse_x, mouse_y), (0, 0, 255),
                       markerType=cv2.MARKER_CROSS, markerSize=15, thickness=1)

        # 2. 좌표 텍스트 위치 조절 (화면 밖으로 나가지 않도록)
        text_x = mouse_x + 15 if mouse_x + 160 < MAP_WIDTH else mouse_x - 160
        text_y = mouse_y - 10 if mouse_y - 20 > 0 else mouse_y + 20

        # 3. 변환된 (X, Y) 직교 좌표 표시
        cv2.putText(display_img, f"({cartesian_x}, {cartesian_y}) px", (text_x, text_y),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 1, cv2.LINE_AA)

        cv2.imshow("Warped Top-View Map", display_img)

        # 'q' 키를 누르거나 키 입력을 받으면 루프 종료
        if cv2.waitKey(20) & 0xFF == ord('q'):
            break

    cv2.imwrite("warped_map.jpg", warped_map)
    cv2.destroyAllWindows()
else:  
    print("영역이 선택되지 않았습니다.")
    cv2.destroyAllWindows()
