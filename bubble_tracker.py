import cv2
import os
import numpy as np
import csv
import sys
import tkinter as tk
from tkinter import filedialog


# ==========================================
# 辅助函数：绘制多边形 ROI
# ==========================================
def get_polygon_roi(frame, window_name):
    points = []
    display_frame = frame.copy()

    def mouse_callback(event, x, y, flags, param):
        if event == cv2.EVENT_LBUTTONDOWN:
            points.append((x, y))
            cv2.circle(display_frame, (x, y), 3, (0, 0, 255), -1)
            if len(points) > 1:
                cv2.line(display_frame, points[-2], points[-1], (0, 255, 0), 2)
            cv2.imshow(window_name, display_frame)

    cv2.namedWindow(window_name)
    cv2.setMouseCallback(window_name, mouse_callback)
    cv2.imshow(window_name, display_frame)

    print(f"\n---> 请在弹出的 '{window_name}' 窗口中用鼠标【左键】画框。")
    print("绘制完成后，按【ENTER】或【SPACE】键确认。")

    while True:
        key = cv2.waitKey(10) & 0xFF
        if key in [13, 32]: break
    cv2.destroyWindow(window_name)
    return np.array(points, dtype=np.int32)


# ==========================================
# 主程序
# ==========================================
# 1. 自由选择视频文件
root = tk.Tk()
root.withdraw()
video_path = filedialog.askopenfilename(
    title="请选择微流控视频文件",
    filetypes=[("视频文件", "*.avi *.mp4 *.mkv"), ("所有文件", "*.*")]
)

if not video_path:
    print("未选择视频文件，程序退出。")
    sys.exit()

print(f"已加载视频: {video_path}")

cap = cv2.VideoCapture(video_path)
fps = cap.get(cv2.CAP_PROP_FPS)
total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
total_seconds = int(total_frames / fps)

ret, first_frame = cap.read()
if not ret:
    print("无法读取视频内容。")
    sys.exit()

# 2. 绘制检测区域
pts_left = get_polygon_roi(first_frame, "Draw Left Channel")
pts_right = get_polygon_roi(first_frame, "Draw Right Channel")

mask_left_poly = np.zeros(first_frame.shape[:2], dtype=np.uint8)
mask_right_poly = np.zeros(first_frame.shape[:2], dtype=np.uint8)
cv2.fillPoly(mask_left_poly, [pts_left], 255)
cv2.fillPoly(mask_right_poly, [pts_right], 255)

# 3. 初始化算法与变量
backSub = cv2.createBackgroundSubtractorMOG2(history=100, varThreshold=50, detectShadows=False)
kernel_close = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (21, 21))

left_results, right_results = [], []
left_occupied, left_temp_max_area, left_start_time = False, 0, 0
right_occupied, right_temp_max_area, right_start_time = False, 0, 0

area_threshold = 2000

# 🌟 新增：预热相关变量
warm_up_frames = int(fps * 0.5)  # 0.5秒对应的帧数
current_frame_count = 0  # 当前处理的帧计数

# 控制面板设置
cv2.namedWindow('Monitoring (ESC to Stop)')


def on_time_trackbar(val):
    global backSub, current_frame_count  # 声明全局变量，以便重置预热
    target_frame = int(val * fps)
    current_pos = int(cap.get(cv2.CAP_PROP_POS_FRAMES))
    if abs(target_frame - current_pos) > fps:
        cap.set(cv2.CAP_PROP_POS_FRAMES, target_frame)
        # 清空背景消除器
        backSub = cv2.createBackgroundSubtractorMOG2(history=100, varThreshold=50, detectShadows=False)
        # 🌟 拖动进度条后，将帧计数清零，强制重新预热 0.5 秒
        current_frame_count = 0
        print(f"\n[提示] 时间已跳转至 {val} 秒附近，正在重新预热背景...")


cv2.createTrackbar('Time(s)', 'Monitoring (ESC to Stop)', 0, total_seconds, on_time_trackbar)
default_delay = int((1000 / fps) * 10)
cv2.createTrackbar('Delay(ms)', 'Monitoring (ESC to Stop)', default_delay, 200, lambda x: None)

print("\n" + "=" * 40)
print("开始分析视频... (按 'ESC' 键可提前保存并退出)")
print("控制台将实时输出记录到的气泡数据：")
print("=" * 40 + "\n")

while True:
    ret, frame = cap.read()
    if not ret: break

    current_frame_count += 1
    current_time = cap.get(cv2.CAP_PROP_POS_MSEC) / 1000.0
    cv2.setTrackbarPos('Time(s)', 'Monitoring (ESC to Stop)', int(current_time))

    blurred = cv2.GaussianBlur(frame, (5, 5), 0)
    # 每一帧都必须经过 backSub，算法才能学习
    fgMask = backSub.apply(blurred)

    # ==========================================
    # 🌟 预热拦截逻辑
    # ==========================================
    if current_frame_count < warm_up_frames:
        # 画出基础边界框和时间
        cv2.polylines(frame, [pts_left], True, (0, 255, 0), 2)
        cv2.polylines(frame, [pts_right], True, (0, 0, 255), 2)
        cv2.putText(frame, f"Time: {current_time:.3f} s", (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)

        # 屏幕上打出预热提示字样 (黄色)
        cv2.putText(frame, "Learning Background...", (20, 80), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2)

        cv2.imshow('Monitoring (ESC to Stop)', frame)
        cv2.imshow('Detection Mask (Internal View)', fgMask)

        current_delay = cv2.getTrackbarPos('Delay(ms)', 'Monitoring (ESC to Stop)')
        if cv2.waitKey(max(1, current_delay)) & 0xFF == 27: break

        # 拦截：跳过本帧的后续气泡检测，进入下一帧
        continue

    # 预热结束，开始正常的二值化和形态学处理
    _, thresh = cv2.threshold(fgMask, 200, 255, cv2.THRESH_BINARY)
    thresh = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel_close)

    roi_left = cv2.bitwise_and(thresh, thresh, mask=mask_left_poly)
    roi_right = cv2.bitwise_and(thresh, thresh, mask=mask_right_poly)

    # ==========================================
    # 左侧处理与实时打印
    # ==========================================
    contours_l, _ = cv2.findContours(roi_left, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    current_area_l = 0
    for cnt in contours_l:
        hull = cv2.convexHull(cnt)
        area = cv2.contourArea(hull)
        current_area_l += area
        if area > 50: cv2.drawContours(frame, [hull], -1, (0, 255, 255), 2)

    if current_area_l > area_threshold:
        if not left_occupied:
            left_start_time = current_time
            left_temp_max_area = current_area_l
            left_occupied = True
        else:
            left_temp_max_area = max(left_temp_max_area, current_area_l)
    else:
        if left_occupied:
            left_results.append((left_start_time, left_temp_max_area))
            print(f"✅ [实时记录] 时间点: {left_start_time:.3f} s | 位置: 左边 | 面积数值: {int(left_temp_max_area)}")
            left_occupied = False
            left_temp_max_area = 0

    # ==========================================
    # 右侧处理与实时打印
    # ==========================================
    contours_r, _ = cv2.findContours(roi_right, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    current_area_r = 0
    for cnt in contours_r:
        hull = cv2.convexHull(cnt)
        area = cv2.contourArea(hull)
        current_area_r += area
        if area > 50: cv2.drawContours(frame, [hull], -1, (255, 255, 0), 2)

    if current_area_r > area_threshold:
        if not right_occupied:
            right_start_time = current_time
            right_temp_max_area = current_area_r
            right_occupied = True
        else:
            right_temp_max_area = max(right_temp_max_area, current_area_r)
    else:
        if right_occupied:
            right_results.append((right_start_time, right_temp_max_area))
            print(f"✅ [实时记录] 时间点: {right_start_time:.3f} s | 位置: 右边 | 面积数值: {int(right_temp_max_area)}")
            right_occupied = False
            right_temp_max_area = 0

    # ==========================================
    # 可视化展示
    # ==========================================
    cv2.polylines(frame, [pts_left], True, (0, 255, 0), 2)
    cv2.polylines(frame, [pts_right], True, (0, 0, 255), 2)
    cv2.putText(frame, f"Time: {current_time:.3f} s", (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
    frame_w = frame.shape[1]
    cv2.putText(frame, f"L Area: {int(current_area_l)}", (frame_w - 280, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.8,
                (0, 255, 0), 2)
    cv2.putText(frame, f"R Area: {int(current_area_r)}", (frame_w - 280, 80), cv2.FONT_HERSHEY_SIMPLEX, 0.8,
                (0, 0, 255), 2)

    cv2.imshow('Monitoring (ESC to Stop)', frame)
    combined_fg = cv2.bitwise_or(roi_left, roi_right)
    cv2.imshow('Detection Mask (Internal View)', combined_fg)

    current_delay = cv2.getTrackbarPos('Delay(ms)', 'Monitoring (ESC to Stop)')
    if cv2.waitKey(max(1, current_delay)) & 0xFF == 27: break

cap.release()
cv2.destroyAllWindows()

# ==========================================
# 4. 统计输出与导出
# ==========================================
print("\n" + "=" * 40)
print("📊 最终统计分析报告")
print("=" * 40)
print(f"🔹 左侧管道 (Left) 气泡总数: {len(left_results)}")
print(f"🔸 右侧管道 (Right) 气泡总数: {len(right_results)}")
print(f"💡 合计检测总次数: {len(left_results) + len(right_results)}")
print("=" * 40)

video_dir = os.path.dirname(video_path)
video_name_stem = os.path.splitext(os.path.basename(video_path))[0]
csv_filename = os.path.join(video_dir, f"bubble_analysis_{video_name_stem}.csv")

with open(csv_filename, mode='w', newline='') as file:
    writer = csv.writer(file)
    writer.writerow(["Time_Seconds", "Channel", "Max_Area_Pixels"])
    all_events = []
    for t, a in left_results: all_events.append([t, "Left", int(a)])
    for t, a in right_results: all_events.append([t, "Right", int(a)])
    all_events.sort(key=lambda x: x[0])
    for row in all_events: writer.writerow(row)

print(f"\n📂 详细数据已自动导出至: {csv_filename}")
print("\n按回车键退出程序...")
input()
