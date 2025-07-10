# yolo_frame_click_tool.py

import os
import os.path as osp
import argparse
import json
import cv2
import imageio
import threading
from ultralytics import YOLO
from tqdm import tqdm

# === グローバル変数 ===
clicked_point = None
latest_boxes = []
latest_classes = []
click_count = 1
stop_requested = False
wq_requested = False
CLASS_NAMES = []
FIXED_CLASS_ID = 0
BASE_DIR = "output_dataset"
CLICKED_OUTPUT_DIR = "output_clicked"
PROGRESS_FILE = "progress.json"

# === フレーム抽出が必要か判定 ===
def should_extract_frames(frame_dir):
    return not (
        os.path.exists(frame_dir) and 
        any(f.endswith(".jpg") for f in os.listdir(frame_dir))
    )

# === FPS選択 ===
def choose_frame_rate():
    print("\nフレームレートを選択してください:")
    print("1. 60fps")
    print("2. 30fps")
    while True:
        choice = input("番号を入力（1 または 2）: ").strip()
        if choice == "1":
            return 60
        elif choice == "2":
            return 30
        else:
            print("無効な入力です。1 または 2 を入力してください。")

# === フレーム抽出 ===
def extract_frames(video_path, frame_rate, output_dir):
    os.makedirs(output_dir, exist_ok=True) 
    reader = imageio.get_reader(video_path)
    meta_data = reader.get_meta_data()
    fps = meta_data.get('fps', 0)
    if fps == 0:
        raise ValueError("[ERROR] 動画のFPSが取得できません。")

    interval = int(round(fps / frame_rate))
    if interval <= 0:
        interval = 1

    selected = []
    for i, img in tqdm(enumerate(reader), total=meta_data['nframes']):
        if i % interval == 0:
            frame_path = os.path.join(output_dir, f"{i:08d}.jpg")
            imageio.imsave(frame_path, img)
            selected.append((i, frame_path))

    if os.path.exists(PROGRESS_FILE):
        os.remove(PROGRESS_FILE)
        print("[INFO] フレーム再生成のため progress.json をリセットしました。")

# === 進捗の保存・読込 ===
def reset_progress():
    if os.path.exists(PROGRESS_FILE):
        os.remove(PROGRESS_FILE)
        print("[INFO] progress.json をリセットしました。")
        
def save_progress(frame_index):
    with open(PROGRESS_FILE, "w") as f:
        json.dump({"last_frame": frame_index, "click_count": click_count}, f)

def load_progress():
    global click_count
    if os.path.exists(PROGRESS_FILE):
        with open(PROGRESS_FILE, "r") as f:
            data = json.load(f)
            click_count = data.get("click_count", 1)
            return data.get("last_frame", 0), click_count
    return 0, 1

# === 入力監視 ===
def monitor_input():
    global stop_requested, wq_requested
    while True:
        user_input = input()
        if user_input.strip().lower() == "wq":
            wq_requested = True
            stop_requested = True
            break
        elif user_input.strip().lower() == "w":
            stop_requested = True
            break

# === マウスクリックイベント ===
def mouse_callback(event, x, y, flags, param):
    global clicked_point
    if event == cv2.EVENT_LBUTTONDOWN:
        clicked_point = (x, y)

# === メイン処理 ===
def main(video_path):
    global clicked_point, latest_boxes, latest_classes, click_count, CLASS_NAMES, stop_requested, wq_requested
    stop_requested = False
    wq_requested = False
    threading.Thread(target=monitor_input, daemon=True).start()

    os.makedirs(os.path.join(BASE_DIR, "images", "train"), exist_ok=True)
    os.makedirs(os.path.join(BASE_DIR, "images", "val"), exist_ok=True)
    os.makedirs(os.path.join(BASE_DIR, "labels", "train"), exist_ok=True)
    os.makedirs(os.path.join(BASE_DIR, "labels", "val"), exist_ok=True)
    os.makedirs(CLICKED_OUTPUT_DIR, exist_ok=True)

    model = YOLO("yolov8n.pt")
    CLASS_NAMES = model.names

    last_frame_idx, click_count = load_progress()
    frame_dir = os.path.splitext(os.path.basename(video_path))[0] + "_frames"
    frames = sorted([
        (int(os.path.splitext(f)[0]), os.path.join(frame_dir, f))
        for f in os.listdir(frame_dir) if f.endswith(".jpg")
    ])

    window_name = "YOLOv8 Click Tool"
    cv2.namedWindow(window_name)
    cv2.setMouseCallback(window_name, mouse_callback)

    total_frames = len(frames)
    current_index = 0
    for idx, (frame_idx, _) in enumerate(frames):
        if frame_idx >= last_frame_idx:
            current_index = idx
            break

    while current_index < total_frames:
        if stop_requested:
            if wq_requested:
                reset_progress()
            else:
                save_progress(frames[current_index][0])
                print("[INFO] 進捗を保存しました。")
            print("[INFO] 終了します。")
            break

        frame_index, frame_path = frames[current_index]
        frame = cv2.imread(frame_path)
        results = model(frame)
        annotated = results[0].plot()

        boxes = results[0].boxes
        if boxes is not None:
            latest_boxes = boxes.xyxy.cpu().numpy()
            latest_classes = boxes.cls.cpu().numpy()
        else:
            latest_boxes = []
            latest_classes = []

        while True:
            cv2.imshow(window_name, annotated)
            key = cv2.waitKey(1) & 0xFF

            if key == ord("q"):
                save_progress(frame_index)
                cv2.destroyAllWindows()
                return

            elif key == ord("f"):
                current_index += 1
                break

            elif key == ord("d"):
                if current_index > 0:
                    current_index -= 1
                break

            elif clicked_point is not None:
                x_click, y_click = clicked_point
                h, w, _ = frame.shape
                found = False

                for box, cls_id in zip(latest_boxes, latest_classes):
                    x1, y1, x2, y2 = box
                    if x1 <= x_click <= x2 and y1 <= y_click <= y2:
                        x_center = ((x1 + x2) / 2) / w
                        y_center = ((y1 + y2) / 2) / h
                        box_width = (x2 - x1) / w
                        box_height = (y2 - y1) / h

                        split_type = "train" if (click_count - 1) % 10 < 9 else "val"
                        filename_base = f"{click_count:06d}"
                        label_path = os.path.join(BASE_DIR, "labels", split_type, f"{filename_base}.txt")
                        image_path = os.path.join(BASE_DIR, "images", split_type, f"{filename_base}.jpg")
                        clicked_path = os.path.join(CLICKED_OUTPUT_DIR, f"{filename_base}.jpg")

                        with open(label_path, "w") as f:
                            f.write(f"{FIXED_CLASS_ID} {x_center:.6f} {y_center:.6f} {box_width:.6f} {box_height:.6f}\n")
                        cv2.imwrite(image_path, frame)
                        cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), (0, 0, 255), 2)
                        cv2.imwrite(clicked_path, frame)

                        print(f"[INFO] 保存: {filename_base}")
                        click_count += 1
                        found = True
                        break

                if not found:
                    print("[WARN] 検出範囲にクリックがありませんでした")

                clicked_point = None
                break

    if wq_requested:
        reset_progress()
    else:
        save_progress(frame_index)
    cv2.destroyAllWindows()


# === エントリポイント ===
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("video_path", help="動画ファイルのパス")
    args = parser.parse_args()

    frame_folder = os.path.splitext(os.path.basename(args.video_path))[0] + "_frames"

    if should_extract_frames(frame_folder):
        selected_rate = choose_frame_rate()
        extract_frames(args.video_path, selected_rate, frame_folder)
    else:
        print("[INFO] フレーム画像が既に存在します。切り出し処理をスキップします。")

    main(args.video_path)
