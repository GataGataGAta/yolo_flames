import os
import os.path as osp
import argparse
import json
import cv2
import imageio
import threading
import tkinter as tk
from tkinter import ttk
from PIL import Image, ImageTk
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
CURRENT_FRAME = None
CURRENT_ANNOTATED_FRAME = None
current_index = 0
frames = []
total_frames = 0
window_name = "YOLOv8 Click Tool"
video_path = None  # 動画パスを保持
frame_dir = None # フレームディレクトリを保持
model = None # YOLO モデルを保持

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

    global frames
    global total_frames
    selected = []
    for i, img in tqdm(enumerate(reader), total=meta_data['nframes']):
        if i % interval == 0:
            frame_path = os.path.join(output_dir, f"{i:08d}.jpg")
            imageio.imsave(frame_path, img)
            selected.append((i, frame_path))

    frames = sorted([
        (int(os.path.splitext(f)[0]), os.path.join(frame_dir, f))
        for f in os.listdir(frame_dir) if f.endswith(".jpg")
    ])

    total_frames = len(frames)

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

# === マウスクリックイベント ===
def mouse_callback(event, x, y, flags, param):
    global clicked_point
    if event == cv2.EVENT_LBUTTONDOWN:
        clicked_point = (x, y)

# === フレーム表示の更新 ===
def update_frame_display(frame):
    global CURRENT_FRAME, CURRENT_ANNOTATED_FRAME
    results = model(frame)
    annotated = results[0].plot()
    CURRENT_ANNOTATED_FRAME = annotated
    CURRENT_FRAME = frame

    boxes = results[0].boxes
    global latest_boxes, latest_classes
    if boxes is not None:
        latest_boxes = boxes.xyxy.cpu().numpy()
        latest_classes = boxes.cls.cpu().numpy()
    else:
        latest_boxes = []
        latest_classes = []

    img = Image.fromarray(cv2.cvtColor(annotated, cv2.COLOR_BGR2RGB))
    imgtk = ImageTk.PhotoImage(image=img)
    image_label.imgtk = imgtk
    image_label.config(image=imgtk)

# === フレーム送り ===
def next_frame():
    global current_index
    if current_index < total_frames - 1:
        current_index += 1
        frame_index, frame_path = frames[current_index]
        frame = cv2.imread(frame_path)
        update_frame_display(frame)

# === フレーム戻し ===
def prev_frame():
    global current_index
    if current_index > 0:
        current_index -= 1
        frame_index, frame_path = frames[current_index]
        frame = cv2.imread(frame_path)
        update_frame_display(frame)

# === 保存処理 ===
def save_label():
    global clicked_point, latest_boxes, latest_classes, click_count, FIXED_CLASS_ID

    if clicked_point is not None and CURRENT_FRAME is not None:
        x_click, y_click = clicked_point
        h, w, _ = CURRENT_FRAME.shape
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
                cv2.imwrite(image_path, CURRENT_FRAME)
                cv2.rectangle(CURRENT_FRAME, (int(x1), int(y1)), (int(x2), int(y2)), (0, 0, 255), 2)
                cv2.imwrite(clicked_path, CURRENT_FRAME)

                print(f"[INFO] 保存: {filename_base}")
                click_count += 1
                found = True
                next_frame()
                break

        if not found:
            print("[WARN] 検出範囲にクリックがありませんでした")

        clicked_point = None

# === GUIイベント ===
def on_mouse_click(event):
    global clicked_point
    clicked_point = (event.x, event.y)
    save_label()

def on_closing():
    global stop_requested
    stop_requested = True
    root.destroy()

def set_class_id(event):
    global FIXED_CLASS_ID
    try:
        FIXED_CLASS_ID = int(class_id_combo.get())
    except ValueError:
        print("[ERROR] 無効なクラスIDです。")

def save_and_quit():
    save_progress(frames[current_index][0])
    print("[INFO] 進捗を保存しました。")
    print("[INFO] 終了します。")
    on_closing()

def quit_without_saving():
    global wq_requested
    wq_requested = True
    reset_progress()
    on_closing()

def key_press(event):
    if event.keysym == 'f':
        next_frame()
    elif event.keysym == 'd':
        prev_frame()
    elif event.keysym == 's': # Save on 's' key
        save_label()
    elif event.keysym == 'q':
        save_progress(frames[current_index][0])
        print("[INFO] 進捗を保存しました。")
        print("[INFO] 終了します。")
        on_closing()
    elif event.keysym == 'w': # Check for Control key with 'w'
        global wq_requested
        wq_requested = True
        reset_progress() # リセット処理を呼び出す
        print("[INFO] progress.json をリセットして終了します。")
        on_closing()

# === メイン処理 ===
def main_gui(video_path_arg):
    global video_path, frame_dir, current_index, frames, total_frames, window_name, model
    video_path = video_path_arg

    os.makedirs(os.path.join(BASE_DIR, "images", "train"), exist_ok=True)
    os.makedirs(os.path.join(BASE_DIR, "images", "val"), exist_ok=True)
    os.makedirs(os.path.join(BASE_DIR, "labels", "train"), exist_ok=True)
    os.makedirs(os.path.join(BASE_DIR, "labels", "val"), exist_ok=True)
    os.makedirs(CLICKED_OUTPUT_DIR, exist_ok=True)

    model = YOLO("yolov8n.pt")
    global CLASS_NAMES
    CLASS_NAMES = model.names

    last_frame_idx, click_count = load_progress()

    current_index = 0
    for idx, (frame_idx, _) in enumerate(frames):
        if frame_idx >= last_frame_idx:
            current_index = idx
            break

    # Tkinter GUIの初期化
    global root, image_label, class_id_combo
    root = tk.Tk()
    root.title(window_name)
    root.protocol("WM_DELETE_WINDOW", on_closing)

    # GUI要素の作成
    frame = tk.Frame(root)
    frame.pack(padx=10, pady=10)

    # 左側のクラスID選択タブ
    id_frame = tk.Frame(frame, width=150)
    id_frame.pack(side=tk.LEFT, padx=5)
    id_label = tk.Label(id_frame, text="Class ID:")
    id_label.pack()
    class_id_options = [str(i) for i in range(len(CLASS_NAMES))]  # クラスIDのリスト
    class_id_combo = ttk.Combobox(id_frame, values=class_id_options)
    class_id_combo.pack()
    class_id_combo.set("0")  # デフォルト値を設定
    class_id_combo.bind("<<ComboboxSelected>>", set_class_id) # 選択時にset_class_idを呼ぶ

    # 画像表示ラベル
    image_label = tk.Label(frame)
    image_label.pack(side=tk.LEFT)
    image_label.bind("<Button-1>", on_mouse_click)  # 左クリックイベント

    # フレーム操作ボタン
    button_frame = tk.Frame(root)
    button_frame.pack(pady=5)
    prev_button = tk.Button(button_frame, text="Prev", command=prev_frame)
    prev_button.pack(side=tk.LEFT, padx=5)
    next_button = tk.Button(button_frame, text="Next", command=next_frame)
    next_button.pack(side=tk.LEFT, padx=5)

    # 終了ボタン
    quit_frame = tk.Frame(root)
    quit_frame.pack(pady=5)
    save_quit_button = tk.Button(quit_frame, text="Save & Quit", command=save_and_quit)
    save_quit_button.pack(side=tk.LEFT, padx=5)
    quit_button = tk.Button(quit_frame, text="Quit", command=quit_without_saving)
    quit_button.pack(side=tk.LEFT, padx=5)

    # キーイベントのバインド
    root.bind("<KeyPress>", key_press)

    # 最初のフレームを表示
    frame_index, frame_path = frames[current_index]
    frame = cv2.imread(frame_path)
    update_frame_display(frame)

    root.mainloop()


# === エントリポイント ===
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("video_path", help="動画ファイルのパス")
    args = parser.parse_args()

    video_path = args.video_path
    frame_dir = os.path.splitext(os.path.basename(video_path))[0] + "_frames"

    if should_extract_frames(frame_dir):
        selected_rate = choose_frame_rate()
        extract_frames(video_path, selected_rate, frame_dir)
    else:
        frames = sorted([
            (int(os.path.splitext(f)[0]), os.path.join(frame_dir, f))
            for f in os.listdir(frame_dir) if f.endswith(".jpg")
        ])
        total_frames = len(frames)
        print("[INFO] フレーム画像が既に存在します。切り出し処理をスキップします。")

    main_gui(video_path)