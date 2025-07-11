import os
import os.path as osp
import argparse
import json
import cv2
import imageio
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
frame_dir = None  # フレームディレクトリを保持
model = None  # YOLO モデルを保持
auto_advance = True  # クリック後に自動で次フレームへ進むかのフラグ

# === フレーム抽出が必要か判定 ===
def should_extract_frames(frame_dir):
    return not (
        os.path.exists(frame_dir) and
        any(f.endswith(".jpg") for f in os.listdir(frame_dir))
    )

# === フレーム抽出（動画FPSを自動取得し、指定target_fpsで間引き抽出） ===
def extract_frames(video_path, output_dir):
    os.makedirs(output_dir, exist_ok=True)
    reader = imageio.get_reader(video_path)
    meta_data = reader.get_meta_data()
    video_fps = meta_data.get('fps', 0)
    if video_fps == 0:
        raise ValueError("[ERROR] 動画のFPSが取得できません。")

    print(f"[INFO] 動画FPS: {video_fps} - 全フレームを抽出します。")

    interval = 1  # 間引かずにすべてのフレームを抽出

    for i, img in tqdm(enumerate(reader), total=meta_data['nframes']):
        if i % interval == 0:
            frame_path = os.path.join(output_dir, f"{i:08d}.jpg")
            imageio.imsave(frame_path, img)

    global frames, total_frames
    frames = sorted([
        (int(os.path.splitext(f)[0]), os.path.join(output_dir, f))
        for f in os.listdir(output_dir) if f.endswith(".jpg")
    ])
    total_frames = len(frames)

    if os.path.exists(PROGRESS_FILE):
        os.remove(PROGRESS_FILE)
        print("[INFO] フレーム再生成のため progress.json をリセットしました。")

def extract_or_load_frames(video_path, frame_dir):
    if should_extract_frames(frame_dir):
        extract_frames(video_path, frame_dir)
    else:
        load_frames(frame_dir)
        print("[INFO] フレーム画像が既に存在します。切り出し処理をスキップします。")
        
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

# === フレームリスト読み込み ===
def load_frames(frame_dir):
    global frames, total_frames
    frames = sorted([
        (int(os.path.splitext(f)[0]), os.path.join(frame_dir, f))
        for f in os.listdir(frame_dir) if f.endswith(".jpg")
    ])
    total_frames = len(frames)

# === 出力ディレクトリ準備 ===
def prepare_directories():
    os.makedirs(os.path.join(BASE_DIR, "images", "train"), exist_ok=True)
    os.makedirs(os.path.join(BASE_DIR, "images", "val"), exist_ok=True)
    os.makedirs(os.path.join(BASE_DIR, "labels", "train"), exist_ok=True)
    os.makedirs(os.path.join(BASE_DIR, "labels", "val"), exist_ok=True)
    os.makedirs(CLICKED_OUTPUT_DIR, exist_ok=True)

# === YOLOモデル読み込み ===
def initialize_model():
    global model, CLASS_NAMES
    model = YOLO("yolov8n.pt")
    CLASS_NAMES = model.names
 
# === 最初に表示するフレームの検索関数 ===   
def find_first_person_frame():
    global frames, model, CLASS_NAMES

    print(f"[DEBUG] 総フレーム数: {len(frames)}", flush=True)

    person_class_id = None
    for id, name in CLASS_NAMES.items():
        if name.lower() == "person":
            person_class_id = id
            break

    if person_class_id is None:
        print("[ERROR] 'person' クラスが model.names に存在しません", flush=True)
        return 0

    for idx, (frame_idx, frame_path) in enumerate(frames):
        print(f"[DEBUG] 処理中: {frame_path}", flush=True)
        frame = cv2.imread(frame_path)
        results = model(frame)

        boxes = results[0].boxes
        if boxes is not None and boxes.cls is not None:
            classes = boxes.cls.cpu().numpy()
            if person_class_id in classes:
                print(f"[INFO] 最初に person を検出したフレーム: {frame_idx} (インデックス: {idx})", flush=True)
                return idx

    print("[WARN] person を検出できませんでした。最初のフレームから開始します。", flush=True)
    return 0


# === メインGUI関数 ===
def main_gui(video_path_arg):
    global video_path, frame_dir, current_index, frames, total_frames, window_name, model, click_count, auto_advance_var
    video_path = video_path_arg
    frame_dir = osp.splitext(osp.basename(video_path))[0] + "_frames"

    prepare_directories()
    initialize_model()
    
    load_frames(frame_dir)

    last_frame_idx, click_count = load_progress()

    print("[INFO] person検出フレームの探索を開始します...", flush=True)
    person_start_index = find_first_person_frame()
    print(f"[INFO] person検出後に開始するフレームインデックス: {person_start_index}", flush=True)

    # 最初に起動した時は person_start_index から、再開時は progress.json の位置から開始
    if last_frame_idx == 0:
        current_index = person_start_index
    else:
        current_index = 0
        for idx, (frame_idx, _) in enumerate(frames):
            if frame_idx >= last_frame_idx and idx >= person_start_index:
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

    class_id_options = [str(i) for i in range(len(CLASS_NAMES))]
    class_id_combo = ttk.Combobox(id_frame, values=class_id_options)
    class_id_combo.pack()
    class_id_combo.set("0")
    class_id_combo.bind("<<ComboboxSelected>>", set_class_id)

    # クリック後に次のフレームへ進むかのチェックボックス
    auto_advance_var = tk.BooleanVar(value=True)
    auto_advance_check = tk.Checkbutton(
        id_frame,
        text="クリック後に次のフレームへ進む",
        variable=auto_advance_var
    )
    auto_advance_check.pack(pady=10)

    # 画像表示ラベル
    image_label = tk.Label(frame)
    image_label.pack(side=tk.LEFT)
    image_label.bind("<Button-1>", on_mouse_click)

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

# === GUI関連イベント・関数 ===
def update_frame_display(frame):
    global CURRENT_FRAME, CURRENT_ANNOTATED_FRAME, latest_boxes, latest_classes

    results = model(frame)
    annotated = results[0].plot()
    CURRENT_ANNOTATED_FRAME = annotated
    CURRENT_FRAME = frame

    boxes = results[0].boxes
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

def next_frame():
    global current_index
    if current_index < total_frames - 1:
        current_index += 1
        frame_index, frame_path = frames[current_index]
        frame = cv2.imread(frame_path)
        update_frame_display(frame)

def prev_frame():
    global current_index
    if current_index > 0:
        current_index -= 1
        frame_index, frame_path = frames[current_index]
        frame = cv2.imread(frame_path)
        update_frame_display(frame)

def save_label():
    global clicked_point, latest_boxes, latest_classes, click_count, FIXED_CLASS_ID, auto_advance_var

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
                # 元画像に赤枠を書き込んだものも保存
                annotated_frame = CURRENT_FRAME.copy()
                cv2.rectangle(annotated_frame, (int(x1), int(y1)), (int(x2), int(y2)), (0, 0, 255), 2)
                cv2.imwrite(clicked_path, annotated_frame)

                print(f"[INFO] 保存: {filename_base}")
                click_count += 1
                found = True

                # チェックボックスの値で自動進行制御
                if auto_advance_var.get():
                    next_frame()

                break

        if not found:
            print("[WARN] 検出範囲にクリックがありませんでした")

        clicked_point = None

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
    elif event.keysym == 's':  # Save on 's' key
        save_label()
    elif event.keysym == 'q':
        save_progress(frames[current_index][0])
        print("[INFO] 進捗を保存しました。")
        print("[INFO] 終了します。")
        on_closing()
    elif event.keysym == 'w':  # Reset progress and quit on 'w' key
        global wq_requested
        wq_requested = True
        reset_progress()
        print("[INFO] progress.json をリセットして終了します。")
        on_closing()

# === エントリポイント ===
def main():
    global video_path, frame_dir, frames, total_frames

    parser = argparse.ArgumentParser()
    parser.add_argument("video_path", help="動画ファイルのパス")
    args = parser.parse_args()

    video_path = args.video_path
    frame_dir = os.path.splitext(osp.basename(video_path))[0] + "_frames"

    extract_or_load_frames(video_path, frame_dir)  # ここで frames を準備
    main_gui(video_path)

if __name__ == "__main__":
    main()
