```markdown
# YOLOv8 Click Annotation Tool

This tool allows you to annotate objects in video frames using YOLOv8 object detection.  It provides a simple GUI for navigating frames, selecting object classes, and saving annotations in YOLO format.

## Prerequisites

*   Python 3.6 or higher
*   Required Python packages:
    *   opencv-python
    *   imageio
    *   ultralytics
    *   Pillow
    *   tkinter (usually comes with Python, but may require separate installation on some systems)
    *   tqdm

You can install these packages using pip:

```bash
pip install opencv-python imageio ultralytics Pillow tqdm
```

## Installation

1.  Download the script `yolo_frame_click_tool.py`.

## Usage

1.  **Run the script:**

    ```bash
    python yolo_frame_click_tool.py path/to/your/video.mp4
    ```

    Replace `path/to/your/video.mp4` with the actual path to your video file.

2.  **Frame Extraction:**

    *   If frame images do not exist in `<video_name>_frames` directory, the tool will ask you to select a frame rate (60fps or 30fps). Select a frame rate by entering `1` or `2`.
    *   The tool will extract frames from the video and save them to the `<video_name>_frames` directory.
    *   If the frame images already exist in the directory, the tool will skip frame extraction and proceed to annotation.

3.  **Annotation Process:**

    *   The GUI window will appear, displaying the first frame with YOLOv8 object detection results.
    *   A class ID selection panel is on the left side. Use the Combobox to select the class ID for the object you want to annotate.
    *   Click on the object in the image to annotate it. The tool will:
        *   Save the image to the `output_dataset/images/train` or `output_dataset/images/val` directory.
        *   Create a corresponding YOLO format label file in the `output_dataset/labels/train` or `output_dataset/labels/val` directory.
        *   Save a copy of the clicked image with a red bounding box to `output_clicked` directory.
        *   Automatically advance to the next frame.
    *   Use the following keys to navigate and control the annotation process:
        *   `f`: Next frame
        *   `d`: Previous frame
        *   `s`: Save label for the clicked object
        *   `q`: Save progress and quit
        *   `Ctrl+w`: Reset progress (deletes `progress.json`) and quit

4.  **Output:**

    *   Annotated images and YOLO format label files are saved in the `output_dataset` directory, split into `train` and `val` subdirectories.
    *   Clicked images with bounding boxes are saved in the `output_clicked` directory.
    *   The current progress (last annotated frame and click count) is saved in the `progress.json` file, allowing you to resume annotation later.

## Directory Structure

The tool will create the following directory structure:

```
output_dataset/
├── images/
│   ├── train/
│   │   └── 000001.jpg
│   ├── val/
│   │   └── 000010.jpg
├── labels/
│   ├── train/
│   │   └── 000001.txt
│   ├── val/
│   │   └── 000010.txt
output_clicked/
│   └── 000001.jpg
video_name_frames/
│   └── 00000001.jpg
progress.json
```

## Configuration

*   `CLASS_NAMES`: The class names are automatically loaded from the YOLOv8 model (`yolov8n.pt` by default). You can change the model in the script.
*   `BASE_DIR`: The base directory for output datasets (default: `output_dataset`).
*   `CLICKED_OUTPUT_DIR`: The directory for clicked images (default: `output_clicked`).
*   `PROGRESS_FILE`: The file for saving progress (default: `progress.json`).

## Tips

*   Start with a lower frame rate to reduce the number of frames to annotate.
*   Use the keyboard shortcuts to speed up the annotation process.
*   Regularly save your progress by pressing `q` to avoid losing data.

## Troubleshooting

*   If you encounter errors during frame extraction, make sure you have the necessary codecs installed.
*   If the GUI does not display correctly, check if you have the correct version of Tkinter installed.

## Credits

This tool is based on the YOLOv8 object detection framework by Ultralytics.

```
