# Car Counting using YOLOv8

## Project Overview
This project utilizes YOLOv8 (You Only Look Once version 8) for real-time vehicle detection and counting in video streams. The model can identify various vehicle types, including cars, buses, trucks, and motorcycles. It integrates computer vision techniques to track vehicles across frames and overlays bounding boxes for visualization.

## Technologies Used
- **Model:** YOLOv8 (YOLOv8l)
- **Languages:** Python
- **Libraries:** OpenCV, cvzone
- **Tracking Algorithm:** SORT (Simple Online and Realtime Tracking)
- **Task:** Counting vehicles in a video stream

## Installation

1. Clone the repository:
    ```bash
    https://github.com/ganeshsp7/Car_Counting-Object-Detection-.git
    cd Car_Counting-Object-Detection/Car Counter
    ```

2. Create a virtual environment and activate it:
    ```bash
    python -m venv venv
    source venv/bin/activate  # On Windows, use `venv\Scripts\activate`
    ```

3. Install the required packages:
    ```bash
    pip install -r requirements.txt
    ```

4. Download the YOLOv8 weights and place them in the `Yolo-Weights` directory.

## Usage

1. Place your video file in the `Videos` directory and name it `cars.mp4`.
2. Ensure you have the `mask.png` and `graphics.png` files in the project root.
3. Run the main script:
    ```bash
    python Car Counter/Car_Counter.py
    ```

## Code Explanation
- **Car_Counter.py**: This is the main script that processes the video, applies the YOLOv8 model, tracks vehicles, and displays the count.
- **mask.png**: A mask image used to define the region of interest for vehicle detection.
- **graphics.png**: Overlay graphics for the output video.

## Results
The script will output a video stream with detected vehicles and an on-screen count of the total number of vehicles that have passed through a predefined line.

## Acknowledgements

- [YOLO](https://github.com/ultralytics/yolov5)
- [OpenCV](https://opencv.org/)
- [cvzone](https://github.com/cvzone/cvzone)
- [SORT](https://github.com/abewley/sort)





