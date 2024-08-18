# A Comparative Study of Object Detection Approaches for Fracture Localization in Musculoskeletal Radiographs

## Project Overview

This repository contains the implementation of a graduation project that explores and compares different object detection approaches for the localization of fractures in musculoskeletal radiographs. The study particularly focuses on enhancing and evaluating models within the You Only Look Once (YOLO) family, including novel architectures integrated with various attention mechanisms.

### Key Features
- **FracAtlas Dataset**: A comprehensive dataset with 4,083 X-ray scans, specifically annotated for fracture localization.
- **YOLO Models**: Implementation and comparison of multiple YOLO models, including YOLOv8, YOLOv9, and YOLOv10, with various attention mechanisms like CBAM, GAM, ECA, and SA.
- **Advanced Neural Network Architectures**: The YOLOv8-AM, YOLOv9, and YOLOv10 models are enhanced with attention mechanisms to improve fracture detection accuracy.

## Project Structure

- **`/data`**: Contains the FracAtlas dataset used for training and evaluation.
- **`/models`**: Includes the implementations of YOLOv8, YOLOv9, and YOLOv10 along with the attention mechanisms.
- **`/scripts`**: Python scripts for training models, evaluating performance, and running inference.
- **`/results`**: Stores the results of the experiments, including evaluation metrics and model outputs.

## Installation

To set up the project locally, follow these steps:

1. **Clone the Repository**:
    ```bash
    git clone https://github.com/your-username/your-repository.git
    cd your-repository
    ```

2. **Install Dependencies**:
    Install the required Python libraries using pip:
    ```bash
    pip install -r requirements.txt
    ```

3. **Download the Dataset**:
    Download the FracAtlas dataset from Roboflow:
    ```python
    from roboflow import Roboflow

    rf = Roboflow(api_key="your-api-key")
    project = rf.workspace("your-workspace").project("fractured-6vb1c")
    version = project.version(19)
    dataset = version.download("yolov8-obb")
    ```

4. **Prepare the Dataset**:
    Make sure the dataset is properly organized and the `data.yaml` file is configured correctly.

## Usage

### Training the Models

You can train any of the YOLO models with the following commands:

1. **YOLOv8**:
    ```bash
    python start_train.py --model ./ultralytics/cfg/models/v8/yolov8m.yaml --data_dir ./data/FracAtlas/data.yaml
    ```

2. **YOLOv9**:
    ```bash
    yolo task=detect mode=train model=yolov9c.pt data=./data/FracAtlas/data.yaml epochs=100 imgsz=640
    ```

3. **YOLOv10**:
    ```bash
    yolo task=detect mode=train epochs=100 batch=16 plots=True model=./weights/yolov10n.pt data=./data/FracAtlas/data.yaml
    ```

### Evaluating the Models

After training, you can evaluate the models using standard metrics like mAP, precision, and recall. The evaluation scripts will generate confusion matrices and other visualizations to aid in analysis.

## Results

The key results of our experiments include:
- **YOLOv8-SA** achieved the highest mAP50 (0.583) among the YOLOv8 variants.
- **YOLOv9** demonstrated the best overall performance with an mAP50 of 0.571 and mAP(50-95) of 0.277.
- **YOLOv10** offered a strong balance between accuracy and computational efficiency, making it suitable for real-time applications.

## Future Work

The project lays the foundation for further exploration into:
- Applying these models to more diverse datasets to enhance generalizability.
- Integrating these detection systems into clinical workflows for real-time fracture detection and diagnosis.

## Contributing

Contributions are welcome! Please submit a pull request or open an issue to discuss any changes or improvements.
