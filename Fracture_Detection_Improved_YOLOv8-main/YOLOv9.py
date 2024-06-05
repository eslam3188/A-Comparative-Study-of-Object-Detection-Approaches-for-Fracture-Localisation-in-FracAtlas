!git clone https://github.com/eslam3188/A-Comparative-Study-of-Object-Detection-Approaches-for-Fracture-Localisation-in-FracAtlas.git

%cd /content/A-Comparative-Study-of-Object-Detection-Approaches-for-Fracture-Localisation-in-FracAtlas/Fracture_Detection_Improved_YOLOv8-main

!pip install roboflow

from roboflow import Roboflow
rf = Roboflow(api_key="8em4DWcN6y4rVfn5bwZa")
project = rf.workspace("gp-f9rwi").project("fractured-6vb1c")
version = project.version(19)
dataset = version.download("yolov8-obb")

from ultralytics import YOLO
!yolo task=detect mode=train model=yolov9c.pt data={dataset.location}/data.yaml epochs=100 imgsz=640