import os
HOME = os.getcwd()
print(HOME)

# Install YOLOv10
!pip install -q git+https://github.com/THU-MIG/yolov10.git

!pip install -q supervision roboflow

# Download pre-trained weights
!mkdir -p {HOME}/weights
!wget -P {HOME}/weights -q https://github.com/THU-MIG/yolov10/releases/download/v1.1/yolov10n.pt
!wget -P {HOME}/weights -q https://github.com/THU-MIG/yolov10/releases/download/v1.1/yolov10s.pt
!wget -P {HOME}/weights -q https://github.com/THU-MIG/yolov10/releases/download/v1.1/yolov10m.pt
!wget -P {HOME}/weights -q https://github.com/THU-MIG/yolov10/releases/download/v1.1/yolov10b.pt
!wget -P {HOME}/weights -q https://github.com/THU-MIG/yolov10/releases/download/v1.1/yolov10x.pt
!wget -P {HOME}/weights -q https://github.com/THU-MIG/yolov10/releases/download/v1.1/yolov10l.pt
!ls -lh {HOME}/weights

# Download dataset from Roboflow Universe
!mkdir {HOME}/datasets
%cd {HOME}/datasets

!pip install roboflow

from roboflow import Roboflow
rf = Roboflow(api_key="8em4DWcN6y4rVfn5bwZa")
project = rf.workspace("gp-f9rwi").project("fractured-6vb1c")
version = project.version(19)
dataset = version.download("yolov8")

# Custom Training
%cd {HOME}

!yolo task=detect mode=train epochs=100 batch=16 plots=True \
model={HOME}/weights/yolov10n.pt \
data={dataset.location}/data.yaml