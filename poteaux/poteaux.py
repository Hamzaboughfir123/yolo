from ultralytics import YOLO

# Load a pretrained model on
model = YOLO('poteau.pt')

# Define path to the image file
source = 'images/mdc_photo_mdc_340e3a_1703671997707_1703672007873.jpg'

# Run inference on the source
result = model.predict(source, save=True, imgsz=640, conf=0.3, iou=0.5, augment=True, max_det=2)# the result will be saved in a folder
