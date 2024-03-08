from ultralytics import YOLO

# Load a pretrained model on
model = YOLO('best.pt')

# Define path to the image file
source = 'images/mdc_photo_mdc_0ae41b_1702634985847_1702635006860.jpg'

# Run inference on the source
result = model.predict(source, save=True, imgsz=640, conf=0.7, iou=0.6, augment=True, max_det=2, save_crop=True)# the result will be saved in a folder
