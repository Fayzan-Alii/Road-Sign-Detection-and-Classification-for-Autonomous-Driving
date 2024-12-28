import streamlit as st
from ultralytics import YOLO
import torch
from torchvision import transforms
from torchvision.models.detection import FasterRCNN
from torchvision.models.detection.rpn import AnchorGenerator
from torchvision.models import mobilenet_v3_small
from collections import OrderedDict
from PIL import Image
import numpy as np
import cv2
import gc
from torchvision.ops import MultiScaleRoIAlign


class MobileNetBackbone(torch.nn.Module):
    def __init__(self, pretrained=True):
        super().__init__()
        
        mobile_net = mobilenet_v3_small(pretrained=pretrained)
        self.features = mobile_net.features
        self.out_channels = 576
        
    def forward(self, x):
        x = self.features(x)
        return OrderedDict([('0', x)])

def create_fasterrcnn_model(num_classes, pretrained=True):
    backbone = MobileNetBackbone(pretrained=pretrained)
    anchor_generator = AnchorGenerator(
        sizes=((32, 64, 128),),
        aspect_ratios=((0.5, 1.0, 2.0),)
    )
    roi_pooler = MultiScaleRoIAlign(
        featmap_names=['0'], output_size=5, sampling_ratio=2
    )
    
    model = FasterRCNN(
        backbone=backbone,
        num_classes=num_classes,
        rpn_anchor_generator=anchor_generator,
        box_roi_pool=roi_pooler
    )
    return model


def clear_memory():
    gc.collect()
    torch.cuda.empty_cache()


def load_yolov8_model(model_path):
    model = YOLO(model_path)  
    return model


def predict_yolov8(image, model):
    results = model(image)
    return results


def predict_fasterrcnn(image, model, device):
    transform = transforms.Compose([
        transforms.Resize((300, 300)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    image_tensor = transform(image).unsqueeze(0).to(device)
    with torch.no_grad():
        prediction = model(image_tensor)
    return prediction


def draw_boxes_yolov8(image, results):
    img = np.array(image)
    
    
    for result in results[0].boxes:  
        bbox = result.xywh[0]  
        conf = result.conf[0]  
        cls = result.cls[0]  
        
        
        if conf > 0.5:
            x1, y1, x2, y2 = result.xyxy[0]  
            cv2.rectangle(img, (int(x1), int(y1)), (int(x2), int(y2)), (0, 255, 0), 2)
    
    return img


def draw_boxes_fasterrcnn(image, predictions):
    img = np.array(image)
    for element in predictions[0]['boxes']:
        cv2.rectangle(img, 
                      (int(element[0]), int(element[1])), 
                      (int(element[2]), int(element[3])), 
                      (0, 255, 0), 
                      2)
    return img


st.title("Object Detection with YOLOv8 and Faster R-CNN")
st.write("Upload an image and choose a detection model (YOLOv8 or Faster R-CNN).")


uploaded_image = st.file_uploader("Choose an image...", type=["jpg", "png", "jpeg", "ppm"])


if uploaded_image is not None:
    
    if uploaded_image.name.lower().endswith(".ppm"):
        image = Image.open(uploaded_image)
    else:
        image = Image.open(uploaded_image)  
    
    
    model_choice = st.selectbox("Select Model", ["YOLOv8", "Faster R-CNN"])

    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    if model_choice == "YOLOv8":
        
        model_path = 'fine_tuned_yolov8_gtsdb.pt'  
        model = load_yolov8_model(model_path)

        
        st.write("Processing image with YOLOv8...")
        results = predict_yolov8(image, model)

        
        result_img = draw_boxes_yolov8(image, results)

    elif model_choice == "Faster R-CNN":
        
        num_classes = 44  
        model = create_fasterrcnn_model(num_classes=num_classes).to(device)
        model.eval()  
        
        
        st.write("Processing image with Faster R-CNN...")
        predictions = predict_fasterrcnn(image, model, device)

        
        result_img = draw_boxes_fasterrcnn(image, predictions)
    
    
    st.image(result_img, caption="Detection Result", use_column_width=True)
