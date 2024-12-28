# Road Sign Detection and Classification for Autonomous Driving

## Project Overview
This project focuses on developing a computer vision module for autonomous vehicles that detects and classifies road signs from images captured by dashboard cameras. By accurately identifying road signs like speed limits, stop signs, and yield signs, this module enhances vehicle safety and ensures compliance with traffic regulations.

The project employs Convolutional Neural Networks (CNNs) for classification and state-of-the-art object detection algorithms like YOLO and Faster RCNN for real-time detection of road signs.

## Features
- **Custom CNN Model for Classification**: Designed and trained CNN architectures to classify road signs with high accuracy.
- **Object Detection with YOLO and Faster RCNN**: Fine-tuned pre-trained models to detect road signs in real-world images.
- **Evaluation Metrics**: Used mAP and IoU for evaluating model performance.
- **Frontend Integration**: Built an interface for users to upload images, select detection models, and view results.

## Technologies Used
- **Programming Languages**: Python
- **Libraries and Frameworks**: PyTorch, OpenCV, Matplotlib
- **Object Detection Models**: YOLO, Faster RCNN
- **Visualization Tools**: Confusion Matrix, Bounding Box Visualizations

## Dataset
The project uses the German Traffic Sign Detection Benchmark (GTSDB) dataset, which contains images of road signs with bounding boxes and labels.

