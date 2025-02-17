# Wagon-Number-Recognition-in-Real-Time


## Overview

This project implements an automated Wagon Identification Number Recognition system using deep learning and computer vision techniques. The system is designed to accurately detect and recognize 11-digit identification numbers from freight train wagons in real-time.


## Features

- **Automated Detection & Recognition:** Uses YOLOv8 for wagon number detection and recognition.
- **High Accuracy:** Custom-trained YOLOv8 model with over 91% accuracy for complete wagon ID recognition.
- **Real-Time Processing:** Executes within **1.5ms per image**, making it suitable for real-world deployment.
- **Robust to Environmental Variations:** Handles diverse lighting conditions, image quality, and font variations.
- **Dataset Augmentation:** Includes techniques like brightness adjustment and noise augmentation.


## Workflow

### Preprocessing
Before being processed by the deep learning model, train images undergo a 90-degree clockwise rotation to ensure proper alignment. Various image processing techniques, including erosion, dilation, binarization, and noise removal, were explored but were ultimately excluded due to their detrimental impact on model performance. Consequently, the final implementation retains only the rotation step for preprocessing.

### Wagon Number Detection
A YOLOv8m model was trained to detect and isolate wagon identification numbers from wagon images. This step is critical in filtering out extraneous text and markings present on the wagons that are irrelevant to the identification process.

### Wagon Number Recognition
The cropped wagon number images are subsequently processed using a custom-trained YOLOv8m model for digit recognition. The model identifies and extracts each digit, sorting them in a top-to-bottom, left-to-right order. The recognized digits are systematically stored in a structured dictionary format, where each wagon number is mapped accordingly. Additionally, delineation detection is performed to associate all images belonging to the same wagon, ensuring accurate tracking and identification.
