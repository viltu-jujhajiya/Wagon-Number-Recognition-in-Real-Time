### Import Libraries
import os
import cv2
import numpy as np
from PIL import Image
from ultralytics import YOLO


### Import trained models
'''For YOLO model training ultralytics' custom training code was used, and all models were YOLOv8m.'''
delineation_detector_model = YOLO("Delineation_Detection_best_weights.pt") # For freight train delineation detection
wagon_number_detector_model = YOLO("Number_Detection_best_weights.pt")  # For wagon number detection
digit_recognizer_model = YOLO("Digit_Recognition_best_weights.pt") # For digit recognition from detection wagon number


### Wagon Number Detector
def wagon_number_detector(wagon_images):
    results = []
    for img in wagon_images:
        
        detector_result = wagon_number_detector_model(img, conf=0.6)
        
        if len(detector_result[0].boxes.data)>0:
            im = np.array(img)
            im = im[int(detector_result[0].boxes.data[0][1]):int(detector_result[0].boxes.data[0][3]), int(detector_result[0].boxes.data[0][0]):int(detector_result[0].boxes.data[0][2])]
            im = Image.fromarray(im)
            results.append(im)
    return results


### Wagon Number Recognizer
def wagon_number_recognizer(wagon_number_crops):
    results = []
    for crop in wagon_number_crops:
        recognizer_result = digit_recognizer_model(crop, conf=0.6)
        
        if len(recognizer_result[0].boxes.data)>0:    
            num = np.array(recognizer_result[0].boxes.data).copy()
            digits = []
            k=0
            while True:
                rows=[] 
                i=0
                j=0
                remove_rows = []
                minimum=min(num[:,1:2])
                
                for row in num:
                    if row[1]<=minimum+70:
                        row[1]=minimum
                        rows.append(row)
                        remove_rows.append(i)
                    i+=1
                    
                rows.sort(key = lambda x: x[0])
                for digit in rows:
                    digits.append(digit[5])
                num = np.delete(num, remove_rows, axis=0)
                if len(num)==0:
                    break
                k+=1
                if k>5:
                    break
            results.append(digits)
    return results


### Postprocessing
def wagon_number_processesor(wagon_numbers):
    results = {}
    for wagon_count in wagon_numbers.keys():
        maximum_length = 0
        best_res = []
        for wagon_number in wagon_numbers[wagon_count]:
            if len(wagon_number)>=maximum_length:
                maximum_length = len(wagon_number)
                best_res = wagon_number
        
        results[wagon_count] = best_res
    return results  


### Main class  
class WNR_main():
    def __init__(self, wagon_image_path):
        self.wagon_image_path = wagon_image_path
    
    def recognizer(self):
        wagon_number_crops = {}
        wagon_count = 0
        wagon_identification_numbers = {}
        temp = []
        delineation_variable = 0
        k=0
        for img in os.listdir(self.wagon_image_path):
            if k<150:
                k+=1
                continue
            img_path = os.path.join(self.wagon_image_path, img)
            image = Image.open(img_path)
            image = image.rotate(-90)
            
            delineation_result = delineation_detector_model(image, conf=0.6)
            
            if len(delineation_result[0].boxes.data)>0:
                if delineation_variable>0:
                    delineation_variable-=1
                else:
                    delineation_variable=5
                    wagon_number_crops[wagon_count] = wagon_number_detector(temp.copy())
                    wagon_identification_numbers[wagon_count] = wagon_number_recognizer(wagon_number_crops[wagon_count].copy())
                    temp.clear()
                    wagon_count+=1
            else:
                if delineation_variable>0:
                    delineation_variable-=1
            if delineation_variable<2:
                temp.append(image)
            print(k)
            k+=1
            if k>=500:
                break
        return wagon_identification_numbers