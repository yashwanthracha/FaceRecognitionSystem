import pickle
import faiss
import cv2
import numpy as np
from sklearn.preprocessing import normalize
from facenet_model import resnet_inception
from tools import utilities
import os
from mtcnn import MTCNN
from facenet_model import resnet_inception
import csv
from datetime import datetime

import cv2
import pickle
import faiss
from mtcnn import MTCNN

from sklearn.preprocessing import normalize
from pathlib import Path

class Infer():
    def __init__(self):
        self.face_model = resnet_inception.InceptionResNetV1(input_shape=(160, 160, 3), classes=128)
        self.detect_model = MTCNN(min_face_size=20, scale_factor=0.7, steps_threshold=[0.6, 0.7, 0.7])
        self.csv_file = 'mini_project\\results.csv'
        self.faiss_index = faiss.read_index('mini_project\\faiss_index.idx')
        with open('mini_project\\metadata.pkl', 'rb') as f:
            self.metadata = pickle.load(f)

    def face_recognition(self, crop_img, k=1):
        # Normalize and preprocess the cropped image
        self.face_model.load_weights('mini_project\\facenet_model\\facenet_keras_weights.h5')
        # crop_img = cv2.resize(crop_img, (160, 160), interpolation=cv2.INTER_AREA)
        crop_img = cv2.resize(crop_img, (160, 160), interpolation=cv2.INTER_LANCZOS4)
        mean, std = crop_img.mean(), crop_img.std()
        crop_img = (crop_img - mean) / std

        embedding = self.face_model.predict(np.expand_dims(crop_img, axis=0))[0]  # Remove reshape
        query_embedding = normalize(embedding.reshape(1, -1), norm="l2")
        # Search in FAISS index
        #Eyes Mouth cheeks
        D, I = self.faiss_index.search(query_embedding, k)
        
        # Retrieve and print metadata
        results = []
        for dist, idx in zip(D[0], I[0]):
            if dist > 0.60: 
                label=self.metadata[idx]["label"]
            else:
                label = "unknown"
            results.append({
                "label": label,
                "score": dist
            })
        return results

    def main(self, img_path):
        tool = utilities()
        query_img = cv2.imread(img_path)
        resized_image = tool.resize_with_padding(query_img, (416, 416))
        detections = self.detect_model.detect_faces(resized_image)
        k = 0
        all_results = []
        for detection in detections:
            confidence = detection["confidence"]
            if confidence > 0.65:
                bbox = detection["box"]
                x, y, w, h = bbox
                img_crop = resized_image[y:y + h, x:x + w]
                cv2.imwrite(f"{k}.jpg", cv2.cvtColor(img_crop, cv2.COLOR_RGB2BGR))
                recognition_results = self.face_recognition(img_crop)
                all_results.append({
                    "bbox": bbox,
                    "confidence": confidence,
                    "recognition_results": recognition_results
                })
        return all_results
    
    def save_results_to_csv(self, all_results):
        file_exists = os.path.isfile(self.csv_file)
        with open(self.csv_file, mode='a', newline='') as file:
            writer = csv.writer(file)
            for result in all_results:
                bbox = result["bbox"]
                confidence = result["confidence"]
                for recog_res in result["recognition_results"]:
                    label = recog_res["label"]
                    score = recog_res["score"]
                    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                    writer.writerow([label, timestamp, "Present"])

# Example usage
if __name__=="__main__":
    infer = Infer()
    results = infer.main("mini_project\\test_images\\IMG20240820151332.jpg")
    infer.save_results_to_csv(results)
    for res in results:
        print("Bounding Box:", res["bbox"])
        print("Confidence:", res["confidence"])
        print("Recognition Results:")
        for recog_res in res["recognition_results"]:
            print(f"  Label: {recog_res['label']}, Score: {recog_res['score']}")

