import cv2
import glob
import json
import numpy as np
import os
from tqdm import tqdm
from keras.models import load_model
from mtcnn import MTCNN
from pathlib import Path
from sklearn.preprocessing import normalize
from facenet_model import resnet_inception
import faiss
import pickle
# from models import resnet_inception
import cv2
import shutil
from tools import utilities
class AttendanceDetection:
    def __init__(self, img_path, weight_path, output_path,crop_img=True,annotated_img = True,data= None):
        """
        Initializes the FaceIdentification class.

        Args:
            img_path (str): Path to the image directory.
            weight_path (str): Path to the weight file.
            output_path (str): Path to the output directory.
        """
        self.img_path = img_path
        self.weight_path = weight_path
        self.output_path = output_path
        self.crop_img = crop_img
        self.annotated_img = annotated_img
        self.data = data
    def directory_remover(self):
        shutil.rmtree("output\\annotated_img")
        shutil.rmtree("output\\cropped_img")

    def face_detection(self):
        """
        Performs face detection on images in the specified image path and saves the bounding box information in YOLO format.

        Returns:
            list: A list of bounding boxes of the detected faces.
        """
        bboxes = []
        face_detector = MTCNN()
        target_size = (416,416)

        for img_path in tqdm(glob.glob(os.path.join(self.img_path, "*\*.jpg")), desc="Face detection is in progress"):
            # Extract image name and label
            split_path = Path(img_path)
            img_name = split_path.stem
            label = split_path.parent.stem
    
            
            # Read and process the image
            img = cv2.imread(img_path)
            img = utilities().resize_with_padding(image=img, target_size=target_size)

            img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            detections = face_detector.detect_faces(img_rgb)
            img_height, img_width, _ = img.shape

            bbox_list = []
            annotated_img = img.copy()

            for idx,detection in enumerate(detections):
                confidence = detection["confidence"]
                if confidence > 0.85:
                    bbox = detection["box"]
                    x, y, w, h = bbox
                    x_center = (x + w / 2) / img_width
                    y_center = (y + h / 2) / img_height
                    width = w / img_width
                    height = h / img_height

                    bbox_list.append(f"0 {x_center} {y_center} {width} {height}\n")
                    bboxes.append(bbox)

                    # Optionally save the cropped face images
                    if self.crop_img:
                        img_crop = img_rgb[y:y + h, x:x + w]
                        # img_crop = cv2.resize(img_crop, (160, 160), interpolation=cv2.INTER_AREA)
                        img_crop = cv2.resize(img_crop, (160, 160), interpolation=cv2.INTER_LANCZOS4)
                        crop_img_dir = os.path.join(self.output_path, "cropped_img", label)
                        os.makedirs(crop_img_dir, exist_ok=True)
                        crop_img_path = os.path.join(crop_img_dir, f"{img_name}_{idx}.jpg")
                        cv2.imwrite(crop_img_path, cv2.cvtColor(img_crop, cv2.COLOR_RGB2BGR))

                    # Draw the bounding box on the annotated image
                    top_left = (x, y)
                    bottom_right = (x + w, y + h)
                    cv2.rectangle(annotated_img, top_left, bottom_right, (0, 255, 0), 2)

            # Write all bounding boxes to a single file
            if bbox_list:
                label_dir = os.path.join(self.output_path, "labels", label)
                os.makedirs(label_dir, exist_ok=True)
                label_path = os.path.join(label_dir, f"{img_name}.txt")
                with open(label_path, 'w') as f:
                    f.writelines(bbox_list)

            # Optionally save the annotated image
            if self.annotated_img:
                annotated_img_dir = os.path.join(self.output_path, "annotated_img", label)
                os.makedirs(annotated_img_dir, exist_ok=True)
                annotated_img_path = os.path.join(annotated_img_dir, f"{img_name}.jpg")
                cv2.imwrite(annotated_img_path, annotated_img)
        return bboxes
    
    
    def face_recognition(self):
        """
        Performs face recognition using the FaceNet model and saves the embeddings and labels.
        """
        d = 128  # Dimensionality of FaceNet embeddings (128 or 512 depending on the model)
        index = faiss.IndexFlatIP(d)
        #DL Architecture
        face_model = resnet_inception.InceptionResNetV1(
            input_shape=(160, 160, 3),  # Set the input shape to match the resized image size
            classes=128
        )
        face_model.load_weights('mini_project\\facenet_model\\facenet_keras_weights.h5')

        embeddings = []
        metadata = []

        for crop_img_path in tqdm(glob.glob(os.path.join(self.output_path, "cropped_img/*/*.jpg")),
                                 desc="Embedding extraction is in progress"):
            label = os.path.basename(os.path.dirname(crop_img_path))
            crop_img = cv2.imread(crop_img_path)
            crop_img = cv2.resize(crop_img, (160, 160), interpolation=cv2.INTER_AREA)
            mean, std = crop_img.mean(), crop_img.std()
            crop_img = (crop_img - mean) / std
            embedding = face_model.predict(np.expand_dims(crop_img, axis=0))[0]  # Remove reshape
            #L2 Norm
            normalized_embedding = normalize(embedding.reshape(1, -1), norm="l2")  # Reshape here
            embeddings.append(normalized_embedding)
            metadata.append({"label": label})

        #Final Embeddings
        embeddings = np.vstack(embeddings).astype('float32')

        # Add embeddings to FAISS index
        index.add(embeddings)

        # Save the FAISS index and metadata
        faiss.write_index(index, 'mini_project\\faiss_index.idx')
        with open('mini_project\\metadata.pkl', 'wb') as f:
            pickle.dump(metadata, f)


    
if __name__=="__main__":
    with open('mini_project\\names_ids.json', 'r') as f:
        data = json.load(f)
    face_id = AttendanceDetection(img_path="pre_processed_data", weight_path="mini_project\\facenet_model\\facenet_keras_weights.h5", output_path="mini_project\\post_processed_data",data=data)
    face_id.face_detection() #MTCNN
    face_id.face_recognition() #FaceNet
