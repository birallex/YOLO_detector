import os
import sys
from PIL import Image
import numpy as np
from keras_yolo3.yolo import YOLO

class TagDetector:
    def __init__(self, confidence_level = 0.4, save_img = False, gpu = 0, postfix = "_processed"):
        self.anchors_path = os.path.join("keras_yolo3", "model_data", "yolo_anchors.txt")
        self.model_folder = os.path.join("Data", "Model_Weights")
        self.model_weights = os.path.join(self.model_folder, "trained_weights_final.h5")
        self.model_classes = os.path.join(self.model_folder, "data_classes.txt")
        self.confidence = confidence_level
        self.save_img = save_img
        self.gpu = gpu
        self.postfix = postfix
        self.yolo = YOLO(
            **{
                "model_path": self.model_weights,
                "anchors_path": self.anchors_path,
                "classes_path": self.model_classes,
                "score": self.confidence,
                "gpu_num": 0,
                "model_image_size": (416, 416),
            }
        )

    def detect_from_file(self, img_path, save_img_path=""):
        try:
            image = Image.open(img_path)
            if image.mode != "RGB":
                image = image.convert("RGB")
        except:
            print("File Open Error! Try again!")
            return None

        prediction, new_image = self.yolo.detect_image(image)

        img_out = self.postfix.join(os.path.splitext(os.path.basename(img_path)))

        if self.save_img:
            new_image.save(os.path.join(save_img_path, img_out))

        return prediction

    
    def detect_from_array(self, image, save_img_path=""):
        prediction, new_image = self.yolo.detect_image(image)
        if self.save_img:
            new_image.save(os.path.join(save_img_path, "img"))
        return prediction


if __name__ == "__main__":
    detector = TagDetector(save_img=True)
    img_path = "1.jpg"
    model_prediction = detector.detect_from_file(img_path)
    prediction = model_prediction[0]
    print("\tx: {}\n \ty: {} \n \tw: {} \n \th: {} \n \tconf: {}".format(
            prediction[0], prediction[1], 
            prediction[2], prediction[3], 
            prediction[5]
        )
    )


