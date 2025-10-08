#====================================
#--- Importing Libraries and Paths --
#====================================

from ultralytics import YOLO
import os
import cv2
import numpy as np
from paths import *

model_path = YOLO_MODEL_PATH
test_images_dir = INPUT_IMAGE_PATH
mask_save_dir = OUTPUT_MASKS_PATH
bbox_save_dir = OUTPUT_BBOX_PATH
conf_score = CONF
os.makedirs(mask_save_dir, exist_ok=True)
os.makedirs(bbox_save_dir, exist_ok=True)


#====================================
#---------- YOLO Detection ----------
#====================================

yolo_model = YOLO(model_path)
results = yolo_model.predict(source=test_images_dir, conf=conf_score, imgsz=640)

for result in results:
    img = result.orig_img.copy()
    mask_img = np.ones_like(img) * 255

    if result.masks is not None:
        mask_array = result.masks.data.cpu().numpy()
        for m in mask_array:
            m_resized = cv2.resize(m, (img.shape[1], img.shape[0]))
            m_binary = (m_resized > 0.5).astype(np.uint8) * 255
            m_binary = 255 - m_binary
            m_binary_3ch = cv2.merge([m_binary, m_binary, m_binary])
            mask_img = cv2.bitwise_and(mask_img, m_binary_3ch)

    img_with_boxes = img.copy()
    for box in result.boxes.xyxy.cpu().numpy():
        x1, y1, x2, y2 = map(int, box)
        cv2.rectangle(img_with_boxes, (x1, y1), (x2, y2), (0,255,0), 2)

    mask_save_path = os.path.join(mask_save_dir, os.path.basename(result.path).replace(".jpg", "_mask.png"))
    cv2.imwrite(mask_save_path, mask_img)

    bbox_save_path = os.path.join(bbox_save_dir, os.path.basename(result.path))
    cv2.imwrite(bbox_save_path, img_with_boxes)

print("All test image results saved.")