#====================================
#------------- README ---------------
#====================================

# Below listed are the paths of the directories that will be used for
# the images to be used in the model, and the output directories that
# will be created.

# You must change these paths according to your directory, otherwise the
# model wouldn't function properly.

# You may also change the confidence score of the model.

#====================================
#------------------------------------
#====================================

# Base Directory of the Project

BASE_DIR = "/home/user/underwater_crack_detection/"

# Model and Test Paths

YOLO_MODEL_PATH = f"{BASE_DIR}/models/best.pt"
INPUT_IMAGE_PATH = f"{BASE_DIR}/test_images"

# You need not require change the following paths

OUTPUT_MASKS_PATH = f"{BASE_DIR}/test_masks_yolo"
OUTPUT_BBOX_PATH = f"{BASE_DIR}/test_bbox_images"

# Model Confidence Score

CONF = 0.25              # You may change it according to your preference