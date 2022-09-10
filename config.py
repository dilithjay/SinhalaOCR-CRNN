import os

DATA_DIR = os.environ.get('DATA_DIR', "D:/DocumentAI/SinhalaOCR/data/word_imgs")
TARGET_JSON_PATH = os.environ.get('TARGET_JSON_PATH', 'D:/DocumentAI/SinhalaOCR/word_img_text.json')
BATCH_SIZE = os.environ.get('BATCH_SIZE', 8)
IMAGE_WIDTH = os.environ.get('IMAGE_WIDTH', 150)
IMAGE_HEIGHT = os.environ.get('IMAGE_HEIGHT', 32)
NUM_WORKERS = os.environ.get('NUM_WORKERS', 2)
EPOCHS = os.environ.get('NUM_WORKERS', 100)
DEVICE = os.environ.get('DEVICE', 'cpu')
MODEL_SAVE_PATH = os.environ.get('MODEL_SAVE_PATH', "D:/DocumentAI/captcha-recognition-pytorch/models/model.pt")
