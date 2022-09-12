import os

DATA_DIR = os.environ.get('DATA_DIR', "D:/DocumentAI/SinhalaOCR/data/word_imgs")
TARGET_JSON_PATH = os.environ.get('TARGET_JSON_PATH', 'D:/DocumentAI/SinhalaOCR/word_img_text.json')
BATCH_SIZE = int(os.environ.get('BATCH_SIZE', 8))
IMAGE_WIDTH = int(os.environ.get('IMAGE_WIDTH', 150))
IMAGE_HEIGHT = int(os.environ.get('IMAGE_HEIGHT', 32))
NUM_WORKERS = int(os.environ.get('NUM_WORKERS', 2))
EPOCHS = int(os.environ.get('EPOCHS', 100))
LR = float(os.environ.get("LR", 1e-4))
DEVICE = os.environ.get('DEVICE', 'cpu')
MODEL_SAVE_PATH = os.environ.get('MODEL_SAVE_PATH', "D:/DocumentAI/captcha-recognition-pytorch/models/model.pt")
