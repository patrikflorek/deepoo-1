from pathlib import Path

IMG_SIZE = (224, 224)
BATCH_SIZE = 4
EPOCHS = 50
NUM_CLASSES = 1

DATA_DIR = Path(__file__).resolve().parent.parent / 'data' / 'normalized'
TRAIN_IMG_DIR = DATA_DIR / 'train' / 'images'
TRAIN_MASK_DIR = DATA_DIR / 'train' / 'masks'
TEST_IMG_DIR = DATA_DIR / 'test' / 'images'
TEST_MASK_DIR = DATA_DIR / 'test' / 'masks'
MODEL_DIR = Path(__file__).resolve().parent.parent / 'models'
MODEL_DIR.mkdir(parents=True, exist_ok=True)
