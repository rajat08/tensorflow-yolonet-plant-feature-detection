import os

DATA_PATH = 'training'
NETWORK_PATH = 'network'
CACHE_PATH = os.path.join(NETWORK_PATH, 'cache')
OUTPUT_DIR = os.path.join(NETWORK_PATH, 'output')
WEIGHTS_DIR = os.path.join(NETWORK_PATH, 'weights')
LABEL_PATH = 'Anemone_canadensis_label.txt'
WEIGHTS_FILE = None
CLASSES = ['bud', 'flower', 'fruit', 'Others', 'Others', 'Others',
           'Others', 'Others', 'Others', 'Others', 'Others', 'Others', 'Others', 'Others', 'Others', 'Others', 'Others', 'Others', 'Others', 'Others']
IMAGE_SIZE = 448
CELL_NUM = 7

CENTERS_PER_CELL = 2
ALPHA = 0.1
DISP_CONSOLE = False
OBJECT_SCALE = 2.0
NOOBJECT_SCALE = 0.5
CLASS_SCALE = 1.0
COORD_SCALE = 5.0

GPU = ''
LEARNING_RATE = 0.0001
DECAY_STEPS = 30000
DECAY_RATE = 0.1
STAIRCASE = True

BATCH_SIZE = 16
MAX_ITER = 30000
SUMMARY_ITER = 100
SAVE_ITER = 10000

THRESHOLD = 0.5  
DIST_THRESHOLD = 0.5 
