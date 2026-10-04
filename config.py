import os


# Folders, files routes:
DATA_FOLDER = 'data'
ORIGINAL_DATA_FOLDER = 'original'
PAIRS_DATA_FOLDER = 'pairs'
PREPARED_DATA_FOLDER = 'prepared'
TRAIN_FILE_NAME = 'train.csv'
DEV_FILE_NAME = 'dev.csv'
TRAIN_PAIRED_FILE_NAME = 'train_paired_data.csv'
DEV_TEST_PAIRED_FILE_NAME = 'dev_paired_data.csv'
TRAIN_PREPARED_FILE_NAME = 'train_prepared.csv'
DEV_PREPARED_FILE_NAME = 'dev_prepared.csv'
TEST_PREPARED_FILE_NAME = 'test_prepared.csv'
LOG_FILE_NAME = "experiments_log.csv"

# Splits:
SPLIT_FOLDER_PATH = 'split'
TRAIN_SLICES_FOLDER_PATH = os.path.join(SPLIT_FOLDER_PATH, 'train_slices')
TRAIN_SPLIT_FILE = 'train.csv'
DEV_SPLIT_FILE = 'dev.csv'
TEST_SPLIT_FILE = 'test.csv'

# Test prediction results:
TEST_RESULTS_FOLDER = 'test'
TEST_RESULTS_FILE = 'predictions.csv'

# Transformer Model params:
MODEL_NAME = "bert-base-uncased"
SAVE_BEST_PATH = 'best_model.pth'
BATCH_SIZE = 16
MAX_LENGTH = 512
EPOCHS = 10
LEARNING_RATE = 5e-5

# SVM params:
ENCODING_MODEL = 'sentence-transformers/all-MiniLM-L6-v2' # for embeddings
KERNEL = 'rbf'
C = 1.0
GAMMA = 'scale'
MAX_FEATURES = 1000

# Experiment:
TRAIN_SPLIT_SIZES = (25, 50, 100, 200, 300, 500, 700, 900, 1100, 1300, 1500, 1750, 2000)
RANDOM_SEEDS = (7, 10, 35)

# Min MeanGrade:
MIN_FUNNY_THRESHOLD = 1.2 # minimal mean funniness score (meanGrade) for texts to select from the original dataset

# Flags:
DO_PAIRS = False # if true then each text from the original dataset will be transformed to pair <original text>, <edited text> with all the other fields remaining 
DO_TRANSFORM_PAIRS = False  # if ture then the paired data will be transofrmed to well-known classification format: <text>, <label>
CREATE_SPLITS = True  # If you need to split the dataset into train, validation, and test sets. Otherwise, we load all three parts directly from the corresponding files.
DELETE_OLD_REPORT = True  # if you need to delete the old log file before running experiments


