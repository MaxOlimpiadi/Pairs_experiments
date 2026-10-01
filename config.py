
# Folders, files routes:
DATA_FOLDER = 'data'
ORIGINAL_DATA_FOLDER = 'original'
PAIRS_DATA_FOLDER = 'pairs'
PREPARED_DATA_FOLDER = 'prepared'
TRAIN_FILE_NAME = 'train.csv'
DEV_FILE_NAME = 'dev.csv'
TRAIN_PAIRED_FILE_NAME = 'train_paired_data.csv'
DEV_PAIRED_FILE_NAME = 'dev_paired_data.csv'

# Min MeanGrade:
MIN_FUNNY_THRESHOLD = 1.2 # minimal mean funniness score (meanGrade) for texts to select from the original dataset

# Flags:
DO_PAIRS = True # TRUE = each text from the original dataset will be transformed to pair <original text>, <edited text> with all the other fields remaining 
DO_TRANSFORM_PAIRS = True