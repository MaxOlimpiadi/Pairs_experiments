import os
import pandas as pd
import re
from collections import namedtuple
from sklearn.model_selection import train_test_split
import random

from config import (
    DATA_FOLDER, 
    PAIRS_DATA_FOLDER,
    PREPARED_DATA_FOLDER,
    ORIGINAL_DATA_FOLDER,
    TRAIN_FILE_NAME,
    DEV_FILE_NAME,
    MIN_FUNNY_THRESHOLD,
    TRAIN_PAIRED_FILE_NAME,
    DEV_TEST_PAIRED_FILE_NAME,
    RANDOM_SEEDS,
    TRAIN_SLICES_FOLDER_PATH,
    TRAIN_SPLIT_SIZES,
    SPLIT_FOLDER_PATH,
    DEV_SPLIT_FILE,
    TEST_SPLIT_FILE, 
    TRAIN_PREPARED_FILE_NAME,
    DEV_PREPARED_FILE_NAME,
    TEST_PREPARED_FILE_NAME,
    
)




def create_folders():
    pairs_folder_path = os.path.join(DATA_FOLDER, PAIRS_DATA_FOLDER)
    prepared_folder_path = os.path.join(DATA_FOLDER, PREPARED_DATA_FOLDER)
    for folder_path in [pairs_folder_path, prepared_folder_path]:
        os.makedirs(folder_path, exist_ok = True)


def load_data(file_path):
    header_df = pd.read_csv(file_path, nrows = 0) # read only header of the .csv file
    dict_types = {}
    if "grades" in header_df.columns:
        dict_types["grades"] = str
    df = pd.read_csv(file_path, dtype = dict_types)
    return df


def save_data(df, file_path):
    df.to_csv(file_path, index = False)


# TODO: maybe to add some logic like "at least 3 grades are more than 1"
def check_grades(grades: str):
    if grades == '0':
        return False
    else: 
        return True


def get_pair_data_instance(row) -> tuple[str, str, str]:
    text = row.original
    match_container = re.search(r"<([^<>]*?)/>", row.original)
    if match_container is None:
        return '', '', '' #marking was not found
    marked = match_container.group(0)
    word_original = match_container.group(1)
    original_text = text.replace(marked, word_original, 1)
    humorous_text = text.replace(marked, row.edit, 1)
    return (
        original_text,
        humorous_text,
        word_original
    )
    

def get_train_pairs_data():
    file_path = os.path.join(DATA_FOLDER, ORIGINAL_DATA_FOLDER, TRAIN_FILE_NAME)
    df = load_data(file_path)
    df_selected = df[ df['meanGrade'] >= MIN_FUNNY_THRESHOLD]
    print(df_selected.head(20))
    
    ids, original_texts, humorous_texts, original_words, edit_words, grades, mean_grades = [], [], [], [], [], [], []
    no_marking_count = 0 # count of rows with no marking
    
    for row in df_selected.itertuples(index = False):
        if not check_grades(str(row.grades)):
            continue
        
        original_text, humorous_text, word_original = get_pair_data_instance(row) 
        
        if not humorous_text:
            no_marking_count += 1
            continue
        
        ids.append(row.id)
        original_texts.append(original_text)
        humorous_texts.append(humorous_text)    # without marking symbols < />
        original_words.append(word_original)
        edit_words.append(row.edit)
        grades.append(row.grades)
        mean_grades.append(row.meanGrade)
        
    df_paired_train = pd.DataFrame(
        {
            "id": ids,
            "original": original_texts,
            "humorous": humorous_texts,
            "word_to_edit": original_words,
            "changed_word": edit_words,
            "grades": grades,
            "meanGrade": mean_grades
        }
    )
    
    print(f'Total count of rows in old train data: {len(df)}')
    print(f'Total count of rows in processed train data: {len(df_paired_train)}')
    print(f'Count of no-marking rows in old train data: {no_marking_count}\n')
    
    save_data(df_paired_train, os.path.join(DATA_FOLDER, PAIRS_DATA_FOLDER, TRAIN_PAIRED_FILE_NAME))
        

def get_dev_pairs_data():
    file_path = os.path.join(DATA_FOLDER, ORIGINAL_DATA_FOLDER, DEV_FILE_NAME)
    df = load_data(file_path)
    
    ids, original_texts, humorous_texts, original_words, edit_words = [], [], [], [], []
    no_marking_count = 0 # count of rows with no marking
    for row in df.itertuples(index = False):
        original_text, humorous_text, word_original = get_pair_data_instance(row)
        if not humorous_text:
            no_marking_count += 1
            continue
        
        ids.append(row.id)
        original_texts.append(original_text) # without marking symbols < />
        humorous_texts.append(humorous_text)
        original_words.append(word_original)
        edit_words.append(row.edit)
   
    df_paired_dev = pd.DataFrame(
        {
            "id": ids,
            "original": original_texts,
            "humorous": humorous_texts,
            "word_to_edit": original_words,
            "changed_word": edit_words
        }
    )
    print(f'Total count of rows in old dev data: {len(df)}')
    print(f'Total count of rows in processed dev data: {len(df_paired_dev)}')
    print(f'Count of no-marking rows in old dev data: {no_marking_count}')
    save_data(df_paired_dev, os.path.join(DATA_FOLDER, PAIRS_DATA_FOLDER, DEV_TEST_PAIRED_FILE_NAME))




def pairs_to_classification(df_pairs) -> pd.DataFrame:
    #preparing data for forming final dataframe:
    positive_instances = pd.DataFrame({
        "text": df_pairs['humorous'].unique(),
        "label": 1  # pandas broadcasting gonna work and assign label 1 to every row 
    })
    
    negative_instances = pd.DataFrame({
        "text": df_pairs["original"].unique(),
        "label": 0
    })
    
    # combined final dataframe:
    result_df = (
        pd.concat([positive_instances, negative_instances], ignore_index = True, axis = 0)
    )
    
    #shuffle:
    result_df = result_df.sample(frac = 1, random_state = 42).reset_index(drop = True)
    
    print(f'\n\t\t{len(positive_instances)} positive and {len(negative_instances)} negative')
    
    return result_df
    
    
def prepare_train_data(file_name):
    file_path = os.path.join(DATA_FOLDER, PAIRS_DATA_FOLDER, file_name)
    df_pairs = load_data(file_path) 
    print('\n Train subset:')
    train_df = pairs_to_classification(df_pairs)
    
    save_data(train_df, os.path.join(DATA_FOLDER, PREPARED_DATA_FOLDER, TRAIN_PREPARED_FILE_NAME))
    
    return train_df
    
    

def prepare_dev_test_data(file_name) -> tuple[pd.DataFrame, pd.DataFrame]:
    file_path = os.path.join(DATA_FOLDER, PAIRS_DATA_FOLDER, file_name)
    df = load_data(file_path)
    
    all_originals = df['original'].unique()
    
    # spliting original texts equally into dev and test:
    dev_originals, test_originals = train_test_split(
        all_originals,
        test_size = 0.5,
        random_state = 42
    )
    
    df_dev = df[ df['original'].isin(dev_originals) ]
    
    df_test = df[ df['original'].isin(test_originals) ]
      
    print('\n Dev subset:')
    final_dev_df = pairs_to_classification(df_dev)
    print('\n Test subset:')
    final_test_df = pairs_to_classification(df_test)

    save_data(final_dev_df, os.path.join(DATA_FOLDER, PREPARED_DATA_FOLDER, DEV_PREPARED_FILE_NAME))
    save_data(final_test_df, os.path.join(DATA_FOLDER, PREPARED_DATA_FOLDER, TEST_PREPARED_FILE_NAME))
    
    return final_dev_df, final_test_df



# getting a convinient structure of global train data for further slicing:
def get_samples_by_class(global_train_texts, global_train_labels, random_state):
    samples_by_class = {
        "positive": [],
        "negative": []
    }
    for t, l in zip(global_train_texts, global_train_labels):
        if l == 1:
            samples_by_class["positive"].append((t, l))
        elif l == 0:
            samples_by_class["negative"].append((t,l))  
        else: 
            raise ValueError(f"Unexpected label: {l}")
    
    # shuffle, taking care not to alter the global random state:
    rng = random.Random(random_state)
    rng.shuffle(samples_by_class["positive"])
    rng.shuffle(samples_by_class["negative"])
    
    return samples_by_class



def create_training_slice(samples_by_class, size, random_state):
    n_positive = size // 2
    n_negative = size - n_positive

    # Ensure there are enough samples of each class for the requested slice size:
    if len(samples_by_class["positive"]) < n_positive:
        raise ValueError(
            f"Not enough positive samples: "
            f"requested {n_positive}, "
            f"available {len(samples_by_class['positive'])}"
        )
    if len(samples_by_class["negative"]) < n_negative:
        raise ValueError(
            f"Not enough negative samples: "
            f"requested {n_negative}, "
            f"available {len(samples_by_class['negative'])}"
        )
    
    positive_slice_tuples = samples_by_class["positive"][:n_positive]
    negative_slice_tuples = samples_by_class["negative"][:n_negative]
    
    current_data = positive_slice_tuples + negative_slice_tuples

    rng = random.Random(random_state)
    rng.shuffle(current_data)
    
    split_texts = []
    split_labels = []
    for t, l in current_data:
        split_texts.append(t)
        split_labels.append(l)
            
    return split_texts, split_labels



def create_splits(train_df, dev_df, test_df):
    
    global_train_texts = train_df['text']
    global_train_labels = train_df['label']
    
    dev_texts = dev_df['text']
    dev_labels = dev_df['label']
    
    test_texts = test_df['text']
    test_labels = test_df['label']
    
     
    for seed in RANDOM_SEEDS:
        seed_folder_path = os.path.join(TRAIN_SLICES_FOLDER_PATH, f'Seed_{seed}')
        samples_by_class = get_samples_by_class(global_train_texts, global_train_labels, seed)
        for size in TRAIN_SPLIT_SIZES:
            train_texts, train_labels = create_training_slice(samples_by_class, size, seed)
            save_split(train_texts, train_labels, seed_folder_path, f'train_{size}.csv')

    save_split(dev_texts, dev_labels, SPLIT_FOLDER_PATH, DEV_SPLIT_FILE)
    save_split(test_texts, test_labels, SPLIT_FOLDER_PATH, TEST_SPLIT_FILE)
   
    
  
def save_split(texts, labels, folder_path, filename):
    os.makedirs(folder_path, exist_ok=True) # if the folder doesn`t exist, it will be created`
    df = pd.DataFrame(
        {
            'text': texts,
            'label': labels
        }    
    )
    full_path = os.path.join(folder_path, filename)
    df.to_csv(full_path, index = False)


