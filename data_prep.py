import os
import pandas as pd
import re
from collections import namedtuple

from config import (
    DATA_FOLDER, 
    PAIRS_DATA_FOLDER,
    PREPARED_DATA_FOLDER,
    ORIGINAL_DATA_FOLDER,
    TRAIN_FILE_NAME,
    DEV_FILE_NAME,
    MIN_FUNNY_THRESHOLD,
    TRAIN_PAIRED_FILE_NAME,
    DEV_PAIRED_FILE_NAME,
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



def get_pair_data_instance(row: namedtuple) -> tuple[str, str, str]:
    text = row.original
    match_container = re.search(r"<([^<>]*?)/>", row.original)
    if match_container is None:
        return None, None, None
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
    save_data(df_paired_dev, os.path.join(DATA_FOLDER, PAIRS_DATA_FOLDER, DEV_PAIRED_FILE_NAME))



def get_prepared_data(file_name, mode) -> pd.DataFrame:
    file_path = os.path.join(DATA_FOLDER, PAIRS_DATA_FOLDER, file_name)
    df = load_data(file_path)
    
    positive_instances = pd.DataFrame({
        "text": df['humorous'],
        "label": 1  # pandas broadcasting gonna work and make this unique label for all the texts 
    })
    
    negative_instances = pd.DataFrame({
        "text": df["original"],
        "label": 0
    })
    
    final_df = pd.concat([positive_instances, negative_instances], ignore_index = True, axis = 0)
    final_df = final_df.sample(frac = 1, random_state = 42).reset_index(drop = True)
    save_path = os.path.join(DATA_FOLDER, PREPARED_DATA_FOLDER, f'{mode}_prepared.csv')
    
    save_data(final_df, save_path)
    
    return final_df