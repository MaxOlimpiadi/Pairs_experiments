# -*- coding: utf-8 -*-
"""
Created on Tue Sep 29 14:36:44 2026

@author: MAKSIM
"""

import pandas as pd
import os
import re


DATA_FOLDER = 'data'
TRAIN_FILE_NAME = 'train.csv'
DEV_FILE_NAME = 'dev.csv'



def load_data(file_path):
    df = pd.read_csv(file_path, dtype = {"grades": str})
    return df


def save_data(df, file_path):
    df.to_csv(file_path, index = False)


# TODO: maybe to add some logic like "at least 3 grades are more than 1"
def check_grades(grades: str):
    if grades == '0':
        return False
    else: 
        return True
    

def get_text_pairs_data(df: pd.DataFrame) -> pd.DataFrame:
    df_selected = df[ df['meanGrade'] >= 1.2]
    print(df_selected.head(20))
    
    ids, original_texts, humorous_texts, original_words, edit_words, grades, mean_grades = [], [], [], [], [], [], []
    no_marking_count = 0 # count of rows with no marking
    for row in df_selected.itertuples(index = False):
        if not check_grades(str(row.grades)):
            continue
        text = row.original
        match_container = re.search(r"<([^<>]*?)/>", row.original)
        if match_container is None:
            no_marking_count += 1
            continue
        marked = match_container.group(0)
        word = match_container.group(1)
        
        ids.append(row.id)
        original_texts.append(text.replace(marked, word, 1))
        humorous_texts.append(text.replace(marked, row.edit, 1))
        original_words.append(word)
        edit_words.append(row.edit)
        grades.append(row.grades)
        mean_grades.append(row.meanGrade)
        
    df_paired_data = pd.DataFrame(
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
    
    print(f'Total count of rows in old data: {len(df)}')
    print(f'Total count of rows in processed data: {len(df_paired_data)}')
    print(f'Count of no-marking rows in old data: {no_marking_count}')
    
    return df_paired_data
        
        
    

def main():
    file_path = os.path.join(DATA_FOLDER, TRAIN_FILE_NAME) 
    df = load_data(file_path)
    df_paired_data = get_text_pairs_data(df)
    save_data(df_paired_data, os.path.join(DATA_FOLDER, 'train_paired_data.csv'))
    
    
    
    

    
    

        
    



if __name__ == '__main__':
    main()