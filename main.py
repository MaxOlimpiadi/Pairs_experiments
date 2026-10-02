# -*- coding: utf-8 -*-
"""
Created on Tue Sep 29 14:36:44 2026

@author: MAKSIM
"""

import os
from sentence_transformers import SentenceTransformer

from config import (
    DATA_FOLDER,
    PREPARED_DATA_FOLDER,
    TRAIN_PAIRED_FILE_NAME, 
    DEV_TEST_PAIRED_FILE_NAME, 
    TRAIN_PREPARED_FILE_NAME,
    DEV_PREPARED_FILE_NAME,
    TEST_PREPARED_FILE_NAME,
    DO_PAIRS, 
    DO_TRANSFORM_PAIRS,
    DELETE_OLD_REPORT,
    LOG_FILE_NAME,
    CREATE_SPLITS,
    ENCODING_MODEL 
)

from data_prep import create_folders, get_train_pairs_data, get_dev_pairs_data, prepare_train_data, prepare_dev_test_data, load_data, create_splits
from experiments import do_svm_experiments, do_transformer_experiments
    

def main():
    create_folders()
    if DO_PAIRS:
        get_train_pairs_data()
        get_dev_pairs_data()
    if DO_TRANSFORM_PAIRS:
        train_df = prepare_train_data(TRAIN_PAIRED_FILE_NAME)
        dev_df, test_df = prepare_dev_test_data(DEV_TEST_PAIRED_FILE_NAME)
    if DELETE_OLD_REPORT and os.path.exists(LOG_FILE_NAME): # delete the entire log before a new series of experiments
        os.remove(LOG_FILE_NAME) 
    if CREATE_SPLITS:
        train_df = load_data(os.path.join(DATA_FOLDER, PREPARED_DATA_FOLDER, TRAIN_PREPARED_FILE_NAME))
        dev_df = load_data(os.path.join(DATA_FOLDER, PREPARED_DATA_FOLDER, DEV_PREPARED_FILE_NAME))
        test_df = load_data(os.path.join(DATA_FOLDER, PREPARED_DATA_FOLDER, TEST_PREPARED_FILE_NAME))
        create_splits(train_df, dev_df, test_df)

    embedding_model = SentenceTransformer(ENCODING_MODEL)
    do_svm_experiments('embeddings', embedding_model)
    do_svm_experiments('tf-idf')
    do_transformer_experiments()  
        
    

if __name__ == '__main__':
    main()