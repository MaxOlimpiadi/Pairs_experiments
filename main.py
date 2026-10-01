# -*- coding: utf-8 -*-
"""
Created on Tue Sep 29 14:36:44 2026

@author: MAKSIM
"""

import pandas as pd
import os


from config import (
    TRAIN_PAIRED_FILE_NAME, 
    DEV_PAIRED_FILE_NAME, 
    DO_PAIRS, 
    DO_TRANSFORM_PAIRS 
)

from data_prep import create_folders, get_train_pairs_data, get_dev_pairs_data, get_prepared_data

    

def main():
    create_folders()
    if DO_PAIRS:
        get_train_pairs_data()
        get_dev_pairs_data()
    if DO_TRANSFORM_PAIRS:
        get_prepared_data(TRAIN_PAIRED_FILE_NAME, 'train')
        get_prepared_data(DEV_PAIRED_FILE_NAME, 'dev')

        
    

if __name__ == '__main__':
    main()