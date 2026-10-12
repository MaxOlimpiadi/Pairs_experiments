# -*- coding: utf-8 -*-
"""
Created on Sun Oct 11 19:17:07 2026

@author: zidan
"""

import pandas as pd
import os
from bertopic import BERTopic
from sentence_transformers import SentenceTransformer
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction import text
from transformers import pipeline

from config import (
    DATA_FOLDER, 
    PAIRS_DATA_FOLDER,
    TRAIN_PAIRED_FILE_NAME,
    TOPICS,
    topic_classifier,
)

from data_prep import load_data




def get_topic(text_to_asses):  
    
    result = topic_classifier(
        text_to_asses, 
        candidate_labels = TOPICS,
        multi_label = True
    )
    
    sorted_tuples = sorted(
        zip(result["labels"], result["scores"]), 
        key = lambda x: x[1], 
        reverse = True
    )
    
    return {topic: round(score, 3) for topic, score in sorted_tuples[:3]}
    
    
    





def main():
    file_path = os.path.join(DATA_FOLDER, PAIRS_DATA_FOLDER, TRAIN_PAIRED_FILE_NAME)
    df = load_data(file_path)
    
    small_df = df[:100]
    small_df = small_df.drop(columns = ['word_to_edit', 'changed_word', 'grades', 'meanGrade'])
    small_df["topic_original"] = small_df["original"].apply(get_topic)
    small_df["topic_funny"] = small_df["humorous"].apply(get_topic)
    
    small_df.to_csv("small_df_topics.csv", index = False)
    
    #res = get_topic("President discusses new economic reforms")
    
    # for topic, score in zip(res["labels"], res["scores"]):
    #     print(f"{topic}: {score:.3f}")
    



if __name__ == '__main__':
    main()