import pandas as pd
import numpy as np
import re
import datetime
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score
from matplotlib import pyplot as plt

df = pd.read_csv("profiles.csv")

def filter_essay_text(text):
    # replace html entities and newlines with space
    filtered_text = re.sub(r'<br />\n|\n|&amp;|&quot;', ' ', text)
    # eliminate non-word characters
    filtered_text = re.sub(r'[^\w^ ]', '', filtered_text)
    return filtered_text

def get_essays_word_list(essays):
    word_list = []
    for essay in essays:
        if (type(essay) == str):
            word_list += filter_essay_text(essay).split()
    return word_list

def get_concatenated_and_filtered_essays(essays, word_combination_level=1):
    concatenated_text = ''
    word_list = get_essays_word_list(essays)
    word_list_len = len(word_list)
    for i in range(0, word_list_len):
        if ((i + word_combination_level) < word_list_len):
            for j in range(0, word_combination_level):
                concatenated_text += word_list[i+j]
            concatenated_text += ' '
    # remove the extra space from the end of the string
    concatenated_text = concatenated_text[:-1]
    return concatenated_text

def get_concatenated_and_filtered_essays_from_row(row, word_combination_level=1):
    essays = []
    for i in range(0, 10):
        key = 'essay' + str(i)
        essays.append(row[key])
    return get_concatenated_and_filtered_essays(essays, word_combination_level)

def print_metrics(c_matrix):
    accuracy = (c_matrix[0][0] + c_matrix[1][1]) / (c_matrix[0][0] + c_matrix[0][1] + c_matrix[1][0] + c_matrix[1][1])
    precision = c_matrix[0][0] / (c_matrix[0][0] + c_matrix[0][1])
    recall = c_matrix[0][0] / (c_matrix[0][0] + c_matrix[1][0])
    f1 = 2*precision*recall/(precision+recall)
    print("confusion matrix: ", c_matrix)
    print("accuracy: ", accuracy)
    print("precision: ", precision)
    print("recall: ", recall)
    print("f1: ", f1)

print("Concatenating and filtering essays...")
df['filtered_essays'] = df.apply(lambda row: get_concatenated_and_filtered_essays_from_row(row), axis=1)
df['sex_int'] = df.apply(lambda row: 0 if row['sex'] == 'm' else 1, axis=1)

# This random_state variable is reused for the various scikit_learn model constructors.
random_state = 50

print("Running train_test_split...")
train_data, test_data, train_labels, test_labels = train_test_split(df['filtered_essays'], df['sex_int'], test_size=0.2, random_state=random_state)

print("Computing count vectors...")
counter = CountVectorizer()
counter.fit(train_data)
train_counts = counter.transform(train_data)
test_counts = counter.transform(test_data)
print("Finished computing count vectors.")

