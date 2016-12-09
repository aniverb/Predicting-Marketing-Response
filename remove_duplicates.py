import pandas as pd
import numpy as np
import os

def main():
    train = pd.read_csv(os.path.expanduser('~/Downloads/train_split/train_a.csv.clean'))
    col_num = train.shape[1]
    col_names = list(train.columns.values)
    repeats = []
    for k in range(col_num):
        data = list(train[col_names[k]])
        for m in range(k + 1, col_num):
            if list(train[col_names[m]]) == data:
                print str(col_names[m]) + " is a repeat of " + str(col_names[k])
                repeats.append(col_names[m])
    train.drop(set(repeats), 1, inplace=True)
    train.to_csv('train_no_duplicates.csv', index=False)
main()
