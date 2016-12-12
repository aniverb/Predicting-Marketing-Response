'''
python remove_duplicates <input csv> <output csv>
'''

import pandas as pd
import numpy as np
import os
import sys

# args: input output
def main():
    train = pd.read_csv(sys.argv[1], header=0, index_col = False)
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
    train.to_csv(sys.argv[2], index=False)
main()
