import pandas as pd
import numpy as np
import os

def main():
    train = pd.read_csv(os.path.expanduser('~/Downloads/train_split/train_a.csv'))
    rows = train.shape[0]
    find_nans = pd.isnull(train).sum() > 0.5*rows # Find features where more than 50% are NAs to get rid of
    keep_cols = []
    for k in list(find_nans.index):
        if find_nans[k] == False:
            keep_cols.append(k)
    train = train[keep_cols] # Keep features that have less than 50% NAs
    col_names = list(train.columns.values)
    for k in range(len(col_names)):
        print k
        data = train[col_names[k]]
        if type(data[0]) == int or type(data[0]) == np.float64 or type(data[0]) == float or type(data[0]) == long:
            replacement = np.nanmedian(data) # If feature is numerical, replace with median of all instances
        elif type(data[0]) == str:
            replacement = max(set(list(data)), key=list(data).count) # If feature is categorical, replace with mode of all instances
        for m in range(len(data)):
            if pd.isnull(data[m]):
                train.set_value(m, col_names[k], replacement)
    for k in range(len(col_names)):
        data = train[col_names[k]]
        if data[0] == '[]':
            del train[col_names[k]] # Delete feature where all instances are empty lists
    train.to_csv('train_a_cleaned.csv')
main()
