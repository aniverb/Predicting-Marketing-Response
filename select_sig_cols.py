import pandas as pd
import numpy as np
import os

def main():
    train = pd.read_csv(os.path.expanduser('~/Downloads/train_split/train_a_no_dup.csv'))
    sig_num = pd.read_csv(os.path.expanduser('~/Downloads/sig_numer_col.csv'))
    sig_cat = pd.read_csv(os.path.expanduser('~/Downloads/catSigColFisher.csv'))
    sig_cols = list(sig_num.columns.values) + list(sig_cat['x'])
    sig_cols = ['ID'] + sig_cols
    sig_cols.append('target')
    newtrain = train[sig_cols]
    newtrain.to_csv(os.path.expanduser('~/Downloads/train_split/train_a_sig.csv'), index=False)
main()
