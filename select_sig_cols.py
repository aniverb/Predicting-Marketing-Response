import pandas as pd
import numpy as np
import os

def main():
    train = pd.read_csv(os.path.expanduser('~/Downloads/train_split/train_a_no_dup'))
    sig_num = pd.read_csv(os.path.expanduser('~/Downloads/sig_numer_col.csv'))
    sig_cat = pd.read_csv(os.path.expanduser('~/Downloads/catSigColFisher.csv'))
    sig_cols = list(sig_num.columns.values) + list(sig_cat['x'])
    newtrain = train[sig_cols]
    newtrain.to_csv(os.path.expanduser('~/Downloads/train_split/train_a_sig.csv'), index=False)
main()
