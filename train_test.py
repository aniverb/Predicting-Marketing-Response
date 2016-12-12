import pandas as pd
import numpy as np
import os
import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_curve

def main():
    train = pd.read_csv(os.path.expanduser('~/Downloads/train_a_reduced_200.csv'))
    test = pd.read_csv(os.path.expanduser('~/Downloads/test_200.csv'))
    target_train = train['target']
    target_test = test['target']
    col_names = list(train.columns.values)
    cat_cols = col_names[-26:]
    train = train.ix[:,2:len(col_names)]
    train = pd.get_dummies(train, columns = cat_cols)
    LR = LogisticRegression(warm_start = True, solver = 'sag', penalty = 'l2', C = 0.211)
    LRfit = LR.fit(train, target_train)
    test = test.ix[:,2:len(col_names)]
    test = pd.get_dummies(test, columns = cat_cols)
    LRpred = LR.predict(test)
    fpr, tpr, _ = roc_curve(target_test, LRfit.decision_function(test))
    plt.plot(fpr, tpr)
    plt.show()
main()
