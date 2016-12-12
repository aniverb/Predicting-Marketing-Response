import pandas as pd
import numpy as np
from sklearn.model_selection import GridSearchCV
from sklearn.linear_model import LogisticRegression
import sys

'''
python log_reg numer cat idtarget
'''

df_array = []
dev_numer = pd.read_csv(sys.argv[1], index_col=False, header=0)
dev_cat = pd.read_csv(sys.argv[2], index_col=False, header=0)
idtarget = pd.read_csv(sys.argv[3], index_col=False, header=0)

df_array.append(pd.get_dummies(dev_cat, columns=dev_cat.columns))
df_array.append(dev_numer)
df = pd.concat(df_array, axis=1)

params = {'C': np.linspace(0.001,1.0,20)}
lr = LogisticRegression(random_state=1111, warm_start=True, solver='sag', penalty='l2')
clf = GridSearchCV(lr, params, scoring='accuracy', verbose=3, n_jobs=4)
clf.fit(df, idtarget['target'])
print clf.best_score_
print clf.best_params_