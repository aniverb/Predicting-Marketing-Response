import pandas as pd
pd.set_option('display.max_columns', None)
import numpy as np
from sklearn import ensemble
from sklearn.model_selection import GridSearchCV

#Read cleaned data, remove ID column and target column
dev = pd.read_csv('C:\\Users\\aniverb\\Documents\\Grad_School\\JHU\\436 - Data Mining\\Project\\Springleaf data\\clean\\dev_200.csv')
parameters = {'n_estimators': np.arange(10, 300, 10), 'max_features':np.arange(1, 30, 2)} #['log2', 'sqrt'] 7.8 and 15
rf = ensemble.RandomForestClassifier(random_state=121216, warm_start=True)
clf = GridSearchCV(rf, parameters, cv=10, scoring='accuracy', verbose=3)
clf.fit(dev.ix[:,2:], dev.target)
cvRes=clf.cv_results_
sorted(cvRes.keys())
clf.best_score_
clf.best_params_

