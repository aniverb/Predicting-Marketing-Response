import pandas as pd
pd.set_option('display.max_columns', None)
import numpy as np
from sklearn import ensemble
from sklearn.model_selection import GridSearchCV
from sklearn.preprocessing import OneHotEncoder
from sklearn.feature_extraction import DictVectorizer

#Read cleaned data, remove ID column and target column
dev = pd.read_csv('C:\\Users\\aniverb\\Documents\\Grad_School\\JHU\\436 - Data Mining\\Project\\Springleaf data\\clean\\dev_200.csv')
devDumR = pd.read_csv('C:\\Users\\aniverb\\Documents\\Grad_School\\JHU\\436 - Data Mining\\Project\\Springleaf data\\clean\\devDum.csv') #dummied in R

devCat=dev.ix[:,-26:]
devCatDict=devCat.to_dict(orient = 'records')
v = DictVectorizer(sparse=False)
devCatVec=v.fit_transform(devCatDict)
enc = OneHotEncoder()
enc.fit(devCatVec) #problem with negatives

coln=devCat.columns.values.tolist()
catDum=pd.get_dummies(devCat, prefix=coln[-26:]) #did not work as expected

newdev=pd.concat([dev.ix[:,:-26],devDumR], axis=1)
parameters = {'n_estimators': np.arange(10, 200, 10), 'max_features':['log2', 'sqrt']} #np.arange(1, 30, 2), ['log2', 'sqrt'] 7.8 and 15
rf = ensemble.RandomForestClassifier(random_state=121216, warm_start=True)
clf = GridSearchCV(rf, parameters, cv=5, scoring='accuracy', verbose=3)
clf.fit(newdev.ix[:,2:], newdev.target)
cvRes=clf.cv_results_
sorted(cvRes.keys())
clf.best_score_
clf.best_params_

