import numpy as np
import pandas as pd
% pylab inline

#Import train_a, remove ID column and target column
train = pd.read_csv('train_a.csv') 
train = train.drop(train.columns[[0]], axis=1)
target = train.iloc[:,train.shape[1]-1] #Create separate "target" vector
train.drop(labels='target', axis=1, inplace=True)

#Nole's code to remove columns with >50% NAs
find_nans = pd.isnull(train).sum() > 0.5*train.shape[0]
keep_cols = []
for k in list(find_nans.index):
    if find_nans[k] == False:
        keep_cols.append(k)
        
train = train[keep_cols]

from scipy.stats import itemfreq
from scipy.stats import chi2_contingency

#Matrix of categorical variables with 25 or fewer categories
catcols = train.ix[:, train.apply(lambda x: x.nunique()) <= 25]

pvals = []

#Crosstab between column category values and target values; calculate p-val
#using chi-square test
for i in range(catcols.shape[1]):
    obs = pd.crosstab(index = catcols.iloc[:,i], columns = target)
    pvals.append((catcols.columns[i], chi2_contingency(obs)[1]))
    
#All p-values
pvals = pd.DataFrame(pvals, columns = ['Feature', 'p-value'])

print pvals.loc[pvals.iloc[:,1]<0.05]
