import numpy as np
import pandas as pd
from numba import *
from scipy.stats import chi2_contingency
from scipy.stats import mannwhitneyu
pd.set_option('display.max_columns', None)
import cProfile

#Read cleaned data, remove ID column and target column
train = pd.read_csv('C:\\Users\\aniverb\\Documents\\Grad_School\\JHU\\436 - Data Mining\\Project\\Springleaf data\\clean\\train_a_cleaned.csv')
'''DtypeWarning: Columns (9,10,11,12,13,44,182,202,205,206,208,212,215) have mixed types. Specify dtype option on import or set low_memory=False.
  data = self._reader.read(nrows)'''

train.shape #(72615, 1910)
rows=train.shape[0]

for i in [9,10,11,12,13,44,182,202,205,206,208,212,215]:
    categ = train.ix[:,i].value_counts()
    print {"Feature index": i, "% Not Empty": round(sum(categ) / float(rows), 5), "Category Counts": categ}

exclude = train[[9, 10, 11, 12, 13, 44, 182, 202, 205, 206, 208, 212, 215]].columns.values  # temporarily remove b/c of mixed type

for i in exclude:
    del train[i]

train.shape #(72615, 1897)

train = train.drop(train.columns[[0, 1]], axis=1)

@jit("void(f4[:, :])")
def getCat(data):
    cols=data.shape[1]
    cat_list=np.empty((1,cols), dtype=int)
    count=0
    c=data.columns
    for i in range(cols):
        if ((data[c[i]].nunique())<=25) & ((data[c[i]].nunique())>1):
            cat_list[0,count]=i
            count+=1
    return cat_list[0,0:count-1] #b/c target is last col

cat_cols_ix=getCat(train)
cat_cols_ix=cat_cols_ix.tolist() #indicies

target = train.iloc[:,train.shape[1]-1] #Create separate "target" vector
#catcols = train.ix[:, train.apply(lambda x: x.nunique()) <= 25] #too slow
catcols = train[cat_cols_ix]
catcols.shape #(72615, 787)

catcolsNames=catcols.columns.values
trainNames=train.columns.values
trainWilc=train
nonCate=[i for i in trainNames if i not in catcolsNames]
trainWilc=trainWilc[nonCate]
trainWilc.shape #(72615, 1108)

@jit("void(f4[:, :])")
def getNum(data):
    cols=data.shape[1]
    cat_list=np.empty((1,cols), dtype=int)
    count=0
    c=data.columns
    for i in range(cols):
        if ((data[c[i]].dtype) == np.int64) | ((data[c[i]].dtype)  == np.float64) & ('' not in list(data[c[i]])) & ((data[c[i]].nunique())>1):
            cat_list[0,count]=i
            count+=1
    return cat_list[0,0:count]

trainWilc_ix=getNum(trainWilc)
trainWilc_ix=trainWilc_ix.tolist()
trainWilc=trainWilc[trainWilc_ix]
trainWilc.shape #(72615, 1060)


@jit 
def mw_test(data, i):
    group1 = data.target == 0
    group1 = (data.ix[group1]).ix[:, i]
    group2 = data.target == 1
    group2 = (data.ix[group2]).ix[:, i]
    p_value = mannwhitneyu(group1, group2)[1]
    return p_value

cProfile.run('mw_test(trainWilc, 0)')	#test
	
@jit("void(f4[:, :])")
def getSigNumFeat(data):
    cols=data.shape[1]
    ix_list=np.empty((1,cols), dtype=int)
    count=0
    for i in range(cols-1):
        p_value=mw_test(data, i)
        if p_value<.05:
            ix_list[0, count] = i
            count+=1
    return ix_list[0,0:count]

test=getSigNumFeat(trainWilc.ix[:,range(50, 100)+[1059]])
trainWilcTest=trainWilc.ix[:, [0,1,1057]]
cProfile.run('getSigNumFeat(trainWilcTest)')
getSigNumFeat(trainWilcTest)

trainWilcT_ix=getSigNumFeat(trainWilc)
# Matrix of categorical variables with 25 or fewer categories
train.drop(labels='target', axis=1, inplace=True)
pvals = []

# Crosstab between column category values and target values; calculate p-val
# using chi-square test
for i in range(catcols.shape[1]):
    obs = pd.crosstab(index=catcols.iloc[:, i], columns=target)
    pvals.append((catcols.columns[i], chi2_contingency(obs)[1]))

# All p-values
pvals = pd.DataFrame(pvals, columns=['Feature', 'p-value'])

print pvals.loc[pvals.iloc[:, 1] < 0.05]

