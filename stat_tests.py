import pandas as pd
pd.set_option('display.max_columns', None)
import numpy as np
from numba import *
from scipy.stats import *
import cProfile

#Read cleaned data, remove ID column and target column
train = pd.read_csv('C:\\Users\\aniverb\\Documents\\Grad_School\\JHU\\436 - Data Mining\\Project\\Springleaf data\\clean\\train_a_no_dup.csv')

print train.shape #(72615, 1805)
train = train.drop(train.columns[[0]], axis=1)

@jit("void(f4[:, :])")
def getCat(data):
    cols=data.shape[1]
    cat_list=np.empty((1,cols), dtype=int)
    count=0
    c=data.columns
    for i in range(cols):
        if ((data[c[i]].nunique())<=25):
            cat_list[0,count]=i
            count+=1
    return cat_list[0,0:count-1] #b/c target is last col

cat_cols_ix=getCat(train)
cat_cols_ix=cat_cols_ix.tolist() #indicies

target = train.iloc[:,train.shape[1]-1] #Create separate "target" vector
#catcols = train.ix[:, train.apply(lambda x: x.nunique()) <= 25] #too slow
catcols = train[cat_cols_ix] # Matrix of categorical variables with 25 or fewer categories
print catcols.shape #(72615, 770)

catcolsNames=catcols.columns.values
trainNames=train.columns.values
trainWilc=train
nonCate=[i for i in trainNames if i not in catcolsNames]
trainWilc=trainWilc[nonCate]
print trainWilc.shape #(72615, 1035)

@jit("void(f4[:, :])")
def getNum(data):
    cols=data.shape[1]
    cat_list=np.empty((1,cols), dtype=int)
    count=0
    c=data.columns
    for i in range(cols):
        if ((data[c[i]].dtype) == np.int64) | ((data[c[i]].dtype)  == np.float64):
            cat_list[0,count]=i
            count+=1
    return cat_list[0,0:count]

trainWilc_ix=getNum(trainWilc)
trainWilc_ix=trainWilc_ix.tolist()
trainWilc=trainWilc[trainWilc_ix]
print trainWilc.shape #(72615, 1035)

#wilcoxon test
@jit
def mw_test(data, i):
    group1 = data[:,-1] == 0
    group1 = data[group1, i]
    group2 = data[:,-1] == 1
    group2 = data[group2, i]
    p_value = mannwhitneyu(group1, group2)[1]
    return p_value

@jit("void(f4[:, :])")
def getSigNumFeat(data):
    data=np.array(data)
    cols=data.shape[1]
    ix_list=np.empty((1,cols), dtype=int)
    count=0
    for i in range(cols-1):
        p_value=mw_test(data, i)
        if p_value<.05/(cols-1):
            ix_list[0, count] = i
            count+=1
    return ix_list[0,0:count]

#cProfile.run('getSigNumFeat(trainWilc.ix[:, [0,1039]])')
#0.02 seconds

numSigCol_ix=getSigNumFeat(trainWilc)
print len(numSigCol_ix) #992 (out of 1034)
numSigCol_ix=numSigCol_ix.tolist()
numSigCol=trainWilc.columns[numSigCol_ix]
numSigCol=pd.DataFrame(columns=numSigCol)
numSigCol.to_csv("sig_numer_col.csv", index=False, header=True)

# Doing chi-squared test
#pvals = []
# Crosstab between column category values and target values; calculate p-val
#for i in range(catcols.shape[1]):
#    obs = pd.crosstab(index=catcols.iloc[:, i], columns=target)
#    pvals.append((catcols.columns[i], chi2_contingency(obs)[1]))

#cProfile.run('pd.crosstab(index=catcols.iloc[:, 0], columns=target)')
#chi_test=pd.crosstab(index=catcols.iloc[:, 0], columns=target)
#cProfile.run('chi2_contingency(chi_test)[1]')
#.028 sec

# All p-values
#pvals = pd.DataFrame(pvals, columns=['Feature', 'p-value'])
#catSigCol=pvals.loc[pvals.iloc[:, 1] < 0.05]
#print len(catSigCol) #685 (out of 770)
#catSigCol=pd.DataFrame(columns=catSigCol['Feature'])
#catSigCol.to_csv("sig_cat_col.csv", index=False, header=True)
