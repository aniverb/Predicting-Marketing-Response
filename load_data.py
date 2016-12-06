import pandas as pd
import numpy as np
import time, sys
from matplotlib import pyplot

NULL_SYMBOL = ['NA', 'na', 'NaN', 'nan', 'NULL', 'null', 'Nan']
NA_THRESH = 25 # Drop col with more than NA_THRESH null vals
CAT_THRESH = 50 # Ignore categorical with more than this many categories

def update_progress(progress, status=""):
    # adapted From http://stackoverflow.com/questions/3160699/python-progress-bar
    barLength = 50 
    if isinstance(progress, int):
        progress = float(progress)
    if not isinstance(progress, float):
        progress = 0
        status = "Error: progress var must be float\r\n"
    block = int(round(barLength*progress))
    text = "\rPercent: [{0}] {1:.1f}% {2}".format( "#"*block + "-"*(barLength-block), progress*100, status)
    sys.stdout.write(text)
    sys.stdout.flush()

def open_data(filename):
    with open(filename, 'r') as f:
        df = pd.read_csv(filename, header=0, index_col=0, true_values=['true'], na_values=NULL_SYMBOL, false_values=['false'])
        drop = []
        for col in df:
            # Drop cols with high NA usage
            pct_na = 100 * float(df[col].count()) / len(df[col])
            if pct_na < NA_THRESH:
                df.drop(col, 1, inplace=True)
                drop.append(col)
        
        to_append = []
        ntot = df.shape[1]
        cnt = 0
        for col in df:
            cnt += 1
            update_progress(float(cnt)/ntot, col)
            if col == 'ID' or col == 'target':
                pass
            elif np.issubdtype(df[col].dtype, np.number):
                # Fill NA's with median for all numeric types
                df[col].fillna((df.median()))
            elif df[col].dtype == object:
                vct = df[col].value_counts(sort=False)
                # Limit categorical to limited set sizes to prevent blowup
                if len(vct) < CAT_THRESH:
                    dum = pd.get_dummies(df[col], prefix=col)
                    to_append.append(dum)
                df.drop(col, 1, inplace=True)
            else:
                # Drop all non numeric and categorical data
                df.drop(col, 1, inplace=True)

        to_append.append(df)
        df = pd.concat(to_append, axis=1)

        return df, drop


    

if __name__ == '__main__':
    if len(sys.argv) < 2:
        print "No data file provided."
    out = sys.argv[1] + '.clean'
    if len(sys.argv) > 2:
        out = sys.argv[2]

    df, drop  = open_data(sys.argv[1])
    print "\n Writing to: %s" % out
    df.to_csv(out)




