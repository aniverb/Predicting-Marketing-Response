import pandas as pd
import numpy as np
import time, sys

NULL_SYMBOL = ['NA', 'na', 'NaN', 'nan', 'NULL', 'null', 'Nan', '', '[]', '-1']


def open_data(filename):
    with open(filename, 'r') as f:
        df = pd.read_csv(filename, header=0, index_col=False, true_values=['true'], na_values=NULL_SYMBOL, false_values=['false'])
        drop = []
        for col in df:
            # Drop cols with high NA usage
            if 100 * float(df[col].count()) / len(df[col]) < 50:
                df.drop(col, 1, inplace=True)
                drop.append(col)
        
        keep = []
        for col in df:
            if col == 'ID' or col == 'target' or df[col].dtype == bool:
                pass
            elif np.issubdtype(df[col].dtype, np.number):
                # Fill NA's with median for all numeric types
                df[col].fillna(df[col].median(), inplace=True)
            elif df[col].dtype == object:
                df[col].fillna(max(set(list(df[col])), key=list(df[col]).count), inplace=True)
                if len(df[col].value_counts(sort=False)) > 25:
                    df.drop(col, 1, inplace=True)
                    continue

            else:
                df.drop(col, 1, inplace=True)
                continue

            if len(df[col].value_counts(sort=False)) == 1:
                df.drop(col, 1, inplace=True)


        keep.append(df)
        df = pd.concat(keep, axis=1)

        dup = []
        for i,lab in enumerate(df):
            a = list(df[lab].data)
            for blab in df.columns[:i]:
                if a == list(df[blab].data):
                    dup.append(lab)

        print dup
        df.drop(dup, 1, inplace=True)


        return df, drop


    

if __name__ == '__main__':
    if len(sys.argv) < 2:
        print "No data file provided."
    out = sys.argv[1] + '.clean'
    if len(sys.argv) > 2:
        out = sys.argv[2]

    df, drop  = open_data(sys.argv[1])
    print "\n Writing to: %s" % out
    df.to_csv(out, index=False)




