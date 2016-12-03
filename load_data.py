import pandas as pd
import sys
from matplotlib import pyplot

NULL_SYMBOL = ['NA', 'na', 'NaN', 'nan', 'NULL', 'null', 'Nan']
THRESH = 80


def get_type(col):
    try:
        return pd.to_datetime(col, format='%d%b%y:%H:%M:%S')
    except:
        pass
    try:
        return pd.to_numeric(col)
    except:
        pass

    check_r = col.data[0][0]
    for rec in col.data:
        if rec not in NULL_SYMBOL:
            check_r = rec[0]
            break

    if check_r == '[':
        ret = pd.Series([[]]*len(col))
        for i,r in enumerate(col):
            r = r[1:-1]
            s = r.split(',')
            for it in s:
                ret[i].append(it)   
        return ret  

    return col

def open_data(filename):
    with open(filename, 'r') as f:
        df = pd.read_csv(filename, dtype=object, header=0, index_col=0, true_values=['true'], na_values=NULL_SYMBOL, false_values=['false'])
        df = df.apply(lambda x: get_type(x))
        drop = []
        for col in df:
            pct_na = 100 * float(df[col].count()) / len(df[col])
            if pct_na < THRESH:
                df.drop(col, 1)
                drop.append(col)
        return df, drop


    

if __name__ == '__main__':
    a = [[1, '2', 'true', 'false', '[]', '[1,2]', '12SEP12:00:00:00', 'string']]
    df = pd.DataFrame(a)
    df = df.replace({'true' : True, 'false' : False})
    df = df.apply(lambda x: get_type(x))
    for col in df.columns:
        x = get_type(df[col])

    df,drop = open_data('/Data/Data Mining/train.csv')
    print "%d cols dropped" % len(drop)




