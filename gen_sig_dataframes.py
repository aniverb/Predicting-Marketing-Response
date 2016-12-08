import pandas as pd
import numpy as np
import time, sys

def get_sig(filename, cat_file, numer_file):
    sigcat = set()
    signumer = set()

    with open(cat_file, 'r') as f:
        sigcat = set(''.join(f.readlines()).strip().split(','))
    with open(numer_file, 'r') as f:
        signumer = set(''.join(f.readlines()).strip().split(','))

    with open(filename, 'r') as f:
        df = pd.read_csv(filename, header=0, index_col=False)

        catdf = []
        numerdf = []
        for col in df:
            if col in sigcat:
                catdf.append(df[col])
            elif col in signumer:
                numerdf.append(df[col])


        return pd.concat([df['ID'], df['target']], axis=1), pd.concat(catdf, axis=1), pd.concat(numerdf, axis=1)


    

if __name__ == '__main__':
    if len(sys.argv) < 2:
        print "No data file provided."

    name = sys.argv[1]
    idtarg, catdf, numerdf  = get_sig(name, sys.argv[2], sys.argv[3])
    idtarg.to_csv(name + '.idtarget', index=False)
    catdf.to_csv(name + '.cat', index=False)
    numerdf.to_csv(name + '.numer', index=False)




