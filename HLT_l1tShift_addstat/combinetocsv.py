import pandas as pd
import numpy as np
from functools import reduce

# write to csv?
dflist = []
keepcol = [ 'run_number', 'label']
for i in range(1,5):
    fn = 'HLTPhysics{}.parquet'.format(i)
    df = pd.read_parquet(fn)
    notscore = []
    for col in df.columns:
        if ('score' not in col) and (col not in keepcol):
            notscore.append(col)
    df = df.drop(columns=notscore)
    print(df.shape)
    dflist.append(df)

df_merged = reduce(lambda  left,right: pd.merge(left,right,on=['run_number', 'label'], how='outer'), dflist)

df_merged.to_csv('L1T_HLTPhysics.csv')
