import pandas as pd
import lightgbm as lgb
import numpy as np
import pickle
from sklearn.metrics import confusion_matrix, precision_score, recall_score, roc_auc_score
import random
from datetime import datetime, timedelta
# %%
print('loading Data...')
df=pd.read_csv('''_data.csv')
df=df.replace(np.NaN,-1)
# %%
col=[]
for c in df.columns:
    col.append(c.split('.')[1])
df.columns=col
# %%
print('categorical to numeric...')
cat=[]
with open('./maps/cat.pkl', 'rb') as fin:
    cat = pickle.load(fin)
for c in cat:
    print(c)
    mapp={}
    with open('./maps/'+str(c+'OH')+'.pkl', 'rb') as fin:
        mapp = pickle.load(fin)
    mk=list(mapp.keys())
    uni=df[c].unique()
    diff=list(set(uni)-set(mk))
    mv=list(mapp.values())
    for d in diff:
        mapp[d]=random.choice(mv)
    df[c+'OH']=df[c].apply(lambda x: mapp[x])
# %%
print('dropping columns...')
to_drop=[
]
msisdn=df['msisdn_a'].values
df=df.drop(to_drop,axis=1)
print('columns dropped...')
# %%
id=df['id'].values
#y_test=df['active'].values
x_train=df.drop(['id'],axis=1)
with open('./maps/x_cols.pkl', 'rb') as fin:
    x_cols=pickle.load(fin)
x_train=x_train[x_cols]
# %%
