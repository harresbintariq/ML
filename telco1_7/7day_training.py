himport pandas as pd
import lightgbm as lgb
import numpy as np
from sklearn.model_selection import StratifiedKFold
import pickle
# %%gini
def gini(actual, pred, cmpcol = 0, sortcol = 1):
    assert( len(actual) == len(pred) )
    all = np.asarray(np.c_[ actual, pred, np.arange(len(actual)) ], dtype=np.float)
    all = all[ np.lexsort((all[:,2], -1*all[:,1])) ]
    totalLosses = all[:,0].sum()
    giniSum = all[:,0].cumsum().sum() / totalLosses
    
    giniSum -= (len(actual) + 1) / 2.
    return giniSum / len(actual)
 
def gini_normalized(a, p):
    return gini(a, p) / gini(a, a)

def gini_lgb(preds, dtrain):
    y = list(dtrain.get_label())
    score = gini(y, preds) / gini(y, y)
    return 'gini', score, True
# %%
print('loading data...')
df=pd.read_csv('''_data.csv')
df=df.replace(np.NaN,-1)
# %%
print('Columns drops and Categoricals...')
col=[]
for c in df.columns:
    col.append(c.split('.')[1])
df.columns=col
# %%
df=df.drop(['msisdn','access_method_id','msisdn_a','msisdn_prefix'], axis=1)
to_drop=[]
count=0
for c in df.columns:
    if(len(df[c].unique())==1):
        to_drop.append(c)
        print(c)
        count+=1
print(count)
df=df.drop(to_drop,axis=1)
cat=[]
for c in df.columns:
    if(str(df[c].dtype)=='object'):
        print(c)
        cat.append(c)
        uni=df[c].unique()
        mapp={}
        count=0
        for v in uni:
            mapp[v]=count
            count+=1
        df[c+'OH']=df[c].apply(lambda x: mapp[x])
        with open('./maps/'+str(c+'OH')+'.pkl', 'wb') as fout:
            pickle.dump(mapp, fout)
with open('./maps/cat.pkl', 'wb') as fout:
    pickle.dump(cat, fout)
df=df.drop(cat,axis=1)
# %%
id=df['id'].values
y_train=df['active'].values
x_train=df.drop(['active','id'],axis=1)
x_cols=x_train.columns
with open('./maps/x_cols.pkl', 'wb') as fout:
    pickle.dump(x_cols, fout)
# corr_dict={}
# for c in x_train.columns:
#     a=np.corrcoef(y_train,x_train[c])
#     corr_dict[c]=a[0,1]
# from collections import OrderedDict
# from operator import itemgetter
# d = OrderedDict(sorted(corr_dict.items(), key=itemgetter(1)))
params = {
          'boosting_type': 'gbdt',
          'objective': 'binary',
          'num_leaves': 64,#31
          'learning_rate': 0.01,
          'max_bin': 255,
          'reg_alpha': 1, 
          'reg_lambda': 1,
          'min_data_in_leaf':5000,
          'feature_fraction':0.7,#0.6
          'bagging_freq':1,
          'bagging_fraction':0.8,#0.6
          }
print('training model....')
count=0
folds=5
skf=StratifiedKFold(n_splits=folds,random_state=87,shuffle=False)
g=0
for k, (train_idx, test_idx) in enumerate(skf.split(x_train, y_train)):
    model = lgb.train(params, lgb.Dataset(x_train.iloc[train_idx].copy(), label=y_train[train_idx]), 1500, 
                   lgb.Dataset(x_train.iloc[test_idx].copy(), label=y_train[test_idx]), verbose_eval=10,
                   early_stopping_rounds=100)
    pred = model.predict(x_train.iloc[test_idx], num_iteration=model.best_iteration)
    p_test = (np.exp(pred) - 1.0).clip(0,1)
    print('Iteration Gini: ', gini_normalized(y_train[test_idx],p_test))
    g+=gini_normalized(y_train[test_idx],p_test)
    count+=1
    print('Iterations Done: ', count)
    imp=model.feature_importance()
    imp_idx=sorted(range(len(imp)), key=lambda k: imp[k])
    most_imp=x_train.columns[imp_idx[-1]]
    with open('./models/modnc'+str(count)+'.pkl', 'wb') as fout:
          pickle.dump(model, fout)
g_avg=g/folds
print('Avg Gini: ', g_avg)
            


