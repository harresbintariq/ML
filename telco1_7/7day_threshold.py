import pandas as pd
import lightgbm as lgb
import numpy as np
from sklearn.model_selection import StratifiedKFold
import pickle
from sklearn.metrics import confusion_matrix, precision_score, recall_score, roc_auc_score
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
df=pd.read_csv('''_data.csv')
df=df.replace(np.NaN,-1)
# %%
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
df=df.drop(cat,axis=1)
# %%
id=df['id'].values
y_train=df['active'].values
x_train=df.drop(['active','id'],axis=1)
# corr_dict={}
# for c in x_train.columns:
#     a=np.corrcoef(y_train,x_train[c])
#     corr_dict[c]=a[0,1]
# from collections import OrderedDict
# from operator import itemgetter
# d = OrderedDict(sorted(corr_dict.items(), key=itemgetter(1)))
count=1
folds=5
skf=StratifiedKFold(n_splits=folds,random_state=87,shuffle=False)
for k, (train_idx, test_idx) in enumerate(skf.split(x_train, y_train)):
    print(count)    
    with open('./models/mod'+str(count)+'.pkl', 'rb') as fin:
              model = pickle.load(fin)
    pred = model.predict(x_train.iloc[test_idx], num_iteration=model.best_iteration)
    p_test =1-(np.exp(pred) - 1.0).clip(0,1)
    count+=1
    y_train_inv=1-y_train
    churn_rate=np.sum(y_train_inv)/len(y_train_inv)
    idx=sorted(range(len(p_test)), key=lambda j: p_test[j])[-100000:-1]
    top_decile_pred_rate=np.sum(y_train_inv[idx])/len(idx)
    lift=top_decile_pred_rate/churn_rate
    print('lift: ', lift)
    for i in np.arange(0,1,0.02):
        p_test_bin=(p_test>i)
        print(i)
        cm=confusion_matrix(y_train_inv[test_idx],p_test_bin)
        print('Accuracy: ',(cm[0,0]+cm[1,1])/np.sum(cm))
        print(precision_score(y_train_inv[test_idx],p_test_bin))
        print(recall_score(y_train_inv[test_idx],p_test_bin))
        print(cm)
            
