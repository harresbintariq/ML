import pandas as pd
import lightgbm as lgb
import numpy as np
import pickle
from sklearn.metrics import confusion_matrix, precision_score, recall_score, roc_auc_score
import random
from datetime import datetime, timedelta
# %%
print('loading Data...')
df=pd.read_csv('''_topred_data.csv')
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
x_test=df.drop(['id'],axis=1)
with open('./maps/x_cols.pkl', 'rb') as fin:
    x_cols=pickle.load(fin)
x_test=x_test[x_cols]
# if(x_cols==x_test.columns):
#     print('good cols...')
# else:
#     x_test=x_test[[x_cols]]
# corr_dict={}
# for c in x_test.columns:
#     a=np.corrcoef(y_test,x_test[c])
#     corr_dict[c]=a[0,1]
# from collections import OrderedDict
# from operator import itemgetter
# d = OrderedDict(sorted(corr_dict.items(), key=itemgetter(1)))
#y_test_inv=1-y_test
p_test=np.zeros((x_test.shape[0],5))
print('predicting...')
for count in range(1,6):
    print(count)    
    with open('./models/modn'+str(count)+'.pkl', 'rb') as fin:
              model = pickle.load(fin)
    pred = model.predict(x_test, num_iteration=model.best_iteration)
    p_test[:,count-1] =1-pred#1-(np.exp(pred) - 1.0).clip(0,1)
p_test_avg=np.mean(p_test,axis=1)
thr=0.95
p_test_bin=(p_test_avg>=thr).astype(int)
sub=pd.DataFrame()
sub['access_method_id']=id
msisdn=['92'+str(x).split('.')[0] for x in msisdn]
sub['msisdn']=msisdn
sub['churn']=p_test_bin
sub['insert_datetime']=str(datetime.now())
sub['reference_date']=str(datetime.now()-timedelta(days=2)).split(' ')[0]
file_name='''_7day_pred.csv'
sub.to_csv('~/''_7day_dor/'+file_name, index=False)
# save as csv then load in appropriate table
# %%
# for i in np.arange(0,1,0.05):
#     p_test_bin=(p_test_avg>i)
#     print(i)
#     cm=confusion_matrix(y_test_inv,p_test_bin)
#     print('Accuracy: ',(cm[0,0]+cm[1,1])/np.sum(cm))
#     print(precision_score(y_test_inv,p_test_bin))
#     print(recall_score(y_test_inv,p_test_bin))
#     print(cm)
            
