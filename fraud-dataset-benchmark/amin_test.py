#%%
from fdb.datasets import FraudDatasetBenchmark
import lightgbm as lgb
from sklearn.metrics import average_precision_score
#%%
# all_keys = ['fakejob', 'vehicleloan', 'malurl', 'ieeecis', 'ccfraud', 'fraudecom', 'twitterbot', 'ipblock'] 
key = 'vehicleloan'

obj = FraudDatasetBenchmark(
    key=key,
    load_pre_downloaded=False,  # default
    delete_downloaded=False,  #################################### default
    add_random_values_if_real_na = { 
        "EVENT_TIMESTAMP": True, 
        "LABEL_TIMESTAMP": True,
        "ENTITY_ID": True,
        "ENTITY_TYPE": True,
        "ENTITY_ID": True,
        "EVENT_ID": True
        } # default
    )
print("########## Object key: ##########")
print(obj.key)

print('########## Train set: ########## ')
print(obj.train.head())
print(len(obj.train.columns))
print(obj.train.shape)
#%%
num_round = 10
# obj.train.drop(columns=['EVENT_ID', 'ENTITY_ID', 'EVENT_TIMESTAMP', 'ENTITY_TYPE'], inplace=True)
# obj.train.drop(columns=['LABEL_TIMESTAMP'], inplace=True)
train_data = lgb.Dataset(obj.train, label=obj.train['EVENT_LABEL'])
param = {'num_leaves': 31, 'objective': 'binary', 'n_estimators':1000}
param['metric'] = 'auc'
bst = lgb.train(param, train_data, num_round,)


print('Test set: ')
print(obj.test.head())
print(obj.test.shape)

#%%
# obj.test.drop(columns=['EVENT_ID', 'ENTITY_ID', 'EVENT_TIMESTAMP', 'ENTITY_TYPE'], inplace=True)
# obj.test.drop(columns=['LABEL_TIMESTAMP'], inplace=True)

ypred = bst.predict(obj.test, predict_disable_shape_check=True)

print('Test scores')
print(obj.test_labels.head())
print(obj.test_labels['EVENT_LABEL'].value_counts())
print(obj.train['EVENT_LABEL'].value_counts(normalize=True))
print('=========')

# %%
obj.eval(ypred)
# %%
