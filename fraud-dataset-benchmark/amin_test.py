from fdb.datasets import FraudDatasetBenchmark

# all_keys = ['fakejob', 'vehicleloan', 'malurl', 'ieeecis', 'ccfraud', 'fraudecom', 'twitterbot', 'ipblock'] 
key = 'ieeecis'

obj = FraudDatasetBenchmark(
    key=key,
    load_pre_downloaded=False,  # default
    delete_downloaded=True,  # default
    add_random_values_if_real_na = { 
        "EVENT_TIMESTAMP": True, 
        "LABEL_TIMESTAMP": True,
        "ENTITY_ID": True,
        "ENTITY_TYPE": True,
        "ENTITY_ID": True,
        "EVENT_ID": True
        } # default
    )
print(obj.key)

print('Train set: ')
print(obj.train.head())
print(len(obj.train.columns))
print(obj.train.shape)

print('Test set: ')
print(obj.test.head())
print(obj.test.shape)

print('Test scores')
print(obj.test_labels.head())
print(obj.test_labels['EVENT_LABEL'].value_counts())
print(obj.train['EVENT_LABEL'].value_counts(normalize=True))
print('=========')
