# %%
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split

feature_table = pd.read_csv('feature_table_0.csv')

labels = {}
with open('arxiv.label', 'r') as f:
    for line in f.readlines():
        label = line[:-1].split('\t')
        labels[label[0]] = int(label[1])

train_features = pd.DataFrame(columns=feature_table.columns)
for p in labels.keys() :
    train_features = train_features.append(feature_table.loc[feature_table['pattern'] == p])

train_labels = []
for p in train_features['pattern']:
    train_labels.append(labels[p])
# train_features['label'] = train_labels

rf = RandomForestRegressor(n_estimators = 1000, random_state = 42)
train_features = train_features.drop('pattern', axis=1)
rf.fit(train_features, train_labels)

# test the model trained
test_features = pd.DataFrame(columns=feature_table.columns)
test_phrases = []
with open('test_phrases.txt', 'r') as f:
    test_phrases = eval(f.read())
for tp in test_phrases:
    test_features = test_features.append(feature_table.loc[feature_table['pattern'] == tp])
# test_features = test_features.drop('pattern', axis=1)
predict = rf.predict(test_features.drop('pattern', axis=1))
test_features['predict'] = predict
result = list(test_features.sort_values('predict', ascending=False)['pattern'])
# %%
