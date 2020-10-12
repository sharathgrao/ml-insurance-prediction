import pandas as pd
import numpy as np
from tqdm import tqdm
import copy
import seaborn as sns
import matplotlib.pyplot as plt
from scipy.stats import norm
from datetime import date
import pickle
from sklearn.model_selection import train_test_split
from catboost import CatBoostClassifier
from xgboost import XGBClassifier, XGBRegressor
from lightgbm import LGBMClassifier
from sklearn.model_selection import StratifiedKFold
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import LabelEncoder


##Read Train, Test data and SampleSubmission
train = pd.read_csv("./Train.csv")
test = pd.read_csv("./Test.csv")
submission = pd.read_csv("./SampleSubmission.csv")

##EDA is performed in the jupyter notebook to understand the data
##Below branch code, occupation code and marital status are modified such that least frequently occuring
##categories are combined to single category 'others'

##Replace < 1% categories among occupation code as others
ov = train['occupation_code'].value_counts()
replace_oc_cat = []

for k, v in ov.items():
  if v < 500:
    replace_oc_cat.append(k)

##Replacement to be done on both train and test data
train["occupation_code"] = train.apply(lambda x: "others" if x["occupation_code"] in replace_oc_cat else x["occupation_code"], axis= 1)
test["occupation_code"] = test.apply(lambda x: "others" if x["occupation_code"] in replace_oc_cat else x["occupation_code"], axis= 1)

##Replace < 1% categories among branch code as others
bc = train["branch_code"].value_counts()
replace_bc_cat = []

for k, v in bc.items():
  if v < 500:
    replace_bc_cat.append(k)

##Replacement to be done on both train and test data
train["branch_code"] = train.apply(lambda x: "others" if x["branch_code"] in replace_bc_cat else x["branch_code"], axis= 1)
test["branch_code"] = test.apply(lambda x: "others" if x["branch_code"] in replace_bc_cat else x["branch_code"], axis= 1)

##Replace < 1% categories among marital status as others
ms = train["marital_status"].value_counts()
replace_ms_cat = []


for k, v in ms.items():
  if v < 500:
    replace_ms_cat.append(k)

##Replacement to be done on both train and test data
train["marital_status"] = train.apply(lambda x: "others" if x["marital_status"] in replace_ms_cat else x["marital_status"], axis= 1)
test["marital_status"] = test.apply(lambda x: "others" if x["marital_status"] in replace_ms_cat else x["marital_status"], axis= 1)

#As there are on;y 2 null values in join date, the 2 rows are dropped from the train data
train = train.dropna()


##Below code converts the training data into test data format
##by making one of the products purchased by the user as 0 and add that as new entry.
##Above procedure is repeated for all purchased products of each user one by one
##And the modified product is added under the column name product_pred 
##And the data is assigned order under the column name ID2 
X_train = []
X_train_columns = train.columns

c = 0
for v in train.values:
    info = v[:8]
    binary = v[8:]
    index = [k for k, i in enumerate(binary) if i == 1]

    for i in index:
        c+=1
        for k in range(len(binary)):
            if k == i:
                binary_transformed = list(copy.copy(binary))
                binary_transformed[i] = 0
                X_train.append(list(info) + binary_transformed + [X_train_columns[8+k]] + [c])

X_train = pd.DataFrame(X_train)
X_train.columns = ['ID', 'join_date', 'sex', 'marital_status', 'birth_year', 'branch_code',
       'occupation_code', 'occupation_category_code', 'P5DA', 'RIBP', '8NN1',
       '7POT', '66FJ', 'GYSR', 'SOP4', 'RVSZ', 'PYUQ', 'LJR9', 'N2MW', 'AHXO',
       'BSTQ', 'FM3X', 'K6QO', 'QBOL', 'JWFN', 'JZ9D', 'J9JW', 'GHYX', 'ECY3', 'product_pred', 'ID2']


##As the test data is the reference format, we add just order to the data under column name ID2
##To create submission file, we store the already purchased products of each user in sampleSubmission format in 'true_values'
X_test = []
true_values = []
c = 0
for v in test.values:
    c += 1
    info = v[:8]
    binary = v[8:]
    index = [k for k, i in enumerate(binary) if i == 1]
    X_test.append(list(info) + list(binary) + [c])
    for k in test.columns[8:][index]:
        true_values.append(v[0] + ' X ' + k)

X_test = pd.DataFrame(X_test)
X_test.columns = ['ID', 'join_date', 'sex', 'marital_status', 'birth_year', 'branch_code',
       'occupation_code', 'occupation_category_code', 'P5DA', 'RIBP', '8NN1',
       '7POT', '66FJ', 'GYSR', 'SOP4', 'RVSZ', 'PYUQ', 'LJR9', 'N2MW', 'AHXO',
       'BSTQ', 'FM3X', 'K6QO', 'QBOL', 'JWFN', 'JZ9D', 'J9JW', 'GHYX', 'ECY3', 'ID2']


##The order of the features is modified to get the categorical features in the end
features_train = []
features_test = []
columns = []

append_features = ['P5DA', 'RIBP', '8NN1', '7POT', '66FJ', 'GYSR', 'SOP4', 'RVSZ', 'PYUQ', 'LJR9', 
'N2MW', 'AHXO','BSTQ', 'FM3X', 'K6QO', 'QBOL', 'JWFN', 'JZ9D', 'J9JW', 'GHYX', 
'ECY3', 'ID', 'ID2', 'join_date', 'sex', 'marital_status', 'branch_code', 'occupation_code', 'occupation_category_code',
'birth_year']
for v in append_features:
    features_train.append(X_train[v].values.reshape(-1, 1))
    features_test.append(X_test[v].values.reshape(-1, 1))
    columns.append(np.array([v]))

y_train = X_train[['product_pred']]

features_train = np.concatenate(features_train, axis=1)
features_test = np.concatenate(features_test, axis=1)
columns = np.concatenate(np.array(columns))

X_train = pd.DataFrame(features_train)
X_train.columns = columns
X_test = pd.DataFrame(features_test)
X_test.columns = columns



##split features further
##join_date converted to day, month, dayofweek and year
##passed_years is the number of years since user has purchased the insurance products
##date_diff is the age of the user when the product is purchased
X_train['day'] = X_train['join_date'].apply(lambda x: int(x.split('/')[0]) if (x == x) else np.nan)
X_train['month'] = X_train['join_date'].apply(lambda x: int(x.split('/')[1]) if (x == x) else np.nan)
X_train['year'] = X_train['join_date'].apply(lambda x: int(x.split('/')[2]) if (x == x) else np.nan)
X_train['passed_years'] = date.today().year - pd.to_datetime(X_train['join_date']).dt.year
X_train.loc[:, 'dayofweek'] = pd.to_datetime(X_train['join_date']).dt.dayofweek


X_test.loc[:, 'dayofweek'] = pd.to_datetime(X_test['join_date']).dt.dayofweek
X_test['day'] = X_test['join_date'].apply(lambda x: int(x.split('/')[0]) if (x == x) else np.nan)
X_test['month'] = X_test['join_date'].apply(lambda x: int(x.split('/')[1]) if (x == x) else np.nan)
X_test['year'] = X_test['join_date'].apply(lambda x: int(x.split('/')[2]) if (x == x) else np.nan)
X_test['passed_years'] = date.today().year - pd.to_datetime(X_test['join_date']).dt.year

X_train['join_date'] = X_train['join_date'].fillna(X_train['join_date'].mode()[0])
st_date = pd.to_datetime(X_train['join_date']).min()
X_train['join_date'] = (pd.to_datetime(X_train['join_date']) - st_date).dt.days
X_train['join_date'] = X_train['join_date'].astype(int)

X_test['join_date'] = X_test['join_date'].fillna(X_test['join_date'].mode()[0])
X_test['join_date'] = (pd.to_datetime(X_test['join_date']) - st_date).dt.days
X_test['join_date'] = X_test['join_date'].astype(int)

X_train['date_diff'] = X_train['year'] - X_train['birth_year']
X_test['date_diff'] = X_test['year'] - X_test['birth_year']


##Fill nan values with mode value 
X_train['day'] = X_train['day'].fillna(X_train['day'].mode()[0])
X_train['month'] = X_train['month'].fillna(X_train['month'].mode()[0])
X_train['year'] = X_train['year'].fillna(X_train['year'].mode()[0])
X_train['date_diff'] = X_train['date_diff'].fillna(X_train['date_diff'].mode()[0])
X_train['passed_years'] = X_train['passed_years'].fillna(X_train['passed_years'].mode()[0])
X_train['dayofweek'] = X_train['dayofweek'].fillna(X_train['dayofweek'].mode()[0])


X_test['day'] = X_test['day'].fillna(X_test['day'].mode()[0])
X_test['month'] = X_test['month'].fillna(X_test['month'].mode()[0])
X_test['year'] = X_test['year'].fillna(X_test['year'].mode()[0])
X_test['date_diff'] = X_test['date_diff'].fillna(X_test['date_diff'].mode()[0])
X_test['passed_years'] = X_test['passed_years'].fillna(X_test['passed_years'].mode()[0])
X_test['dayofweek'] = X_test['dayofweek'].fillna(X_test['dayofweek'].mode()[0])

##Label encoding the target feature
le = LabelEncoder()

le.fit(y_train.iloc[:,0])
y_train = pd.DataFrame(le.transform(y_train.iloc[:,0]))
y_train.columns = ['target']

X_train['birth_year'] = X_train['birth_year'].astype(int)
X_test['birth_year'] = X_test['birth_year'].astype(int)

all_data = X_train.append(X_test)

# Remove skewness from join_date
numeric_col = ['join_date','date_diff','birth_year']
skew = all_data[numeric_col].skew()
skew = skew[abs(skew) > 0.75]
print(skew)

all_data['join_date'] = np.square(all_data['join_date'])

X_train = all_data[:X_train.shape[0]]
X_test = all_data[-X_test.shape[0]:]

##Applying one hot encoding for categorical features
data = X_train.append(X_test)
data = pd.get_dummies(data, columns=['sex', 'marital_status', \
                                     'branch_code','occupation_code',\
                                     'occupation_category_code','month','year','passed_years'])
X_train = data[:X_train.shape[0]]
X_test = data[-X_test.shape[0]:]


##Exclude the features if the sum of the entire column in train data is 0
remove_features = []
for i in X_train.columns:
    if X_train[i].sum()==0:
        remove_features.append(i)

cat_features = list(X_train.columns[28:])

for r in remove_features:
  cat_features.remove(r)

##Initialize catboost and xgb classifiers. The parameter variations have been explored in the notebook
catb_model = CatBoostClassifier(random_state=1, max_depth=3, task_type='CPU', iterations=1900, learning_rate=0.2)
xgb_model = XGBClassifier(random_state=1, max_depth=3, tree_method='hist', n_estimators=150)

catb_model.fit(X_train.drop(columns=['ID', 'ID2','day']), y_train,verbose=100, cat_features=cat_features)

##All the features to be converted to int for xgb classifier
features = ['P5DA', 'RIBP', '8NN1', '7POT', '66FJ', 'GYSR', 'SOP4', 'RVSZ', 'PYUQ', 'LJR9', 'N2MW', 'AHXO', 'BSTQ', 'FM3X', 'K6QO', 'QBOL', 'JWFN', 'JZ9D', 'J9JW', 'GHYX', 'ECY3', 'birth_year']
X_train[features] = X_train[features].astype(int)
X_test[features] = X_test[features].astype(int)

xgb_model.fit(X_train.drop(columns=['ID', 'ID2','day']), y_train,verbose=100)

predicts = []
predicts.append(catb_model.predict_proba(X_test.drop(columns=['ID','ID2','day'], axis=1)))
predicts.append(xgb_model.predict_proba(X_test.drop(columns=['ID','ID2','day'], axis=1)))

##Predictions from both the models are averaged and output is generated.
y_test = pd.DataFrame(np.mean(predicts, axis=0))
y_test.columns = le.inverse_transform(y_test.columns)

##The prediction is converted to submission format to check the score on the test data
answer_mass = []
for i in range(X_test.shape[0]):
    id = X_test['ID'].iloc[i]
    
    for c in y_test.columns:
            answer_mass.append([id + ' X ' + c, y_test[c].iloc[i]])
            
df_answer = pd.DataFrame(answer_mass)
df_answer.columns = ['ID X PCODE', 'Label']
for i in range(df_answer.shape[0]):
    if df_answer['ID X PCODE'].iloc[i] in true_values:
        df_answer['Label'].iloc[i] = 1.0

df_answer.reset_index(drop=True, inplace=True)
df_answer.to_csv('submission3.csv', index=False)

##Models are saved to disk using pickle
pickle.dump(catb_model, open("insurance_catb.pkl", 'wb'))
pickle.dump(xgb_model, open("insurance_xgb.pkl", 'wb'))


