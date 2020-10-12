import pickle
import pandas as pd
from datetime import date
import numpy as np

catb_model = pickle.load(open("insurance_catb.pkl", 'rb'))
xgb_model = pickle.load(open("insurance_xgb.pkl", 'rb'))

def predict_product(sample):
      features_test = []
      columns = []
      append_features = ['P5DA', 'RIBP', '8NN1', '7POT', '66FJ', 'GYSR', 'SOP4', 'RVSZ', 'PYUQ', 'LJR9', 
      'N2MW', 'AHXO','BSTQ', 'FM3X', 'K6QO', 'QBOL', 'JWFN', 'JZ9D', 'J9JW', 'GHYX', 
      'ECY3', 'ID', 'join_date', 'sex', 'marital_status', 'branch_code', 'occupation_code', 'occupation_category_code',
      'birth_year']

      for v in append_features:
            features_test.append([sample[v]])
            columns.append(np.array([v]))

      features_test = np.concatenate(features_test, axis=1)
      columns = np.concatenate(np.array(columns))

      sample = pd.DataFrame(features_test)
      sample.columns = columns

      ##Features similar to train data feature engineering are generated
      sample['dayofweek'] = pd.to_datetime(sample['join_date']).dt.dayofweek
      sample['day'] = sample['join_date'].apply(lambda x: float(x.split('/')[0]) if (x == x) else np.nan)
      sample['month'] = sample['join_date'].apply(lambda x: float(x.split('/')[1]) if (x == x) else np.nan)
      sample['year'] = sample['join_date'].apply(lambda x: float(x.split('/')[2]) if (x == x) else np.nan)
      sample['passed_years'] = float(date.today().year - pd.to_datetime(sample['join_date']).dt.year)
      sample['date_diff'] = sample['year'] - sample['birth_year']
      sample['join_date'] = (pd.to_datetime(sample['join_date']) - pd.to_datetime("01/01/1970")).dt.days
      sample['join_date'] = sample['join_date'].astype(int)
      sample['birth_year'] = sample['birth_year'].astype(int)

      ##One hot encoding is performed
      sample = pd.get_dummies(sample, columns=['sex', 'marital_status', \
                                           'branch_code','occupation_code',\
                                           'occupation_category_code','month','year','passed_years'])


      ##Based on the cat features, we will create the other features required for prediction and mark the categories other than inputs as 0
      cat_features = ['sex_F', 'sex_M', 'marital_status_F', 'marital_status_M',
             'marital_status_S', 'marital_status_U', 'marital_status_others',
             'branch_code_1X1H', 'branch_code_30H5', 'branch_code_49BM',
             'branch_code_748L', 'branch_code_94KC', 'branch_code_E5SW',
             'branch_code_O67J', 'branch_code_UAOD', 'branch_code_XX25',
             'branch_code_ZFER', 'branch_code_others', 'occupation_code_0B60',
             'occupation_code_0FOI', 'occupation_code_0KID', 'occupation_code_0OJM',
             'occupation_code_0ZND', 'occupation_code_2A7I', 'occupation_code_31JW',
             'occupation_code_8CHJ', 'occupation_code_93OJ', 'occupation_code_9F96',
             'occupation_code_BIA0', 'occupation_code_BP09', 'occupation_code_BPSA',
             'occupation_code_E2MJ', 'occupation_code_HSI5', 'occupation_code_JBJP',
             'occupation_code_QZYX', 'occupation_code_SST3', 'occupation_code_UJ5T',
             'occupation_code_others', 'occupation_category_code_56SI',
             'occupation_category_code_90QI', 'occupation_category_code_AHH5',
             'occupation_category_code_JD7X', 'occupation_category_code_L44T',
             'occupation_category_code_T4MS', 'month_1.0', 'month_2.0', 'month_3.0',
             'month_4.0', 'month_5.0', 'month_6.0', 'month_7.0', 'month_8.0',
             'month_9.0', 'month_10.0', 'month_11.0', 'month_12.0', 'year_2010.0',
             'year_2011.0', 'year_2012.0', 'year_2013.0', 'year_2014.0',
             'year_2015.0', 'year_2016.0', 'year_2017.0', 'year_2018.0',
             'year_2019.0', 'year_2020.0', 'passed_years_0.0', 'passed_years_1.0',
             'passed_years_2.0', 'passed_years_3.0', 'passed_years_4.0',
             'passed_years_5.0', 'passed_years_6.0', 'passed_years_7.0',
             'passed_years_8.0', 'passed_years_9.0', 'passed_years_10.0']
      # print(len(cat_features))

      ##The features order has to follow the same order as that of train data so we generate new sample from sample
      new_sample = sample[sample.columns[:27]]

      for c in cat_features:
            if c not in sample.columns:
                  new_sample[c] = 0
                  new_sample[c] = new_sample[c].astype('uint8')
            else:
                  new_sample[c] = sample[c]

      modified_cols = ['branch_code', 'occupation_code', 'marital_status']
      for s in sample.columns:
            for c in modified_cols:
                  if c in s and  s not in cat_features:
                        new_sample[c + "_others"] = 1
                        new_sample[c + "_others"] = new_sample[c + "_others"].astype('uint8')

      # print(new_sample.columns[:35])
      predicts = []
      predicts.append(catb_model.predict_proba(new_sample.drop(columns=['ID','day'], axis=1)))

      features_to_int = ['P5DA', 'RIBP', '8NN1', '7POT', '66FJ', 'GYSR', 'SOP4', 'RVSZ', 'PYUQ', 'LJR9', 'N2MW', 'AHXO', 'BSTQ', 'FM3X', 'K6QO', 'QBOL', 'JWFN', 'JZ9D', 'J9JW', 'GHYX', 'ECY3', 'birth_year', 'date_diff']
      new_sample[features_to_int] = new_sample[features_to_int].astype('int')

      predicts.append(xgb_model.predict_proba(new_sample.drop(columns=['ID','day'], axis=1)))


      products = new_sample.columns[:21]

      ##Both predictions are averaged to get the final output
      y_test = pd.DataFrame(np.mean(predicts, axis=0))
      y_test.columns = products

      return y_test

if __name__ == '__main__':
      test = pd.read_csv("../Test.csv")
      sample = test.truncate(before=0, after=0)
      resp = predict_product(sample)
      print(resp)
