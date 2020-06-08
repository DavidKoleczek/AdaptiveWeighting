from datetime import datetime, timedelta
import random

import pandas as pd
import numpy as np
import lightgbm as lgb
import xgboost as xgb
from xgboost import XGBRegressor
import tensorflow as tf
from sklearn import *

from mk_datasets import *

def add_days_to_string(string, days):
    temp = datetime.strptime(string, '%Y-%m-%d') + timedelta(days = days)
    return datetime.strftime(temp, '%Y-%m-%d')


# Uncomment this is for some reason the precomputed data does not exist.
# Note it will take a while to run (few hours)
"""
## Precomputing FE1 datasets
import datetime 
from mk_datasets import feature_engineer1

curr_date = '2017-03-02'
while curr_date != '2017-04-13':
    temp = feature_engineer1(False, curr_date)
    temp.to_csv('data/feature_engineered/fe1_' + curr_date + '.csv', index = False)
    curr_date = add_days_to_string(curr_date, 7)
"""



lgbm_learning_params_1 = { #NOTE: Comments to the right are default parameter values for reference
    'verbose' : 0,
    'device_type' : 'cpu',
    'gpu_use_dp' : False,
    'force_row_wise' : True,
    'seed' : random.randrange(2147483646),
    
    'objective' : 'poisson',
    'metric' : 'l2',
    'num_iterations' : 1200,
    'boosting' : 'gbdt', #gbdt
    'learning_rate' : 0.08, #0.1
    'num_leaves' : 31, #31
    'tree_learner' : 'serial', #serial

    'max_depth' : -1, #-1
    'min_data_in_leaf' : 30, #20
    'min_sum_hessian_in_leaf' : 0.001, #0.001
    'bagging_fraction' : .8, #1
    'bagging_freq' : 1, #0
    'feature_fraction' : 1, #1
    'max_delta_step' : 0, #0
    'lambda_l1' : 0, #0
    'lambda_l2' : 0.1, #0
    'min_gain_to_split' : 0, #0

    'min_data_per_group': 100, #100
    'max_cat_threshold' : 32, #32
    'cat_l2' : 10, #10
    'cat_smooth' : 10, #10
    'max_cat_to_onehot' : 4, #4
    'top_k' : 20, #20

    #Dart Params
    'drop_rate' : 0.1, #0.1
    'max_drop' : 25, #50
    'skip_drop' : .5, #0.5
    'xgboost_dart_mode' : False, 
    'uniform_drop' : False, #False

    'max_bin' : 255, #255
}

lgbm_learning_params_2 = { #NOTE: Comments to the right are default parameter values for reference
    'verbose' : -1,
    'device_type' : 'cpu',
    'gpu_use_dp' : False,
    'force_row_wise' : True,
    'seed' : random.randrange(2147483646),
    
    'objective' : 'regression',
    'metric' : 'l2',
    'num_iterations' : 20000,
    'boosting' : 'gbdt', #gbdt
    'learning_rate' : 0.007, #0.1
    'num_leaves' : 31, #31
    'tree_learner' : 'serial', #serial

    'max_depth' : 5, #-1
    'min_data_in_leaf' : 80, #20
    'min_sum_hessian_in_leaf' : 0.001, #0.001
    'bagging_fraction' : .8, #1
    'bagging_freq' : 1, #0
    'feature_fraction' : 1, #1
    'max_delta_step' : 0, #0
    'lambda_l1' : 0, #0
    'lambda_l2' : 0, #0
    'min_gain_to_split' : 0, #0

    'min_data_per_group': 100, #100
    'max_cat_threshold' : 32, #32
    'cat_l2' : 10, #10
    'cat_smooth' : 10, #10
    'max_cat_to_onehot' : 4, #4
    'top_k' : 20, #20

    #Dart Params
    'drop_rate' : 0.1, #0.1
    'max_drop' : 25, #50
    'skip_drop' : .5, #0.5
    'xgboost_dart_mode' : False, 
    'uniform_drop' : False, #False

    'max_bin' : 255, #255
}


xgb_params = { # https://xgboost.readthedocs.io/en/latest/parameter.html
    
    'objective' : 'reg:squarederror', # reg:squarederror
    'eval_metric' : 'rmse',
    'seed' : random.randrange(2147483646),
    
    'booster' : 'gbtree', # gbtree
    'verbosity' : 1, # 1
    'validate_parameters' : True, # False
    
    'learning_rate' : 0.02, # 0.3
    'min_split_loss' : 0, # 
    'max_depth' : 11, # 6
    'min_child_weight' : 16, # 1
    'max_delta_step' : 0, # 0, 0.7 in possion
    
    'subsample' : 0.9, # 1
    'sampling_method' : 'uniform', # uniform
    'colsample_bytree' : 1, # 
    'colsample_bylevel' : 0.1, # 1
    'colsample_bynode' : 1, # 1
    
    'lambda' : 1, # 1, L2
    'alpha' : 0, # 0, L1 
    
    'tree_method' : 'gpu_hist', # auto
    'gpu_id' : 0,
    'sketch_eps' : 0.03, # 0.03
    'scale_pos_weight' : 1, # 1
    
    'grow_policy' : 'depthwise', # depthwise
    'max_leaves' : 0, # 0
    'max_bin' : 256, #256
    
    'num_parallel_tree' : 1, #1
}


train_end = '2017-03-02'
test_start = '2017-03-04'
test_period_end = add_days_to_string(test_start, 7)
final_end = '2017-04-15'

all_preds = pd.DataFrame()

while test_start != final_end:
    df1 = pd.read_csv('data/feature_engineered/fe1_' + add_days_to_string(test_start, -2) + '.csv')
    
    df1['visitors_log1p'] = np.log1p(df1['visitors'])
    df1 = df1[(df1['is_test'] == False) & (df1['is_outlier'] == False) & (df1['was_nil'] == False)]
    df1['day_of_week'] = df1['day_of_week'].astype('category').cat.codes.astype('int16')
    df1['air_genre_name'] = df1['air_genre_name'].astype('category').cat.codes.astype('int16')

    to_drop = ['is_test', 'test_number', 'was_nil',
               'is_outlier', 'visitors_capped', 'air_area_name',
               'station_id', 'station_latitude', 'station_longitude', 'station_vincenty',
               'station_great_circle', 'visitors_capped_log1p', 
               'latitude_str', 'longitude_str']
    df1 = df1.drop(to_drop, axis = 1)
    
    x_train = df1[df1.visit_date <= train_end]
    x_train = x_train.dropna()
    y_train = x_train['visitors_log1p']
    x_train = x_train.drop(['visitors_log1p', 'visit_date', 'air_store_id', 'visitors'], axis = 1)

    x_test = df1[(df1.visit_date > test_start) & (df1.visit_date <= test_period_end)]
    x_test = x_test.fillna(0)
    y_test = x_test['visitors_log1p']
    # Save data needed to know what we are actually predicting
    pred_ids = x_test.air_store_id
    visit_dates = x_test.visit_date
    x_test = x_test.drop(['visitors_log1p', 'visit_date', 'air_store_id', 'visitors'], axis = 1)

    cat_features = ['day_of_week', 'is_holiday', 'prev_day_is_holiday',
                'next_day_is_holiday', 'air_genre_name', 'latitude', 'longitude',
                'is_weekend', 'day_of_month']
    
    xgbm_x_train = xgb.DMatrix(data = x_train, label=y_train)
    xgbm_x_test = xgb.DMatrix(data = x_test)
    xgbm = xgb.train(params = xgb_params, dtrain = xgbm_x_train, num_boost_round = 700, verbose_eval = 100)
    preds = xgbm.predict(xgbm_x_test)
    pred_df = pd.DataFrame(data = {'air_store_id': pred_ids, 'visit_date': visit_dates, 'model_name': 'XGB2', 'pred': preds,})
    all_preds = pd.concat([all_preds, pred_df])
    
    lgbm_x_train = lgb.Dataset(data = x_train, label = y_train, categorical_feature = cat_features)
    lgbm = lgb.train(params = lgbm_learning_params_1, train_set = lgbm_x_train, categorical_feature = cat_features, verbose_eval = 100)
    preds = lgbm.predict(x_test)
    pred_df = pd.DataFrame(data = {'air_store_id': pred_ids, 'visit_date': visit_dates, 'model_name': 'LGBM1', 'pred': preds,})
    all_preds = pd.concat([all_preds, pred_df])
    
    lgbm_x_train = lgb.Dataset(data = x_train, label = y_train, categorical_feature = cat_features)
    lgbm = lgb.train(params = lgbm_learning_params_2, train_set = lgbm_x_train, categorical_feature = cat_features, verbose_eval = 1000)
    preds = lgbm.predict(x_test)
    pred_df = pd.DataFrame(data = {'air_store_id': pred_ids, 'visit_date': visit_dates, 'model_name': 'LGBM2', 'pred': preds,})
    all_preds = pd.concat([all_preds, pred_df])
    
    
    train, test = feature_engineer2(train_end, test_start, test_period_end)
    col = [c for c in train if c not in ['id', 'air_store_id', 'visit_date','visitors']]

    nn_model = get_nn_complete_model(train)
    nn_train, nn_test = feature_engineer2_nn(train_end, test_start, test_period_end)

    model1 = ensemble.GradientBoostingRegressor(learning_rate=0.2, random_state=random.randrange(2147483646), n_estimators=200, subsample=0.8, max_depth=10)
    model2 = neighbors.KNeighborsRegressor(n_jobs=-1, n_neighbors=4)
    model3 = XGBRegressor(learning_rate=0.02, random_state=2, n_estimators=1000, subsample=0.8, colsample_bytree=0.75, max_depth =10, reg_lambda=.25)
    model4 = get_nn_complete_model(train, hidden2_neurons=12)

    model1.fit(train[col], np.log1p(train['visitors'].values))
    model2.fit(train[col], np.log1p(train['visitors'].values))
    model3.fit(train[col], np.log1p(train['visitors'].values))

    for i in range(5):
        model4.fit(nn_train[0], nn_train[1], epochs=3, verbose=1, batch_size=256, shuffle=True, validation_split=0.15)
        model4.fit(nn_train[0], nn_train[1], epochs=8, verbose=0, batch_size=256, shuffle=True)


    preds1 = model1.predict(test[col])
    pred_df = pd.DataFrame(data = {'air_store_id': test.air_store_id, 'visit_date': test.visit_date, 'model_name': 'GBM', 'pred': preds1,})
    all_preds = pd.concat([all_preds, pred_df])

    preds2 = model2.predict(test[col])
    pred_df = pd.DataFrame(data = {'air_store_id': test.air_store_id, 'visit_date': test.visit_date, 'model_name': 'KNNR', 'pred': preds2,})
    all_preds = pd.concat([all_preds, pred_df])

    preds3 = model3.predict(test[col])
    pred_df = pd.DataFrame(data = {'air_store_id': test.air_store_id, 'visit_date': test.visit_date, 'model_name': 'XGB1', 'pred': preds3,})
    all_preds = pd.concat([all_preds, pred_df])

    preds4 = pd.Series(model4.predict(nn_test[0]).reshape(-1)).clip(0, 6.8).values
    pred_df = pd.DataFrame(data = {'air_store_id': test.air_store_id, 'visit_date': test.visit_date, 'model_name': 'NN', 'pred': preds4,})
    all_preds = pd.concat([all_preds, pred_df])
    
    
    train_end = add_days_to_string(train_end, 7)
    test_start = add_days_to_string(test_start, 7)
    test_period_end = add_days_to_string(test_period_end, 7)

    

a = all_preds.copy()
a['visit_date'] = pd.to_datetime(all_preds['visit_date'])
a = a.pivot_table(values = 'pred', index = ['air_store_id', 'visit_date'], columns = 'model_name').reset_index()
a.to_csv('data/output/all_preds.csv', index = False)