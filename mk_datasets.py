import glob, re, random

import numpy as np
import pandas as pd
from scipy import optimize
from sklearn import *
from datetime import datetime
from xgboost import XGBRegressor
import tensorflow as tf

def feature_engineer1(to_csv, train_end_date):
    # train_end_date is the last day we want to train on. String of type 'yyyy-mm-dd'
    
    ### Feature engineering part 1 courtesy of https://github.com/MaxHalford/kaggle-recruit-restaurant/blob/master/Solution.ipynb, the 8th place solution

    ## Load visit data. 
    air_visit = pd.read_csv('data/air_visit_data.csv')
    air_visit.index = pd.to_datetime(air_visit['visit_date'])

    # Resampled by day in order to calculate rolling features based on time, was_nil keeps track of this. 
    air_visit = air_visit.groupby('air_store_id').apply(lambda g: g['visitors'].resample('1d').sum()).reset_index()
    air_visit['visit_date'] = air_visit['visit_date'].dt.strftime('%Y-%m-%d')
    air_visit['was_nil'] = air_visit['visitors'].isnull()
    air_visit['visitors'].fillna(0, inplace=True)


    ## Load calendar data
    date_info = pd.read_csv('data/date_info.csv')
    date_info.rename(columns={'holiday_flg': 'is_holiday', 'calendar_date': 'visit_date'}, inplace=True)
    date_info['prev_day_is_holiday'] = date_info['is_holiday'].shift().fillna(0)
    date_info['next_day_is_holiday'] = date_info['is_holiday'].shift(-1).fillna(0)


    ## Using preprocessed weather data from https://www.kaggle.com/huntermcgushion/rrv-weather-data 
    air_store_info = pd.read_csv('data/weather/air_store_info_with_nearest_active_station.csv')


    ## Test set 
    submission = pd.read_csv('data/sample_submission.csv')
    submission['air_store_id'] = submission['id'].str.slice(0, 20)
    submission['visit_date'] = submission['id'].str.slice(21)
    submission['is_test'] = True
    submission['visitors'] = np.nan
    submission['test_number'] = range(len(submission))


    ## Merging train and test to compute features in one go
    data = pd.concat((air_visit, submission.drop('id', axis='columns')))
    data['is_test'].fillna(False, inplace=True)
    data = pd.merge(left=data, right=date_info, on='visit_date', how='left')
    data = pd.merge(left=data, right=air_store_info, on='air_store_id', how='left')
    data['visitors'] = data['visitors'].astype(float)
    
    data.loc[data.visit_date > train_end_date, 'visitors'] = np.NaN


    ## Handling weather data
    weather_dfs = []

    for path in glob.glob('data/weather/stations/*.csv'):
        weather_df = pd.read_csv(path)
        weather_df['station_id'] = path.split('\\')[-1].rstrip('.csv')
        weather_dfs.append(weather_df)

    weather = pd.concat(weather_dfs, axis='rows')
    weather.rename(columns={'calendar_date': 'visit_date'}, inplace=True)
    weather['visit_date'] = pd.to_datetime(weather['visit_date'])

    # Replacing missing weather values with global daily average.
    means = weather.groupby('visit_date')[['avg_temperature', 'precipitation']].mean().reset_index()
    means.rename(columns={'avg_temperature': 'global_avg_temperature', 'precipitation': 'global_precipitation'}, inplace=True)
    weather = pd.merge(left=weather, right=means, on='visit_date', how='left')
    weather['avg_temperature'].fillna(weather['global_avg_temperature'], inplace=True)
    weather['precipitation'].fillna(weather['global_precipitation'], inplace=True)

    weather = weather[['visit_date', 'station_id', 'avg_temperature', 'precipitation']]


    ## Cleaning up data a bit
    data['visit_date'] = pd.to_datetime(data['visit_date'])
    data.index = data['visit_date']
    data.index = data.index.rename("date_idx")
    data.sort_values(['air_store_id', 'visit_date'], inplace=True)

    ## Feature Engineering - Outlier detection
    def find_outliers(series):
        return (series - series.mean()) > 2.4 * series.std()

    def cap_values(series):
        outliers = find_outliers(series)
        max_val = series[~outliers].max()
        series[outliers] = max_val
        return series

    stores = data.groupby('air_store_id')
    data['is_outlier'] = stores.apply(lambda g: find_outliers(g['visitors'])).values
    data['visitors_capped'] = stores.apply(lambda g: cap_values(g['visitors'])).values
    data['visitors_capped_log1p'] = np.log1p(data['visitors_capped'])


    ## Temporal Features
    data['is_weekend'] = data['day_of_week'].isin(['Saturday', 'Sunday']).astype(int)
    data['day_of_month'] = data['visit_date'].dt.day


    ## Exponentially Weighted Means w/ alpha parameter optimized using differential evolution
    def calc_shifted_ewm(series, alpha, adjust=True):
        return series.shift().ewm(alpha=alpha, adjust=adjust).mean()

    def find_best_signal(series, adjust=False, eps=10e-5):
        def f(alpha):
            shifted_ewm = calc_shifted_ewm(series=series, alpha=min(max(alpha, 0), 1), adjust=adjust)
            corr = np.mean(np.power(series - shifted_ewm, 2))
            return corr

        res = optimize.differential_evolution(func=f, bounds=[(0 + eps, 1 - eps)])

        return calc_shifted_ewm(series=series, alpha=res['x'][0], adjust=adjust)

    roll = data.groupby(['air_store_id', 'day_of_week']).apply(lambda g: find_best_signal(g['visitors_capped']))
    data['optimized_ewm_by_air_store_id_&_day_of_week'] = roll.sort_index(level=['air_store_id', 'date_idx']).values

    roll = data.groupby(['air_store_id', 'is_weekend']).apply(lambda g: find_best_signal(g['visitors_capped']))
    data['optimized_ewm_by_air_store_id_&_is_weekend'] = roll.sort_index(level=['air_store_id', 'date_idx']).values

    roll = data.groupby(['air_store_id', 'day_of_week']).apply(lambda g: find_best_signal(g['visitors_capped_log1p']))
    data['optimized_ewm_log1p_by_air_store_id_&_day_of_week'] = roll.sort_index(level=['air_store_id', 'date_idx']).values

    roll = data.groupby(['air_store_id', 'is_weekend']).apply(lambda g: find_best_signal(g['visitors_capped_log1p']))
    data['optimized_ewm_log1p_by_air_store_id_&_is_weekend'] = roll.sort_index(level=['air_store_id', 'date_idx']).values


    ## Naive rolling features
    def extract_precedent_statistics(df, on, group_by):

        df.sort_values(group_by + ['visit_date'], inplace=True)

        groups = df.groupby(group_by, sort=False)

        stats = {'mean': [], 'median': [], 'std': [], 'count': [], 'max': [], 'min': []}

        exp_alphas = [0.1, 0.25, 0.3, 0.5, 0.75]
        stats.update({'exp_{}_mean'.format(alpha): [] for alpha in exp_alphas})

        for _, group in groups:

            shift = group[on].shift()
            roll = shift.rolling(window=len(group), min_periods=1)

            stats['mean'].extend(roll.mean())
            stats['median'].extend(roll.median())
            stats['std'].extend(roll.std())
            stats['count'].extend(roll.count())
            stats['max'].extend(roll.max())
            stats['min'].extend(roll.min())

            for alpha in exp_alphas:
                exp = shift.ewm(alpha=alpha, adjust=False)
                stats['exp_{}_mean'.format(alpha)].extend(exp.mean())

        suffix = '_&_'.join(group_by)

        for stat_name, values in stats.items():
            df['{}_{}_by_{}'.format(on, stat_name, suffix)] = values

    extract_precedent_statistics(df=data, on='visitors_capped', group_by=['air_store_id', 'day_of_week'])
    extract_precedent_statistics(df=data, on='visitors_capped', group_by=['air_store_id', 'is_weekend'])
    extract_precedent_statistics(df=data, on='visitors_capped', group_by=['air_store_id'])

    extract_precedent_statistics(df=data, on='visitors_capped_log1p', group_by=['air_store_id', 'day_of_week'])
    extract_precedent_statistics(df=data, on='visitors_capped_log1p', group_by=['air_store_id', 'is_weekend'])
    extract_precedent_statistics(df=data, on='visitors_capped_log1p', group_by=['air_store_id'])


    ## Merge in weather data
    data = data.merge(right = weather, how = 'left', on = ['visit_date', 'station_id'])

    if to_csv:
        data.to_csv("data/feature_engineered/fe1.csv", index = False)
    else:
        return data
    
    
    
def feature_engineer2(train_end, test_start, test_end):
    ## Data Ingestion and Feature Engineering

    data = {
        'tra': pd.read_csv('data/air_visit_data.csv'),
        'as': pd.read_csv('data/air_store_info.csv'),
        'hs': pd.read_csv('data/hpg_store_info.csv'),
        'ar': pd.read_csv('data/air_reserve.csv'),
        'hr': pd.read_csv('data/hpg_reserve.csv'),
        'id': pd.read_csv('data/store_id_relation.csv'),
        'tes': pd.read_csv('data/sample_submission.csv'),
        'hol': pd.read_csv('data/date_info.csv').rename(columns={'calendar_date':'visit_date'})
    }

    data['hr'] = pd.merge(data['hr'], data['id'], how='inner', on=['hpg_store_id'])

    for df in ['ar','hr']:
        data[df]['visit_datetime'] = pd.to_datetime(data[df]['visit_datetime'])
        data[df]['visit_dow'] = data[df]['visit_datetime'].dt.dayofweek
        data[df]['visit_datetime'] = data[df]['visit_datetime'].dt.date
        data[df]['reserve_datetime'] = pd.to_datetime(data[df]['reserve_datetime'])
        data[df]['reserve_datetime'] = data[df]['reserve_datetime'].dt.date
        data[df]['reserve_datetime_diff'] = data[df].apply(lambda r: (r['visit_datetime'] - r['reserve_datetime']).days, axis=1)
        # Exclude same-week reservations
        data[df] = data[df][data[df]['reserve_datetime_diff'] > data[df]['visit_dow']]
        tmp1 = data[df].groupby(['air_store_id','visit_datetime'], as_index=False)[['reserve_datetime_diff', 'reserve_visitors']].sum().rename(columns={'visit_datetime':'visit_date', 'reserve_datetime_diff': 'rs1', 'reserve_visitors':'rv1'})
        tmp2 = data[df].groupby(['air_store_id','visit_datetime'], as_index=False)[['reserve_datetime_diff', 'reserve_visitors']].mean().rename(columns={'visit_datetime':'visit_date', 'reserve_datetime_diff': 'rs2', 'reserve_visitors':'rv2'})
        data[df] = pd.merge(tmp1, tmp2, how='inner', on=['air_store_id','visit_date'])

    data['tra']['visit_date'] = pd.to_datetime(data['tra']['visit_date'])
    data['tra']['dow'] = data['tra']['visit_date'].dt.dayofweek
    data['tra']['doy'] = data['tra']['visit_date'].dt.dayofyear
    data['tra']['year'] = data['tra']['visit_date'].dt.year
    data['tra']['month'] = data['tra']['visit_date'].dt.month
    data['tra']['week'] = data['tra']['visit_date'].dt.week
    data['tra']['visit_date'] = data['tra']['visit_date'].dt.date

    data['tes']['visit_date'] = data['tes']['id'].map(lambda x: str(x).split('_')[2])
    data['tes']['air_store_id'] = data['tes']['id'].map(lambda x: '_'.join(x.split('_')[:2]))
    data['tes']['visit_date'] = pd.to_datetime(data['tes']['visit_date'])
    data['tes']['dow'] = data['tes']['visit_date'].dt.dayofweek
    data['tes']['doy'] = data['tes']['visit_date'].dt.dayofyear
    data['tes']['year'] = data['tes']['visit_date'].dt.year
    data['tes']['month'] = data['tes']['visit_date'].dt.month
    data['tes']['week'] = data['tes']['visit_date'].dt.week
    data['tes']['visit_date'] = data['tes']['visit_date'].dt.date

    unique_stores = data['tes']['air_store_id'].unique()
    stores = pd.concat([pd.DataFrame({'air_store_id': unique_stores, 'dow': [i]*len(unique_stores)}) for i in range(7)], axis=0, ignore_index=True).reset_index(drop=True)


    tmp = data['tra'].groupby(['air_store_id','dow'], as_index=False)['visitors'].min().rename(columns={'visitors':'min_visitors'})
    stores = pd.merge(stores, tmp, how='left', on=['air_store_id','dow']) 
    tmp = data['tra'].groupby(['air_store_id','dow'], as_index=False)['visitors'].mean().rename(columns={'visitors':'mean_visitors'})
    stores = pd.merge(stores, tmp, how='left', on=['air_store_id','dow'])
    tmp = data['tra'].groupby(['air_store_id','dow'], as_index=False)['visitors'].median().rename(columns={'visitors':'median_visitors'})
    stores = pd.merge(stores, tmp, how='left', on=['air_store_id','dow'])
    tmp = data['tra'].groupby(['air_store_id','dow'], as_index=False)['visitors'].count().rename(columns={'visitors':'count_observations'})
    stores = pd.merge(stores, tmp, how='left', on=['air_store_id','dow']) 

    stores = pd.merge(stores, data['as'], how='left', on=['air_store_id']) 
    # NEW FEATURES FROM Georgii Vyshnia
    stores['air_genre_name'] = stores['air_genre_name'].map(lambda x: str(str(x).replace('/',' ')))
    stores['air_area_name'] = stores['air_area_name'].map(lambda x: str(str(x).replace('-',' ')))
    lbl = preprocessing.LabelEncoder()
    for i in range(10):
        stores['air_genre_name'+str(i)] = lbl.fit_transform(stores['air_genre_name'].map(lambda x: str(str(x).split(' ')[i]) if len(str(x).split(' '))>i else ''))
        stores['air_area_name'+str(i)] = lbl.fit_transform(stores['air_area_name'].map(lambda x: str(str(x).split(' ')[i]) if len(str(x).split(' '))>i else ''))
    stores['air_genre_name'] = lbl.fit_transform(stores['air_genre_name'])
    stores['air_area_name'] = lbl.fit_transform(stores['air_area_name'])

    data['hol']['visit_date'] = pd.to_datetime(data['hol']['visit_date'])
    data['hol']['day_of_week'] = lbl.fit_transform(data['hol']['day_of_week'])
    data['hol']['visit_date'] = data['hol']['visit_date'].dt.date
    train = pd.merge(data['tra'], data['hol'], how='left', on=['visit_date']) 
    test = pd.merge(data['tes'], data['hol'], how='left', on=['visit_date']) 

    train = pd.merge(train, stores, how='inner', on=['air_store_id','dow']) 
    test = pd.merge(test, stores, how='left', on=['air_store_id','dow'])

    for df in ['ar','hr']:
        train = pd.merge(train, data[df], how='left', on=['air_store_id','visit_date']) 
        test = pd.merge(test, data[df], how='left', on=['air_store_id','visit_date'])

    train['id'] = train.apply(lambda r: '_'.join([str(r['air_store_id']), str(r['visit_date'])]), axis=1)

    train['total_reserv_sum'] = train['rv1_x'] + train['rv1_y']
    train['total_reserv_mean'] = (train['rv2_x'] + train['rv2_y']) / 2
    train['total_reserv_dt_diff_mean'] = (train['rs2_x'] + train['rs2_y']) / 2

    test['total_reserv_sum'] = test['rv1_x'] + test['rv1_y']
    test['total_reserv_mean'] = (test['rv2_x'] + test['rv2_y']) / 2
    test['total_reserv_dt_diff_mean'] = (test['rs2_x'] + test['rs2_y']) / 2


    train['date_int'] = train['visit_date'].apply(lambda x: x.strftime('%Y%m%d')).astype(int)
    test['date_int'] = test['visit_date'].apply(lambda x: x.strftime('%Y%m%d')).astype(int)
    train['var_max_lat'] = train['latitude'].max() - train['latitude']
    train['var_max_long'] = train['longitude'].max() - train['longitude']
    test['var_max_lat'] = test['latitude'].max() - test['latitude']
    test['var_max_long'] = test['longitude'].max() - test['longitude']

    train['lon_plus_lat'] = train['longitude'] + train['latitude'] 
    test['lon_plus_lat'] = test['longitude'] + test['latitude']

    lbl = preprocessing.LabelEncoder()
    train['air_store_id2'] = lbl.fit_transform(train['air_store_id'])
    test['air_store_id2'] = lbl.transform(test['air_store_id'])

    col = [c for c in train if c not in ['id', 'air_store_id', 'visit_date','visitors']]
    train = train.fillna(-1)
    test = test.fillna(-1)

    train.visit_date = pd.to_datetime(train.visit_date)
    test = train[(train.visit_date > test_start) & (train.visit_date <= test_end)]
    train = train[train.visit_date <= train_end]

    return train, test


def feature_engineer2_nn(train_end, test_start, test_end):
    
    train, test = feature_engineer2(train_end, test_start, test_end)
    
    value_col = ['holiday_flg','min_visitors','mean_visitors','median_visitors', 'count_observations',
             'rs1_x','rv1_x','rs2_x','rv2_x','rs1_y','rv1_y','rs2_y','rv2_y','total_reserv_sum','total_reserv_mean',
             'total_reserv_dt_diff_mean','date_int','var_max_lat','var_max_long','lon_plus_lat']

    nn_col = value_col + ['dow', 'year', 'month', 'air_store_id2', 'air_area_name', 'air_genre_name',
    'air_area_name0', 'air_area_name1', 'air_area_name2', 'air_area_name3', 'air_area_name4',
    'air_area_name5', 'air_area_name6', 'air_genre_name0', 'air_genre_name1',
    'air_genre_name2', 'air_genre_name3', 'air_genre_name4']


    X = train.copy()
    X_test = test[nn_col].copy()

    value_scaler = preprocessing.MinMaxScaler()
    for vcol in value_col:
        X[vcol] = value_scaler.fit_transform(X[vcol].values.astype(np.float64).reshape(-1, 1))
        X_test[vcol] = value_scaler.transform(X_test[vcol].values.astype(np.float64).reshape(-1, 1))

    X_train = list(X[nn_col].T.values)
    Y_train = np.log1p(X['visitors']).values
    nn_train = [X_train, Y_train]
    nn_test = [list(X_test[nn_col].T.values)]
    
    return nn_train, nn_test



def get_nn_complete_model(train, hidden1_neurons=35, hidden2_neurons=15):
    """ Keras neural network model
    
    Args:
        train: train dataframe(used to define the input size of the embedding layer)
        hidden1_neurons: number of neurons in the first hidden layer
        hidden2_neurons: number of neurons in the first hidden layer
        
    Returns:
        return 'keras neural network model'
    """
    
    tf.keras.backend.clear_session()

    air_store_id = tf.keras.Input(shape=(1,), dtype='int32', name='air_store_id')
    air_store_id_emb = tf.keras.layers.Embedding(len(train['air_store_id2'].unique()) + 1, 15, input_shape=(1,), name='air_store_id_emb')(air_store_id)
    air_store_id_emb = tf.keras.layers.Flatten(name='air_store_id_emb_flatten')(air_store_id_emb)

    dow = tf.keras.Input(shape=(1,), dtype='int32', name='dow')
    dow_emb = tf.keras.layers.Embedding(8, 3, input_shape=(1,), name='dow_emb')(dow)
    dow_emb = tf.keras.layers.Flatten(name='dow_emb_flatten')(dow_emb)

    month = tf.keras.Input(shape=(1,), dtype='int32', name='month')
    month_emb = tf.keras.layers.Embedding(13, 3, input_shape=(1,), name='month_emb')(month)
    month_emb = tf.keras.layers.Flatten(name='month_emb_flatten')(month_emb)

    air_area_name, air_genre_name = [], []
    air_area_name_emb, air_genre_name_emb = [], []
    for i in range(7):
        area_name_col = 'air_area_name' + str(i)
        air_area_name.append(tf.keras.Input(shape=(1,), dtype='int32', name=area_name_col))
        tmp = tf.keras.layers.Embedding(len(train[area_name_col].unique()), 3, input_shape=(1,),
                        name=area_name_col + '_emb')(air_area_name[-1])
        tmp = tf.keras.layers.Flatten(name=area_name_col + '_emb_flatten')(tmp)
        air_area_name_emb.append(tmp)

        if i > 4:
            continue
        area_genre_col = 'air_genre_name' + str(i)
        air_genre_name.append(tf.keras.Input(shape=(1,), dtype='int32', name=area_genre_col))
        tmp = tf.keras.layers.Embedding(len(train[area_genre_col].unique()), 3, input_shape=(1,),
                        name=area_genre_col + '_emb')(air_genre_name[-1])
        tmp = tf.keras.layers.Flatten(name=area_genre_col + '_emb_flatten')(tmp)
        air_genre_name_emb.append(tmp)

    air_genre_name_emb = tf.keras.layers.concatenate(air_genre_name_emb)
    air_genre_name_emb = tf.keras.layers.Dense(4, activation='sigmoid', name='final_air_genre_emb')(air_genre_name_emb)

    air_area_name_emb = tf.keras.layers.concatenate(air_area_name_emb)
    air_area_name_emb = tf.keras.layers.Dense(4, activation='sigmoid', name='final_air_area_emb')(air_area_name_emb)
    
    air_area_code = tf.keras.Input(shape=(1,), dtype='int32', name='air_area_code')
    air_area_code_emb = tf.keras.layers.Embedding(len(train['air_area_name'].unique()), 8, input_shape=(1,), name='air_area_code_emb')(air_area_code)
    air_area_code_emb = tf.keras.layers.Flatten(name='air_area_code_emb_flatten')(air_area_code_emb)
    
    air_genre_code = tf.keras.Input(shape=(1,), dtype='int32', name='air_genre_code')
    air_genre_code_emb = tf.keras.layers.Embedding(len(train['air_genre_name'].unique()), 5, input_shape=(1,),
                                   name='air_genre_code_emb')(air_genre_code)
    air_genre_code_emb = tf.keras.layers.Flatten(name='air_genre_code_emb_flatten')(air_genre_code_emb)

    
    holiday_flg = tf.keras.Input(shape=(1,), dtype='float32', name='holiday_flg')
    year = tf.keras.Input(shape=(1,), dtype='float32', name='year')
    min_visitors = tf.keras.Input(shape=(1,), dtype='float32', name='min_visitors')
    mean_visitors = tf.keras.Input(shape=(1,), dtype='float32', name='mean_visitors')
    median_visitors = tf.keras.Input(shape=(1,), dtype='float32', name='median_visitors')
    count_observations = tf.keras.Input(shape=(1,), dtype='float32', name='count_observations')
    rs1_x = tf.keras.Input(shape=(1,), dtype='float32', name='rs1_x')
    rv1_x = tf.keras.Input(shape=(1,), dtype='float32', name='rv1_x')
    rs2_x = tf.keras.Input(shape=(1,), dtype='float32', name='rs2_x')
    rv2_x = tf.keras.Input(shape=(1,), dtype='float32', name='rv2_x')
    rs1_y = tf.keras.Input(shape=(1,), dtype='float32', name='rs1_y')
    rv1_y = tf.keras.Input(shape=(1,), dtype='float32', name='rv1_y')
    rs2_y = tf.keras.Input(shape=(1,), dtype='float32', name='rs2_y')
    rv2_y = tf.keras.Input(shape=(1,), dtype='float32', name='rv2_y')
    total_reserv_sum = tf.keras.Input(shape=(1,), dtype='float32', name='total_reserv_sum')
    total_reserv_mean = tf.keras.Input(shape=(1,), dtype='float32', name='total_reserv_mean')
    total_reserv_dt_diff_mean = tf.keras.Input(shape=(1,), dtype='float32', name='total_reserv_dt_diff_mean')
    date_int = tf.keras.Input(shape=(1,), dtype='float32', name='date_int')
    var_max_lat = tf.keras.Input(shape=(1,), dtype='float32', name='var_max_lat')
    var_max_long = tf.keras.Input(shape=(1,), dtype='float32', name='var_max_long')
    lon_plus_lat = tf.keras.Input(shape=(1,), dtype='float32', name='lon_plus_lat')

    date_emb = tf.keras.layers.concatenate([dow_emb, month_emb, year, holiday_flg])
    date_emb = tf.keras.layers.Dense(5, activation='sigmoid', name='date_merged_emb')(date_emb)

    cat_layer = tf.keras.layers.concatenate([holiday_flg, min_visitors, mean_visitors,
                    median_visitors, # max_visitors, 
                    count_observations, rs1_x, rv1_x,
                    rs2_x, rv2_x, rs1_y, rv1_y, rs2_y, rv2_y,
                    total_reserv_sum, total_reserv_mean, total_reserv_dt_diff_mean,
                    date_int, var_max_lat, var_max_long, lon_plus_lat,
                    date_emb, air_area_name_emb, air_genre_name_emb,
                    air_area_code_emb, air_genre_code_emb, air_store_id_emb])

    m = tf.keras.layers.Dense(hidden1_neurons, name='hidden1', kernel_initializer=tf.keras.initializers.RandomNormal(mean=0.0, stddev=0.05, seed=None))(cat_layer)
    m = tf.keras.layers.PReLU()(m)
    m = tf.keras.layers.BatchNormalization()(m)
    
    m1 = tf.keras.layers.Dense(hidden2_neurons, name='sub1')(m)
    m1 = tf.keras.layers.PReLU()(m1)
    m = tf.keras.layers.Dense(1, activation='relu')(m1)

    inp_ten = [
        holiday_flg, min_visitors, mean_visitors, median_visitors, # max_visitors, 
        count_observations,
        rs1_x, rv1_x, rs2_x, rv2_x, rs1_y, rv1_y, rs2_y, rv2_y, total_reserv_sum, total_reserv_mean,
        total_reserv_dt_diff_mean, date_int, var_max_lat, var_max_long, lon_plus_lat,
        dow, year, month, air_store_id, air_area_code, air_genre_code
    ]
    inp_ten += air_area_name
    inp_ten += air_genre_name
    model = tf.keras.Model(inp_ten, m)
    model.compile(loss='mse', optimizer='rmsprop', metrics=['acc'])

    return model