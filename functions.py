# -*- coding: utf-8 -*-
import os
import re
import sys
import dill
import yaml
import numpy as np
import pandas as pd
from datetime import timedelta, datetime
from creator import FeaturesCreator
from sklearn.pipeline import Pipeline
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import mean_squared_error


params = yaml.safe_load(open('params/params.yaml'))

for model_name in params['models']:
    exec(params['models'][model_name]['import'])

COUNTRIES = params['countries']
splits_count = params['data_params']['splits_count']


# Function that collects all json files into one pandas.DataFrame
def collect_data(path):

    if not os.path.isdir(path):
        raise(Exception('Directory does not exist'))

    columns = ['country', 'customer_id', 'invoice', 'price', 'stream_id',
               'times_viewed', 'year', 'month', 'day']
    json_list = [i for i in os.listdir(path) if re.fullmatch(r'.+\.json$', i)]

    data = pd.DataFrame(columns=columns)

    if len(json_list) < 1:
        raise(Exception('No file to load'))

    for file_name in json_list:
        df = pd.read_json(os.path.join(path, file_name))
        df.columns = columns
        data = data.append(df)

    data['date'] = pd.to_datetime(data.year.astype('str')+'-'+data.month.astype('str')+'-'+data.day.astype('str'))
    data['country'] = data.country.str.lower()

    return data.sort_values(by='date').reset_index(drop=True)


'''
Function that transforms collected pandas.DataFrame
to new DataFrame with new features for country
'''
def transform_data(original_data, country):
    df = original_data.copy()

    start_date = df.date.min()
    end_date = df.date.max()
    dates_range = pd.date_range(start_date, end_date)

    if country != 'all':
        if country not in df.country.unique():
            raise Exception('Country not found')
        df = df[df.country == country]

    df['invoice'] = df.invoice.str.extract(r'([0-9]+)')

    df_to_join = pd.DataFrame({'date': dates_range})
    df_to_join['month_day'] = df_to_join.date.dt.strftime('%m-%d')

    df = df.groupby('date').agg({'country': len,
                                 'invoice': lambda x: len(np.unique(x)),
                                 'stream_id': lambda x: len(np.unique(x)),
                                 'times_viewed': np.sum,
                                 'price': np.sum}).reset_index()

    df.rename(columns={'invoice': 'unique_invoices',
                       'times_viewed': 'total_views',
                       'price': 'revenue',
                       'country': 'purchases',
                       'stream_id': 'total_streams'}, inplace=True)

    df = pd.merge(df_to_join, df, how='left', on='date').fillna(0)         
    return df

'''
Function that returns dictionary {country: pandas.DataFrame}
If there is already a saved pandas.DataFrame by country, the function will simply load it. 
If there is no pandas.DataFrame, then the function will create it and save it.

If renew_all_csv = True, function collects and transforms for all countries in COUNTRIES
If country=None, function returns dictionary with all countries in COUNTRIES
'''
def get_data(path, country=None, renew_all_csv=False):
    
    data_csv_path = os.path.join(path, 'convert_data')
    
    if not os.path.isdir(data_csv_path):
        os.mkdir(data_csv_path)
           
    countries_to_csv = set(COUNTRIES) if renew_all_csv \
                    else set(COUNTRIES).difference([i.split('.')[0] for i in os.listdir(data_csv_path)]) 
    
    if len(countries_to_csv) > 0:
        print(f'...Collect json files from {path}')
        df = collect_data(path)
        print('...Create csv files')
        for c in countries_to_csv:
            data_country = transform_data(original_data=df, country=c)
            data_country.to_csv(os.path.join(data_csv_path, c)+'.csv', index=False)
            print(f'......csv file for {c} created successfuly')
        print(f'...all csv files from {path} created successfuly\n')
    
    if country:
        return {country: pd.read_csv(os.path.join(data_csv_path, country.lower())+'.csv', parse_dates=['date'])}
        
    return {i.split('.')[0]: pd.read_csv(os.path.join(data_csv_path, i), parse_dates=['date']) for i in os.listdir(data_csv_path)}



# Function that collect ,transform and split data to X_train, X_test, y_train, y_test for country
# If renew_data=True, all csv files recreated
def get_train_test_data(country, renew_data=False):
    
    train_data = get_data(params['path']['train'], country, renew_data)[country][['date', 'revenue']]
    train_data.set_index('date', inplace=True, drop=True)
    test_data = get_data(params['path']['test'], country, renew_data)[country][['date', 'revenue']]
    test_data.set_index('date', inplace=True, drop=True)
    
    all_data = train_data.append(test_data)
    all_data['y'], all_data['sum_2_to_29'] = None, None
    
    for date in all_data.index[:-29]:
        mask = np.in1d(all_data.index, pd.date_range(date, date+timedelta(30), closed='left'))
        all_data.loc[date, 'y'] = all_data[mask].revenue.sum()
        mask = np.in1d(all_data.index, pd.date_range(date+timedelta(1), date+timedelta(29), closed='left'))
        all_data.loc[date, 'sum_2_to_29'] = all_data[mask].revenue.sum()
    
    all_data.dropna(inplace=True)
    all_data['last_revenue'] = all_data.y - all_data.revenue - all_data.sum_2_to_29
    
    
    X_train = all_data[train_data.index.min():train_data.index.max()]
    y_train = all_data[train_data.index.min():train_data.index.max()].y
    X_test = all_data[test_data.index.min():]
    y_test = all_data[test_data.index.min():].y
    
    return X_train, X_test, y_train, y_test
'''
A function that trains models from the params list for the country. 
Models are trained on GridSearchCV
splits_count is defined in params.yaml
Models with the best parameters are saved, 
for each of them the training results are saved
The model with the lowest rmse on the test gets the TEST tag
'''
def fit_models_(country, renew_data, standard_parameters=False):
    
    X_train, X_test, y_train, y_test = get_train_test_data(country, renew_data)
    
    split_index = np.linspace(365, X_train.shape[0], splits_count+1).astype('int')
    cv = [[np.arange(split_index[i]), np.arange(split_index[i], split_index[i+1])]
              for i in range(len(split_index)-1)]
    
    
    for model_name in params['models']:
        print(f'......Train {model_name.split("(")[0]} model')
        creator = FeaturesCreator()
        model = eval(model_name)
       
        pipline =Pipeline(steps=[('creator', creator),
                                 ('model', model)])
        
        param_grid = get_param_grid(model_name, standard_parameters)
       
        gcv = GridSearchCV(estimator=pipline,
                           param_grid=param_grid,
                           cv=cv, 
                           scoring='neg_root_mean_squared_error')
        
        gcv.fit(X_train, y_train)
       
        best_estimator = gcv.best_estimator_ 
        train_results = get_train_results(X_train, X_test, y_train, y_test, best_estimator)
        test_error = train_results['test_error']
        
        print(f'.........RMSE for {model_name.split("(")[0]} on validation = {test_error}')
        
        X = X_train.append(X_test)
        y = y_train.append(y_test)
        
        best_estimator.fit(X, y)
        
        save_model(best_estimator, train_results, model_name.split('(')[0], country)
        
    set_model_to_test_version(os.path.join(params['path']['models'], country))
    return None

'''
Function, at the output of which we get the param_grid for the model. 
If standard_parameters=True, then the standard parameters of the model will be output.
If False, then parameters are loaded from params.yaml
'''
def get_param_grid(model_name, standard_parameters):
    if standard_parameters:
        param_grid = {f'model__{key}': [value] 
                for key, value in eval(model_name).get_params().items()}
        
        param_grid.update({f'creator__{key}': [value] 
                for key, value in FeaturesCreator().get_params().items()})
    else:
        param_grid = params['models'][model_name]['params']
        param_grid.update(params['creator'])        
    return param_grid

'''

A function that generates a dictionary with the results of model training. 
If fit_model=True, model refi on train data.
The input is the model and the data. The output dictionary:
    y_train - target variable on training
    y_test - target variable on test
    y_train_mean - mean value of the target variable on training
    y_test_mean - mean value of the target variable on test
    train_preds - predictions on train
    test_preds - predictions on test
    train_error - rmse on train
    test_error - rmse on test
    base_train_error - baseline error on train (model always predict mean train value)
    base_test_error - baseline error on test (model always predict mean train value)
'''
def get_train_results(X_train, X_test, y_train, y_test, model, fit_model=False):
    if fit_model:
        model.fit(X_train, y_train)
    train_preds = model.predict(X_train)
    test_preds = model.predict(X_test)
    train_error = mean_squared_error(y_train, train_preds)**0.5
    test_error = mean_squared_error(y_test, test_preds)**0.5
    base_train_error = np.mean((y_train - y_train.mean())**2)**0.5
    base_test_error = np.mean((y_test - y_train.mean())**2)**0.5
    
    dct = {'y_train': y_train,
           'y_test': y_test,
           'y_train_mean': y_train.mean(),
           'y_test_mean': y_test.mean(),
           'train_preds': train_preds,
           'test_preds': test_preds,
           'train_error': train_error,
           'test_error': test_error,
           'base_train_error': base_train_error,
           'base_test_error': base_test_error}
    return dct


# A function that determines the latest version of models for a country    
def get_last_model_version(path):
    return max([int(re.findall(r'\d+', i.split('_')[5])[0]) for i in os.listdir(path)])

# A function that sets the model with the smallest error as the test model
def set_model_to_test_version(path):
    
    def rename_model(model_name, replace_sample):
        if replace_sample == ('SL__', 'TEST__'):
            print(f'...Set {candidate} as test model')
        old_model_name = os.path.join(path, model_name)
        new_model_name = os.path.join(path, model_name.replace(*replace_sample))
        os.rename(old_model_name, new_model_name)

    
    full_models_list = os.listdir(path)
    sl_models_list = [i for i in full_models_list if i.startswith('SL__')]

    sl_models_errors = [int(re.findall(r'\d+', i.split('_')[6])[0]) for i in sl_models_list]
    candidate = sl_models_list[np.argmin(sl_models_errors)]
    
    
    if len([i for i in full_models_list if i.startswith('TEST__')]) == 0:
        rename_model(candidate, ('SL__', 'TEST__'))
    else:
        current_test_model = [i for i in full_models_list if i.startswith('TEST__')][0]
        current_test_model_error = int(re.findall(r'\d+', current_test_model.split('_')[6])[0])
        candidate_error = np.min(sl_models_errors)    
        
        if candidate_error < current_test_model_error:
            rename_model(candidate, ('SL__', 'TEST__'))
            rename_model(current_test_model, ('TEST__', 'SL__'))
        else: 
            print('...The quality of the new models is worse than the current test model')
    return None


# Function that saves the model
def save_model(model, train_results, start_name, country):
    model_path = os.path.join(params['path']['models'], country)
    train_results_path = os.path.join(params['path']['train_results'], country)
    error = train_results['test_error']
    
    if not os.path.isdir(model_path):
        os.mkdir(model_path)
        model_version = 0
    else:
        if len(os.listdir(model_path)) == 0:
            model_version = 0
        else:
            model_version = get_last_model_version(model_path)
    
    date_tag = datetime.now().strftime('%Y%m%d_%H%M%S')
    
    model_name = f'SL__{start_name}_{date_tag}_v{model_version+1}_err{int(error)}.pkl'
    train_results_name = f'{start_name}_{date_tag}_v{model_version+1}_err{int(error)}.pkl'
    
    with open (os.path.join(model_path, model_name), 'wb') as f:
        dill.dump(model ,f)
        
    save_train_results(train_results, train_results_path, train_results_name)
    return None

# Function that saves training results
def save_train_results(results, path, name):
    if not os.path.isdir(path):
        os.mkdir(path)
    full_name = os.path.join(path, name)
    with open(full_name, 'wb') as f:
        dill.dump(results, f)
    return None

'''
Function that loads the model for the country. 
If the version of the model is specified, then the model with this version 
is loaded, if not, then the test model is loaded
'''
def load_model(country, version=None):
    path = os.path.join(params['path']['models'], country.lower())
    
    if country.lower() not in COUNTRIES:
        print('...For this country no models. You can load models for countries in this list:')
        print('......', ', '.join(COUNTRIES))
        return None
    
    if not os.path.isdir(path):
        os.mkdir(path)
    
    test_list = [i for i in os.listdir(path) if i.startswith('TEST__')]   
    
    if version:
        models_vers = [re.findall(r'\d+', i.split('_')[5])[0] for i in os.listdir(path)]
        if str(version) not in models_vers:
            print(f'...Model with version {version} is not exist')
            return None
        model_name = os.listdir(path)[models_vers.index(str(version))]

    else:
        if len(test_list) == 0:
            fit_models_(country, renew_data=False)
        test_list = [i for i in os.listdir(path) if i.startswith('TEST__')]
        model_name = test_list[0]
        
    model = dill.load(open(os.path.join(path, model_name), 'rb'))
    
    return model

# Function that predicts the target variable
def predict(country, target_date, model=None):
    if not model:
        model = load_model(country, version=None)
    predictions = []
       
    if target_date in model['Creator'].X.index:
        predictions.append(model['Creator'].X.loc[target_date, 'y'])
        dates_range_to_predict = [target_date]
    else:
        
        start_date_to_predict = model['Creator'].start_date + timedelta(1)
        dates_range_to_predict = pd.date_range(start_date_to_predict, target_date)
        
        for date_to_predict in dates_range_to_predict:
            
            X = model['Creator'].X
            
            revenue = X.y[-29] - X.y[-30] + X.revenue[-30]
            sum_2_to_29 = X.y[-1] - X.revenue[-1] - revenue
            
            sample = pd.DataFrame({'revenue': revenue, 
                                   'sum_2_to_29': sum_2_to_29, 
                                   'y': None,
                                   'last_revenue': None}, index=[date_to_predict])
            
            prediction = model.predict(sample)[0]
            sample.loc[date_to_predict, 'y'] = prediction

            sample['last_revenue'] = sample.y - sample.sum_2_to_29 - sample.revenue
            
            model['Creator'].X = model['Creator'].X.append(sample)
            model['Creator'].X_to_top_test = model['Creator'].X_to_top_test.iloc[1:].append(sample)
            predictions.append(prediction)
               
    return pd.DataFrame({'preds': predictions}, index=dates_range_to_predict)


'''
Checking models for a country. If there are no models, then you can optionally
train models with standard parameters. If we receive a refusal from training, 
then the execution of the program stops.
'''
def check_models(country):
    path = os.path.join(params['path']['models'], country)
    
    if not os.path.isdir(path) or len(os.listdir(path))==0:
        print(f'For {country.upper()} not exist fitted models. ',
              'Do you want fit models with standard parameters ',
              f'for {country.upper()}? [Y] - yes, [N] - no')
        ans = input().lower()
        if ans == 'y':
            fit_models_(country, renew_data=False, standard_parameters=True)
        else:
            print(f'Exit, please fit models for {country.upper()}')
            sys.exit()
    
    if not [i for i in os.listdir(path) if i.startswith('TEST_')]:
        set_model_to_test_version(path)
    
    return None

'''
Validate learning outcomes for all models for a country.
If there are no results for some model, then the results are collected
'''
def check_train_results(country):
    models_path = os.path.join(params['path']['models'], country)
    train_results_path = os.path.join(params['path']['train_results'], country)
    
    if not os.path.isdir(train_results_path):
        os.mkdir(train_results_path)
        train_results_list = []
    else:  
        train_results_list = os.listdir(train_results_path)
        
    models_list = os.listdir(models_path)
    lost_train_results = get_lost_train_results(models_list, train_results_list)
        
    if lost_train_results:
        X_train, X_test, y_train, y_test = get_train_test_data(country)
        for model_name in lost_train_results:
            print(f'Collect train_results for {model_name} for {country}')
            train_results_name = model_name.split('__')[1]
            model = dill.load(open(os.path.join(models_path, model_name), 'rb'))
            train_results = get_train_results(X_train, X_test, y_train, y_test, model, fit_model=True)
            save_train_results(train_results, train_results_path, train_results_name)
    return None


# Function that determines for which models there are no training results
def get_lost_train_results(models_list, results_list):
    needed_results_list = set([i.split('__')[1] for i in models_list])
    results_list = set(results_list)
    differences = needed_results_list.difference(results_list)
    models_without_results = [i for i in models_list if i.split('__')[1] in differences]
    return models_without_results