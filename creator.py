# -*- coding: utf-8 -*-

from sklearn.base import BaseEstimator, TransformerMixin
import numpy as np
import pandas as pd
from datetime import timedelta


class FeaturesCreator(BaseEstimator, TransformerMixin):
    
    def __init__(self, 
                 num_lags=30,
                 window_size=1, 
                 num_quantiles=3, # number of quantiles previouse lags
                 target_season_features=['month', 'day_of_week'], # list values 'month', 'day', 'day_of_week'
                 type_of_target_season_features='mean', # 'mean', 'one-hot'
                 target_measure_features = ['min', 'max', 'median'], # 'min', 'max', 'median'
                 last_revenue_season_features = ['month', 'day_of_week'] # 'month', 'day', 'day_of_week'
                ):
        # number of target variable lags
        self.num_lags = num_lags
        self.window_size = window_size
        # number of quantiles by lags
        self.num_quantiles = num_quantiles
        # list of time units by which the target variable will be encoded
        self.target_season_features = target_season_features
        # type of encoding seasonal variables: one-hot or mean
        self.type_of_target_season_features = type_of_target_season_features
        # list of descriptive statistics by which the target variable will be coded
        self.target_measure_features = target_measure_features
        # list of time units by which the 30th day variable will be encoded
        self.last_revenue_season_features = last_revenue_season_features
        
        # Dictionary of functions to _add_target_measure_features
        self.target_measure_functions = dict({'median': np.median,
                                              'max': np.max,
                                              'min': np.min})
        
        self.__num_one_hot_features = dict({'day': 31, 'month': 12, 'day_of_week': 7})
        self.__quantiles = np.linspace(0, 1, num_quantiles+1, endpoint=False)[1:]
    
    def fit(self, X, y=None):
        # save X_train
        self.X = X
        # save X_train indices
        self.__train_indices = X.index.tolist()
        
        # Revenue mean values by monthes to 'mean_by_month' feature
        self.__revenue_month_mean = dict(X.last_revenue.groupby(X.index.month).mean())
        
        # Mean value of revenue by day for add first rows to train data
        self.__revenue_day_mean = dict(X.revenue.groupby(X.index.strftime('%m-%d')).mean())
        
        # Mean target values for season mean encode
        self.__target_season_mean_day = dict(X.y.groupby(X.index.day).mean())
        self.__target_season_mean_day_of_week = dict(X.y.groupby(X.index.day_of_week).mean())
        self.__target_season_mean_month = dict(X.y.groupby(X.index.month).mean())
        # functions for seasonal encoding target variable 
        self.__target_season_functions = dict({
            'day': [lambda x: self.__target_season_mean_day[x.day], lambda x: x.day-1],
            'day_of_week': [lambda x: self.__target_season_mean_day_of_week[x.day_of_week], lambda x: x.day_of_week],
            'month': [lambda x: self.__target_season_mean_month[x.month], lambda x: x.month-1]})
        
        # Mean last_revenue (30th day) for season mean encode
        self.__last_revenue_season_mean_day = dict(X.last_revenue.groupby(X.index.day).mean())
        self.__last_revenue_season_mean_day_of_week = dict(X.last_revenue.groupby(X.index.day_of_week).mean())
        self.__last_revenue_season_mean_month = dict(X.last_revenue.groupby(X.index.month).mean())
        # functions for seasonal encoding last revenue (30th day)
        self.__last_revenue_season_functions = dict({
            'day': lambda x: self.__last_revenue_season_mean_day[x.day],
            'day_of_week': lambda x: self.__last_revenue_season_mean_day_of_week[x.day_of_week],
            'month': lambda x: self.__last_revenue_season_mean_month[x.month]})
        # Save last rows for X_test. Needed to crate lags on test df       
        self.X_to_top_test = X[-self.num_lags:]
        self.start_date = X.index.max()
        return self
    
    
    def transform(self, X):
        X = X.copy()
        # add rows to df to create lags wihtout data loss
        X = self._add_rows_to_df(X)
        # add features to df
        X = self._add_features(X)
        X.dropna(inplace=True)
        X = self._add_target_season_features(X)
        X = self._add_target_measure_features(X)
        X = self._add_quantile_features(X)
        X = self._add_last_revenue_season_features(X)
        return X

    def _add_first_rows_to_train(self, X):
        # for train df add rows by mean on day '%m-%d'
        for date in reversed(pd.date_range(X.index.min()-timedelta(self.num_lags), X.index.min(), closed='left')):
            revenue = self.__revenue_day_mean[date.strftime('%m-%d')]
            y = X[:29].revenue.sum() + revenue
            sum_2_to_29 = X[:28].revenue.sum()
            df_row = pd.DataFrame({'revenue': [revenue], 'y': [y], 'sum_2_to_29': [sum_2_to_29]}, index=[date])
            X = df_row.append(X)
        return X
    
    def _add_rows_to_df(self, X):
        # if indices of input df match with indices of df that had fit
        if self.__train_indices == X.index.tolist():
            X = self._add_first_rows_to_train(X)
        else:
            X = self.X_to_top_test.append(X)
        return X
    
    
    def _add_target_season_features(self, X):
        X_output = X.copy()
        if self.target_season_features:
            if self.type_of_target_season_features == 'mean':
                for agg_func in self.target_season_features:
                    X_output[f'{agg_func}_mean_target'] = X_output.index.map(
                                                            self.__target_season_functions[agg_func][0])
            
            elif self.type_of_target_season_features == 'one-hot':
                for agg_func in self.target_season_features:
                    add = np.zeros((len(X), self.__num_one_hot_features[agg_func]))
                    add[range(len(X)), X.index.map(self.__target_season_functions[agg_func][1])] = 1
                    add = pd.DataFrame(add, index=X.index, 
                            columns=[f'{agg_func}_{i+1}' for i in range(self.__num_one_hot_features[agg_func])])
                    X_output = pd.concat([X_output, add], axis=1)
        return X_output
    
    def _add_target_measure_features(self, X):
        X_output = X.copy()
        
        if self.target_measure_features:
            for feature in self.target_measure_features:
                X_output[f'{feature}_lags'] = self.target_measure_functions[feature](
                                                X_output[[f'y_lag_{i}' for i in range(1, self.num_lags+1)]], axis=1)
        return X_output
    
    def _add_quantile_features(self, X):
        X_output = X.copy()
        if self.num_quantiles and self.num_quantiles>0:
            for idx, q in enumerate(self.__quantiles):
                if q==0.5 and 'median_lags' in X_output.columns:
                    continue
                X_output[f'q{idx+1}'] = np.quantile(X_output, q, axis=1)
        return X_output
    
    def _add_last_revenue_season_features(self, X):
        X_output = X.copy()
        if self.last_revenue_season_features:
            for agg_func in self.last_revenue_season_features:
                X_output[f'{agg_func}_mean_last_revenue'] = X_output.index.map(
                                                              self.__last_revenue_season_functions[agg_func])
        return X_output
        
    
    def _add_features(self, X):
        X['monthes'] = X.index.map(
            lambda x: [date.month for date in pd.date_range(x, x+timedelta(30), closed='left')])
        
        X['mean_by_month'] = X.monthes.map(
            lambda x: np.mean([self.__revenue_month_mean[i] for i in x]))        
        
        X['monthes_window'] =  X.index.map(
            lambda x: [date.month for date in pd.date_range(x+timedelta(29-self.window_size), 
                                                            x+timedelta(29+self.window_size))])
        
        X['mean_month_window'] = X.monthes_window.map(
            lambda x: np.mean([self.__revenue_month_mean[i] for i in x]))
              
        for lag in range(1, self.num_lags+1):
            X[f'y_lag_{lag}'] = X.y.shift(lag)

        X.drop(['monthes', 'monthes_window', 'y', 'last_revenue'], axis=1, inplace=True)
        return X
