# -*- coding: utf-8 -*-
import pandas as pd
from functions import predict, check_models, check_train_results
from datetime import timedelta, datetime
import numpy as np
import yaml
import sys
import warnings

warnings.filterwarnings('ignore')


params = yaml.safe_load(open('params/params.yaml'))
COUNTRIES = params['countries']
country = None
date = None

if __name__ == '__main__':
    for c in COUNTRIES:
        check_models(c)
        check_train_results(c)
    try:
        print('For exit press Ctrl + "C"')
        while True:
            while country not in COUNTRIES:
                country = input('Enter country. Country = ')
                if country not in COUNTRIES:
                    print('\nIncorrect country value')
                    print('Valid country values: ')
                    print(COUNTRIES)
            while not date:
                try:
                    input_date = input('\nEnter date in format YYYY-MM-DD. Date = ')
                    datetime.strptime(input_date, '%Y-%m-%d')
                    if pd.to_datetime(input_date) >= params['data_params']['first_date']:
                        date = input_date
                    else:
                        print('\nLater dates used in training')
                        print('Models trained on data:')
                        print(f'Since {params["data_params"]["first_date"]}')
                        print(f'To {params["data_params"]["last_date"]}')
                except ValueError:
                    print('\nIncorrect format of date.')
                    print('Please enter date in format YYYY-MM-DD')
               
            prediction = round(predict(country, pd.to_datetime(date)).preds[-1], 2)
        
            print(f'\nPrediction on date={date} for country={country.upper()} is {prediction}\n')
        
            country = None
            date = None
    except KeyboardInterrupt:
        print('\n\nExit')
        sys.exit()