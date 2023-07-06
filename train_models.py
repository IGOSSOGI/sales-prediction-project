# -*- coding: utf-8 -*-

from functions import fit_models_
import yaml
import sys
import warnings
warnings.filterwarnings('ignore')

params = yaml.safe_load(open('params/params.yaml'))
COUNTRIES = params['countries']

args = sys.argv

if len(sys.argv) == 2:
    if sys.argv[1] in ['False', 'True']:
        country = None
        renew_data = eval(sys.argv[1])
    else:
        country = sys.argv[1]
        renew_data = False
elif len(sys.argv) > 2:
    country = sys.argv[1]
    renew_data = eval(sys.argv[2])
else:
    country = None
    renew_data = False
    
countries = [country] if country else COUNTRIES


if __name__ == '__main__':
    for country in countries:
        if country not in COUNTRIES:
            print(f'For {country} impossible fit models')
            break
        print(f'...Train models for country: {country}')
        fit_models_(country, renew_data=renew_data)
        renew_data = False
        print('...Models trained successfuly\n')