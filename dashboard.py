# -*- coding: utf-8 -*-
import os
import pandas as pd
import dash
import dash_core_components as dcc
import dash_html_components as html
from dash.dependencies import Output, Input
from functions import predict, check_models, check_train_results
from datetime import timedelta
import numpy as np
import yaml
import dill
import warnings


warnings.filterwarnings('ignore')

params = yaml.safe_load(open('params/params.yaml'))
COUNTRIES = params['countries']

# Load last date of data on which the models were trained
last_date = pd.to_datetime(params['data_params']['last_date'])

min_date = last_date + timedelta(1)
max_date = last_date + timedelta(365)
start_date = last_date + timedelta(3)


external_stylesheets = [
    {
        "href": "https://fonts.googleapis.com/css2?"
                "family=Lato:wght@400;700&display=swap",
        "rel": "stylesheet",
    },
]
app = dash.Dash(__name__, external_stylesheets=external_stylesheets)
app.title = "Countries Sales"

app.title = 'Dashboard: Sales Predictions'
app.layout = html.Div(
    children=[
        html.Div(
            children=[
                html.P(children="Dashboard", className="header-title"),
                html.H1(
                    children="Sales Predictions", className="header-title"
                ),
                html.P(
                    children="Sales prediction"
                    " by country for 30 days"
                    " starting from the specified date",
                    className="header-description",
                ),
            ],
            className="header",
        ),
        html.Div(
            children=[
                html.Div(
                    children=[
                        html.Div(children="Country", className="menu-title"),
                        dcc.Dropdown(
                            id='country-filter',
                            options=[
                                {'label': country.upper(), "value": country}
                                for country in COUNTRIES
                            ],
                            value=COUNTRIES[0],
                            clearable=False,
                            className='dropdown',
                        ),
                    ]
                ),
                html.Div(
                    children=[
                        html.Div(children="Date", className="menu-title"),
                        dcc.DatePickerSingle(
                            id='date-v',
                            min_date_allowed = min_date,
                            max_date_allowed = max_date,
                            date = start_date
                        ),
                    ]
                ),
                html.Div(
                    children=[
                        html.Div(children="Model_version", className="menu-title"),
                        dcc.Dropdown(
                            id='model_choice',
                            clearable=False,
                            className='dropdown',
                        ),
                    ]
                ),

            ],
            className='menu'
        ),

        html.Div(
            children=[
                html.Div(
                    children=[
                        html.H1(
                            id='preds_graph_title', className="prediction-title", 
                        ),                        
                        dcc.Graph(
                        id='prediction_graph',
                        config={'displayModeBar':True},                       
                    ),
                    ],
                    className='prediction'
                )
            ],
        ),
        html.Div(
            children=[
                html.Div(
                    children=[
                        html.H1(
                            id='train_test_graph_title', className="prediction-title", 
                        ),                        
                        dcc.Graph(
                        id='train_test_graph',
                        config={'displayModeBar':True},                       
                    ),
                    ],
                    className='prediction'
                )
            ],
        ),

        html.Div(
            children=[
                html.Div(        
                    children=[               
                        html.H1(
                            id='train_graph_title', className="prediction-title"
                        ),
                        dcc.Graph(
                        id='train_data_graph',
                        config={'displayModeBar': True},
                    ),
                    ],
                    className='prediction'
                )
            ],
        ),
        html.Div(
            children=[
                html.Div(        
                    children=[               
                        html.H1(
                            id='test_graph_title', className="prediction-title"
                        ),
                        dcc.Graph(
                        id='test_data_graph',
                        config={'displayModeBar': True},
                    ),
                    ],
                    className='prediction'
                )
            ],
        ),
    ]
)

@app.callback(
    [
        Output('prediction_graph', 'figure'),
        Output('train_test_graph', 'figure'),
        Output('train_data_graph', 'figure'),
        Output('test_data_graph', 'figure'),
        Output('preds_graph_title', 'children'),
        Output('train_test_graph_title', 'children'),
        Output('train_graph_title', 'children'),
        Output('test_graph_title', 'children'),
        Output('model_choice', 'options'),
        Output('model_choice', 'value'),
    ],
    
    [
        Input('country-filter', 'value'),
        Input('date-v', 'date'),
        Input('model_choice', 'value')
    ],
)
def update_charts(country, date, model_name):
    models_path = os.path.join(params['path']['models'], country)
    
    # Default model in dash is test model
    if not model_name or model_name not in os.listdir(models_path):
        model_name = [i for i in os.listdir(models_path) if i.startswith('TEST_')][0]
    # load train results for choice model
    data = dill.load(open(os.path.join(params['path']['train_results'], country, model_name.split('__')[1]), 'rb'))
    
    # Load last rows train data
    num_days_predict = 2 * len(pd.date_range(data['y_test'].index.max(), date))
    last_test_rows = data['y_test'][-num_days_predict:]
    
    # load model
    model = dill.load(
        open(os.path.join(
            params['path']['models'], country, model_name), 'rb'))
    
    # make predictions
    ans = predict(country, date, model)
    ans = data['y_test'][-1:].append(ans.preds)
    
    # load train results
    train_preds = data['train_preds']
    test_preds = data['test_preds']
    
    # load rmse's from train results
    train_rmse = round(data['train_error'], 2)
    test_rmse = round(data['test_error'], 2)
    
    # Prediction chart
    prediction_figure = {
        'data': [
            {
                'x': last_test_rows.index,
                'y': last_test_rows.values,
                'type': 'lines',
                'hovertemplate': '$%{y:.2f}<extra></extra>',
                'name': 'Last predicted values'

            },
            {
                'x': ans.index,
                'y': ans.values,
                'type': 'lines',
                'hovertemplate': '$%{y:.2f}<extra></extra>',
                'name': 'Current predicted values'
            }
        ],
        'layout': {
            'title': {
                'text': f'Prediction on date {pd.Timestamp(date).date()} = {round(ans.values[-1], 2)}',
                'x': 0.5,
                'xanchor': 'centr',
                'font': {'size': 26, 'color': '#555555', 'family': 'Courier'}
            },
            'xaxis': {'fixedrange': False, 'title': 'Date'},
            'yaxis': {'tickprefix': '$', 'fixedrange': False, 'title': 'Revenue'},
            'colorway': ['#17B897', 'FF0050'],
            'hovermode': 'x'
        },
    }
    # train and test data chart
    train_test_figure = {
        'data': [
            {
                'x': data['y_train'].index,
                'y': data['y_train'].values,
                'type': 'lines',
                'hovertemplate': '$%{y:.2f}<extra></extra>',
                'name': 'Train Data'
            },
            {
                'x': data['y_test'].index,
                'y': data['y_test'].values,
                'type': 'lines',
                'hovertemplate': '$%{y:.2f}<extra></extra>',
                'name': 'Test Data'
            },            
        ],
        'layout': {
            'title': {
                'text': f'Mean on Train data = {round(data["y_train_mean"], 2)}, Mean on Test data = {round(data["y_test_mean"], 2)}',
                'x': 0.5,
                'xanchor': 'centr',
                'font': {'size': 26, 'color': '#555555', 'family': 'Courier'}
            },
            'xaxis': {'fixedrange': False, 'title': 'Date'},
            'yaxis': {'tickprefix': '$', 'fixedrange': False, 'title': 'Revenue'},
            'colorway': ['#17B897', 'FF0050'],
            'hovermode': 'x'
        },
    }
    
    # train results chart
    train_data_figure = {
        'data': [
            {
                'x': data['y_train'].index,
                'y': data['y_train'].values,
                'type': 'lines',
                'hovertemplate': '$%{y:.2f}<extra></extra>',
                'name': 'Train Data'
            },
            {
                'x': data['y_train'].index,
                'y': train_preds,
                'type': 'lines',
                'hovertemplate': '$%{y:.2f}<extra></extra>',
                'name': 'Predictions',
                'opacity': 0.5            
            },
            {
                'x': data['y_train'].index,
                'y': np.full(len(data['y_train']), data['y_train'].mean()),
                'type': 'lines',
                'hovertemplate': '$%{y:.2f}<extra></extra>',
                'name': 'BaseLine',
                'opacity': 0.4            
            },
        ],
        'layout': {
            'title': {
                'text': f'Loss: RMSE on train data = {train_rmse}, Baseline Loss = {round(data["base_train_error"], 2)}',
                'x': 0.5,
                'xanchor': 'centr',
                'font': {'size': 26, 'color': '#555555', 'family': 'Courier'}
            },
            'xaxis': {'fixedrange': False, 'title': 'Date'},
            'yaxis': {'tickprefix': '$', 'fixedrange': False, 'title': 'Revenue'},
            'colorway': ['#17B897', '#FF0000', 'FF0050'],
            'hovermode': 'x'
        },
    }
    # validation results chart
    test_data_figure = {
        'data': [
            {
                'x': data['y_test'].index,
                'y': data['y_test'].values,
                'type': 'lines',
                'hovertemplate': '$%{y:.2f}<extra></extra>',
                'name': 'Test Data'
            },
            {
                'x': data['y_test'].index,
                'y': test_preds,
                'type': 'lines',
                'hovertemplate': '$%{y:.2f}<extra></extra>',
                'name': 'Predictions',
                'opacity': 0.5            
            },
            {
                'x': data['y_test'].index,
                'y': np.full(len(data['y_test']), data['y_train'].mean()),
                'type': 'lines',
                'hovertemplate': '$%{y:.2f}<extra></extra>',
                'name': 'BaseLine',
                'opacity': 0.4            
            },
            
        ],
        'layout': {
            'title': {
                'text': f'Loss: RMSE on test data = {test_rmse}, Baseline Loss = {round(data["base_test_error"], 2)}',
                'x': 0.5,
                'xanchor': 'centr',
                'font': {'size': 26, 'color': '#555555', 'family': 'Courier'}
            },
            'xaxis': {'fixedrange': False, 'title': 'Date'},
            'yaxis': {'tickprefix': '$', 'fixedrange': False, 'title': 'Revenue'},
            'colorway': ['#17B897', '#FF0000', 'FF0050'],
            'hovermode': 'x'
        },
    }
    
    model_choice = [
            {'label': f'{model.split("_")[2]}_{model.split("_")[6]} (TEST)' if 'TEST' in model else f'{model.split("_")[2]}_{model.split("_")[6]}',
             'value': model}
            for model in os.listdir(os.path.join(params['path']['models'], country))
        ]
    
    model_value = model_name
    
    preds_graph_title = f'Predictions for {country.upper()} since {(data["y_test"].index.max()+timedelta(1)).date()} to {pd.Timestamp(date).date()}'
    train_test_graph_title = f'Train and test data for {country.upper()}'
    train_graph_title = f'Train results and predictions for {country.upper()}'
    test_graph_title = f'Test results and predictions for {country.upper()}' 
 
    return (prediction_figure, train_test_figure, train_data_figure, 
            test_data_figure, preds_graph_title, train_test_graph_title, 
            train_graph_title, test_graph_title, model_choice, model_value)
    
if __name__ == "__main__":
    for country in COUNTRIES:
        check_models(country)
        check_train_results(country)
    app.run_server(debug=True, host = '0.0.0.0', port=8080, use_reloader=False)