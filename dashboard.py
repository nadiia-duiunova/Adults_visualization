import dash
import pandas as pd
import numpy as np
import os
from dash import html, dcc, Input, Output, State
import dash_bootstrap_components as dbc
import shap
import datetime
from PIL import Image
import plotly.express as px
from joblib import load
import matplotlib
from common_functions import cluster_categorical, cluster_education

matplotlib.use('agg')

#load data from souce
DATA_DIR = 'data'
ASSETS_DIR = 'assets'
countries_file = 'country.csv'
data_file = 'clean_data.csv'
data = pd.read_csv(os.path.join(DATA_DIR, data_file))
TARGET = 'Income'

# create data to feed to model
model_data = data.copy()
model_data = cluster_education(model_data)
model_data = cluster_categorical(model_data)
model_data = model_data[model_data['Workclass'] != 'Without-pay']

X_model = model_data.drop(columns=[TARGET, 'Education-Num'])
y_model = model_data[TARGET]

# load model
model = load('rand_forest_full.joblib')
model.fit(X_model, y_model)

#get Labels for Categorical features
#presonal info
sex_options = data['Sex'].unique()
ethnic_group_options = data['Ethnic group'].unique()

# cluster countries
countries = pd.read_csv(os.path.join(DATA_DIR, countries_file))
country_options = countries['value'].unique()
developed_countries = []
for i in range(countries.shape[0]):
    if countries['developed'][i]:
        developed_countries.append(countries['value'][i])

#family
marital_satus_options = ['Married', 'Single']
relationschip_options = data['Relationship'].unique()

#create app
app = dash.Dash(__name__, external_stylesheets=[dbc.themes.BOOTSTRAP])
server = app.server

#create layout
app.layout = html.Div(
    children = [
        html.Div(
            children = [
                html.H1("Let's check your income"),
                html.P("Hello dear visitor! I am happy to have you here and welcome to my little project.")
            ],
            className='header'
        ),
        html.Div(
            className = 'body',
            children = [
                html.Div(
                    className='project_description',
                    children = [
                        html.P(['Thank you for your interest in my income prediction project. I am  happy to assist you in predicting your income class based on the parameters you provide. Random Forest model, built on the ', 
                               html.A(children = ['Adults dataset'], href = 'https://archive.ics.uci.edu/ml/datasets/Adult', target="_blank", className='link'),
                                ' from the 90s, will analyze the data you provide and provide you with a prediction of your income class, along with the probability of that value being true. '
                                'Additionally, a graph will be generated to show which parameters influenced the decision the most.']), 
                        html.P('In order to proceed with the prediction, I kindly ask you to answer the following questions, all of which are mandatory: ')
                    ]
                ),
                html.Div(
                    className='questionnaire',
                    children = [
                        html.Div(
                            className='col_left',
                            children = [
                                html.Div(
                                className='personal_info',
                                    children = [
                                        html.Br(),
                                        html.H3('Personal Info'),
                                        html.Label("Select your age"),
                                        dcc.Slider(
                                            className='slider',
                                            id = 'age_input',
                                            min=18, max=99, step=1, #this dataset can only predict income for adult people
                                            marks={18: '18',99: '99'},
                                            tooltip={"placement": "bottom", "always_visible": True},
                                            value = 35,
                                            included=False
                                        ),
                                        html.Br(),
                                        html.Label("Select your Sex"),
                                        dcc.RadioItems(
                                            className='radio',
                                            id = 'sex_input',
                                            options = sex_options,
                                            inline = False
                                        ),
                                        html.Br(),
                                        html.Label("Select your Country"),
                                        dcc.Dropdown(
                                            className='dropdown',
                                            id = 'country_input',
                                            options = country_options, 
                                            clearable = False
                                        ),
                                        html.Br(),
                                        html.Label("Select your Ethnic Group"),
                                        dcc.Dropdown(
                                            className='dropdown',
                                            id = 'ethnic_group_input',
                                            options = ethnic_group_options,
                                            clearable = False
                                        )
                                    ]
                                ),
                                html.Div(
                                    className = 'family_info',
                                    children = [
                                        html.Br(),
                                        html.H3('Family Info'),
                                        html.Br(),
                                        html.Label("Select your Marital Status"),
                                        dcc.Dropdown(
                                            className='dropdown',
                                            id = 'marital_status_input',
                                            options = marital_satus_options,
                                            searchable=False,
                                            clearable = False
                                        ),
                                        html.Br(),
                                        html.Label("Do you live with your family?"),
                                        dcc.RadioItems(
                                            className='radio',
                                            id = 'relationschip_input',
                                            options = [{'label': 'Yes', 'value': 'Family'},
                                                       {'label': 'No', 'value': 'Not-in-Family'}
                                                    ],
                                            inline = False
                                        )
                                    ]
                                )
                            ]
                        ),
                        html.Div(
                            className='col_right',
                            children = [
                                html.Div(
                                    className = 'work_education_info',
                                    children = [
                                        html.H3('Education and profesional Info'),
                                        html.Label("Select your education"),
                                        dcc.Slider(
                                            className='slider',
                                            id = 'education_input',
                                            min = 0, max = 3, step = 1,
                                            marks={
                                                0: {'label': 'Undergraduated'},
                                                1: {'label': 'High school'},
                                                2: {'label': 'Some college'},
                                                3: {'label': 'University degree'}
                                            },
                                            value = 2, #default value is 'HS-grad' as it's the most popular value
                                            included=False
                                        ),
                                        html.Br(),
                                        html.Label("How many hours per week do you work?"),
                                        dcc.Slider(
                                            className='slider',
                                            id = 'hpw_input',
                                            min=0, max=100, step=1,
                                            marks = {0: '0', 40: '40', 100: '100'},
                                            tooltip={"placement": "bottom", "always_visible": True},
                                            value = 40, #default value is 40 as it's the most popular value
                                            included=False
                                        ), 
                                        html.Br(),
                                        html.Label("Select your workclass"),
                                        dcc.Dropdown(
                                            className='dropdown',
                                            id = 'workclass_input',
                                            options = [
                                                {'label': 'Private', 'value': 'Private'},
                                                {'label': 'Unincorporated self employment', 'value': 'Self-emp-not-inc'},
                                                {'label': 'Incorporated self employment', 'value': 'Self-emp-inc'},
                                                {'label': 'Local government', 'value': 'Local-gov'},
                                                {'label': 'State government', 'value': 'State-gov'},
                                                {'label': 'Federal government', 'value': 'Federal-gov'},
                                                {'label': 'Without pay', 'value': 'Without-pay'}
                                            ],
                                            searchable=False,
                                            clearable = False
                                        ),
                                        html.Br(),
                                        html.Label("Select your occupation"),
                                        dcc.Dropdown(
                                            className='dropdown',
                                            id = 'occupation_input',
                                            options = [
                                                {'label': 'Profesional specialty', 'value': 'Prof-specialty'},
                                                {'label': 'Executional manager', 'value': 'Exec-managerial'},
                                                {'label': 'Administrative, clerical', 'value': 'Adm-clerical'},
                                                {'label': 'Sales', 'value': 'Sales'},
                                                {'label': 'Machine inspection', 'value': 'Machine-op-inspct'},
                                                {'label': 'Craft repair', 'value': 'Craft-repair'},
                                                {'label': 'Transportation', 'value': 'Transport-moving'},
                                                {'label': 'Handler, creaner', 'value': 'Handlers-cleaners'},
                                                {'label': 'Farming, fishing', 'value': 'Farming-fishing'},
                                                {'label': 'Technical support', 'value': 'Tech-support'},
                                                {'label': 'Protective service', 'value': 'Protective-serv'},
                                                {'label': 'Armed forces', 'value': 'Armed-Forces'},
                                                {'label': 'Private house service', 'value': 'Priv-house-serv'},
                                                {'label': 'Other services', 'value': 'Other-service'}
                                            ],
                                            searchable=False,
                                            clearable = False
                                        )
                                    ]
                                ),
                                html.Div(
                                    className = 'capital_operations_info',
                                    children = [
                                        html.Br(),
                                        html.H3('Capital operations Info'),
                                        html.P('*please try to calculate total sum of your capital operations. If in total you have gained the capital, put the total sum to the first input. If in total you have lost - put this value to the second input (without minus)',
                                               className='capital_explanation'),
                                        html.Label("Gained capital, USD"),
                                        html.Br(),
                                        dcc.Input(
                                            className = 'num_input',
                                            id = 'capital_gain_input',
                                            type="number",
                                            min=0, step=1,
                                            value = 0, #default value is 0 as it's the most popular value
                                            debounce=True
                                        ), 
                                        html.Br(),
                                        html.Br(),
                                        html.Label("Lost capital, USD"),
                                        html.Br(),
                                        dcc.Input(
                                            className = 'num_input',
                                            id = 'capital_loss_input',
                                            type="number",
                                            min=0, step=1,
                                            value = 0, #default value is 0 as it's the most popular value
                                            debounce=True
                                        )
                                    ]
                                )
                            ]
                        )
                    ]
                ),
                html.Div(
                    className='submit-button',
                    children=[dbc.Button('Submit', id='submit-btn', size = 'lg', color ='success', n_clicks=0)]
                ),
                html.Div(
                    className = 'output',
                    children = [
                        html.H3(id='output_text'),
                        html.Div(
                            children = [html.H5(id='explanation_text')],
                            className='explanation_text'
                        ),
                        html.Div(id='output_graph')
                    ]
                )
            ]
        ),
        html.Div(
            className = 'footer',
            children = [
                html.P([
                    'Developed by Nadiia Duiunonova in 2023 based on Adults dataset from ', 
                    html.A(children = ['USI'], href = 'https://archive.ics.uci.edu/ml/datasets/Adult', target="_blank", className='link')
                ]),
                dcc.Link(
                    className = 'link',
                    href='https://github.com/nadiia95/Adults_pet_project', 
                    children = [
                        html.Img(
                            alt = 'GitHub',
                            src=os.path.join(ASSETS_DIR, 'github.svg')
                        )
                    ]
                ),
                dcc.Link(
                    className = 'link',
                    href='https://www.linkedin.com/in/nadiia-duiunova/', 
                    children = [
                        html.Img(
                            alt = 'LinkedIn',
                            src=os.path.join(ASSETS_DIR, 'linkedin.png')
                        )
                    ]
                )
            ]
        )
    ]
)

@app.callback(Output('output_text', 'children'),
    Output('explanation_text', 'children'),
    Output('output_graph', 'children'),
    Input('submit-btn', 'n_clicks'),
    State('age_input', 'value'),
    State('workclass_input', 'value'),
    State('education_input', 'value'),
    State('marital_status_input', 'value'),
    State('occupation_input', 'value'),
    State('relationschip_input', 'value'),
    State('ethnic_group_input', 'value'),
    State('sex_input', 'value'),
    State('country_input', 'value'),
    State('capital_gain_input', 'value'),
    State('capital_loss_input', 'value'),
    State('hpw_input', 'value')
)
def update_output_div(n_clicks, age_input, workclass_input, education_input, marital_satus_input, occupation_input, relationschip_input,
                      ethnic_group_input, sex_input, country_input, capital_gain_input, capital_loss_input, hpw_input):
    
    if n_clicks >0 :
        if workclass_input and marital_satus_input and occupation_input and relationschip_input and ethnic_group_input and sex_input and country_input:
            
            # if the work class is 'without pay' the model wont even start training, because it is impossible to predict income without work in our case
            if workclass_input == 'Without-pay':
                return ['Unfortunatelly, we cannot predict your income if you do not earn money', '', '']
            else:
                # transform country input to the one, model understands: developed or developing
                if country_input in developed_countries:
                    country_input = 'Developed'
                else:
                    country_input = 'Developing'

                #create a datapoint to predict
                X_predictable = pd.DataFrame([[age_input, workclass_input, education_input, marital_satus_input, occupation_input, relationschip_input, ethnic_group_input, 
                                   sex_input, capital_gain_input, capital_loss_input, hpw_input, country_input]], columns=X_model.columns)

                # get prideicted class (more or less than 50K)
                income_prediction = model.predict(X_predictable)[0]
                # get probabilities of both classes
                prediction_prob = model.predict_proba(X_predictable)

                
                # create an explainer to get shapley values for every feature
                explainer = shap.TreeExplainer(model['randomforestclassifier'])
                data_point = model['columntransformer'].transform(X_predictable)
                x_columns_names = model['columntransformer'].get_feature_names_out()
                data_point = pd.DataFrame(data_point, columns = x_columns_names)

                # at this point there are 31 features, as column transformer was applied. 31 features will later be transformed back to those 12, user actually gave.
                shap_values = explainer.shap_values(data_point)

                if income_prediction == '<=50K':
                    income_group = '50K or below'
                    explanation = 'Why so? On the left side of the graph (represented in red), you can see all the variables that may be hindering your financial growth. On the right side (shown in blue), you will find the parameters that can push you towards higher earnings.'
                    prediction_prob = float(prediction_prob[0][0])*100
                    shap_values = shap_values[0]
                    expected_value = explainer.expected_value[0]
                else:
                    income_group = 'above 50K'
                    explanation = 'Why so? First of all, congrats! \U0001F973 You are in the minority of people, who managed to do it over 50K, which is around 25%. On the left side of the graph (represented in red), you will find the parameters that can push you towards higher earnings. On the right side (shown in blue), you can see all the variables that may be hindering your financial growth.'
                    prediction_prob = float(prediction_prob[0][1])*100
                    shap_values = shap_values[1]
                    expected_value = explainer.expected_value[1]

                probability_of_group = round(float(prediction_prob), 4)
                prediction_text = f'With the probability of {probability_of_group}% your income would be {income_group}.'

                # inverse transformation to original columns
                # initialize first 4 not oneHotEncoded categories of standard scaler
                n_categories = [1,1,1,1]

                #append number of categories in each categorical feature
                categorical_features_list = ['Workclass', 'Marital Status', 'Occupation', 'Relationship', 'Ethnic group', 'Sex', 'Country']
                for cat in categorical_features_list:
                    n = X_predictable[cat].nunique()-1 # one column was dropped for each categorical feature during transforming to aviod correlation
                    n_categories.append(n)

                # append last column (education), as it was skipped by column transformer
                n_categories.append(1) 

                # transform list of 31 shap values to 12 values according to number of one-hot-encoded features per original feature
                new_shap_values = []
                for values in shap_values:
                    #split shap values into a list for each feature
                    values_split = np.split(values , np.cumsum(n_categories))
                    
                    #sum values within each list
                    values_sum = [sum(l) for l in values_split]
                    new_shap_values.append(values_sum)

                new_shap_values = np.array([new_shap_values[0][:-1]], dtype=object)

                # create a force plot and store it to assets directory
                shap.force_plot(expected_value, new_shap_values, X_predictable, matplotlib=True, show=False)

                assets_location = '/Users/nadiiaduiunova/adult_no_api/assets/graphs'

                # create unique name for every graph
                timestamp = datetime.datetime.now()
                graph_name = f'{timestamp}.png'
                graph_path = os.path.join(assets_location, graph_name)

                matplotlib.pyplot.savefig(graph_path)
                
                # show the saved graph
                img = np.array(Image.open(graph_path))

                fig = px.imshow(img, color_continuous_scale="gray")
                fig.update_layout(coloraxis_showscale=False)
                fig.update_xaxes(showticklabels=False)
                fig.update_yaxes(showticklabels=False)

                return prediction_text, explanation, dcc.Graph(figure=fig)
        else: 
            return ['Please fill all the fields and press Submit', '', '']
    else:
        return ['', '', '']


if __name__ == '__main__':
    app.run_server(debug=True)