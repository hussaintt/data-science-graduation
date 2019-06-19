import dash
import dash_core_components as dcc
import dash_html_components as html
from pip._vendor.html5lib._trie import py
from plotly.tools import mpl_to_plotly
import dash_core_components as dcc
import pandas as pd
import matplotlib.pyplot as plt
import pprint
import numpy as np
import plotly.graph_objs as go
from dash.dependencies import Output,Input


# Matplotlib
import matplotlib.pyplot as plt

# Plotly
import plotly.plotly as py
import plotly.tools as tls

df= pd.read_csv("D:\\data anlysis\\data sets\\NCHS_-_Leading_Causes_of_Death__United_States.csv")


my_df=df[['Year','Cause Name','State','Deaths']]
new_df1 =my_df.groupby('Year')
Year_df=new_df1.sum()
Year_df.sort_values('Year',ascending=False,inplace=True)
#plt.show()
##################################################
my_df2=df[['Year','Cause Name','State','Age-adjusted Death Rate']]
new_df2 =my_df2.groupby('Year')
Year_df2=new_df2.mean()
Year_df2.sort_values('Year',ascending=False,inplace=True)
##################STATE DATA FRAME################
new_df2 =my_df.groupby('State')
State_df=new_df2.sum()
State_df.drop(columns=['Year'],inplace=True)
State_df.sort_values('Deaths',ascending=False,inplace=True)
#State_df.plot.bar(figsize=(14,6))
topdf=State_df[1:11]
####################Cause name data frame with total deaths ##############################

new_df3=my_df.groupby('Cause Name')
Cause_df=new_df3.sum()
Cause_df.drop(columns=['Year'],inplace=True)
Cause_df.sort_values('Deaths',ascending=False,inplace=True)
print(Cause_df)

###################adjsusted death rate of canser by compared to year ##################################
dfAge=df[['Year','Cause Name','State','Age-adjusted Death Rate']]
Canser_df2=dfAge[dfAge['Cause Name']=='Cancer']
cn2=Canser_df2.groupby('Year')
cn3=cn2.mean()
print(cn3)
################################################################################################
Canser_State=dfAge[dfAge['Cause Name']=='Cancer']
cn2s=Canser_State.groupby('State')
cn3s=cn2s.mean()
print(cn3s)
###########################Heart disease adjusted death rate by year ###########################
Heart_df=dfAge[dfAge['Cause Name']=='Heart disease']
Hr1=Heart_df.groupby('Year')
Hr2=Hr1.mean()

#####################Alzheimer's disease adjusted death rate by year #######################

AL_df=dfAge[dfAge['Cause Name']=="Alzheimer's disease"]
al1=AL_df.groupby('Year')
al2=al1.mean()

###################new data frame for relation between state and the year and the adjusted death rate ############
state=dfAge['State'].unique().tolist()
causes=dfAge['Cause Name'].unique().tolist()
year=dfAge['Year'].unique().tolist()

graph_df=pd.DataFrame(index=year,columns = state)
graph_df.sort_index(inplace=True)
gb = dfAge.groupby('State')
for s in state :
    x=gb.get_group(s).set_index('Year').sort_index()
    allcauses=x[x['Cause Name']=='All causes']

    graph_df[s]=allcauses['Age-adjusted Death Rate'].tolist()

newgh=graph_df[["District of Columbia",'Arizona','Washington','New York','North Carolina']]
####################################################################################################################


###################new data frame for relation between cause of death and the adjusted death rate ##################
graph_df2=pd.DataFrame(index=year,columns = causes)
graph_df2.sort_index(inplace=True)

gb = dfAge.groupby('Cause Name')
for s in causes :
    x=gb.get_group(s).set_index('Year').sort_index()
    allcauses=x[x['State']=='United States']
    graph_df2[s]=allcauses['Age-adjusted Death Rate'].tolist()

no_all=graph_df2
no_all.drop(columns=['All causes'],inplace=True)
no_allCause=Cause_df

print(newgh)

no_allCause = Cause_df[Cause_df.index != 'All causes']

app = dash.Dash()




app.layout= html.Div([
html.H1('Data Analysis on causes of death in us from 1999-2016'),

#dcc.Graph( id="plot"),
html.Div([
    html.Div([
        html.H3('Choose one of those Bar charts  '),
        dcc.Dropdown(
            id='first-dropdown1',
            options=[
                {'label': 'Total deaths in every state', 'value': 'State-Deaths'},
                {'label': 'Total deaths in us every year', 'value': 'Year-Deaths-Bar'},
                {'label': 'The death rate in every year', 'value': 'Year-Deaths-Bar-rate'},
                {'label': 'Total deaths by every Cause of death', 'value': 'Cause-Deaths-Bar'},
                {'label': 'The death rate of each Cause of death every year ', 'value': 'All-causes'},
                {'label': 'Comparison between the death rate of Canser and Heart disease ', 'value': 'canser-vs-heart'},
                {'label': "Influenza and pneumonia vs Diabetes vs Alzheimer's disease", 'value': 'Three-Causes'}

            ],
            value='State-Deaths'
        ),

        dcc.Graph(id="plot"),

html.Div([
dcc.Graph(
        id="Year-Deaths-",

        figure={
            'data': [go.Pie(labels=no_allCause.index, values=no_allCause['Deaths'].tolist(),
                            marker={'colors': ['#EF963B', '#C93277', '#349600', '#EF533B', '#57D4F1']},
                            textinfo='label')],
        'layout':{'title':'Deaths by every Cause of death'}
           }
    )]),
    ], className="six columns"),

        html.Div([
            html.H3('Choose one of those Scatter charts '),
        dcc.Dropdown(
                id='first-dropdown2',
         options=[
                     {'label': 'Total deaths in us every year', 'value': 'Year-Deaths'},
                     {'label': 'Total deaths in every state', 'value': 'State-Deaths-Chart'},
                     {'label': 'Canser death rate', 'value': 'Canser-Rate-Chart'},
                     {'label': 'Heart death rate', 'value': 'Heart-Rate-Chart'},
                     {'label': 'Alzheimer s disease death rate', 'value': 'AL-Rate-Chart'},
                     {'label': 'States and the and avarage death rate ', 'value': 'compare-Rate-Chart'}

         ],
        value='Year-Deaths'
    ),


            dcc.Graph( id="plot2"),
html.Div([
dcc.Graph(
        id="Year-Deaths-Ba",
        figure={
            'data': [go.Pie(labels=topdf.index, values=topdf['Deaths'].tolist(),
                            marker={'colors': ['#EF963B', '#C93277', '#349600', '#EF533B', '#57D4F1']},
                            textinfo='label')],
            'layout': {'title': 'Deaths in Top 10 states'}

        }
    )]),



        ], className="six columns")




],className="row"),








app.css.append_css({
    'external_url': 'https://codepen.io/chriddyp/pen/bWLwgP.css'
})


])

#df.iplot(kind='bar', filename='cufflinks/grouped-bar-chart')

@app.callback(dash.dependencies.Output('plot','figure'),
              [dash.dependencies.Input('first-dropdown1','value')]
              )


def update_fig_1(input):

    if(input=="Year-Deaths-Bar"):
        figure = {
            'data': [go.Bar(
                x=Year_df.index,
                y=Year_df['Deaths']
            )],
            'layout': {
                'title': 'Total deaths in us every year'
            }}
    if (input == "Year-Deaths-Bar-rate"):
        figure = {
            'data': [go.Bar(
                x=Year_df2.index,
                y=Year_df2['Age-adjusted Death Rate']
            )],
            'layout': {
                'title': 'The death rate in every year'
            }}
    if(input=="State-Deaths"):
            figure={

                'data':[go.Bar(
                    x=topdf.index,
                    y=topdf['Deaths']

                )],
                'layout': {
                    'title': 'Total deaths in every state'
                }


            }
    if(input=='Cause-Deaths-Bar'):
            figure = {
               'data': [go.Bar(
                x=Cause_df.index,
                y=Cause_df['Deaths']

            )],
            'layout': {
                'title': 'Total deaths by every Cause of death'
            }}

    if(input=='All-causes'):
        trase0=go.Bar(x=no_all.index, y=no_all['Cancer'],name="Cancer")
        trase1 = go.Bar(x=no_all.index, y=no_all['Heart disease'],name="Heart disease")
        trase2 = go.Bar(x=no_all.index, y=no_all['Stroke'],name='Stroke')
        trase3 = go.Bar(x=no_all.index, y=no_all["Alzheimer's disease"],name="Alzheimer's disease")
        trase4 = go.Bar(x=no_all.index, y=no_all['Diabetes'],name='Diabetes')
        trase5 =go.Bar(x=no_all.index, y=no_all['Influenza and pneumonia'],name='Influenza and pneumonia')
        trase6 = go.Bar(x=no_all.index, y=no_all['Kidney disease'],name='Kidney disease')
        trase7 = go.Bar(x=no_all.index, y=no_all['Suicide'],name='Suicide')
        data=[trase0,trase1,trase2,trase3,trase4,trase5,trase6,trase6,trase7]
        figure = {
            'data':data ,
            'layout': {
                'title': 'The death rate of each Cause of death every year'
            }}
    if (input == "canser-vs-heart"):

            trase0 = go.Bar(x=no_all.index, y=no_all['Cancer'], name="Cancer")
            trase1 = go.Bar(x=no_all.index, y=no_all['Heart disease'], name="Heart disease")
            data = [trase0, trase1]
            figure = {
                'data': data,
                'layout': {
                    'title': 'Comparison between the death rate of Canser and Heart disease'
                }}
    if (input == "Three-Causes"):

            trase0 = go.Bar(x=no_all.index, y=no_all['Diabetes'], name="Diabetes")
            trase1 = go.Bar(x=no_all.index, y=no_all['Influenza and pneumonia'], name="Influenza and pneumonia")
            trase2 = go.Bar(x=no_all.index, y=no_all["Alzheimer's disease"], name="Alzheimer's disease")

            data = [trase2, trase0,trase1]
            figure = {
                'data': data,
                'layout': {
                    'title': "Influenza and pneumonia vs Diabetes vs Alzheimer's disease"
                }}
    return figure

@app.callback(dash.dependencies.Output('plot2','figure'),
              [dash.dependencies.Input('first-dropdown2','value')]
              )

def update_fig_2(input):
    if(input=="Year-Deaths"):
        figure = {
            'data': [go.Scatter(
                x=Year_df.index,
                y=Year_df['Deaths']
            )],
            'layout': {
                'title': 'Total deaths in us every year'
            }}
    if (input == "State-Deaths-Chart"):
        figure = {

            'data': [go.Scatter(
                x=topdf.index,
                y=topdf['Deaths']

            )],
            'layout': {
                'title': 'Total deaths in every state'
            }}
    if(input=='Canser-Rate-Chart'):
        figure = {

            'data': [go.Scatter(
                x=cn3.index,
                y=cn3['Age-adjusted Death Rate']

            )],
            'layout': {
                'title': 'The average adjusted rate of canser in every year'
            }}

    if (input == 'Heart-Rate-Chart'):
        figure = {

            'data': [go.Scatter(
                x=Hr2.index,
                y=Hr2['Age-adjusted Death Rate']

            )],
            'layout': {
                'title': 'The average adjusted rate of Heart disease in every year'
            }}

    if (input == "AL-Rate-Chart"):
        figure = {

            'data': [go.Scatter(
                x=al2.index,
                y=al2['Age-adjusted Death Rate']

            )],
            'layout': {
                'title': 'The average adjusted rate of Alzheimer s disease in every year'
            }}

    if (input == "compare-Rate-Chart"):
        District_of_Columbia=go.Scatter(x=newgh.index,y=newgh['District of Columbia'],name="District of Columbia")
        Arizona=go.Scatter(x=newgh.index,y=newgh['Arizona'],name="Arizona")
        New_York=go.Scatter(x=newgh.index,y=newgh['New York'],name="New York")
        North_Carolina=go.Scatter(x=newgh.index,y=newgh['North Carolina'],name="North Carolina")

        data=[District_of_Columbia,Arizona,New_York,North_Carolina]



        figure = {

            'data': data ,
            'layout': {
                'title': "The death rate of some states in every year"
            }}
    return figure



if __name__ == '__main__':
    app.run_server(port=3030)
