import pandas as pd
import dash
import dash_bootstrap_components as dbc
from dash import dcc
from dash import html
from dash.dependencies import Input, Output
import plotly.graph_objects as go
import plotly.express as px
from wordcloud import WordCloud
import matplotlib.pyplot as plt
import seaborn as sns
import io
import base64
from wordcloud import STOPWORDS

data = pd.read_csv('bfro_cleaned.csv')

data['date'] = pd.to_datetime(data['date'])
data['year'] = data['date'].dt.year
data['month'] = data['date'].dt.month_name()

data['year'] = pd.to_numeric(data['year'], errors='coerce')
data = data.dropna(subset=['year'])
data['year'] = data['year'].astype(int)

min_year = data['year'].min()
max_year = data['year'].max()

start_decade = (min_year // 10) * 10
marks = {year: str(year) for year in [1960, 1970, 1980, 1990, 2000, 2010]}

valid_data = data.dropna(subset=['latitude', 'longitude'])

state_options = [{"label": state, "value": state} for state in sorted(valid_data['state'].unique())]


app = dash.Dash(__name__, external_stylesheets=[dbc.themes.BOOTSTRAP])

server = app.server

app.layout = dbc.Container([
    dbc.Row([
        dbc.Col([
            html.H1("Bigfoot Sightings", style={'textAlign': 'center', 'color': '#4287f5', 'display': 'inline-block', 'verticalAlign': 'middle'}),
            html.Img(src='/assets/bf.png', style={'height': '50px', 'display': 'inline-block', 'verticalAlign': 'middle', 'marginLeft': '10px'})
        ], width=12, style={'textAlign': 'center'})
    ], className="mb-3"),

    dbc.Row([
        dbc.Col([
            dbc.Card([ 
                dbc.CardBody([
                    dbc.Row([ 
                        dbc.Col([dcc.Dropdown(id="state-filter", options=state_options, multi=True, placeholder="Filter by state")], width=12)
                    ], className="mb-3"),
                    dbc.Row([ 
                        dbc.Col([dcc.RangeSlider(id="date-slider", min=min_year, allowCross=False, max=max_year, step=1, marks=marks, value=[min_year, max_year], tooltip={"placement": "bottom", "always_visible": True})], width=12)
                    ], className="mb-3"),
                    dbc.Row([ 
                        dbc.Col(dcc.Graph(id="map"), width=12),
                    ], className="mb-3"),
                    dbc.Row([ 
                        dbc.Col(dbc.Card(dbc.CardBody(html.Div(id="info-text-box", style={'overflowY': 'scroll', 'maxHeight': '150px'})), body=True, style={'marginTop': '10px'}),)
                    ], className="mb-3"),
                ])
            ], body=True),
        ], width=12, lg=6),

        dbc.Col([
            dbc.Card([ 
                html.Div([ 
                    dcc.RadioItems(id='histogram-toggle', options=[
                        {'label': 'By Year', 'value': 'year'},
                        {'label': 'By State', 'value': 'state'}
                    ], value='year', labelStyle={'display': 'inline-block', 'paddingRight': '20px'})
                ]), 
                html.Div(id='bar-plot')
            ], body=True, style={'marginBottom': '20px'}),
            dbc.Card(html.Div(id="word-cloud"), body=True, style={'marginBottom': '20px'}),
        ], width=12, lg=6)
    ]),

    dbc.Row([ 
        dbc.Col(dbc.Card(dcc.Graph(id="heatmap"), body=True, style={'marginBottom': '20px'}), width=12)
    ])
], fluid=True, style={'backgroundColor': '#d5e6f7'})



@app.callback(
    [Output("map", "figure"),
     Output("word-cloud", "children"),
     Output("info-text-box", "children"),
     Output("bar-plot", "children"),
     Output("heatmap", "figure")],
    [Input("state-filter", "value"),
     Input("date-slider", "value"),
     Input("map", "clickData"),
     Input("histogram-toggle", "value")]
)
def update_map_and_visuals(selected_states, date_range, click_data, histogram_toggle):
    filtered_data = data.copy()

    if selected_states:
        filtered_data = filtered_data[filtered_data['state'].isin(selected_states)]

    filtered_data = filtered_data[(filtered_data['year'] >= date_range[0]) & (filtered_data['year'] <= date_range[1])]
    
    map_fig = px.scatter_mapbox(
        filtered_data,
        lat="latitude",
        lon="longitude",
        color="classification",
        hover_name="number",
        hover_data={
            "state": True,
            "county": True,
            "date": True,
            "latitude": False,
            "longitude": False
        },
        zoom=2.5,
        height=300,
        color_discrete_sequence=["#1f77b4", "#ff7f0e", "#2ca02c"],
    )

    zoom_level = 2.5
    map_center = dict(lat=37.0902, lon=-95.7129)

    if selected_states:
        state_data = valid_data[valid_data['state'].isin(selected_states)]
        lat_min, lat_max = state_data['latitude'].min(), state_data['latitude'].max()
        lon_min, lon_max = state_data['longitude'].min(), state_data['longitude'].max()
        zoom_level = 5
        map_center = dict(lat=(lat_min + lat_max) / 2, lon=(lon_min + lon_max) / 2)

    map_fig.update_layout(
        mapbox_style="carto-positron",
        hoverlabel=dict(
            bgcolor="white",
            font_size=12,
            font_family="Arial",
            bordercolor="gray",
            namelength=-1,
            align="left"
        ),
        legend=dict(
            x=0.3,
            y=0.3,
            title=None,
            xanchor='right',
            yanchor='top',
            bgcolor='rgba(255, 255, 255, 0.5)'
        ),
        margin={"l": 0, "r": 0, "t": 0, "b": 0},
        mapbox=dict(
            zoom=zoom_level,
            center=map_center,
        ),
        height=300
    )

    text = " ".join(filtered_data['title'].dropna())
    stopwords = set(STOPWORDS)
    stopwords.update(['report', 'near', 'sighting', 'possible', 'see'])
    text = " ".join(filtered_data['title'].dropna().str.lower())
    wordcloud = WordCloud(width=800, height=400, background_color='white', colormap='copper_r', max_words=50, stopwords=stopwords, random_state=42).generate(text)

    img_buf = io.BytesIO()
    wordcloud.to_image().save(img_buf, format='PNG')
    wordcloud_img_str = base64.b64encode(img_buf.getvalue()).decode('utf-8')
    wordcloud_html = html.Div([
        html.H5("Wordcloud for sighting reports in selected year range", style={'textAlign': 'center'}),
        html.Img(src=f"data:image/png;base64,{wordcloud_img_str}", style={'width': '100%'})
    ])

    sightings_per_year = filtered_data.groupby('year').size().reset_index(name='total_sightings')

    fig = go.Figure()

    if histogram_toggle == 'year':
        fig.add_trace(go.Bar(
            x=sightings_per_year['year'],
            y=sightings_per_year['total_sightings'],
            name='Total Sightings',
            marker=dict(color='#6042f5', opacity=0.7)
        ))
        fig.update_layout(
            title=dict(
            text='Bigfoot Sightings by Year',
            x=0.5,
            xanchor='center'
            ),
            xaxis_title="Year",
            yaxis_title='Total Sightings',
            bargap=0.2,
            height=300,
            margin={"l": 0, "r": 0, "t": 30, "b": 0}
        )
        fig.update_traces(marker=dict(color='#331317', opacity=0.7))
        
        fig.add_vrect(
            x0=1973, x1=1978,
            fillcolor="LightSalmon", opacity=0.5,
            layer="below", line_width=0, 
            legendgroup="1993-2002", name="six million dollar Era"
        )
        fig.add_annotation(
            x=1990,
            y=75,
            text="X-files span <br> (Science Fiction TV Series)",
            showarrow=False,
            yshift=10,
            textangle=-90,
            font=dict(color="#f5704e")
        )

        fig.add_vrect(
            x0=1993, x1=2002,
            fillcolor="LightSalmon", opacity=0.5,
            layer="below", line_width=0, 
            legendgroup="1993-2002", name="X-Files Era"
        )

        fig.add_annotation(
            x=1970,
            y=75,
            text="Six Million Dollar Man span <br> (Science Fiction TV Series)",
            showarrow=False,
            yshift=10,
            textangle=-90,
            font=dict(color="#f5704e")
        )


      
    else:
        sightings_per_state = filtered_data.groupby('state').size().reset_index(name='total_sightings')
        sightings_per_state_sorted = sightings_per_state.sort_values(by='total_sightings', ascending=False)

        top_15_states = sightings_per_state_sorted
        fig.add_trace(go.Bar(
            x=top_15_states['state'],
            y=top_15_states['total_sightings'],
            name='Total Sightings by State',
            marker=dict(color='#6042f5', opacity=0.7)
        ))

        fig.update_layout(
            title=dict(
                text='Bigfoot sightings by State',
                x=0.5,
                xanchor='center'
            ),
            xaxis_title=None,
            yaxis_title='Total Sightings',
            bargap=0.2,
            height=300,
            margin={"l": 0, "r": 0, "t": 30, "b": 0},
            xaxis=dict(tickmode='array', tickvals=sightings_per_state_sorted['state'])
        )
        fig.update_traces(marker=dict(color='#331317', opacity=0.7))

     

    histogram_html = dcc.Graph(figure=fig)

    info_text = ""

    if click_data:
        unique_id = click_data['points'][0]['hovertext']
        clicked_row = valid_data[valid_data['number'] == unique_id].iloc[0]

        title = clicked_row['title']
        observed = clicked_row['observed']
        date_text = clicked_row['date'].strftime("%B %d, %Y")
        confidence = clicked_row['classification']

        info_text = html.Div([
            html.H6(title),
            html.P(f"Date: {date_text}"),
            html.P(f"Sighting confidence: {confidence}"),
            html.P(observed)
        ], style={'backgroundColor': '#f0f0f0', 'padding': '5px', 'borderRadius': '5px'})
    else:
        info_text = html.Div([
            html.P("Click an observation to see the sighting report")
        ], style={'backgroundColor': '#f0f0f0', 'padding': '5px', 'borderRadius': '5px'})

    visibility_data = filtered_data.groupby(['weather_class', 'year']).size().unstack(fill_value=0)
    
    visibility_data_normalized = visibility_data.div(visibility_data.sum(axis=0), axis=1)

    heatmap_fig = px.imshow(
        visibility_data_normalized,
        labels=dict(x="Year", y="Weather classification", color="Proportion"),
        color_continuous_scale="solar"
    )
    heatmap_fig.update_layout(
        title={
            'text': "Normalized sightings by weather classification",
            'y': 0.95,
            'x': 0.5,
            'xanchor': 'center',
            'yanchor': 'top'
        },
        margin={"t": 50},
    )
    heatmap_fig.update_layout(height=350)

    return map_fig, wordcloud_html, info_text, histogram_html, heatmap_fig


if __name__ == "__main__":
    app.run_server(debug=True)
