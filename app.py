import os
import re
import io
import base64
import pandas as pd
import numpy as np
import spacy
from flask import Flask, render_template, request, jsonify
from transformers import pipeline
import joblib
from fuzzywuzzy import fuzz, process
import plotly.express as px

# Dash imports
from dash import Dash, dcc, html, Input, Output, State

# Initialize Flask app
app = Flask(__name__)
UPLOAD_FOLDER = 'uploads'
if not os.path.exists(UPLOAD_FOLDER):
    os.makedirs(UPLOAD_FOLDER)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

# Load NLP models and summarizer
nlp = spacy.load("en_core_web_sm")
summarizer = pipeline("summarization", model="facebook/bart-large-cnn")

# Load trained model if available
MODEL_PATH = 'query_model.pkl'
model = joblib.load(MODEL_PATH) if os.path.exists(MODEL_PATH) else None

# Global variables for data storage
df = None              # Main DataFrame from uploaded file
last_query_df = None   # Most recent query result

# -----------------------
# Flask Endpoints
# -----------------------

@app.route('/')
def index():
    return render_template('index.html')

# Upload endpoint supporting Excel, CSV, and JSON
@app.route('/upload', methods=['POST'])
def upload_file():
    global df, last_query_df
    if 'file' not in request.files:
        return jsonify({'error': 'No file uploaded'}), 400
    file = request.files['file']
    if file.filename == '':
        return jsonify({'error': 'No selected file'}), 400

    file_path = os.path.join(app.config['UPLOAD_FOLDER'], file.filename)
    file.save(file_path)
    
    ext = os.path.splitext(file.filename)[1].lower()
    try:
        if ext in ['.xls', '.xlsx']:
            df = pd.read_excel(file_path)
        elif ext == '.csv':
            df = pd.read_csv(file_path)
        elif ext == '.json':
            df = pd.read_json(file_path)
        else:
            return jsonify({'error': 'Unsupported file type.'}), 400
    except Exception as e:
        return jsonify({'error': 'Error reading file: ' + str(e)}), 500
    
    # Normalize column names and convert date column if present.
    df.columns = df.columns.str.strip().str.lower()
    if 'date' in df.columns:
        df['date'] = pd.to_datetime(df['date'], errors='coerce')
    last_query_df = None
    return jsonify({'message': 'File uploaded successfully', 'columns': df.columns.tolist()})

# Query processing endpoint
@app.route('/query', methods=['POST'])
def process_query():
    global df, last_query_df
    if df is None or df.empty:
        return jsonify({'error': 'No file uploaded or file is empty!'}), 400
    
    data = request.json
    query = data.get('query', '').strip()
    sort_by = data.get('sort_by', None)
    sort_order = data.get('sort_order', 'desc')
    
    if not query:
        return jsonify({'error': 'Empty query'}), 400

    # Use spaCy to extract entities from the query
    doc = nlp(query)
    entities = {ent.label_: ent.text for ent in doc.ents}
    
    # Optional month filtering if a month is mentioned and a 'date' column exists
    months = ['january','february','march','april','may','june',
              'july','august','september','october','november','december']
    month_dict = {m: i+1 for i, m in enumerate(months)}
    filter_month = next((m for m in months if m in query.lower()), None)
    if filter_month and 'date' in df.columns:
        df_query = df[df['date'].dt.month == month_dict[filter_month]]
    else:
        df_query = df.copy()
    
    # Map common keywords to DataFrame columns
    column_mapping = {
        "sales": "sales",
        "product": "product",
        "profit": "profit",
        "discount": "discount band",
        "units sold": "units sold",
        "revenue": "gross sales",
        "cost": "cogs",
        "date": "date",
        "region": "region",
        "category": "product category"
    }
    matched_columns = []
    for keyword, col in column_mapping.items():
        if keyword in query.lower() or any(ent.text.lower() == keyword for ent in doc.ents):
            matched_columns.append(col)
    if not matched_columns:
        closest = process.extractOne(query, df.columns.tolist(), scorer=fuzz.partial_ratio)
        if closest and closest[1] >= 80:
            matched_columns.append(closest[0])
    if not matched_columns:
        return jsonify({'message': 'No relevant columns found. Please check your query or file.'}), 400
    
    # Determine query intent
    intent = "default"
    if 'count' in query.lower():
        intent = "count"
    elif 'total' in query.lower() or 'sum' in query.lower():
        intent = "sum"
    elif 'average' in query.lower() or 'avg' in query.lower():
        intent = "average"
    elif 'group by' in query.lower() or 'divide by' in query.lower():
        intent = "group"
    elif 'filter' in query.lower():
        intent = "filter"
    elif 'maximum' in query.lower() or 'min' in query.lower():
        intent = "extreme"
    elif re.search(r'top (\d+)', query.lower()):
        intent = "top_n"
    elif re.search(r'least (\d+)', query.lower()) or re.search(r'bottom (\d+)', query.lower()):
        intent = "least_n"
    
    try:
        if intent == "sum":
            query_df = pd.DataFrame([df_query[matched_columns].sum()])
        elif intent == "average":
            query_df = pd.DataFrame([df_query[matched_columns].mean()])
        elif intent == "count":
            query_df = pd.DataFrame([df_query[matched_columns].count()])
        elif intent == "group":
            group_column = 'product' if 'product' in df_query.columns else df_query.columns[0]
            query_df = df_query.groupby(group_column)[matched_columns].sum().reset_index()
        elif intent == "filter":
            query_df = df_query[matched_columns].head(10)
        elif intent == "extreme":
            if 'maximum' in query.lower():
                query_df = pd.DataFrame([df_query[matched_columns].max()])
            else:
                query_df = pd.DataFrame([df_query[matched_columns].min()])
        elif intent == "top_n":
            match = re.search(r'top (\d+)', query.lower())
            top_n = int(match.group(1)) if match else 10
            query_df = df_query.sort_values(by=matched_columns, ascending=False).head(top_n)[matched_columns]
        elif intent == "least_n":
            match = re.search(r'least (\d+)', query.lower()) or re.search(r'bottom (\d+)', query.lower())
            least_n = int(match.group(1)) if match else 10
            query_df = df_query.sort_values(by=matched_columns, ascending=True).head(least_n)[matched_columns]
        else:
            query_df = df_query[matched_columns].head(10)
    except Exception as e:
        return jsonify({'error': f"Error processing query: {str(e)}"}), 500
    
    if sort_by is None and matched_columns:
        sort_by = matched_columns[0]
    if sort_by and sort_by in query_df.columns:
        ascending = sort_order == 'asc'
        query_df = query_df.sort_values(by=sort_by, ascending=ascending)
    
    last_query_df = query_df.copy()
    result = query_df.to_dict(orient='records')
    return jsonify({'results': result})

# Summarize endpoint
@app.route('/summarize', methods=['POST'])
def summarize_text():
    data = request.json
    text = data.get('text', '').strip()
    if not text:
        return jsonify({'error': 'No text provided for summarization'}), 400
    try:
        summary = summarizer(text, max_length=150, min_length=50, do_sample=False)
        return jsonify({'summary': summary[0]['summary_text']})
    except Exception as e:
        return jsonify({'error': f"Error during summarization: {str(e)}"}), 500

# File insights endpoint – renders an insights template with details
@app.route('/insights', methods=['GET'])
def file_insights():
    global df
    if df is None or df.empty:
        return "No file uploaded or file is empty!", 400
    shape = df.shape
    columns = df.columns.tolist()
    dtypes = df.dtypes.astype(str).to_dict()
    missing = df.isnull().sum().to_dict()
    unique = df.nunique().to_dict()
    numeric_columns = df.select_dtypes(include=['number']).columns
    description = df[numeric_columns].describe().to_dict() if len(numeric_columns) > 0 else {}
    head = df.head(5).to_dict(orient='records')
    insights = {
        'shape': shape,
        'columns': columns,
        'dtypes': dtypes,
        'missing': missing,
        'unique': unique,
        'describe': description,
        'head': head
    }
    return render_template('insights.html', insights=insights)

# Unified Chart Generation Endpoint
@app.route('/generate_chart', methods=['GET'])
def generate_chart_endpoint():
    global df, last_query_df
    # Use query results if available; otherwise, use the full DataFrame.
    data_source = last_query_df if (last_query_df is not None and not last_query_df.empty) else df
    if data_source is None or data_source.empty:
        return jsonify({'error': 'No data available for chart generation'}), 400

    chart_type = request.args.get('type', 'bar').strip().lower()
    title = request.args.get('title', 'Chart Title')
    # For XY charts:
    x = request.args.get('x')
    y = request.args.get('y')
    # For single column charts:
    column = request.args.get('column')

    try:
        if x:  # XY chart mode
            if x not in data_source.columns:
                return jsonify({'error': f'X column "{x}" not found.'}), 400
            if not y:
                return jsonify({'error': 'Y parameter is required for XY chart.'}), 400
            if y not in data_source.columns:
                return jsonify({'error': f'Y column "{y}" not found.'}), 400
            fig = generate_plotly_chart(data_source, x=x, y=y, chart_type=chart_type, title=title)
        elif column:  # Single column mode
            if column not in data_source.columns:
                return jsonify({'error': f'Column "{column}" not found in the data.'}), 400
            fig = generate_plotly_chart(data_source, x=column, chart_type=chart_type, title=title)
        else:
            return jsonify({'error': 'Please provide parameters for chart generation: either "column" for single chart or "x" and "y" for an XY chart.'}), 400

        # Generate chart image using Plotly (requires kaleido)
        img_bytes = fig.to_image(format="png")
        img_base64 = base64.b64encode(img_bytes).decode('utf-8')
        return jsonify({'chart_url': f"data:image/png;base64,{img_base64}"})
    except Exception as e:
        return jsonify({'error': f"Chart generation failed: {str(e)}"}), 500

# Shared chart-generation function using Plotly Express
def generate_plotly_chart(data, x=None, y=None, chart_type="scatter", title="Chart"):
    chart_type = chart_type.lower()
    if chart_type == "scatter":
        if y:
            fig = px.scatter(data, x=x, y=y, title=title, template="plotly_white")
        else:
            fig = px.scatter(data, x=x, title=title, template="plotly_white")
    elif chart_type == "line":
        if y:
            fig = px.line(data, x=x, y=y, title=title, markers=True, template="plotly_white")
        else:
            fig = px.line(data, x=x, title=title, markers=True, template="plotly_white")
    elif chart_type == "bar":
        if y:
            fig = px.bar(data, x=x, y=y, title=title, template="plotly_white")
        else:
            fig = px.bar(data, x=x, title=title, template="plotly_white")
    elif chart_type == "histogram":
        fig = px.histogram(data, x=x, title=title, template="plotly_white")
    elif chart_type == "pie":
        if y and pd.api.types.is_numeric_dtype(data[y]):
            fig = px.pie(data, names=x, values=y, title=title, template="plotly_white")
        else:
            pie_data = data[x].value_counts().reset_index()
            pie_data.columns = [x, 'count']
            fig = px.pie(pie_data, names=x, values='count', title=title, template="plotly_white")
    else:
        raise ValueError("Unsupported chart type")
    return fig

# -----------------------
# Dash Dashboard Integration
# -----------------------

external_stylesheets = ["/static/style.css"]
dash_app = Dash(__name__, server=app, url_base_pathname='/dashboard/', external_stylesheets=external_stylesheets)

def serve_layout():
    global df
    options = [{'label': col, 'value': col} for col in df.columns] if df is not None else []
    return html.Div(
        className='dash-container',
        style={'width': '100vw', 'maxWidth': '100vw', 'margin': '0'},
        children=[
            html.Div("Dashboard", className="dash-header"),
            html.Div([
                html.A("Home", href="/", className="dash-nav"),
                html.A("File Insights", href="/insights", className="dash-nav")
            ], className="dash-nav"),
        # Chart Mode Selection
        html.Div([
            html.Label("Chart Mode:"),
            dcc.RadioItems(
                id='chart-mode',
                options=[
                    {'label': 'Single Column', 'value': 'single'},
                    {'label': 'XY Chart', 'value': 'xy'}
                ],
                value='single',
                labelStyle={'display': 'inline-block', 'margin-right': '20px'}
            )
        ], style={'margin-bottom': '50px'}),
        # Filter Input
        html.Div([
            html.Label("Filter Data (optional, e.g., sales > 100):"),
            dcc.Input(id='filter-input', type='text', placeholder="Enter filter condition", style={'width': '50%'})
        ], style={'margin-bottom': '20px'}),
        # Single Column Chart Inputs
        html.Div(id='chart-inputs-container', children=[
            html.Div([
                html.Label("Column:"),
                dcc.Dropdown(
                    id='column-dropdown',
                    options=options,
                    placeholder="Select Column"
                )
            ], style={'margin-bottom': '10px'}),
            html.Div([
                html.Label("Chart Type:"),
                dcc.Dropdown(
                    id='chart-type-dropdown',
                    options=[
                        {'label': 'Bar', 'value': 'bar'},
                        {'label': 'Line', 'value': 'line'},
                        {'label': 'Pie', 'value': 'pie'},
                        {'label': 'Histogram', 'value': 'histogram'}
                    ],
                    value='bar'
                )
            ], style={'margin-bottom': '10px'})
        ]),
        # XY Chart Inputs
        html.Div(id='xy-inputs-container', style={'display': 'none', 'margin-bottom': '10px'}, children=[
            html.Div([
                html.Label("X-Axis:"),
                dcc.Dropdown(
                    id='x-axis-dropdown',
                    options=options,
                    placeholder="Select X-Axis Column"
                )
            ], style={'margin-bottom': '10px'}),
            html.Div([
                html.Label("Y-Axis:"),
                dcc.Dropdown(
                    id='y-axis-dropdown',
                    options=options,
                    placeholder="Select Y-Axis Column"
                )
            ], style={'margin-bottom': '10px'}),
            html.Div([
                html.Label("Chart Type:"),
                dcc.Dropdown(
                    id='xy-chart-type-dropdown',
                    options=[
                        {'label': 'Scatter', 'value': 'scatter'},
                        {'label': 'Line', 'value': 'line'},
                        {'label': 'Bar', 'value': 'bar'},
                        {'label': 'Histogram', 'value': 'histogram'},
                        {'label': 'Pie', 'value': 'pie'}
                    ],
                    value='scatter'
                )
            ])
        ]),
        html.Button("Generate Chart", id='generate-chart-btn', n_clicks=0, style={'margin-top': '20px'}),
        dcc.Graph(id='unified-chart')
    ])

dash_app.layout = serve_layout

@dash_app.callback(
    [Output('chart-inputs-container', 'style'),
     Output('xy-inputs-container', 'style')],
    [Input('chart-mode', 'value')]
)
def toggle_chart_mode(mode):
    if mode == 'single':
        return {'display': 'block', 'margin-bottom': '10px'}, {'display': 'none'}
    else:
        return {'display': 'none'}, {'display': 'block', 'margin-bottom': '10px'}

@dash_app.callback(
    Output('unified-chart', 'figure'),
    [Input('generate-chart-btn', 'n_clicks')],
    [State('chart-mode', 'value'),
     State('column-dropdown', 'value'),
     State('chart-type-dropdown', 'value'),
     State('x-axis-dropdown', 'value'),
     State('y-axis-dropdown', 'value'),
     State('xy-chart-type-dropdown', 'value'),
     State('filter-input', 'value')]
)
def update_unified_chart(n_clicks, mode, column, single_chart_type, x_axis, y_axis, xy_chart_type, filter_condition):
    if not n_clicks:
        return {}
    global df
    if df is None or df.empty:
        return {}
    # Apply filter if provided
    data_source = df.copy()
    if filter_condition:
        try:
            data_source = data_source.query(filter_condition)
        except Exception as e:
            print("Filter error:", e)
    try:
        if mode == 'single':
            if not column:
                return {}
            title = f"{single_chart_type.capitalize()} Chart: {column}"
            fig = generate_plotly_chart(data_source, x=column, chart_type=single_chart_type, title=title)
        else:
            if not x_axis or not y_axis:
                return {}
            title = f"{xy_chart_type.capitalize()} Chart: {y_axis} vs {x_axis}"
            fig = generate_plotly_chart(data_source, x=x_axis, y=y_axis, chart_type=xy_chart_type, title=title)
        fig.update_layout(margin=dict(l=40, r=40, t=40, b=40))
        return fig
    except Exception as e:
        print("Chart generation error:", e)
        return {}

if __name__ == '__main__':
    app.run(debug=True)
