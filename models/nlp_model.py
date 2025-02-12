import spacy
import pandas as pd
import re

nlp = spacy.load("en_core_web_sm")

OPERATIONS = {
    "sum": "sum",
    "average": "mean",
    "mean": "mean",
    "total": "sum",
    "count": "count",
    "minimum": "min",
    "maximum": "max",
    "highest": "max",
    "lowest": "min"
}

def preprocess_query(query):
    """Clean and normalize the query."""
    query = query.lower().strip()
    query = re.sub(r'[^a-zA-Z0-9 ]', '', query)  
    return query

def extract_query_details(query, df):
    """Extract relevant details from the query using NLP."""
    doc = nlp(query)
    columns = [col.lower() for col in df.columns]  
    detected_columns = []
    operation = None
    condition = None
    value = None
    
    for token in doc:
        word = token.text.lower()
        if word in OPERATIONS:
            operation = OPERATIONS[word]
        if word in columns:
            detected_columns.append(word)
        if token.like_num:
            value = token.text
        if word in ["greater", "less", "equal", "more", "above", "below"]:
            condition = word
    
    return detected_columns, operation, condition, value

def process_query(query, df):
    """Process the query and return results from the DataFrame."""
    query = preprocess_query(query)
    columns, operation, condition, value = extract_query_details(query, df)
    
    if not columns:
        return "Could not determine relevant column(s) in the query."
    
    result = None
    if operation:
        result = df[columns[0]].agg(operation)
    elif condition and value:
        if condition in ["greater", "more", "above"]:
            result = df[df[columns[0]] > float(value)]
        elif condition in ["less", "below"]:
            result = df[df[columns[0]] < float(value)]
        elif condition in ["equal"]:
            result = df[df[columns[0]] == float(value)]
    else:
        result = df[columns]
    
    return result

if __name__ == "__main__":
    sample_data = {
        "sales": [100, 200, 300, 400],
        "profit": [10, 20, 30, 40]
    }
    df = pd.DataFrame(sample_data)
    user_query = "What is the total sales?"
    output = process_query(user_query, df)
    print(output)
