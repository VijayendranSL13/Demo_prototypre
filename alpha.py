import json
import pandas as pd
import streamlit as st
from transformers import BertTokenizer, BertForTokenClassification
import torch
import plotly.express as px
import plotly.graph_objects as go
import re

# Load the tokenizer and model
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
model = BertForTokenClassification.from_pretrained('model_classify_columns_V12')

def predict_entities_each_column(text):
    # Tokenize and encode the input
    inputs = tokenizer(text, return_tensors='pt', padding=True, truncation=True, max_length=512)
    
    # Get predictions from the model
    with torch.no_grad():
        outputs = model(**inputs)
    
    # Get the predicted labels for each token
    predictions = outputs.logits.argmax(dim=2)
    
    # Decode the predicted labels back to original label names
    predicted_labels = [model.config.id2label[label_id.item()] for label_id in predictions[0]]
    
    # Use set to remove duplicate labels
    unique_labels = list(set(predicted_labels))
    
    return unique_labels

# Load JSON file function
def load_json(file):
    return json.load(file)

# Load and parse the requirement.txt file
def load_requirements(file_path):
    lines = file_path.read().decode('utf-8').splitlines()  # Read lines from the uploaded file
    requirements = {}
    keys = ['metadata', 'schemas', 'name', 'tables', 'sourceName', 'dataType']
    
    for i, line in enumerate(lines):
        if i < len(keys):  # Ensure we don't exceed the keys list
            requirements[keys[i]] = line.strip()  # Store each line under its corresponding key
            
    return requirements

# Extract tables and columns function with dynamic keys
def extract_all_tables_and_columns(data, requirements):
    # Retrieve keys from the requirements dictionary
    metadata_key = requirements.get('metadata', 'metadata')
    schemas_key = requirements.get('schemas', 'schemas')
    tables_key = requirements.get('tables', 'tables')
    name_key = requirements.get('name', 'name')
    source_name_key = requirements.get('sourceName', 'sourceName')
    data_type_key = requirements.get('dataType', 'dataType')

    # Check if metadata exists
    metadata = data.get(metadata_key, {})
    
    # Check if schemas exist
    schemas = metadata.get(schemas_key, [])
    
    predictions = []

    if not schemas:
        st.write("No schemas found.")
        return pd.DataFrame(predictions)

    for schema in schemas:
        schema_name = schema.get(name_key, 'Unknown')
        tables = schema.get(tables_key, [])
        
        if not tables:
            st.write(f"No tables found in schema {schema_name}.")
            continue
        
        for table in tables:
            # Ensure the table is a dictionary
            if isinstance(table, dict):
                table_name = table.get(source_name_key, 'Unknown')
                columns = table.get('columns', [])
                
                if not columns:
                    st.write(f"No columns found in table {table_name}.")
                    continue
                
                for column in columns:
                    # Ensure the column is a dictionary
                    if isinstance(column, dict):
                        column_name = column.get(name_key, 'Unknown')
                        data_type = column.get(data_type_key, 'Unknown')
                        
                        # Predict labels for each column name
                        predicted_label = predict_entities_each_column(column_name)
                        predictions.append({
                            'Schema': schema_name,
                            'Table': table_name,
                            'Column': column_name,
                            'Data Type': data_type,
                            'Predicted Label': ', '.join(predicted_label)
                        })
                    else:
                        st.write(f"Column data is not a dictionary: {column}")

            else:
                st.write(f"Table data is not a dictionary: {table}")

    return pd.DataFrame(predictions)

# Streamlit app structure
st.title("JSON Table Column Extractor and Predictor")

# File upload
uploaded_file = st.file_uploader("Upload a JSON file", type="json")
req_file = st.file_uploader("Upload requirement.txt", type="txt")

if uploaded_file is not None and req_file is not None:
    # Load JSON data
    data = load_json(uploaded_file)
    requirements = load_requirements(req_file)

    st.write("Loaded Requirements Metadata:", requirements)  # Display loaded requirements for verification

    # Extract tables and columns with predictions
    df_predictions = extract_all_tables_and_columns(data, requirements)
    
    # Display DataFrame
    if not df_predictions.empty:
        st.write("Predicted Labels for Columns:")
        st.dataframe(df_predictions)
        
        # Create two columns for charts
        col1, col2 = st.columns(2)
        
        with col1:
            # Generate pie chart for predictions
            prediction_counts = df_predictions['Predicted Label'].value_counts().reset_index()
            prediction_counts.columns = ['Predicted Label', 'Count']
            pie_chart = px.pie(prediction_counts, names='Predicted Label', values='Count', title="Predicted Label Distribution", width=400, height=400)
            st.plotly_chart(pie_chart)

        with col2:
            # Generate line chart for predictions
            line_chart = go.Figure()
            line_chart.add_trace(go.Scatter(x=prediction_counts['Predicted Label'], y=prediction_counts['Count'], mode='lines+markers', name='Predictions'))
            line_chart.update_layout(
                title="Predicted Label Count Over Labels",
                xaxis_title="Predicted Label",
                yaxis_title="Count",
                width=400,
                height=400
            )
            st.plotly_chart(line_chart)
        
        # Generate sorted bar chart with uniform spacing
        st.write("Predicted Label Counts for Each Column")

        # Count the number of occurrences of each predicted label per column
        label_counts = df_predictions.groupby('Column')['Predicted Label'].apply(lambda x: ', '.join(set(x))).reset_index()
        
        # Convert the 'Predicted Label' column into a list of tuples (label, count)
        label_count_dict = label_counts['Predicted Label'].value_counts().reset_index()
        label_count_dict.columns = ['Predicted Label', 'Count']

        # Plotting the bar chart
        bar_chart = px.bar(
            label_count_dict,
            x='Predicted Label',
            y='Count',
            title="Predicted Labels Count for Each Column",
            color='Predicted Label',
            text='Count'  # Add text labels to bars
        )

        # Update layout for consistent bar width and spacing
        bar_chart.update_layout(
            xaxis=dict(tickangle=45),  # Rotate labels if needed
            bargap=0.2,                # Control gap between bars
            bargroupgap=0.1,           # Control gap within grouped bars
            width=1200, 
            height=500,
        )

        st.plotly_chart(bar_chart)

    else:
        st.write("No data to display.")
