import streamlit as st
import pandas as pd
import plotly.express as px
import json

# Load configuration
with open('config.json', 'r') as f:
    config = json.load(f)

# Read average metrics
metrics_file = config.get('metrics_file', r".\main_run_20240616-101926\average_metrics.txt")

# Function to parse the metrics file
def parse_metrics(file_path):
    with open(file_path, 'r') as file:
        lines = file.readlines()
    metrics = {}
    i = 0
    while i < len(lines):
        line = lines[i].strip()
        if 'Average Confusion Matrix' in line:
            i += 1
            matrix = []
            while i < len(lines) and lines[i].strip().startswith('['):
                matrix.extend([float(x) for x in lines[i].strip().replace('[', '').replace(']', '').split()])
                i += 1
            metrics['Average Confusion Matrix'] = matrix
        else:
            if ':' in line:
                key, value = line.split(':')
                metrics[key.strip()] = float(value.strip().replace('%', '').replace('seconds', '').replace('MB', ''))
            i += 1
    return metrics

metrics = parse_metrics(metrics_file)

# Dashboard layout
st.title("Anomaly Detection Dashboard")

st.sidebar.header("Configuration")
st.sidebar.write(f"Train Size Ratio: {config['train_size_ratio']}")
st.sidebar.write(f"Min Epochs: {config['min_epochs']}")
st.sidebar.write(f"Max Epochs: {config['max_epochs']}")
st.sidebar.write(f"Patience: {config['patience']}")
st.sidebar.write(f"Batch Size: {config['batch_size']}")
st.sidebar.write(f"Learning Rate: {config['learning_rate']}")
st.sidebar.write(f"Dataset Percentage: {config['dataset_percentage']}%")

# Display metrics
st.header("Evaluation Metrics")
st.write(f"Average Train Accuracy: {metrics['Average Train Accuracy']:.2f}%")
st.write(f"Average Test Accuracy: {metrics['Average Test Accuracy']:.2f}%")
st.write(f"Average Precision: {metrics['Average Precision']:.2f}%")
st.write(f"Average Recall: {metrics['Average Recall']:.2f}%")
st.write(f"Average Duration: {metrics['Average Duration']:.2f} seconds")
st.write(f"Average Max Memory Usage: {metrics['Average Max Memory Usage']:.2f} MB")

# Display confusion matrix
st.header("Confusion Matrix")
conf_matrix = [[metrics['Average Confusion Matrix'][0], metrics['Average Confusion Matrix'][1]],
               [metrics['Average Confusion Matrix'][2], metrics['Average Confusion Matrix'][3]]]
df_conf_matrix = pd.DataFrame(conf_matrix, index=['Actual Negative', 'Actual Positive'],
                              columns=['Predicted Negative', 'Predicted Positive'])
st.write(df_conf_matrix)

# Create a scatter plot (dummy data for demonstration purposes)
# Since we don't have the detailed results in the metrics file, creating a dummy DataFrame
results = pd.DataFrame({
    'True Label': [0, 0, 1, 1],
    'Predicted Label': [0, 1, 0, 1],
    'Label': ['TN', 'FP', 'FN', 'TP']
})

# Plot results
st.header("Results")
fig = px.scatter(results, x='True Label', y='Predicted Label', color='Label')
st.plotly_chart(fig)
