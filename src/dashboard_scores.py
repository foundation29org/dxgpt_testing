import os
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
# This file is used to generate a dashboard for easily comparing the scores of the predictions between different models.

# Filter data folder for files starting with "scores"
path = "data"
files = [f for f in os.listdir(path) if f.startswith('scores') and f.endswith('.csv')]
print(files)

files_URG = [f for f in files if f.startswith('scores_URG_Torre_Dic_200')]

files_v2 = [f for f in files if f.startswith('scores_v2')]

print("Files Urgencias HM:")
print(files_URG)

print("Files v2:")
print(files_v2)


def get_stats_for_df(df):
    count_p1 = df['Score'].value_counts()['P1']
    count_p5 = df[df['Score'].isin(['P2', 'P3', 'P4', 'P5'])].shape[0]
    count_p0 = df['Score'].value_counts()['P0']

    # Calculate total number of predictions
    total_predictions = count_p1 + count_p5 + count_p0

    # Calculate Strict Accuracy
    strict_accuracy = (count_p1 / total_predictions) * 100

    # Calculate Lenient Accuracy
    lenient_accuracy = ((count_p1 + count_p5) / total_predictions) * 100

    return count_p1, count_p5, count_p0, strict_accuracy, lenient_accuracy

# Now for files in files_v2
for file in files_v2:
    # Load the scores data
    df = pd.read_csv(f'{path}/{file}')
    print(f"Stats for {file}")
    print(get_stats_for_df(df))

print("")

for file in files_URG:
    # Load the scores data
    df = pd.read_csv(f'{path}/{file}')
    print(f"Stats for {file}")
    print(get_stats_for_df(df))

# Create a list to store the data for each file group
data_v2 = []
data_URG = []

# Process files_v2 and store the data
for file in files_v2:
    df = pd.read_csv(f'{path}/{file}')
    count_p1, count_p5, count_p0, strict_accuracy, lenient_accuracy = get_stats_for_df(df)
    data_v2.append([file, count_p1, count_p5, count_p0, strict_accuracy, lenient_accuracy])

# Process files_URG and store the data
for file in files_URG:
    df = pd.read_csv(f'{path}/{file}')
    count_p1, count_p5, count_p0, strict_accuracy, lenient_accuracy = get_stats_for_df(df)
    data_URG.append([file, count_p1, count_p5, count_p0, strict_accuracy, lenient_accuracy])

# Create DataFrames for each file group
df_v2 = pd.DataFrame(data_v2, columns=['File', 'P1', 'P5', 'P0', 'Strict Accuracy', 'Lenient Accuracy'])
df_URG = pd.DataFrame(data_URG, columns=['File', 'P1', 'P5', 'P0', 'Strict Accuracy', 'Lenient Accuracy'])

# Create subplots with 2 rows and 1 column, specifying subplot type as 'domain'
fig = make_subplots(rows=2, cols=1, subplot_titles=('v2 Files', 'URG Files'), specs=[[{'type': 'domain'}], [{'type': 'domain'}]])

# Add the first table to the first subplot
fig.add_trace(
    go.Table(
        header=dict(values=list(df_v2.columns),
                    fill_color='paleturquoise',
                    align='left'),
        cells=dict(values=[df_v2.File, df_v2.P1, df_v2.P5, df_v2.P0, df_v2['Strict Accuracy'], df_v2['Lenient Accuracy']],
                   fill_color='lavender',
                   align='left')
    ),
    row=1, col=1
)

# Add the second table to the second subplot
fig.add_trace(
    go.Table(
        header=dict(values=list(df_URG.columns),
                    fill_color='paleturquoise',
                    align='left'),
        cells=dict(values=[df_URG.File, df_URG.P1, df_URG.P5, df_URG.P0, df_URG['Strict Accuracy'], df_URG['Lenient Accuracy']],
                   fill_color='lavender',
                   align='left')
    ),
    row=2, col=1
)

# Update the layout
fig.update_layout(
    title='Prediction Scores Dashboard',
    height=800
)

# Display the dashboard
fig.show()


# Create lists to store the data for each file group
data_v2 = []
data_v2_improved = []
data_URG = []
data_URG_improved = []

# Process files_v2 and store the data
for file in files_v2:
    df = pd.read_csv(f'{path}/{file}')
    count_p1, count_p5, count_p0, strict_accuracy, lenient_accuracy = get_stats_for_df(df)
    if 'improved' in file:
        data_v2_improved.append([file, count_p1, count_p5, count_p0, strict_accuracy, lenient_accuracy])
    else:
        data_v2.append([file, count_p1, count_p5, count_p0, strict_accuracy, lenient_accuracy])

# Process files_URG and store the data
for file in files_URG:
    df = pd.read_csv(f'{path}/{file}')
    count_p1, count_p5, count_p0, strict_accuracy, lenient_accuracy = get_stats_for_df(df)
    if 'improved' in file:
        data_URG_improved.append([file, count_p1, count_p5, count_p0, strict_accuracy, lenient_accuracy])
    else:
        data_URG.append([file, count_p1, count_p5, count_p0, strict_accuracy, lenient_accuracy])

# Create DataFrames for each file group
df_v2 = pd.DataFrame(data_v2, columns=['File', 'P1', 'P5', 'P0', 'Strict Accuracy', 'Lenient Accuracy'])
df_v2_improved = pd.DataFrame(data_v2_improved, columns=['File', 'P1', 'P5', 'P0', 'Strict Accuracy', 'Lenient Accuracy'])
df_URG = pd.DataFrame(data_URG, columns=['File', 'P1', 'P5', 'P0', 'Strict Accuracy', 'Lenient Accuracy'])
df_URG_improved = pd.DataFrame(data_URG_improved, columns=['File', 'P1', 'P5', 'P0', 'Strict Accuracy', 'Lenient Accuracy'])

# Create subplots with 4 rows and 1 column, specifying subplot type as 'domain'
fig = make_subplots(rows=4, cols=1, subplot_titles=('v2 Files', 'v2 Improved Files', 'URG Files', 'URG Improved Files'),
                    specs=[[{'type': 'domain'}], [{'type': 'domain'}], [{'type': 'domain'}], [{'type': 'domain'}]])

# Add the tables to the respective subplots
fig.add_trace(go.Table(
    header=dict(values=list(df_v2.columns), fill_color='paleturquoise', align='left'),
    cells=dict(values=[df_v2.File, df_v2.P1, df_v2.P5, df_v2.P0, df_v2['Strict Accuracy'], df_v2['Lenient Accuracy']],
               fill_color='lavender', align='left')
), row=1, col=1)

fig.add_trace(go.Table(
    header=dict(values=list(df_v2_improved.columns), fill_color='paleturquoise', align='left'),
    cells=dict(values=[df_v2_improved.File, df_v2_improved.P1, df_v2_improved.P5, df_v2_improved.P0, df_v2_improved['Strict Accuracy'], df_v2_improved['Lenient Accuracy']],
               fill_color='lavender', align='left')
), row=2, col=1)

fig.add_trace(go.Table(
    header=dict(values=list(df_URG.columns), fill_color='paleturquoise', align='left'),
    cells=dict(values=[df_URG.File, df_URG.P1, df_URG.P5, df_URG.P0, df_URG['Strict Accuracy'], df_URG['Lenient Accuracy']],
               fill_color='lavender', align='left')
), row=3, col=1)

fig.add_trace(go.Table(
    header=dict(values=list(df_URG_improved.columns), fill_color='paleturquoise', align='left'),
    cells=dict(values=[df_URG_improved.File, df_URG_improved.P1, df_URG_improved.P5, df_URG_improved.P0, df_URG_improved['Strict Accuracy'], df_URG_improved['Lenient Accuracy']],
               fill_color='lavender', align='left')
), row=4, col=1)

# Update the layout
fig.update_layout(
    title='Prediction Scores Dashboard',
    height=1200
)

# Display the dashboard
fig.show()


# Create subplots with 4 rows and 3 columns
fig = make_subplots(rows=4, cols=3, subplot_titles=('v2 Files', 'v2 Accuracy', 'v2 Scores',
                                                    'v2 Improved Files', 'v2 Improved Accuracy', 'v2 Improved Scores',
                                                    'URG Files', 'URG Accuracy', 'URG Scores',
                                                    'URG Improved Files', 'URG Improved Accuracy', 'URG Improved Scores'),
                    specs=[[{'type': 'domain'}, {'type': 'xy'}, {'type': 'xy'}],
                           [{'type': 'domain'}, {'type': 'xy'}, {'type': 'xy'}],
                           [{'type': 'domain'}, {'type': 'xy'}, {'type': 'xy'}],
                           [{'type': 'domain'}, {'type': 'xy'}, {'type': 'xy'}]])

# Add the tables to the respective subplots
fig.add_trace(go.Table(
    header=dict(values=list(df_v2.columns), fill_color='paleturquoise', align='left'),
    cells=dict(values=[df_v2.File, df_v2.P1, df_v2.P5, df_v2.P0, df_v2['Strict Accuracy'], df_v2['Lenient Accuracy']],
               fill_color='lavender', align='left')
), row=1, col=1)

fig.add_trace(go.Table(
    header=dict(values=list(df_v2_improved.columns), fill_color='paleturquoise', align='left'),
    cells=dict(values=[df_v2_improved.File, df_v2_improved.P1, df_v2_improved.P5, df_v2_improved.P0, df_v2_improved['Strict Accuracy'], df_v2_improved['Lenient Accuracy']],
               fill_color='lavender', align='left')
), row=2, col=1)

fig.add_trace(go.Table(
    header=dict(values=list(df_URG.columns), fill_color='paleturquoise', align='left'),
    cells=dict(values=[df_URG.File, df_URG.P1, df_URG.P5, df_URG.P0, df_URG['Strict Accuracy'], df_URG['Lenient Accuracy']],
               fill_color='lavender', align='left')
), row=3, col=1)

fig.add_trace(go.Table(
    header=dict(values=list(df_URG_improved.columns), fill_color='paleturquoise', align='left'),
    cells=dict(values=[df_URG_improved.File, df_URG_improved.P1, df_URG_improved.P5, df_URG_improved.P0, df_URG_improved['Strict Accuracy'], df_URG_improved['Lenient Accuracy']],
               fill_color='lavender', align='left')
), row=4, col=1)

# Add accuracy comparison for v2 files
fig.add_trace(go.Bar(x=df_v2['File'], y=df_v2['Strict Accuracy'], name='Strict Accuracy', marker_color='blue'), row=1, col=2)
fig.add_trace(go.Bar(x=df_v2['File'], y=df_v2['Lenient Accuracy'], name='Lenient Accuracy', marker_color='green'), row=1, col=2)

# Add score comparison for v2 files
fig.add_trace(go.Bar(x=df_v2['File'], y=df_v2['P1'], name='P1', marker_color='red'), row=1, col=3)
fig.add_trace(go.Bar(x=df_v2['File'], y=df_v2['P5'], name='P5', marker_color='orange'), row=1, col=3)
fig.add_trace(go.Bar(x=df_v2['File'], y=df_v2['P0'], name='P0', marker_color='purple'), row=1, col=3)

# Add accuracy comparison for v2 improved files
fig.add_trace(go.Bar(x=df_v2_improved['File'], y=df_v2_improved['Strict Accuracy'], name='Strict Accuracy', marker_color='blue'), row=2, col=2)
fig.add_trace(go.Bar(x=df_v2_improved['File'], y=df_v2_improved['Lenient Accuracy'], name='Lenient Accuracy', marker_color='green'), row=2, col=2)

# Add score comparison for v2 improved files
fig.add_trace(go.Bar(x=df_v2_improved['File'], y=df_v2_improved['P1'], name='P1', marker_color='red'), row=2, col=3)
fig.add_trace(go.Bar(x=df_v2_improved['File'], y=df_v2_improved['P5'], name='P5', marker_color='orange'), row=2, col=3)
fig.add_trace(go.Bar(x=df_v2_improved['File'], y=df_v2_improved['P0'], name='P0', marker_color='purple'), row=2, col=3)

# Add accuracy comparison for URG files
fig.add_trace(go.Bar(x=df_URG['File'], y=df_URG['Strict Accuracy'], name='Strict Accuracy', marker_color='blue'), row=3, col=2)
fig.add_trace(go.Bar(x=df_URG['File'], y=df_URG['Lenient Accuracy'], name='Lenient Accuracy', marker_color='green'), row=3, col=2)

# Add score comparison for URG files
fig.add_trace(go.Bar(x=df_URG['File'], y=df_URG['P1'], name='P1', marker_color='red'), row=3, col=3)
fig.add_trace(go.Bar(x=df_URG['File'], y=df_URG['P5'], name='P5', marker_color='orange'), row=3, col=3)
fig.add_trace(go.Bar(x=df_URG['File'], y=df_URG['P0'], name='P0', marker_color='purple'), row=3, col=3)

# Add accuracy comparison for URG improved files
fig.add_trace(go.Bar(x=df_URG_improved['File'], y=df_URG_improved['Strict Accuracy'], name='Strict Accuracy', marker_color='blue'), row=4, col=2)
fig.add_trace(go.Bar(x=df_URG_improved['File'], y=df_URG_improved['Lenient Accuracy'], name='Lenient Accuracy', marker_color='green'), row=4, col=2)

# Add score comparison for URG improved files
fig.add_trace(go.Bar(x=df_URG_improved['File'], y=df_URG_improved['P1'], name='P1', marker_color='red'), row=4, col=3)
fig.add_trace(go.Bar(x=df_URG_improved['File'], y=df_URG_improved['P5'], name='P5', marker_color='orange'), row=4, col=3)
fig.add_trace(go.Bar(x=df_URG_improved['File'], y=df_URG_improved['P0'], name='P0', marker_color='purple'), row=4, col=3)

# Update the layout
fig.update_layout(
    title='Prediction Scores Dashboard',
    height=1600,
    showlegend=True
)

# Display the dashboard
fig.show()