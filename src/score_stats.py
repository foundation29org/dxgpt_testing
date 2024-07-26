import os
import pandas as pd

# Load the scores data
# df2 = pd.read_csv('data/scores_RAMEDIS_c3sonnet.csv')

# df = pd.read_csv('data/scores_URG_Torre_Dic_200_gpt4turbo1106.csv')

# df = pd.read_csv('Ruber_cases/scores_RUBER_HHCC_Epilepsy_50_gpt4o_gene.csv')
# df2 = pd.read_csv('Ruber_cases/scores_RUBER_HHCC_Epilepsy_50_gpt4_0613_gene.csv')
df2 = pd.read_csv('Ruber_cases/scores_RUBER_HHCC_Epilepsy_50_gpt4o_text.csv')
# df = pd.read_csv('Ruber_cases/scores_RUBER_HHCC_Epilepsy_50_gpt4_0613_gene_text.csv')
df = pd.read_csv('Ruber_cases/scores_RUBER_HHCC_Epilepsy_50_gpt4_0613_text.csv')
# Summarize the data
# print(df.describe())
# print(df.head())
# print(df.shape)

# Give me the stats for P1, P5 and P0
print(df['Score'].value_counts())
# Give me the stats for P1, P5 and P0 for the first 50 rows
# print(df['Score'].iloc[:50].value_counts())

# Give me the stats for P1, P5 and P0
print(df2['Score'].value_counts())
# Give me the stats for P1, P5 and P0 for the first 50 rows
# print(df2['Score'].iloc[:50].value_counts())

# To calculate the overlapping errors between the two models, we will compare the 'Score' columns from both dataframes (df and df2) to identify common mispredictions.
# First, filter out correct predictions (P1) as we are only interested in errors (P5 and P0).
errors_df = df[df['Score'] != 'P1']
errors_df2 = df2[df2['Score'] != 'P1']

# Now, find the intersection of GT (Ground Truth) values in both error dataframes to identify overlapping errors.
overlapping_errors = pd.merge(errors_df, errors_df2, on='GT', how='inner', suffixes=('_df', '_df2'))

# Display the count of overlapping errors
print(f"Number of overlapping errors: {len(overlapping_errors)}")

# But also for !P5
errors_df = df[df['Score'] == 'P0']
errors_df2 = df2[df2['Score'] == 'P0']

# Now, find the intersection of GT (Ground Truth) values in both error dataframes to identify overlapping errors.
overlapping_errors = pd.merge(errors_df, errors_df2, on='GT', how='inner', suffixes=('_df', '_df2'))

# Display the count of overlapping errors
print(f"Number of overlapping errors: {len(overlapping_errors)}")

#Score
count_p1 = df['Score'].value_counts()['P1']
count_p5 = df[df['Score'].isin(['P2', 'P3', 'P4', 'P5'])].shape[0]
count_p0 = df['Score'].value_counts()['P0']

print(f"Overlapping errors 14 of 23 errors in the 200 predictions: {len(overlapping_errors)/count_p0*100}%")

# Calculate total number of predictions
total_predictions = count_p1 + count_p5 + count_p0

# Calculate Strict Accuracy
strict_accuracy_new = (count_p1 / total_predictions) * 100

# Calculate Lenient Accuracy
lenient_accuracy_new = ((count_p1 + count_p5) / total_predictions) * 100

print(strict_accuracy_new, lenient_accuracy_new)