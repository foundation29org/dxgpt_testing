import os
import pandas as pd

# Load the scores data
df = pd.read_csv('data/scores_medisearch_turbo_v2.csv')

# Summarize the data
print(df.describe())
print(df.head())
print(df.shape)

# Give me the stats for P1, P5 and P0
print(df['Score'].value_counts())
# Give me the stats for P1, P5 and P0 for the first 50 rows
print(df['Score'].iloc[:50].value_counts())

# GPT4 Code Interpreter:
"""
# Check for missing values
missing_values = data.isnull().sum()

# Count the frequency of each score type
score_counts = data['Score'].value_counts()

missing_values, score_counts
"""
"""
import matplotlib.pyplot as plt
import seaborn as sns

# Set style for the plots
sns.set_style("whitegrid")

# Plot the distribution of prediction scores
plt.figure(figsize=(10, 6))
sns.countplot(data=data, x='Score', order=['P0', 'P1', 'P5'])
plt.title('Distribution of Prediction Scores')
plt.xlabel('Prediction Scores')
plt.ylabel('Frequency')
plt.show()

# Plot the distribution of prediction scores for each disease
plt.figure(figsize=(15, 30))
sns.countplot(data=data, y='GT', hue='Score', hue_order=['P0', 'P1', 'P5'])
plt.title('Distribution of Prediction Scores by Disease')
plt.xlabel('Frequency')
plt.ylabel('Disease (GT)')
plt.show()
"""

"""
# Calculate Strict Accuracy (only P1 is considered correct)
strict_accuracy = (data['Score'] == 'P1').sum() / len(data) * 100

# Calculate Lenient Accuracy (both P1 and P5 are considered correct)
lenient_accuracy = ((data['Score'] == 'P1') | (data['Score'] == 'P5')).sum() / len(data) * 100

# Calculate accuracy for each disease
disease_accuracy = data.groupby('GT')['Score'].apply(lambda x: (x == 'P1').sum() / len(x) * 100).reset_index()
disease_accuracy.columns = ['GT', 'Strict_Accuracy']
disease_accuracy['Lenient_Accuracy'] = data.groupby('GT')['Score'].apply(lambda x: ((x == 'P1') | (x == 'P5')).sum() / len(x) * 100).reset_index()['Score']

strict_accuracy, lenient_accuracy, disease_accuracy.sort_values(by='Strict_Accuracy', ascending=False).head()
"""
"""
# Given counts for P1, P5, and P0
count_p1 = 121
count_p5 = 58
count_p0 = 21

# Calculate total number of predictions
total_predictions = count_p1 + count_p5 + count_p0

# Calculate Strict Accuracy
strict_accuracy_new = (count_p1 / total_predictions) * 100

# Calculate Lenient Accuracy
lenient_accuracy_new = ((count_p1 + count_p5) / total_predictions) * 100

strict_accuracy_new, lenient_accuracy_new
"""
#Score
count_p1 = 116
count_p5 = 47
count_p0 = 37

# Calculate total number of predictions
total_predictions = count_p1 + count_p5 + count_p0

# Calculate Strict Accuracy
strict_accuracy_new = (count_p1 / total_predictions) * 100

# Calculate Lenient Accuracy
lenient_accuracy_new = ((count_p1 + count_p5) / total_predictions) * 100

print(strict_accuracy_new, lenient_accuracy_new)