import os
import pandas as pd

# Load the scores data
df = pd.read_csv('data/scores.csv')

# Summarize the data
print(df.describe())
print(df.head())
print(df.shape)

# Give me the stats for P1, P5 and P0
print(df['Score'].value_counts())

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