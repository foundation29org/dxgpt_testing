#for this report, the path is: data/URG_Torre_Dic_2022_IA_GEN.xlsx
import pandas as pd

# Importing the dataset
dataset = pd.read_excel('data/URG_Torre_Dic_2022_IA_GEN.xlsx')

#Perform an exploratory data analysis
print(dataset.head())

print(dataset.describe())

print(dataset.info())

# Select only columns up to 'Evolucion'
dataset = dataset.loc[:, :'Evolucion']

# Drop also the columns: A,B,D,F,G,H by their number
dataset.drop(dataset.columns[[0,1,3,5,6,7]], axis=1, inplace=True)

# Combine all column information into a single 'Medical Report'
dataset['Medical Report'] = dataset.apply(lambda row: ', '.join([f'{i}: {j}' if pd.notnull(j) else f'{i}: NaN' for i, j in zip(row.index, row.values)]), axis=1)

# Drop all other columns
dataset = dataset[['Medical Report']]

# Save the new dataset to a new excel file
dataset.to_excel('data/URG_Torre_Dic_2022_IA_GEN_modified_2.xlsx', index=False)

# Print the new dataset
print(dataset.head())

# # Pass the 'Medical Report' to the AI application
# # (replace 'your_ai_application' with the actual application name and 'process' with the actual method)
# predictions = your_ai_application.process(dataset['Medical Report'])

# # Print the predictions
# print(predictions)