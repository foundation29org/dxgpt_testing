import os
import pandas as pd
import json

data_dir = './data'
files = os.listdir(data_dir)
print(files)

for file in files:
    file_path = os.path.join(data_dir, file)
    if file == 'Final_List_200.json':
        with open(file_path) as json_file:
            data = json.load(json_file)
            df = pd.DataFrame(data)
    else:
        df = pd.read_csv(file_path)
    
    # Perform EDA
    print(df.describe())

names = df['Name'].tolist()
print(names)

# Save names to a file in the /data folder
with open('./data/names.txt', 'w') as f:
    for name in names:
        f.write("%s\n" % name)