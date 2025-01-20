import pandas as pd

def truncate_df(df_path):
    """
    Truncates the dataframe so the diagnosis are always only top 5 predictions.
    """
    df = pd.read_csv(df_path)
    
    # Look for the "+6." in the column "Diagnosis 1" and slice the string from there
    df['Diagnosis 1'] = df['Diagnosis 1'].apply(lambda x: x.split('+6.')[0] if pd.notna(x) and '+6.' in x else x)

    # Save the truncated dataframe to a new CSV file
    # new_df_path = df_path.replace('.csv', '_truncated.csv')
    new_df_path = df_path.replace('SJD_ENG_cases', 'SJD_ENG_cases_final_49')
    df.to_csv(new_df_path, index=False)
    print(f"Truncated dataframe saved to {new_df_path}")

truncate_df('SJD_ENG_cases/diagnoses_SJD_ENG_gpt4o_1_extended_49.csv')
truncate_df('SJD_ENG_cases/diagnoses_SJD_ENG_gpt4o_2_extended_49.csv')
truncate_df('SJD_ENG_cases/diagnoses_SJD_ENG_gpt4o_3_extended_49.csv')

truncate_df('SJD_ENG_cases/diagnoses_SJD_ENG_c35sonnet_new_1_extended_49.csv')
truncate_df('SJD_ENG_cases/diagnoses_SJD_ENG_c35sonnet_new_2_extended_49.csv')
truncate_df('SJD_ENG_cases/diagnoses_SJD_ENG_c35sonnet_new_3_extended_49.csv')

truncate_df('SJD_ENG_cases/diagnoses_SJD_ENG_o1_preview_1_extended_49.csv')
truncate_df('SJD_ENG_cases/diagnoses_SJD_ENG_o1_preview_2_extended_49.csv')
truncate_df('SJD_ENG_cases/diagnoses_SJD_ENG_o1_preview_3_extended_49.csv')