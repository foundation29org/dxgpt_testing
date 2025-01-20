# Load the cases_with_diagnosis.csv file from SJD folder and use DeepL API to translate the cases to English

import pandas as pd
from translate import deepl_translate

# Load the cases_with_diagnosis.csv file from SJD folder
cases_with_diagnosis = pd.read_csv('SJD_cases/cases_with_diagnosis.csv')

# Use DeepL API to translate the cases to English
# Columns: "Caso","Descripción","Descripción Ampliada","Diagnóstico"

# Handle NaN values by converting them to empty strings before translation
# Only translate non-empty strings to avoid ValueError
cases_with_diagnosis['Descripción'] = cases_with_diagnosis['Descripción'].fillna('').apply(lambda x: deepl_translate(x) if x != '' else x)
cases_with_diagnosis['Descripción Ampliada'] = cases_with_diagnosis['Descripción Ampliada'].fillna('').apply(lambda x: deepl_translate(x) if x != '' else x)
cases_with_diagnosis['Diagnóstico'] = cases_with_diagnosis['Diagnóstico'].fillna('').apply(lambda x: deepl_translate(x) if x != '' else x)

cases_with_diagnosis.to_csv('SJD_cases/cases_with_diagnosis_translated.csv', index=False)