import json
import deepl
import os

auth_key = str(os.getenv("DEEPL_API_KEY"))


def deepl_translate(text):
    # auth_key = "YOUR_AUTH_KEY"
    translator = deepl.Translator(auth_key)
    translated_text = translator.translate_text(text, target_lang='EN-US').text
    return translated_text

def translate_diseases(file_path):
    # Cargar los datos del archivo JSON
    with open(file_path, 'r') as file:
        data = json.load(file)

    # Iterar sobre cada enfermedad y traducir su nombre
    for i, entry in enumerate(data):
        entry['name'] = deepl_translate(entry['name'])
        print(entry['name'])
        print(f"Progress: Translated disease {i+1} out of {len(data)}")

    # Guardar los datos traducidos en un nuevo archivo (opcional)
    translated_file_path = file_path.replace('.json', '_translated.json')
    with open(translated_file_path, 'w') as file:
        json.dump(data, file, ensure_ascii=False, indent=4)

    return translated_file_path

# # Usar la función con la ruta de tu archivo JSON corregido
# translated_file_path = translate_diseases('./data/Final_List_200_proms_corrected.json')
# print(f"Archivo traducido guardado en: {translated_file_path}")