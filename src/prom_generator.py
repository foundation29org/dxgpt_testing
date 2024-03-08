import os
import json
import pandas as pd
from dotenv import load_dotenv
from langchain.chat_models import AzureChatOpenAI
from langchain.schema import HumanMessage
from langchain.prompts import PromptTemplate
from langchain.prompts.chat import ChatPromptTemplate, SystemMessagePromptTemplate, HumanMessagePromptTemplate
from tqdm import tqdm

# Load the environment variables from the .env file
load_dotenv()

model = AzureChatOpenAI(
    openai_api_version = str(os.getenv("OPENAI_API_VERSION")),
    deployment_name="nav29",
    temperature=0,
    # request_timeout=128,
)

with open('./data/Final_List_200.json') as json_file:
    diseases_data = json.load(json_file)

PROMPT_TEMPLATE= """Make a list of ten items that are important to a patient with {disease_name}. It should be ten simple sentences that explain a problem that is important to the patient or their caregivers and that is an unmet medical need. They should be items that a drug or treatment could change. When the answer is a set of co-morbidities give the separate items.
Return this python List with the top 10 most probable like "Patient-Reported Outcome Measures (PROMs)".
Return only this list of PROMs, without any other text.
----------------------------------------
Example: [{{ 'name': 'PROM1' }}, {{ 'name': 'PROM2' }}, {{ 'name': 'PROM3' }}, {{ 'name': 'PROM4' }}, {{ 'name': 'PROM5' }}, {{ 'name': 'PROM6' }}, {{ 'name': 'PROM7' }}, {{ 'name': 'PROM8' }}, {{ 'name': 'PROM9' }}, {{ 'name': 'PROM10' }}]
----------------------------------------
PROM List:
"""

# Ejemplo de uso para la primera enfermedad del JSON
# prompt_example = generate_prompt(diseases_data[0]['Name'])

def generate_prompt(disease_name):
    human_message_prompt = HumanMessagePromptTemplate.from_template(PROMPT_TEMPLATE)
    chat_prompt = ChatPromptTemplate.from_messages([human_message_prompt])
    return chat_prompt.format_messages(disease_name=disease_name)

# prom_list = model(formatted_prompt).content

# print(prom_list)

new_diseases_data = []

for i, disease in enumerate(diseases_data):
    print(f"Processing disease {i+1} out of {len(diseases_data)}")
    disease_name = disease['Name']
    prompt = generate_prompt(disease_name)
    try:
        prom_list = model(prompt).content  # Aquí se llama a la API
    except Exception as e:
        print(f"Error al procesar {disease_name}: {e}")
        continue
    new_disease_entry = {
        "id": f"ORPHA:{disease['Code Orpha']}",
        "name": disease_name,
        "items": prom_list
    }
    new_diseases_data.append(new_disease_entry)


new_json_path = './data/Final_List_200_proms.json'
# Guardar en un archivo JSON
with open(new_json_path, 'w') as new_json_file:
    json.dump(new_diseases_data, new_json_file, indent=4)