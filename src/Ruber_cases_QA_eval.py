import os
import re
import yaml
from tqdm import tqdm
import pandas as pd
from langchain_openai import AzureChatOpenAI
from langchain.prompts.chat import ChatPromptTemplate, SystemMessagePromptTemplate, HumanMessagePromptTemplate
from dotenv import load_dotenv

# Load the environment variables from the .env file
load_dotenv()

# Initialize the AzureChatOpenAI model
# This is gpt4-turbo-1106
azure_endpoint=os.getenv("AZURE_ENDPOINT_US")
azure_key=os.getenv("AZURE_KEY_US")
gpt4oazure = AzureChatOpenAI(
    openai_api_version="2024-02-01",
    azure_deployment="gpt-4o",
    azure_endpoint=azure_endpoint,
    api_key=azure_key,
    temperature=0,
    max_tokens=2000,
    model_kwargs={"top_p": 1, "frequency_penalty": 0, "presence_penalty": 0}
)

PROMPT_EVAL_QA = """
Compórtate como un hipotético médico revisando historias clínicas de pacientes.

Este es el caso clínico a evaluar:

<caso_clinico>
{caso_clinico}
</caso_clinico>

Compórtate como un hipotético médico revisando historias clínicas de pacientes. Se te dará un caso clínico con información sobre un paciente y sus síntomas. Deberás evaluar el caso
en base a los criterios siguientes y proporcionar una lista que incluirá si se cumplen o no los criterios. Además, deberás proporcionar una evaluación general del caso terminando con una clasificación.

- Edad del paciente.
- Género del paciente.
- Descripción mínima de los síntomas/signos principales.
- Cuánto tiempo ha estado ocurriendo el problema y sus características o detalles específicos. 
- Enfermedades anteriores, alergias, cirugías. 
- Datos del examen físico. 
- Datos adicionales de resultados de pruebas, siempre y cuando, con los datos anteriores no sea posible establecer un diagnóstico. 

Devolverás SIEMPRE algo como el siguiente output con el YAML dentro y NADA MAS.
Formato ejemplo de output YAML:
<output>
-Edad: <si/no>
-Genero: <si/no>
-Descripcion: <si/no>
-Duracion: <si/no>
-Enfermedades anteriores: <si/no>
-Alergias: <si/no>
-Cirugias: <si/no>
-Examen fisico: <si/no>
-Pruebas adicionales: <si/no>
-Evaluacion: < Conclusion y razonamiento de la clasificacion siguiente >
-Clasificacion: <Muy mala, Mala, Deficiente, Suficiente, Regular, Aceptable, Satisfactoria, Buena, Muy buena, Excelente>
</output>

Asegúrate de que tu respuesta sea clara y concisa. No incluyas información adicional que no se haya solicitado.
Recuerda que el YAML SIEMPRE debe estar en el formato correcto como se muestra arriba en el ejemplo.
"""

def evaluate_medical_case(prompt, case_data, output_file, model=gpt4oazure):
    # Load the cases
    input_path = f'Ruber_cases/{case_data}'
    df = pd.read_excel(input_path)
    # Create a new DataFrame to store the scores of the predictions
    scores_df = pd.DataFrame(columns=['Phenotype', 'Edad', 'Genero', 'Descripcion', 'Duracion', 'Enfermedades anteriores', 'Alergias', 'Cirugias', 'Examen fisico', 'Pruebas adicionales', 'Evaluacion', 'Clasificacion'])

    # Define the chat prompt template
    human_message_prompt = HumanMessagePromptTemplate.from_template(prompt)
    chat_prompt = ChatPromptTemplate.from_messages([human_message_prompt])

    # Iterate over the rows in the data
    for index, row in tqdm(df.iterrows(), total=df.shape[0]):
        description = row["Phenotype"]

        formatted_prompt = chat_prompt.format_messages(caso_clinico=description)

        attempts = 0
        while attempts < 2:
            try:
                evaluation = model(formatted_prompt).content  # Call the model instance directly
                break
            except Exception as e:
                attempts += 1
                print(e)
                if attempts == 2:
                    evaluation = "ERROR"

        match = re.search(r"<output>(.*?)</output>", evaluation, re.DOTALL)

        if match:
            evaluation = match.group(1).strip()
        else:
            print("ERROR: <output> tags not found in the response.")

        # print(evaluation)

        # Add the description to the new DataFrame
        scores_df.loc[index] = [description] + [None] * 11
        try:
            # Now load the YAML evaluation into the DF columns
            yaml_data = yaml.safe_load(evaluation)
            for key, value in yaml_data.items():
                # Remove leading hyphens from keys
                clean_key = key.lstrip('-')
                if clean_key in scores_df.columns:
                    scores_df.loc[index, clean_key] = value
                else:
                    print(f"WARNING: Key '{clean_key}' not found in DataFrame columns.")
        except yaml.YAMLError as e:
            print(f"ERROR: Failed to parse YAML. {e}")
        except Exception as e:
            print(f"ERROR: An unexpected error occurred. {e}")

        print(scores_df.loc[index])

    # Save the scores to a CSV file
    scores_df.to_csv(f'Ruber_cases/{output_file}', index=False)

evaluate_medical_case(PROMPT_EVAL_QA, "Ruber_HHCC_Epilepsy_50.xlsx", "Ruber_HHCC_Epilepsy_50_Eval_1.csv", gpt4oazure)

