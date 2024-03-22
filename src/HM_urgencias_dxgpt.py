import os
import pandas as pd
from dotenv import load_dotenv
from langchain.chat_models import AzureChatOpenAI
from langchain.schema import HumanMessage
from langchain.prompts import PromptTemplate
from langchain.prompts.chat import ChatPromptTemplate, SystemMessagePromptTemplate, HumanMessagePromptTemplate
from tqdm import tqdm


# Load the environment variables from the .env file
load_dotenv()

# Initialize the AzureChatOpenAI model
model = AzureChatOpenAI(
    openai_api_version="2023-06-01-preview",
    azure_deployment="nav29",
    temperature=0,
    max_tokens=800,
    model_kwargs={"top_p": 1, "frequency_penalty": 0, "presence_penalty": 0}
)

# Importing the dataset
dataset = pd.read_excel('data/URG_Torre_Dic_2022_IA_GEN_modified_2.xlsx')

# Create a new DataFrame to store the diagnoses
diagnoses_df = pd.DataFrame(columns=['Description', 'Diagnosis'])

PROMPT_TEMPLATE = "Behave like a hypotethical doctor who has to do a diagnosis for a patient. Give me a list of potential diseases with a short description. Shows for each potential diseases always with '\n\n+' and a number, starting with '\n\n+1', for example '\n\n+23.' (never return \n\n-), the name of the disease and finish with ':'. Dont return '\n\n-', return '\n\n+' instead. You have to indicate which symptoms the patient has in common with the proposed disease and which symptoms the patient does not have in common. The text is \n Symptoms:{description}"

# Define the chat prompt templates
human_message_prompt = HumanMessagePromptTemplate.from_template(PROMPT_TEMPLATE)
chat_prompt = ChatPromptTemplate.from_messages([human_message_prompt])

# Iterate over the rows in the synthetic data
for index, row in tqdm(dataset[:200].iterrows(), total=dataset[:200].shape[0]):
    # Get the ground truth (GT) and the description
    description = row[0]
    # Generate a diagnosis
    diagnoses = []
    # Generate the diagnosis using the GPT-4 model
    formatted_prompt = chat_prompt.format_messages(description=description)
    attempts = 0
    while attempts < 2:
        try:
            diagnosis = model(formatted_prompt).content  # Call the model instance directly
            break
        except Exception as e:
            attempts += 1
            print(e)
            if attempts == 2:
                diagnosis = "ERROR"
    diagnoses.append(diagnosis)

    # Add the diagnoses to the new DataFrame
    diagnoses_df.loc[index] = [description] + diagnoses

# Save the diagnoses to a new CSV file
diagnoses_df.to_csv('data/diagnoses_URG_Torre_Dic_200.csv', index=False)

