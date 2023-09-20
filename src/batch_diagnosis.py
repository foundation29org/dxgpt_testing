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
    openai_api_base=str(os.getenv("OPENAI_API_BASE")),
    openai_api_version=os.getenv("OPENAI_API_VERSION"),
    deployment_name=str(os.getenv("DEPLOYMENT_NAME")),
    openai_api_key=os.getenv("OPENAI_API_KEY"),
    openai_api_type=str(os.getenv("OPENAI_API_TYPE")),
    temperature=0,
    max_tokens=800,
    model_kwargs={"top_p": 1, "frequency_penalty": 0, "presence_penalty": 0}
)

# Load the synthetic data
df = pd.read_csv('data/synthetic_data.csv', sep=',')

# Create a new DataFrame to store the diagnoses
diagnoses_df = pd.DataFrame(columns=['GT', 'Diagnosis 1', 'Diagnosis 2', 'Diagnosis 3', 'Diagnosis 4', 'Diagnosis 5', 'Diagnosis 6'])

PROMPT_TEMPLATE = "Behave like a hypotethical doctor who has to do a diagnosis for a patient. Give me a list of potential rare diseases with a short description. Shows for each potential rare diseases always with '\n\n+' and a number, starting with '\n\n+1', for example '\n\n+23.' (never return \n\n-), the name of the disease and finish with ':'. Dont return '\n\n-', return '\n\n+' instead. You have to indicate which symptoms the patient has in common with the proposed disease and which symptoms the patient does not have in common. The text is \n Symptoms:{description}"

PROMPT_TEMPLATE_MORE = "Behave like a hypotethical doctor who has to do a diagnosis for a patient. Continue the list of potential rare diseases without repeating any disease from the list I give you. If you repeat any, it is better not to return it. Shows for each potential rare diseases always with '\n\n+' and a number, starting with '\n\n+1', for example '\n\n+23.' (never return \n\n-), the name of the disease and finish with ':'. Dont return '\n\n-', return '\n\n+' instead. You have to indicate a short description and which symptoms the patient has in common with the proposed disease and which symptoms the patient does not have in common. The text is \n Symptoms: {description}. Each must have this format '\n\n+7.' for each potencial rare diseases. The list is: {initial_list} "

# Define the chat prompt templates
human_message_prompt = HumanMessagePromptTemplate.from_template(PROMPT_TEMPLATE)
chat_prompt = ChatPromptTemplate.from_messages([human_message_prompt])

human_message_prompt_more = HumanMessagePromptTemplate.from_template(PROMPT_TEMPLATE_MORE)
chat_prompt_more = ChatPromptTemplate.from_messages([human_message_prompt_more])

# Iterate over the rows in the synthetic data
for index, row in tqdm(df.iterrows(), total=df.shape[0]):
    # Get the ground truth (GT) and the descriptions
    gt = row[0]
    descriptions = row[1:]
    # Generate a diagnosis for each description
    diagnoses = []
    for description in descriptions:
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
    diagnoses_df.loc[index] = [gt] + diagnoses

# Save the diagnoses to a new CSV file
diagnoses_df.to_csv('data/diagnoses.csv', index=False)