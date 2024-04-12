import os
import re
import json
import logging
from datasets import load_dataset
import requests
import pyhpo
import pandas as pd
import boto3
from dotenv import load_dotenv
from langchain.chat_models import AzureChatOpenAI, ChatOpenAI
from langchain_community.chat_models import BedrockChat
from langchain.prompts import PromptTemplate
from langchain.prompts.chat import ChatPromptTemplate, SystemMessagePromptTemplate, HumanMessagePromptTemplate
from langchain_community.chat_models.azureml_endpoint import AzureMLChatOnlineEndpoint
from langchain_community.chat_models.azureml_endpoint import (
    AzureMLEndpointApiType,
    CustomOpenAIChatContentFormatter,
)
from langchain_core.messages import HumanMessage
from tqdm import tqdm
import anthropic
import vertexai
from vertexai.preview.generative_models import GenerativeModel
from google.oauth2 import service_account
from open_models import initialize_mistralmoe, initialize_mistral7b

logging.basicConfig(level=logging.INFO)

# Load the environment variables from the .env file
load_dotenv()


def orpha_api_get_disease_name(disease_code):
    """
    Get disease name from Orpha API
    """
    api_key = "f29dev"
    int_code = disease_code.split(":")[1]
    url = f"https://api.orphacode.org/EN/ClinicalEntity/orphacode/{int_code}/Name"
    headers = {
        "accept": "application/json",
        "apiKey": api_key
    }
    
    response = requests.get(url, headers=headers)
    
    if response.status_code == 200:
        data = response.json()
        return data["Preferred term"]
    else:
        return None

def mapping_fn_with_hpo3_plus_orpha_api(data):
    """
    Same as mapping_fn but with HPO3
    This function takes in the dataset and returns the mapped dataset
    Input is a list of Example objects. Output should be another list of Example objects
    Change Phenotype object list to list of texts mapped.
    """
    pyhpo.Ontology()
    mapped_data = []
    for example in data:
        example["Phenotype"] = [pyhpo.Ontology.get_hpo_object(phenotype).name for phenotype in example["Phenotype"]]
        example["RareDisease"] = [orpha_api_get_disease_name(disease) for disease in example["RareDisease"] if disease.startswith("ORPHA:")]
        mapped_data.append(example)

    return mapped_data

def initialize_anthropic_claude(prompt, temperature=0, max_tokens=2000):
    client = anthropic.Anthropic(
        # defaults to os.environ.get("ANTHROPIC_API_KEY")
        api_key=os.environ.get("ANTHROPIC_API_KEY")
    )
    message = client.messages.create(
        model="claude-3-opus-20240229",
        max_tokens=max_tokens,
        temperature=temperature,
        messages=[{"role": "user", "content": prompt}]
    )
    # print(message.content)
    return message

def initialize_bedrock_claude(prompt, temperature=0, max_tokens=2000):
    aws_access_key_id = os.getenv("BEDROCK_USER_KEY")
    aws_secret_access_key = os.getenv("BEDROCK_USER_SECRET")
    region = "us-east-1"

    boto3_session = boto3.Session(
        aws_access_key_id=aws_access_key_id,
        aws_secret_access_key=aws_secret_access_key,
        region_name=region,
    )

    bedrock = boto3_session.client(service_name='bedrock-runtime')

    # body = json.dumps({
    #     "prompt": prompt,
    #     "max_tokens_to_sample": max_tokens,
    #     "top_p": 1,
    #     "temperature": temperature,
    # })

    body = json.dumps({
        "max_tokens": max_tokens,
        "temperature": temperature,
        "messages": [{"role": "user", "content": prompt}],
        "anthropic_version": "bedrock-2023-05-31"
    })

    response = bedrock.invoke_model(
        body=body,
        modelId="anthropic.claude-3-sonnet-20240229-v1:0",
        accept="application/json",
        contentType="application/json",
    )

    # claude3s = BedrockChat(
    #             client = bedrock,
    #             model_id="anthropic.claude-3-sonnet-20240229",
    #             model_kwargs={"temperature": temperature, "max_tokens_to_sample": max_tokens},
    # )

    print(response)

    return json.loads(response.get('body').read())

def initialize_azure_llama2(prompt, temperature=0, max_tokens=800):
    llm = AzureMLChatOnlineEndpoint(
        endpoint_url=os.getenv("AZURE_ML_ENDPOINT"),
        endpoint_api_type="serverless",
        endpoint_api_key=os.getenv("AZURE_ML_API_KEY"),
        content_formatter=CustomOpenAIChatContentFormatter(),
        deployment_name="Llama-2-7b-chat-dxgpt",
        model_kwargs={"temperature": temperature, "max_new_tokens": max_tokens},
    )

    response = llm.invoke(
        [HumanMessage(content=prompt)]
    )

    # logging.warning(response.content)
    return response.content

def initialize_azure_cohere_cplus(prompt, temperature=0, max_tokens=800):
    llm = AzureMLChatOnlineEndpoint(
        endpoint_url=os.getenv("AZURE_ML_ENDPOINT_2"),
        endpoint_api_type="serverless",
        endpoint_api_key=os.getenv("AZURE_ML_API_KEY_2"),
        content_formatter=CustomOpenAIChatContentFormatter(),
        deployment_name="Cohere-command-r-plus-dxgpt",
        model_kwargs={"temperature": temperature, "max_new_tokens": max_tokens},
    )

    response = llm.invoke(
        [HumanMessage(content=prompt)]
    )

    # logging.warning(response.content)
    return response.content

def initialize_gcp_geminipro(prompt, temperature=0, max_tokens=800):
    PROJECT_ID = "nav29-21389"
    REGION = "us-central1"
    credentials = service_account.Credentials.from_service_account_file(
    './nav29-21389-c1a94e300dcb.json')
    vertexai.init(project=PROJECT_ID, location=REGION, credentials=credentials)

    geminipro_model = GenerativeModel("gemini-1.5-pro-preview-0409")
    response = geminipro_model.generate_content([prompt],
                                                generation_config={
            "max_output_tokens": max_tokens,
            "temperature": temperature,
            "top_p": 1,
            "top_k": 32
        },
        safety_settings = {
        },
        stream=False)
    response_text = ""
    if response.to_dict()["candidates"] == []:
        response_text = "No response available due to inappropriate content, request error or safety settings."
    else:
        response_text = response.to_dict()["candidates"][0]["content"]["parts"][0]["text"]
    print(response_text)

    return response_text

# Initialize the AzureChatOpenAI model
# This is gpt4-0613
gpt4 = AzureChatOpenAI(
    openai_api_version=os.getenv("OPENAI_API_VERSION"),
    azure_deployment=os.getenv("DEPLOYMENT_NAME"),
    temperature=0,
    max_tokens=2000,
    model_kwargs={"top_p": 1, "frequency_penalty": 0, "presence_penalty": 0}
)

# Initialize the AzureChatOpenAI model
# This is gpt4-turbo-1106
gpt4turbo = AzureChatOpenAI(
    openai_api_version=os.getenv("OPENAI_API_VERSION"),
    azure_deployment="nav29turbo",
    temperature=0,
    max_tokens=800,
    model_kwargs={"top_p": 1, "frequency_penalty": 0, "presence_penalty": 0}
)

# Initialize the last ChatOpenAI model turbo
# This is gpt4-turbo-0409
model_name = "gpt-4-turbo-2024-04-09"
openai_api_key=os.getenv("OPENAI_API_KEY")
newgpt4turbo = ChatOpenAI(
        openai_api_key = openai_api_key,
        model_name = model_name,
        temperature = 0,
        max_tokens = 800,
    )


PROMPT_TEMPLATE_RARE = "Behave like a hypotethical doctor who has to do a diagnosis for a patient. Give me a list of potential rare diseases with a short description. Shows for each potential rare diseases always with '\n\n+' and a number, starting with '\n\n+1', for example '\n\n+23.' (never return \n\n-), the name of the disease and finish with ':'. Dont return '\n\n-', return '\n\n+' instead. You have to indicate which symptoms the patient has in common with the proposed disease and which symptoms the patient does not have in common. The text is \n Symptoms:{description}"

PROMPT_TEMPLATE = "Behave like a hypotethical doctor who has to do a diagnosis for a patient. Give me a list of potential diseases with a short description. Shows for each potential diseases always with '\n\n+' and a number, starting with '\n\n+1', for example '\n\n+23.' (never return \n\n-), the name of the disease and finish with ':'. Dont return '\n\n-', return '\n\n+' instead. You have to indicate which symptoms the patient has in common with the proposed disease and which symptoms the patient does not have in common. The text is \n Symptoms:{description}"

PROMPT_TEMPLATE_MORE = "Behave like a hypotethical doctor who has to do a diagnosis for a patient. Continue the list of potential rare diseases without repeating any disease from the list I give you. If you repeat any, it is better not to return it. Shows for each potential rare diseases always with '\n\n+' and a number, starting with '\n\n+1', for example '\n\n+23.' (never return \n\n-), the name of the disease and finish with ':'. Dont return '\n\n-', return '\n\n+' instead. You have to indicate a short description and which symptoms the patient has in common with the proposed disease and which symptoms the patient does not have in common. The text is \n Symptoms: {description}. Each must have this format '\n\n+7.' for each potencial rare diseases. The list is: {initial_list} "

PROMPT_TEMPLATE_V2 = """
<prompt> As an AI-assisted diagnostic tool, your task is to analyze the given patient symptoms and generate a list of the top 5 potential diagnoses. Follow these steps:
Carefully review the patient's reported symptoms.
In the <thinking></thinking> tags, provide a detailed analysis of the patient's condition: a. Highlight any key symptoms or combinations of symptoms that stand out. b. Discuss possible diagnoses and why they might or might not fit the patient's presentation. c. Suggest any additional tests or information that could help narrow down the diagnosis.
In the <top5></top5> tags, generate a list of the 5 most likely diagnoses that match the given symptoms: a. Assign each diagnosis a number, starting from 1 (e.g., "\n\n+1", "\n\n+2", etc.). b. Provide the name of the condition, followed by a colon (":"). c. Indicate which of the patient's symptoms are consistent with this diagnosis. d. Mention any key symptoms of the condition that the patient did not report, if applicable.
Remember:

Do not use "\n\n-" in your response. Only use "\n\n+" when listing the diagnoses.
The <thinking> section should come before the <top5> section.
Patient Symptoms:
<patient_description>
{description} 
</patient_description>
</prompt>
"""


def get_diagnosis(prompt, dataframe, output_file, model):
    HM = False # HM Hospitals
    HF = False # Hugging Face Datasets
    if isinstance(dataframe, list):
        HF = True

    # Load the data
    input_path = f'data/{dataframe}'
    if HF:
        df = pd.DataFrame(dataframe)
    elif input_path.endswith('.csv'):
        df = pd.read_csv(input_path, sep=',')
    elif input_path.endswith('.xlsx'):
        df = pd.read_excel(input_path)
        HM = True
    else:
        raise ValueError("Unsupported file extension. Please provide a .csv or .xlsx file.")
        
    # Create a new DataFrame to store the diagnoses
    diagnoses_df = pd.DataFrame(columns=['GT', 'Diagnosis 1'])

    # Define the chat prompt template
    human_message_prompt = HumanMessagePromptTemplate.from_template(prompt)
    chat_prompt = ChatPromptTemplate.from_messages([human_message_prompt])

    # Iterate over the rows in the synthetic data
    for index, row in tqdm(df[:200].iterrows(), total=df[:200].shape[0]):
        # Get the ground truth (GT) and the description
        if HM:
            description = row[0]
        elif HF:
            description = row["Phenotype"]
            gt = row["RareDisease"]
        else:
            gt = row[0]
            description = row[1]
        # Generate a diagnosis
        diagnoses = []
        # Generate the diagnosis using the GPT-4 model
        formatted_prompt = chat_prompt.format_messages(description=description)
        # print(formatted_prompt[0].content)
        attempts = 0
        while attempts < 2:
            try:
                if model == "claude":
                    diagnosis = initialize_bedrock_claude(formatted_prompt[0].content).get("content")[0].get("text")
                    # diagnosis = initialize_anthropic_claude(formatted_prompt[0].content).content[0].text
                    # print(diagnosis)
                elif model == "mistralmoe":
                    diagnosis = initialize_mistralmoe(formatted_prompt[0].content)["outputs"][0]["text"]
                    # print(diagnosis)
                elif model == "mistral7b":
                    diagnosis = initialize_mistral7b(formatted_prompt[0].content)["outputs"][0]["text"]
                    print(diagnosis)
                elif model == "llama2_7b":
                    diagnosis = initialize_azure_llama2(formatted_prompt[0].content)
                elif model == "cohere_cplus":
                    diagnosis = initialize_azure_cohere_cplus(formatted_prompt[0].content)
                elif model == "geminipro":
                    diagnosis = initialize_gcp_geminipro(formatted_prompt[0].content)
                else:
                    diagnosis = model(formatted_prompt).content  # Call the model instance directly
                break
            except Exception as e:
                attempts += 1
                print(e)
                if attempts == 2:
                    diagnosis = "ERROR"
        
        # Extract the content within the <top5> tags using regular expressions
        # print(diagnosis)
        match = re.search(r"<top5>(.*?)</top5>", diagnosis, re.DOTALL)
        # print(match)
        if match:
            diagnosis = match.group(1).strip()
        else:
            print("ERROR: <top5> tags not found in the response.")

        diagnoses.append(diagnosis)
        # print(diagnosis)

        # Add the diagnoses to the new DataFrame
        if HM:
            diagnoses_df.loc[index] = [description] + diagnoses
        else:
            diagnoses_df.loc[index] = [gt] + diagnoses

        # print(diagnoses_df.loc[index])
        # break

    # Save the diagnoses to a new CSV file
    output_path = f'data/{output_file}'
    diagnoses_df.to_csv(output_path, index=False)


datasets = ["RAMEDIS", "MME", "HMS", "LIRICAL", "PUMCH_ADM"]
data = load_dataset('chenxz/RareBench', "RAMEDIS", split='test')

mapped_data = mapping_fn_with_hpo3_plus_orpha_api(data)
print(type(mapped_data))

print(mapped_data[:5])

# Call the get_diagnosis function
# get_diagnosis(PROMPT_TEMPLATE, 'synthetic_data_v2.csv', 'diagnoses_v2_mistral7b.csv', "mistral7b")
get_diagnosis(PROMPT_TEMPLATE, mapped_data, 'diagnoses_RAMEDIS_geminipro15.csv', "geminipro")
# get_diagnosis(PROMPT_TEMPLATE, 'URG_Torre_Dic_2022_IA_GEN_modified_2.xlsx', 'diagnoses_URG_Torre_Dic_200_geminipro15.csv', "geminipro")
