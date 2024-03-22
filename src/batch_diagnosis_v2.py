import os
import re
import json
import pandas as pd
import boto3
from dotenv import load_dotenv
from langchain.chat_models import AzureChatOpenAI, ChatOpenAI
from langchain_community.chat_models import BedrockChat
from langchain.schema import HumanMessage
from langchain.prompts import PromptTemplate
from langchain.prompts.chat import ChatPromptTemplate, SystemMessagePromptTemplate, HumanMessagePromptTemplate
from tqdm import tqdm
import anthropic

from open_models import initialize_mistralmoe, initialize_mistral7b


# Load the environment variables from the .env file
load_dotenv()

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


# Initialize the AzureChatOpenAI model
gpt4 = AzureChatOpenAI(
    openai_api_version=os.getenv("OPENAI_API_VERSION"),
    azure_deployment=os.getenv("DEPLOYMENT_NAME"),
    temperature=0,
    max_tokens=2000,
    model_kwargs={"top_p": 1, "frequency_penalty": 0, "presence_penalty": 0}
)

gpt4turbo = AzureChatOpenAI(
    openai_api_version=os.getenv("OPENAI_API_VERSION"),
    azure_deployment="nav29turbo",
    temperature=0,
    max_tokens=800,
    model_kwargs={"top_p": 1, "frequency_penalty": 0, "presence_penalty": 0}
)

# Initialize the ChatOpenAI model turbo
# model_name = "gpt-4-1106-preview"
# openai_api_key=os.getenv("OPENAI_API_KEY")
# gpt4turbo = ChatOpenAI(
#         openai_api_key = openai_api_key,
#         model_name = model_name,
#         temperature = 0,
#     )

# Initialize the Claude 3 Sonnet model
# claude3s = initialize_bedrock_claude(temperature=0, max_tokens=4096)


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
    HM = False
    # Load the synthetic data
    input_path = f'data/{dataframe}'
    if input_path.endswith('.csv'):
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
                    # diagnosis = initialize_bedrock_claude(formatted_prompt[0].content).get("content")[0].get("text")
                    diagnosis = initialize_anthropic_claude(formatted_prompt[0].content).content[0].text
                    # print(diagnosis)
                elif model == "mistralmoe":
                    diagnosis = initialize_mistralmoe(formatted_prompt[0].content)["outputs"][0]["text"]
                    # print(diagnosis)
                elif model == "mistral7b":
                    diagnosis = initialize_mistral7b(formatted_prompt[0].content)["outputs"][0]["text"]
                    print(diagnosis)
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

    # Save the diagnoses to a new CSV file
    output_path = f'data/{output_file}'
    diagnoses_df.to_csv(output_path, index=False)


# Call the get_diagnosis function
get_diagnosis(PROMPT_TEMPLATE, 'synthetic_data_v2.csv', 'diagnoses_v2_mistral7b.csv', "mistral7b")
