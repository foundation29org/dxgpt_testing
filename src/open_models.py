"""
{
 "modelId": "mistral.mixtral-8x7b-instruct-v0:1",
 "contentType": "application/json",
 "accept": "application/json",
 "body": "{\"prompt\":\"<s>[INST] this is where you place your input text [/INST]\", \"max_tokens\":200, \"temperature\":0.5, \"top_p\":0.9, \"top_k\":50}}"
}

{
 "modelId": "mistral.mistral-7b-instruct-v0:2",
 "contentType": "application/json",
 "accept": "application/json",
 "body": "{\"prompt\":\"<s>[INST] this is where you place your input text [/INST]\", \"max_tokens\":200, \"temperature\":0.5, \"top_p\":0.9, \"top_k\":50}}"
}

{
 "modelId": "meta.llama2-70b-chat-v1",
 "contentType": "application/json",
 "accept": "application/json",
 "body": "{\"prompt\":\"this is where you place your input text\",\"max_gen_len\":512,\"temperature\":0.5,\"top_p\":0.9}"
}
"""
import os
import boto3
import json
from dotenv import load_dotenv
from mistralai import Mistral, UserMessage

# Load the environment variables from the .env file
load_dotenv()


def initialize_mistralmoe(prompt, temperature=0, max_tokens=800):
    aws_access_key_id = os.getenv("BEDROCK_USER_KEY")
    aws_secret_access_key = os.getenv("BEDROCK_USER_SECRET")
    region = "us-east-1"

    boto3_session = boto3.Session(
        aws_access_key_id=aws_access_key_id,
        aws_secret_access_key=aws_secret_access_key,
        region_name=region,
    )

    bedrock = boto3_session.client(service_name="bedrock-runtime")

    prompt = f"<s>[INST] {prompt} [/INST]"

    body = json.dumps({
        "prompt": prompt,
        "max_tokens": max_tokens,
        "top_p": 1,
        "temperature": temperature,
    })

    modelId = "mistral.mixtral-8x7b-instruct-v0:1"

    accept = "application/json"
    contentType = "application/json"

    response = bedrock.invoke_model(
        body=body,
        modelId=modelId,
        accept=accept,
        contentType=contentType
    )

    # print(response.get('body').read())
    return json.loads(response.get('body').read())

def initialize_mistral7b(prompt, temperature=0, max_tokens=800):
    aws_access_key_id = os.getenv("BEDROCK_USER_KEY")
    aws_secret_access_key = os.getenv("BEDROCK_USER_SECRET")
    region = "us-east-1"

    boto3_session = boto3.Session(
        aws_access_key_id=aws_access_key_id,
        aws_secret_access_key=aws_secret_access_key,
        region_name=region,
    )

    bedrock = boto3_session.client(service_name="bedrock-runtime")

    prompt = f"<s>[INST] {prompt} [/INST]"

    body = json.dumps({
        "prompt": prompt,
        "max_tokens": max_tokens,
        "top_p": 1,
        "temperature": temperature,
    })

    modelId = "mistral.mistral-7b-instruct-v0:2"

    accept = "application/json"
    contentType = "application/json"

    response = bedrock.invoke_model(
        body=body,
        modelId=modelId,
        accept=accept,
        contentType=contentType
    )

    # print(response.get('body').read())
    return json.loads(response.get('body').read())


def initialize_mixtral_moe_big(prompt, temperature=0, max_tokens=800):
    api_key = os.environ["MISTRAL_API_KEY"]
    model = "open-mixtral-8x22b"

    client = Mistral(api_key=api_key)

    messages = [
    {
        "role": "user",
        "content": prompt,
    },
 ]

    # No streaming
    chat_response = client.chat.complete(
        model=model,
        messages=messages,
        temperature=temperature,
        max_tokens=max_tokens
    )

    # print(chat_response.choices[0].message.content)
    return chat_response.choices[0].message.content
