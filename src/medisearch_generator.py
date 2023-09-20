import os
import pandas as pd
from dotenv import load_dotenv
from langchain.chat_models import AzureChatOpenAI
from langchain.schema import HumanMessage
from langchain.callbacks.manager import (
    AsyncCallbackManagerForToolRun,
    CallbackManagerForToolRun,
)
from langchain.tools.base import BaseTool
from langchain.agents import Tool, initialize_agent, AgentType

from medisearch_client import MediSearchClient
import uuid
from typing import Dict, List, Optional

# Load the environment variables from the .env file
load_dotenv()

# Create a MediSearch client tool wrapper
class MediSearchRun(BaseTool):
    name = "MediSearch"
    description = """Default tool for medical question-answering search. Use this one first.
    MediSearch is a SOTA medical question-answering system.
    Input should be the question you want to ask MediSearch in English."""
    client: MediSearchClient
    conversation_id: str = str(uuid.uuid4())

    def _run(self, query: str, run_manager: Optional[CallbackManagerForToolRun] = None) -> str:
        responses = self.client.send_user_message(conversation=[query], 
                                                  conversation_id=self.conversation_id,
                                                  should_stream_response=False,
                                                  language="English")
        for response in responses:
            if response["event"] == "llm_response":
                return response["text"]

    def _arun(self, query: str, run_manager: Optional[AsyncCallbackManagerForToolRun] = None) -> str:
        raise NotImplementedError("MediSearchRun does not support async")

# Initialize the AzureChatOpenAI model
model = AzureChatOpenAI(
    openai_api_base=str(os.getenv("OPENAI_API_BASE")),
    openai_api_version=os.getenv("OPENAI_API_VERSION"),
    deployment_name=str(os.getenv("DEPLOYMENT_NAME")),
    openai_api_key=os.getenv("OPENAI_API_KEY"),
    openai_api_type=str(os.getenv("OPENAI_API_TYPE")),
)

# Initialize the Langchain agent with the MediSearch tool
medisearch_tool = MediSearchRun(client=MediSearchClient(api_key=os.getenv("MEDISEARCH_API_KEY")))
agent = initialize_agent(
    [medisearch_tool], model, agent=AgentType.ZERO_SHOT_REACT_DESCRIPTION, verbose=True
)

# Load the list of diseases
with open('./data/clean_eng_names.txt', 'r') as file:
    diseases = file.read().splitlines()

# Initialize lists to store different types of descriptions
detailed, detailed_noise = [], []

for i, disease in enumerate(diseases):  # Limiting to 5 for demonstration
    print(f"Generating synthetic data {i} for {disease}...")
    for _ in range(2):  # Try twice
        try:
            prompt = f"Summarize in one paragraph the chief complaints and notable findings that would be consistent with early stages of {disease}, for a new patient coming to primary care who has no clear diagnosis upon arrival. Do not explicitly state {disease}."
            detailed.append(agent.run(prompt))
            # detailed_noise.append(detailed[-1] + f" Additionally, the patient reports {noise_items[i % len(noise_items)]}.")
            break  # If the model call is successful, break the loop
        except Exception as e:  # Catch any exception
            print(f"Error on attempt description of {disease}: {e}")
            if _ == 1:  # If this is the second attempt
                detailed.append(float('nan'))
                # detailed_noise.append(float('nan'))

    
# Create a DataFrame
df = pd.DataFrame({
    'Disease': diseases, 
    'Detailed': detailed, 
    # 'Detailed with Noise': detailed_noise, 
})

print(df)

# Save the DataFrame to a CSV file
df.to_csv('./data/synthetic_medisearch_data_v2.csv', index=False)