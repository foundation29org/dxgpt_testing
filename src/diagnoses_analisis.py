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
)

# Load the diagnoses data
df = pd.read_csv('data/diagnoses.csv')

# Summarize the data
# print(df.describe())
# print(df.head())
# print(df.shape)

# Now we will analyze the data to see if the GT is in the column 1 of diagnoses (each column has between 3 to 6 diagnoses), between P1 and P5 predictions.
# Then we will create a new DataFrame to store the scores of the predictions.
# We will iterate over the rows in the diagnoses data and we will compare the GT with the predictions column 1 first.
# If the GT is in the predictions column 1, we will store the score as P1 if the first prediction is the GT, and P5 otherwise.
# Create a new DataFrame to store the scores of the predictions
scores_df = pd.DataFrame(columns=['GT', 'Score'])

PROMPT_TEMPLATE = """Behave like a hypotethical doctor who has to classify if a diagnosis is correct or not. You will receive a Ground Thruth (GT) and a series of suggested predictions. You have to classify if the GT is in the predictions or not. If the GT is in the prediction 1, return 'P1'. If the GT is in any other prediction, return 'P5'. If the GT is not in the predictions, return 'P0'. 
----------------------------------------
The text is \n\nGT: {gt}\n\nPredictions:\n\n{predictions}
----------------------------------------
Remember to return the score in the format 'P1', 'P5' or 'P0'. Do not return any other text. NEVER"""
human_message_prompt = HumanMessagePromptTemplate.from_template(PROMPT_TEMPLATE)
chat_prompt = ChatPromptTemplate.from_messages([human_message_prompt])

# Iterate over the rows in the diagnoses data
for index, row in tqdm(df.iterrows(), total=df.shape[0]):
    # Get the ground truth (GT) and the first prediction
    gt = row[0]
    predictions = row[1]

    # Generate a score for the prediction
    formatted_prompt = chat_prompt.format_messages(gt=gt, predictions=predictions)
    attempts = 0
    while attempts < 2:
        try:
            score = model(formatted_prompt).content
            break
        except Exception as e:
            attempts += 1
            print(e)
            if attempts == 2:
                score = "P0"

    print(f"GT: {gt}, Score: {score}")

    # Add the score to the new DataFrame
    scores_df.loc[index] = [gt, score]

# Save the scores to a new CSV file
scores_df.to_csv('data/scores.csv', index=False)