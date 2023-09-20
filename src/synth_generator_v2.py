import os
import pandas as pd
from dotenv import load_dotenv
from langchain.chat_models import AzureChatOpenAI
from langchain.schema import HumanMessage

# Load the environment variables from the .env file
load_dotenv()

# Initialize the AzureChatOpenAI model
model = AzureChatOpenAI(
    openai_api_base=str(os.getenv("OPENAI_API_BASE")),
    openai_api_version=os.getenv("OPENAI_API_VERSION"),
    deployment_name=str(os.getenv("DEPLOYMENT_NAME")),
    openai_api_key=os.getenv("OPENAI_API_KEY"),
    openai_api_type=str(os.getenv("OPENAI_API_TYPE")),
)

# Load the list of diseases
with open('./data/clean_eng_names.txt', 'r') as file:
    diseases = file.read().splitlines()

# Initialize lists to store different types of descriptions
detailed, detailed_noise = [], []

noise_items = ["common cold symptoms", "fatigue from lack of sleep", "stress-induced headaches", 
               "allergies to certain foods", "occasional insomnia", "mild lactose intolerance", 
               "seasonal allergies", "history of smoking", "occasional alcohol consumption", 
               "history of drug use", "anxiety under stressful situations", "occasional heartburn", 
               "mild knee pain from old sports injury"]

for i, disease in enumerate(diseases[:10]):  # Limiting to 5 for demonstration
    print(f"Generating synthetic data {i} for {disease}...")
    for _ in range(2):  # Try twice
        try:
            prompt = f"Summarize in one paragraph the chief complaints and notable findings that would be consistent with early stages of {disease}, for a new patient coming to primary care who has no clear diagnosis upon arrival. Do not explicitly state {disease}."
            detailed.append(model([HumanMessage(content=prompt)]).content)
            # detailed_noise.append(detailed[-1] + f" Additionally, the patient reports {noise_items[i % len(noise_items)]}.")
            break  # If the model call is successful, break the loop
        except Exception as e:  # Catch any exception
            print(f"Error on attempt description of {disease}: {e}")
            if _ == 1:  # If this is the second attempt
                detailed.append(float('nan'))
                # detailed_noise.append(float('nan'))

    
# Create a DataFrame
df = pd.DataFrame({
    'Disease': diseases[:10], 
    'Detailed': detailed, 
    # 'Detailed with Noise': detailed_noise, 
})

print(df)

# Save the DataFrame to a CSV file
df.to_csv('./data/synthetic_data_v2.csv', index=False)
