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
with open('./data/names.txt', 'r') as file:
    diseases = file.read().splitlines()

# Initialize lists to store different types of descriptions
detailed, detailed_noise = [], []
simplified, simplified_noise = [], []
minimal, minimal_noise = [], []

noise_items = ["common cold symptoms", "fatigue from lack of sleep", "stress-induced headaches", 
               "allergies to certain foods", "occasional insomnia", "mild lactose intolerance", 
               "seasonal allergies", "history of smoking", "occasional alcohol consumption", 
               "history of drug use", "anxiety under stressful situations", "occasional heartburn", 
               "mild knee pain from old sports injury"]

for i, disease in enumerate(diseases):  # Limiting to 5 for demonstration
    print(f"Generating synthetic data {i} for {disease}...")
    for description_type in ['Detailed', 'Simplified', 'Minimal']:
        for _ in range(2):  # Try twice
            try:
                if description_type == 'Detailed':
                    prompt = f"Generate ONLY a patient sample description for {disease}, without specifying the disease's name and not being so obvious."
                    detailed.append(model([HumanMessage(content=prompt)]).content)
                    detailed_noise.append(detailed[-1] + f" Additionally, the patient reports {noise_items[i % len(noise_items)]}.")
                elif description_type == 'Simplified':
                    prompt = f"Generate ONLY a simplified patient sample description for {disease}, without specifying the disease's name and not being so obvious."
                    simplified.append(model([HumanMessage(content=prompt)]).content)
                    simplified_noise.append(simplified[-1] + f" Also, the patient mentions {noise_items[i % len(noise_items)]}.")
                elif description_type == 'Minimal':
                    prompt = f"Generate ONLY a minimal patient sample description for {disease}, focusing only on the most critical symptoms, without specifying the disease's name and not being so obvious."
                    minimal.append(model([HumanMessage(content=prompt)]).content)
                    minimal_noise.append(minimal[-1] + f" Plus, the patient feels {noise_items[i % len(noise_items)]}.")
                break  # If the model call is successful, break the loop
            except Exception as e:  # Catch any exception
                print(f"Error on attempt for {description_type} description of {disease}: {e}")
                if _ == 1:  # If this is the second attempt
                    if description_type == 'Detailed':
                        detailed.append(float('nan'))
                        detailed_noise.append(float('nan'))
                    elif description_type == 'Simplified':
                        simplified.append(float('nan'))
                        simplified_noise.append(float('nan'))
                    elif description_type == 'Minimal':
                        minimal.append(float('nan'))
                        minimal_noise.append(float('nan'))
    
# Create a DataFrame
df = pd.DataFrame({
    'Disease': diseases, 
    'Detailed': detailed, 
    'Detailed with Noise': detailed_noise, 
    'Simplified': simplified, 
    'Simplified with Noise': simplified_noise,
    'Minimal': minimal,
    'Minimal with Noise': minimal_noise,
})

print(df)

# Save the DataFrame to a CSV file
df.to_csv('./data/synthetic_data.csv', index=False)
