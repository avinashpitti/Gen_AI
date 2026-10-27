import os
import requests
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

api_key = os.getenv("OPENROUTER_API_KEY")

if not api_key:
    print("Error: OPENROUTER_API_KEY is missing from your .env file.")
    exit(1)

print("--- Fetching Free OpenRouter Models ---\n")

headers = {
    "Authorization": f"Bearer {api_key}"
}

try:
    response = requests.get("https://openrouter.ai/api/v1/models", headers=headers)
    response.raise_for_status()
    models_data = response.json().get("data", [])

    # Filter models that are completely free of cost
    free_models = []
    for model in models_data:
        pricing = model.get("pricing", {})
        prompt_cost = float(pricing.get("prompt", 0))
        completion_cost = float(pricing.get("completion", 0))
        
        # Free models either have 0 cost or :free in their ID
        if (prompt_cost == 0 and completion_cost == 0) or ":free" in model.get("id", ""):
            free_models.append(model.get("id"))

    if free_models:
        for model_id in free_models:
            print(f"Model ID: {model_id}")
        print(f"\nTotal Free Models Found: {len(free_models)}")
    else:
        print("No free models found matching the criteria.")

except requests.exceptions.RequestException as e:
    print(f"Error connecting to OpenRouter: {e}")