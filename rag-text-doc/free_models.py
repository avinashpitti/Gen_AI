import os
from dotenv import load_dotenv
from groq import Groq

# Load GROQ_API_KEY from .env
load_dotenv()

# Initialize Groq client
client = Groq(api_key=os.getenv("GROQ_API_KEY"))

print("--- Fetching Available Groq Models ---\n")

try:
    models = client.models.list()
    for model in models.data:
        print(f"Model ID: {model.id}")
except Exception as e:
    print(f"Error fetching models: {e}")