import os
from dotenv import load_dotenv
from openai import OpenAI
load_dotenv()
# Connect Python to OpenRouter
client = OpenAI(
    base_url="https://openrouter.ai/api/v1",
    api_key=os.getenv("OPENROUTER_API_KEY")
)

# Conversation instructions
messages = [
    {
        "role": "system",
        "content": (
            "You are a friendly Python programming trainer. "
            "Explain concepts in simple English and give "
            "short, practical examples."
        )
    }
]

print("Python Learning Assistant")
print("Type 'exit' to stop.")

while True:
    question = input("\nStudent: ").strip()

    if question.lower() == "exit":
        print("Assistant: Goodbye!")
        break

    if not question:
        continue

    messages.append({
        "role": "user",
        "content": question
    })

    try:
        response = client.chat.completions.create(
            model="inclusionai/ling-3.1-flash",
            messages=messages,
            temperature=0.3,
            max_tokens=300
        )

        answer = response.choices[0].message.content

        print("\nAssistant:", answer)

        messages.append({
            "role": "assistant",
            "content": answer
        })

    except Exception as error:
        print("Error:", error)