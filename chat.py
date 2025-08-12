import base64
import os
from google import genai
from google.genai import types
from dotenv import load_dotenv
load_dotenv()

history = []

def generate():
    client = genai.Client(
        api_key=os.getenv("GEMINI_API_KEY"),
    )

    model = "gemini-2.5-pro"

    print("Bot: Hello, how can I help you?")
    user_input = input("You: ")

    # Append user input to history
    history.append({"role": "user", "parts": [user_input]})

    # Build contents from history
    contents = []
    for msg in history:
        contents.append(
            types.Content(
                role=msg["role"],
                parts=[types.Part.from_text(text=p) for p in msg["parts"]]
            )
        )

    tools = [
        types.Tool(googleSearch=types.GoogleSearch()),
    ]
    generate_content_config = types.GenerateContentConfig(
        thinking_config=types.ThinkingConfig(
            thinking_budget=-1,
        ),
        tools=tools,
    )

    # Streaming output with None check
    full_model_output = ""
    for chunk in client.models.generate_content_stream(
        model=model,
        contents=contents,
        config=generate_content_config,
    ):
        if chunk.text:  # Avoid NoneType errors
            print(chunk.text, end="", flush=True)
            full_model_output += chunk.text

    print()  # newline
    history.append({"role": "model", "parts": [full_model_output]})

if __name__ == "__main__":
    generate()
