import json
import os
from google import genai
from google.genai import types
from config import gemini_api_key, JSON_FILE_PATH

async def llm_model_func(
    prompt, system_prompt=None, history_messages=[], tempurature=0.1,
    max_output_tokens=500, seed=42, presence_penalty=0.2,
    frequency_penalty=0.2, keyword_extraction=False, **kwargs
) -> str:
    client = genai.Client(api_key=gemini_api_key)

    if os.path.exists(JSON_FILE_PATH):
        with open(JSON_FILE_PATH, "r", encoding="utf-8") as file:
            data = json.load(file)
        if data.get("hybrid"):
            for value in data["hybrid"].values():
                original_prompt = value.get("original_prompt")
                return_value = value.get("return")
                history_messages.append({"role": "user", "content": original_prompt})
                history_messages.append({"role": "assistant", "content": return_value})

    combined_prompt = ""
    if system_prompt:
        combined_prompt += f"{system_prompt}\n"
    for msg in history_messages:
        combined_prompt += f"{msg['role']}: {msg['content']}\n"
    combined_prompt += f"user: {prompt}"

    response = client.models.generate_content(
        model="gemini-1.5-flash",
        contents=[combined_prompt],
        config=types.GenerateContentConfig(
            max_output_tokens=max_output_tokens,
            temperature=tempurature,
            seed=seed,
            presence_penalty=presence_penalty,
            frequency_penalty=frequency_penalty,
        )
    )

    return response.text
