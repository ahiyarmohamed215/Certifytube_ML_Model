from __future__ import annotations
import os
from google import genai

def get_gemini_client() -> genai.Client:
    api_key = os.getenv("GEMINI_API_KEY")
    if not api_key:
        raise RuntimeError("GEMINI_API_KEY is missing. Set it in your environment.")
    return genai.Client(api_key=api_key)
