
from langchain_google_genai import ChatGoogleGenerativeAI
from dotenv import load_dotenv
import os

load_dotenv()
api_key = os.getenv("GOOGLE_API_KEY")
if not api_key:
    raise ValueError("GOOGLE_API_KEY not set. Check your .env file.")

model = ChatGoogleGenerativeAI(model="gemini-2.5-flash", temperature='1.5', max_completion_token=3)
result = model.invoke("What is the capital of Nepal")

print(result.content)


