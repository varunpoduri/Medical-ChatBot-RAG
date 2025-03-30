from langchain_groq.chat_models import ChatGroq
import os
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Check if GROQ_API_KEY is set
if os.getenv("GROQ_API_KEY") is None:
    raise ValueError("The GROQ_API_KEY environment variable must be set.")

# Define LLM instance once
llm = ChatGroq(
    model="llama3-70b-8192",
    temperature=0.75,
    api_key=os.getenv("GROQ_API_KEY"),  # Load API key from .env
)
