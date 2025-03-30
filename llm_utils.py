import os
from llm_config import llm 
from langchain.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from operator import itemgetter
from vectordb import get_retriever
from langchain_core.messages import HumanMessage, AIMessage
from models import VectorStore, SearchEngine
from dotenv import load_dotenv

load_dotenv()

retriever = get_retriever()

rag_template_str = (
    "You are an AI-powered medical assistant. Your goal is to provide helpful and general medical information, "
    "but you are NOT a doctor. Always recommend consulting a qualified healthcare professional for medical concerns.\n\n"
    "Base your response strictly on the provided context.\n"
    "Do not diagnose conditions or prescribe medication.\n"
    "If a question is outside the provided medical context, politely state that you cannot provide an answer.\n\n"
    "Context: {context}\n\n"
    "Query: {query}"
)

fallback_prompt = ChatPromptTemplate.from_template(
    (
        "You are a medical assistant AI. You can provide general health information but NOT diagnoses or prescriptions.\n"
        "Encourage users to consult a healthcare professional for medical concerns.\n"
        "If a query is unrelated to health or medicine, politely acknowledge that you cannot assist.\n\n"
        "Current conversation:\n\n{chat_history}\n\n"
        "Human: {query}"
    )
)

rag_prompt = ChatPromptTemplate.from_template(rag_template_str)
rag_chain = rag_prompt | llm | StrOutputParser()

def get_question_router():
    """Creates a question router using LLM and tools."""

    prompt_txt = (
        "You are an AI that routes user queries to either a VectorStore or a SearchEngine.\n"
        "The VectorStore contains medical and healthcare-related information.\n"
        "Use the SearchEngine for general health-related queries that are not found in the VectorStore.\n"
        "If a question is not related to health or medicine, output 'not medical' without using any tool.\n\n"
        "query: {query}"
    )
    return ChatPromptTemplate.from_template(prompt_txt) | llm.bind_tools(tools=[VectorStore, SearchEngine])

question_router = get_question_router()

def run_rag_chain(query: str, retriever) -> str:
    """Runs the RAG chain to generate an answer based on retrieved medical context."""
    context = retriever.invoke(query)
    return rag_chain.invoke({"query": query, "context": context})

fallback_chain = (
    {
        "chat_history": lambda x: "\n".join(
            [
                (
                    f"human: {msg.content}"
                    if isinstance(msg, HumanMessage)
                    else f"AI: {msg.content}"
                )
                for msg in x["chat_history"]
            ]
        ),
        "query": itemgetter("query"),
    }
    | fallback_prompt
    | llm
    | StrOutputParser()
)

def run_fallback_chain(query: str, chat_history=None) -> str:
    """Runs the fallback chain for non-medical queries."""
    if chat_history is None:
        chat_history = []
    return fallback_chain.invoke({"query": query, "chat_history": chat_history})
