from pydantic import BaseModel, Field

class VectorStore(BaseModel):
    """A vector store containing medical information on symptoms, diseases, treatments, medications, diagnostics, nutrition, and general healthcare topics."""
    query: str = Field(..., description="Medical-related query for retrieval from the VectorStore.")

class SearchEngine(BaseModel):
    """A search engine for searching general medical information that is not stored in the VectorStore."""
    query: str = Field(..., description="Medical-related query for external lookup.")

def get_router_prompt(query: str) -> str:
    """Returns a prompt for routing the user query."""

    return (
        "You are an expert in routing user queries to either a VectorStore or a SearchEngine.\n"
        "Use the VectorStore for queries related to symptoms, diseases, treatments, medications, diagnostics, nutrition, and general healthcare topics.\n"
        "Use the SearchEngine for other health-related queries that are not covered by the VectorStore.\n"
        "If a query is not related to medical or healthcare topics, output 'not medical-related' without using any tool.\n\n"
        f"query: {query}"
    )
