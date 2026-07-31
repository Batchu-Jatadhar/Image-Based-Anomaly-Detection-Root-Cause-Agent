import requests
from typing import Dict, Any, List
from src.rag_engine import query_index

SEMANTIC_SCHOLAR_URL = "https://api.semanticscholar.org/graph/v1/paper/search"

def retrieve_all(query: str) -> Dict[str, Any]:
    """
    Retrieves information from both local RAG index and external literature (Semantic Scholar),
    and merges them into a unified format for LLM consumption.
    
    Args:
        query (str): The search query.
        
    Returns:
        Dict[str, Any]: A unified dictionary containing local RAG results and external literature.
    """
    # 1. Search the local guideline RAG index
    try:
        local_results = query_index(query, top_k=3)
    except Exception as e:
        print(f"Failed to query local RAG index (make sure it is built): {e}")
        local_results = []
        
    # 2. Retrieve external literature from Semantic Scholar
    external_results = []
    try:
        # Limit to 3 results to keep context sizes manageable for LLMs
        params = {
            "query": query,
            "limit": 3,
            "fields": "title,abstract,authors,year,url"
        }
        
        # We don't strictly require an API key for basic usage, but rate limits apply
        headers = {}
        
        # If we had a centralized settings variable for Semantic Scholar API key, we could use it:
        # from src.config.settings import SEMANTIC_SCHOLAR_API_KEY
        # if SEMANTIC_SCHOLAR_API_KEY:
        #     headers["x-api-key"] = SEMANTIC_SCHOLAR_API_KEY
        
        response = requests.get(SEMANTIC_SCHOLAR_URL, params=params, headers=headers, timeout=10)
        
        if response.status_code == 200:
            data = response.json()
            for paper in data.get("data", []):
                external_results.append({
                    "title": paper.get("title"),
                    "abstract": paper.get("abstract") or "No abstract available.",
                    "year": paper.get("year"),
                    "authors": [author.get("name") for author in paper.get("authors", [])],
                    "url": paper.get("url")
                })
        else:
            print(f"Semantic Scholar API warning: Status Code {response.status_code}")
            
    except requests.RequestException as e:
        print(f"Network error while fetching from Semantic Scholar: {e}")
    except Exception as e:
        print(f"Unexpected error retrieving external literature: {e}")
        
    # 3 & 4. Merge results into a unified retrieval output and return
    return {
        "query": query,
        "local_guidelines": local_results,
        "external_literature": external_results
    }
