import os
from dotenv import load_dotenv
from openai import OpenAI
from duckduckgo_search import DDGS

load_dotenv()

client = OpenAI(
    base_url=os.getenv("LLM_BASE_URL"),
    api_key=os.getenv("LLM_API_KEY")
)

RESEARCHER_MODEL = os.getenv("RESEARCHER_MODEL", "meta-llama/Llama-3.3-70B-Instruct")

def perform_web_search(query: str, max_results: int = 5) -> str:
    """Helper function to perform a web search and return a formatted string of results."""
    try:
        results = DDGS().text(query, max_results=max_results)
        formatted_results = []
        for r in results:
            formatted_results.append(f"Title: {r['title']}\nSnippet: {r['body']}\n")
        return "\n".join(formatted_results)
    except Exception as e:
        print(f"Warning: Web search failed ({e}). Proceeding without web results.")
        return ""

def gather_research(topic: str) -> str:
    """
    Gathers factual information on a topic using web search and LLM summarization.
    Returns a highly concentrated research brief.
    """
    print(f"Gathering research on: {topic}")
    
    # 1. Get raw data from the web
    web_data = perform_web_search(topic)
    
    # 2. Synthesize into a Research Brief
    system_prompt = """You are an expert researcher for a video production team.
Your job is to take a topic and raw web search results, and synthesize them into a dense, factual "Research Brief".
The brief should include:
- Core facts and timeline
- Surprising or highly engaging hooks/trivia
- Common misconceptions
- Key visual concepts that define this topic

Be concise. Do not write a script, just provide the factual building blocks."""

    user_content = f"Topic: {topic}\n\nWeb Search Results:\n{web_data}"

    response = client.chat.completions.create(
        model=RESEARCHER_MODEL,
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_content}
        ],
        temperature=0.3,
        max_tokens=1024,
    )
    
    return response.choices[0].message.content.strip()

if __name__ == "__main__":
    brief = gather_research("The history of the Colosseum in Rome")
    print("\n--- RESEARCH BRIEF ---")
    print(brief)
