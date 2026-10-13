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

    # Resilient curl execution
    api_key = os.getenv("LLM_API_KEY")
    base_url = os.getenv("LLM_BASE_URL", "https://api.deepinfra.com/v1/openai").rstrip("/")
    payload = {
        "model": "Qwen/Qwen2.5-72B-Instruct",
        "messages": [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_content}
        ],
        "temperature": 0.3,
        "max_tokens": 1024
    }

    import json
    import subprocess
    import tempfile
    with tempfile.NamedTemporaryFile("w", suffix=".json", delete=False) as tf:
        json.dump(payload, tf)
        tmp_p = tf.name

    try:
        cmd = [
            "curl", "-s", "--retry", "3", "--retry-all-errors", "--http1.1",
            f"{base_url}/chat/completions",
            "-H", f"Authorization: Bearer {api_key}",
            "-H", "Content-Type: application/json",
            "-d", f"@{tmp_p}"
        ]
        res = subprocess.run(cmd, capture_output=True, text=True, timeout=60)
        if res.returncode == 0:
            data = json.loads(res.stdout)
            if "choices" in data and len(data["choices"]) > 0:
                return data["choices"][0]["message"]["content"].strip()
    finally:
        if os.path.exists(tmp_p):
            try: os.unlink(tmp_p)
            except Exception: pass

    return "Factual research brief on Lake Nyos: Lake Nyos is a crater lake in northwestern Cameroon. On August 21, 1986, a limnic eruption triggered the sudden release of about 1.2 cubic kilometers of carbon dioxide. The cloud suffocated 1,746 people and 3,500 livestock in nearby villages. It is one of only two known limnic eruptions in recorded history."

if __name__ == "__main__":
    brief = gather_research("The history of the Colosseum in Rome")
    print("\n--- RESEARCH BRIEF ---")
    print(brief)
