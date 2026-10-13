import os
import sys
import json
import subprocess
from typing import Dict, Any, List
from dotenv import load_dotenv

load_dotenv()

SELECTOR_MODEL = os.getenv("RESEARCHER_MODEL", "Qwen/Qwen2.5-72B-Instruct")
LLM_BASE_URL = os.getenv("LLM_BASE_URL", "https://api.deepinfra.com/v1/openai").rstrip("/")
API_KEY = os.getenv("LLM_API_KEY")

TOPIC_SELECTION_SYSTEM_PROMPT = """You are an elite YouTube Content Strategist and Viral Video Algorithm Expert.
Your specialty is high-retention, million-view 2D cartoon animated explainers (in the visual storytelling style of OverSimplified and Kurzgesagt).

Your task is to brainstorm, evaluate, and select a single highest-performing topic for a 3-MINUTE animated explainer video.

CRITERIA FOR A 10/10 TOPIC:
1. Massive Curiosity Gap & Hook: The concept must sound baffling, intense, or impossible at first glance (e.g. 'The Lake That Silenced 1,700 People While They Slept', 'The Day 20,000 Birds Defeated the Military').
2. Visual Storytelling Fit: Easily explained using simple stick-figure characters, physical props, cross-sections, and diagram arrows (matching OverSimplified).
3. 3-Minute Narrative Arc: A tight, punchy story that builds tension, delivers the scientific/historical twist, and concludes with a memorable insight within exactly 180 seconds (~400-450 words total, 6 scenes).
4. High Universal Appeal: Not overly niche; appeals to millions of curious general viewers.

Evaluate at least 4 candidate topics across different genres (Natural Mysteries, Bizarre History, Wild Science, Survival Paradoxes).
Pick the single strongest topic and provide its full production blueprint.

Respond STRICTLY with raw JSON matching this structure:
{
  "candidates": [
    {
      "topic": "Topic Name",
      "viral_score_out_of_100": 95,
      "hook_premise": "Why people will click in 0-5 seconds...",
      "animation_suitability": "How well stick figures & visual diagrams explain it..."
    }
  ],
  "chosen_topic": "The exact selected title",
  "topic_slug": "short_snake_case_slug",
  "logline": "1-sentence summary of the story",
  "target_duration_seconds": 180,
  "target_scenes_count": 6,
  "viral_analysis": "2-3 sentences explaining why this topic will maximize click-through rate and retention"
}
DO NOT wrap in markdown formatting.
"""

def call_selector_llm() -> Dict[str, Any]:
    """Resilient curl execution to select the highest-performing viral topic."""
    payload = {
        "model": "Qwen/Qwen2.5-72B-Instruct",
        "messages": [
            {"role": "system", "content": TOPIC_SELECTION_SYSTEM_PROMPT},
            {"role": "user", "content": "Analyze viral YouTube explainer trends and choose the single best 3-minute explainer topic for production right now."}
        ],
        "temperature": 0.4,
        "max_tokens": 1200
    }

    import tempfile
    with tempfile.NamedTemporaryFile("w", suffix=".json", delete=False) as tf:
        json.dump(payload, tf)
        tmp_p = tf.name

    try:
        cmd = [
            "curl", "-s", "--retry", "3", "--retry-all-errors", "--http1.1",
            f"{LLM_BASE_URL}/chat/completions",
            "-H", f"Authorization: Bearer {API_KEY}",
            "-H", "Content-Type: application/json",
            "-d", f"@{tmp_p}"
        ]
        res = subprocess.run(cmd, capture_output=True, text=True, timeout=60)
        if res.returncode == 0:
            data = json.loads(res.stdout)
            if "choices" in data and len(data["choices"]) > 0:
                raw = data["choices"][0]["message"]["content"].strip()
                if raw.startswith("```json"): raw = raw[7:]
                if raw.startswith("```"): raw = raw[3:]
                if raw.endswith("```"): raw = raw[:-3]
                return json.loads(raw.strip())
    finally:
        if os.path.exists(tmp_p):
            try: os.unlink(tmp_p)
            except Exception: pass

    # Robust default fallback if network timeout occurs
    return {
        "candidates": [],
        "chosen_topic": "The Exploding Lake: The Deadly Mystery of Lake Nyos",
        "topic_slug": "the_deadly_mystery_of_lake_nyos",
        "logline": "In 1986, a silent, invisible cloud erupted from a tranquil volcanic crater lake, mystifying scientists and wiping out an entire valley overnight.",
        "target_duration_seconds": 180,
        "target_scenes_count": 6,
        "viral_analysis": "Combines intense mystery, incredible scientific detective work (limnic eruptions and volcanic gas buildup), and visual cross-section diagrams perfectly suited for OverSimplified animation."
    }

def select_topic_autonomously() -> Dict[str, Any]:
    print("=========================================================================")
    print("🧠 [Topic Research & Viral Strategist Agent]: Brainstorming High-CTR Topics...")
    print("=========================================================================")
    decision = call_selector_llm()
    print(f"\n🏆 CHOSEN TOPIC: {decision['chosen_topic']}")
    print(f"🎯 Logline: {decision.get('logline', '')}")
    print(f"📈 Viral Analysis: {decision.get('viral_analysis', '')}")
    print(f"⏱️ Target Runtime: {decision.get('target_duration_seconds', 180)}s (~{decision.get('target_duration_seconds', 180)/60:.1f} mins) across {decision.get('target_scenes_count', 6)} scenes\n")
    return decision

if __name__ == "__main__":
    select_topic_autonomously()
