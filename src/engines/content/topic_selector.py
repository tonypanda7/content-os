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
Your specialty is high-retention, million-view 2D cartoon animated explainers (in the visual storytelling style of OverSimplified).

Your task is to brainstorm, evaluate, and select a single highest-performing topic for a {target_duration_seconds}-SECOND animated video ({platform}).

CRITERIA FOR A 10/10 VIRAL TOPIC:
1. Instant Hook & Curiosity Loop: The premise must immediately grab mobile attention in the first 0-3 seconds.
2. Visual Storytelling Fit: Easily illustrated using simple stick-figure characters, physical props, cross-sections, and direction arrows (OverSimplified style, pure visual metaphors, zero text).
3. Pacing & Punchline: A tight {target_duration_seconds}-second narrative arc that delivers high value and a surprising scientific or historical twist across exactly {target_scenes_count} scenes.
4. Universal Human Fascination: Bizarre historical anomalies, shocking survival feats, or mind-bending natural phenomena.

Evaluate at least 4 candidate topics across different genres.
Pick the single strongest topic and provide its full production blueprint.

Respond STRICTLY with raw JSON matching this structure:
{{
  "candidates": [
    {{
      "topic": "Topic Name",
      "viral_score_out_of_100": 95,
      "hook_premise": "Why viewers stop scrolling in 0-3 seconds...",
      "animation_suitability": "How stick figures & arrows visually convey it..."
    }}
  ],
  "chosen_topic": "The exact selected title",
  "topic_slug": "short_snake_case_slug",
  "logline": "1-sentence summary of the story",
  "target_duration_seconds": {target_duration_seconds},
  "target_scenes_count": {target_scenes_count},
  "viral_analysis": "2-3 sentences explaining why this topic will maximize retention and shareability"
}}
DO NOT wrap in markdown formatting.
"""

def call_selector_llm(
    target_duration_seconds: int = 60,
    target_scenes_count: int = 3,
    platform: str = "YouTube Shorts (60 seconds)"
) -> Dict[str, Any]:
    """Resilient curl execution to select the highest-performing viral topic."""
    system_prompt = TOPIC_SELECTION_SYSTEM_PROMPT.format(
        target_duration_seconds=target_duration_seconds,
        target_scenes_count=target_scenes_count,
        platform=platform
    )
    payload = {
        "model": SELECTOR_MODEL,
        "messages": [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": f"Analyze current viral {platform} trends and choose the single best {target_duration_seconds}-second animated explainer topic right now."}
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

    # Robust default fallback for 60s Shorts
    return {
        "candidates": [],
        "chosen_topic": "The Man Who Survived Going Inside a Nuclear Particle Accelerator",
        "topic_slug": "the_man_who_survived_a_particle_accelerator",
        "logline": "In 1978, Anatoli Bugorski stuck his head inside a running particle accelerator and saw a flash brighter than a thousand suns—yet somehow survived.",
        "target_duration_seconds": target_duration_seconds,
        "target_scenes_count": target_scenes_count,
        "viral_analysis": "An irresistible curiosity hook that can be explained visually with a cross-section of a synchrotron and cartoon stick-figure particle beam in 60 seconds."
    }

def select_topic_autonomously(
    target_duration_seconds: int = 60,
    target_scenes_count: int = 3,
    platform: str = "YouTube Shorts (60 seconds)"
) -> Dict[str, Any]:
    print("=========================================================================")
    print(f"🧠 [Topic Research & Viral Strategist Agent]: Selecting Topic for {platform}...")
    print("=========================================================================")
    decision = call_selector_llm(
        target_duration_seconds=target_duration_seconds,
        target_scenes_count=target_scenes_count,
        platform=platform
    )
    print(f"\n🏆 CHOSEN TOPIC: {decision['chosen_topic']}")
    print(f"🎯 Logline: {decision.get('logline', '')}")
    print(f"📈 Viral Analysis: {decision.get('viral_analysis', '')}")
    print(f"⏱️ Target Runtime: {decision.get('target_duration_seconds', target_duration_seconds)}s across {decision.get('target_scenes_count', target_scenes_count)} scenes\n")
    return decision

if __name__ == "__main__":
    select_topic_autonomously()
