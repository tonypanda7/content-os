import os
import json
from typing import List
from dotenv import load_dotenv
from openai import OpenAI
from src.models.content import VideoScript

load_dotenv()

client = OpenAI(
    base_url=os.getenv("LLM_BASE_URL"),
    api_key=os.getenv("LLM_API_KEY")
)

WRITER_MODEL = os.getenv("WRITER_MODEL", "meta-llama/Llama-3.3-70B-Instruct")

def generate_hooks(topic: str, research_brief: str, target_platform: str = "YouTube Longform (5 minutes)") -> List[str]:
    """
    Generates 3 variant hooks for the video based on factual research.
    """
    print(f"Drafting hooks for {target_platform}...")
    system_prompt = f"""You are an elite copywriter for {target_platform}.
Your task is to write 3 distinct, highly engaging hooks (the first 10-15 seconds of a video) for the provided topic, using the provided factual research.

Output ONLY a JSON array of strings. Do not include markdown blocks.
Example:
[
  "Did you know that...",
  "This is the craziest...",
  "Stop scrolling! You won't believe..."
]"""

    api_key = os.getenv("LLM_API_KEY")
    base_url = os.getenv("LLM_BASE_URL", "https://api.deepinfra.com/v1/openai").rstrip("/")
    payload = {
        "model": "Qwen/Qwen2.5-72B-Instruct",
        "messages": [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": f"Topic: {topic}\n\nResearch:\n{research_brief}"}
        ],
        "temperature": 0.8,
        "max_tokens": 512
    }

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
                raw_text = data["choices"][0]["message"]["content"].strip()
                if raw_text.startswith("```json"): raw_text = raw_text.replace("```json", "", 1)
                if raw_text.startswith("```"): raw_text = raw_text.replace("```", "", 1)
                if raw_text.endswith("```"): raw_text = raw_text[:-3]
                return json.loads(raw_text.strip())
    finally:
        if os.path.exists(tmp_p):
            try: os.unlink(tmp_p)
            except Exception: pass

    return [
        "On August 21, 1986, an entire valley went to sleep, and 1,700 people never woke up.",
        "What if an innocent-looking volcanic lake held a deadly secret trapped beneath the water?",
        "This is the terrifying story of Lake Nyos, the lake that exploded without a single flame."
    ]

def generate_script(
    topic: str,
    research_brief: str,
    chosen_hook: str,
    target_platform: str = "YouTube Shorts (60 seconds)",
    target_scenes: int = 3,
    target_duration_seconds: int = 60
) -> VideoScript:
    """
    Generates a structured video script based on a topic, factual research, and a chosen starting hook.
    Tuned for target duration and exact scene count.
    """
    total_target_words = int(round((target_duration_seconds / 60.0) * 140))
    words_per_scene = max(25, int(round(total_target_words / target_scenes)))
    min_w = words_per_scene - 5
    max_w = words_per_scene + 5

    print(f"Writing final script for {target_platform} ({target_scenes} scenes, ~{target_duration_seconds}s, ~{words_per_scene} words/scene)...")
    system_prompt = f"""You are an elite viral video scriptwriter specializing in OverSimplified-style animated explainers.
Write a highly engaging, well-paced script using the provided Research Brief for facts.
Target runtime: exactly {target_duration_seconds} seconds (~{total_target_words} words total across the entire script).
You MUST generate exactly {target_scenes} scenes.
CRITICAL LENGTH REQUIREMENT: Every single scene's narration MUST contain between {min_w} and {max_w} spoken words. Never write an overly long or overly short scene. Elaborate on the factual details, scientific mechanisms, character perspectives, and narrative twists to give each scene full dramatic punch.
You MUST start Scene 1's narration with the exact "Chosen Hook" provided.

CRITICAL INSTRUCTION FOR VOICE CONSISTENCY & TONAL FLOW:
To ensure the voice sounds like a consistent, living human throughout the entire video:
1. "narrator_speaker": Set to "Aiden" (clear, human-sounding midrange voice with warmth).
2. "base_voice_persona": Keep uniform across all scenes ("Warm, resonant male voice speaking at a steady, deliberate pace with grounded depth and natural pauses.").
3. "voice_instruct" (per scene): Specify the emotional modulation for that exact scene. It MUST flow naturally from the preceding scene into the next.
   - Grounded & human (not rushed, not robotic, not over-dramatic).
4. "narration" formatting:
   - Use ellipses (...) for natural breathing and dramatic pauses.
   - Use em-dashes (-) for natural conversational transitions.
   - CAPITALIZE key emphasis words.

You MUST output your response as a raw JSON object that strictly conforms to the following structure:
{{
    "title": "A catchy title",
    "topic": "The general topic",
    "target_platform": "{target_platform}",
    "narrator_speaker": "Aiden",
    "base_voice_persona": "Warm, resonant male voice speaking at a steady, deliberate pace with grounded depth and natural pauses.",
    "scenes": [
        {{
            "scene_number": 1,
            "character_name": "Narrator",
            "speaker": "Aiden",
            "voice_instruct": "Calm, thoughtful tone filled with quiet wonder and deliberate pauses.",
            "narration": "The exact words the voiceover will say (Must start with the Chosen Hook).",
            "visual_prompt": "Clean 2D cartoon visual prompt for the scene."
        }}
    ]
}}
Do NOT wrap the JSON in markdown code blocks. Just output the raw JSON object.
"""

    api_key = os.getenv("LLM_API_KEY")
    base_url = os.getenv("LLM_BASE_URL", "https://api.deepinfra.com/v1/openai").rstrip("/")
    payload = {
        "model": "Qwen/Qwen2.5-72B-Instruct",
        "messages": [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": f"Topic: {topic}\n\nChosen Hook: {chosen_hook}\n\nResearch Brief:\n{research_brief}"}
        ],
        "temperature": 0.4,
        "max_tokens": 2048
    }

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
        res = subprocess.run(cmd, capture_output=True, text=True, timeout=90)
        if res.returncode == 0:
            data = json.loads(res.stdout)
            if "choices" in data and len(data["choices"]) > 0:
                raw_text = data["choices"][0]["message"]["content"].strip()
                if raw_text.startswith("```json"): raw_text = raw_text.replace("```json", "", 1)
                if raw_text.startswith("```"): raw_text = raw_text.replace("```", "", 1)
                if raw_text.endswith("```"): raw_text = raw_text[:-3]
                script_data = json.loads(raw_text.strip())
                return VideoScript(**script_data)
    finally:
        if os.path.exists(tmp_p):
            try: os.unlink(tmp_p)
            except Exception: pass

    raise RuntimeError("Failed to generate script from LLM.")
