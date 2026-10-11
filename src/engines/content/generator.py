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

    response = client.chat.completions.create(
        model=WRITER_MODEL,
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": f"Topic: {topic}\n\nResearch:\n{research_brief}"}
        ],
        temperature=0.8,
        max_tokens=512,
    )
    
    raw_text = response.choices[0].message.content.strip()
    if raw_text.startswith("```json"): raw_text = raw_text.replace("```json", "", 1)
    if raw_text.startswith("```"): raw_text = raw_text.replace("```", "", 1)
    if raw_text.endswith("```"): raw_text = raw_text[:-3]
    
    return json.loads(raw_text.strip())

def generate_script(topic: str, research_brief: str, chosen_hook: str, target_platform: str = "YouTube Longform (5 minutes)") -> VideoScript:
    """
    Generates a structured video script based on a topic, factual research, and a chosen starting hook.
    """
    print(f"Writing final script for {target_platform}...")
    system_prompt = f"""You are an expert viral video scriptwriter for {target_platform}.
Write a highly engaging, well-paced script using the provided Research Brief for facts.
You must generate around 8 to 12 scenes (roughly 2-3 minutes).
You MUST start the first scene's narration with the exact "Chosen Hook" provided.

CRITICAL INSTRUCTION FOR VOICE CONSISTENCY & TONAL FLOW:
To ensure the voice sounds like a consistent, living human throughout the entire video:
1. "narrator_speaker": Set to "Aiden" (clear, human-sounding midrange voice with warmth).
2. "base_voice_persona": Keep uniform across all scenes ("Warm, resonant male voice speaking at a steady, deliberate pace with grounded depth and natural pauses.").
3. "voice_instruct" (per scene): Specify the emotional modulation for that exact scene. It MUST flow naturally from the preceding scene into the next.
   - Grounded & human (not rushed, not robotic, not over-dramatic).
   - Examples:
     * "Calm, thoughtful tone filled with quiet wonder and deliberate pauses."
     * "Inquisitive, intriguing tone delivering the mystery with natural depth."
     * "Slightly tense, anticipatory tone, slowing down for emphasis."
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
            "visual_prompt": "A detailed midjourney-style image prompt for the background visual."
        }},
        ...
    ]
}}
Note: If a specific character like "Zeus" speaks, set character_name to "Zeus", speaker to "Uncle_Fu", and give them a distinctive voice_instruct (e.g. "Commanding, booming, and stern tone").
Do NOT wrap the JSON in markdown code blocks. Just output the raw JSON object.
"""

    response = client.chat.completions.create(
        model=WRITER_MODEL,
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": f"Topic: {topic}\n\nChosen Hook: {chosen_hook}\n\nResearch Brief:\n{research_brief}"}
        ],
        temperature=0.7,
        max_tokens=2048,
    )
    
    raw_text = response.choices[0].message.content.strip()
    if raw_text.startswith("```json"): raw_text = raw_text.replace("```json", "", 1)
    if raw_text.startswith("```"): raw_text = raw_text.replace("```", "", 1)
    if raw_text.endswith("```"): raw_text = raw_text[:-3]
    
    script_data = json.loads(raw_text.strip())
    return VideoScript(**script_data)
