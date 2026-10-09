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

CRITICAL INSTRUCTION FOR TTS INFLECTION:
The "narration" field will be fed directly into an AI Text-to-Speech (TTS) engine. To make the TTS sound like a real human with depth, tone, and life, you MUST use expressive formatting:
- Use ellipses (...) for dramatic pauses.
- Use dashes (-) for sudden interruptions or shifts in tone.
- CAPITALIZE specific words to force the TTS to put heavy emphasis on them.
- Use exclamation points (!) and question marks (?) to force pitch changes.
Write the narration exactly as a highly energetic, dramatic storyteller would speak it out loud.

You MUST output your response as a raw JSON object that strictly conforms to the following structure:
{{
    "title": "A catchy title",
    "topic": "The general topic",
    "target_platform": "{target_platform}",
    "scenes": [
        {{
            "scene_number": 1,
            "character_name": "Narrator",
            "narration": "The exact words the voiceover will say (Must start with the Chosen Hook).",
            "visual_prompt": "A detailed midjourney-style image prompt for the background visual."
        }},
        ...
    ]
}}
Note: For most scenes, the `character_name` should be "Narrator", but if a specific character like "Zeus" is speaking a line directly, you can change the `character_name` to "Zeus".
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
