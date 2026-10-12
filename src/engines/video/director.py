import os
import sys
import json
from pathlib import Path
from openai import OpenAI
from dotenv import load_dotenv

# Add project root to python path
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))))
from src.models.storyboard import VideoProductionPlan, SceneStoryboard, VisualBeat

load_dotenv()

client = OpenAI(
    base_url=os.getenv("LLM_BASE_URL", "https://api.deepinfra.com/v1/openai"),
    api_key=os.getenv("LLM_API_KEY")
)

DIRECTOR_MODEL = os.getenv("WRITER_MODEL", "meta-llama/Llama-3.3-70B-Instruct")

DIRECTOR_SYSTEM_PROMPT = """You are an elite YouTube Animation Director specializing in the visual style and comic timing of top channels like OverSimplified and ancient myth animatics.

Given a script with exact audio durations, you will produce the complete VideoProductionPlan JSON.

CRITICAL DIRECTING RULES:
1. Every audio scene (14 to 24 seconds) MUST be broken down into 2 to 3 visual beats (each lasting 4 to 9 seconds).
2. The sum of the beat durations in a scene MUST equal the scene duration.
3. Keep the visual pacing lively and varied:
   - "shot_type": WIDE_NARRATOR_LEFT, WIDE_NARRATOR_RIGHT, CLOSEUP_NARRATOR_CENTER, FULL_CINEMATIC_NO_NARRATOR, or SPLIT_SCREEN_DIAGRAM.
   - "narrator_pose": EXPLAINING, POINTING, AWE_HANDS_UP, SHRUGGING_QUESTIONING, DEADPAN_NEUTRAL, SHOCKED, or HIDDEN.
   - "camera_motion": SLOW_ZOOM_IN, SLOW_ZOOM_OUT, PAN_LEFT_TO_RIGHT, PAN_RIGHT_TO_LEFT, PUNCH_IN, or STATIC.
   - "text_overlay": High-impact 2-4 word UPPERCASE title card for punchy moments (or null).

YOU MUST OUTPUT RAW JSON STRICTLY MATCHING THIS STRUCTURE:
{
  "title": "Title here",
  "topic": "Topic here",
  "art_style_prompt": "Clean 2D cel-shaded comic illustration, 4px crisp black linework, vibrant flat colors, graphic novel aesthetic, hand-drawn cartoon background art",
  "narrator_concept": "Chibi stickman astronomer with an expressive skeleton head, wearing a tweed coat and brass telescope goggles",
  "typography_theme": "Heavy bold comic sans-serif font, vibrant yellow fill, 4px solid black border with subtle drop shadow",
  "scenes": [
    {
      "scene_number": 1,
      "audio_file": "scene_1.mp3",
      "duration": 23.68,
      "beats": [
        {
          "beat_index": 1,
          "start_time": 0.0,
          "end_time": 7.5,
          "duration": 7.5,
          "shot_type": "WIDE_NARRATOR_LEFT",
          "narrator_pose": "AWE_HANDS_UP",
          "narrator_position": "bottom-left",
          "background_prompt": "Vast starry night sky with glowing Milky Way galaxy arching across space, retro observatory dome in distance, 2D comic art",
          "text_overlay": null,
          "camera_motion": "SLOW_ZOOM_IN"
        }
      ]
    }
  ]
}
Do NOT wrap the JSON in markdown blocks. Output only the raw JSON.
"""

def normalize_plan_dict(data: dict, script_title: str, script_topic: str) -> dict:
    """Helper to normalize keys if the model placed them under nested sections."""
    if "visual_theme" in data:
        theme = data["visual_theme"]
        data["art_style_prompt"] = theme.get("art_style_prompt", data.get("art_style_prompt", ""))
        data["narrator_concept"] = theme.get("narrator_concept", data.get("narrator_concept", ""))
        data["typography_theme"] = theme.get("typography_theme", data.get("typography_theme", ""))
        
    data.setdefault("title", script_title)
    data.setdefault("topic", script_topic)
    
    if "scenes" in data:
        for s in data["scenes"]:
            s.setdefault("audio_file", f"scene_{s.get('scene_number', 1)}.mp3")
            if "visual_beats" in s and "beats" not in s:
                s["beats"] = s.pop("visual_beats")
    return data

def direct_storyboard(script_path: Path, output_path: Path) -> VideoProductionPlan:
    print("=========================================")
    print("🎬 Visual Director Agent: Storyboarding")
    print("=========================================")
    
    if not script_path.exists():
        raise FileNotFoundError(f"Script file not found at {script_path}")
        
    with open(script_path, "r") as f:
        script_data = json.load(f)
        
    title = script_data.get("title", "Untitled")
    topic = script_data.get("topic", "General")
    scenes_summary = []
    
    for s in script_data.get("scenes", []):
        scenes_summary.append({
            "scene_number": s["scene_number"],
            "duration": s.get("estimated_duration_seconds", 15.0),
            "narration": s["narration"],
            "audio_file": f"scene_{s['scene_number']}.mp3"
        })
        
    user_prompt = f"""DIRECT THIS SCRIPT INTO A COMPLETE STORYBOARD:
Title: {title}
Topic: {topic}

Scenes with Exact Audio Durations:
{json.dumps(scenes_summary, indent=2)}
"""

    print(f"[Director]: Planning visual grammar, narrator staging, and shot beats for '{title}'...")
    
    response = client.chat.completions.create(
        model=DIRECTOR_MODEL,
        messages=[
            {"role": "system", "content": DIRECTOR_SYSTEM_PROMPT},
            {"role": "user", "content": user_prompt}
        ],
        temperature=0.5,
        max_tokens=4000
    )
    
    raw_text = response.choices[0].message.content.strip()
    if raw_text.startswith("```json"): raw_text = raw_text.replace("```json", "", 1)
    if raw_text.startswith("```"): raw_text = raw_text.replace("```", "", 1)
    if raw_text.endswith("```"): raw_text = raw_text[:-3]
    
    plan_dict = json.loads(raw_text.strip())
    plan_dict = normalize_plan_dict(plan_dict, title, topic)
    
    plan = VideoProductionPlan(**plan_dict)
    
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w") as f:
        f.write(plan.model_dump_json(indent=4))
        
    print(f"✅ Storyboard blueprint complete! Saved to {output_path}")
    print(f"Total directed scenes: {len(plan.scenes)}")
    total_beats = sum(len(s.beats) for s in plan.scenes)
    print(f"Total visual beats: {total_beats} (avg {round(total_beats / len(plan.scenes), 1)} beats per scene)")
    return plan

if __name__ == "__main__":
    script_file = Path("data/script_with_audio.json")
    out_file = Path("data/storyboard.json")
    direct_storyboard(script_file, out_file)
