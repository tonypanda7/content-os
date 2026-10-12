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
    api_key=os.getenv("LLM_API_KEY"),
    max_retries=3,
    timeout=45.0
)

DIRECTOR_MODEL = os.getenv("WRITER_MODEL", "meta-llama/Llama-3.3-70B-Instruct")

DIRECTOR_PROMPT_TEMPLATE = """You are an elite YouTube Animation and Educational Infographic Director (specializing in the style of OverSimplified, Kurzgesagt, and visual explainers).

DIRECT THIS SCENE INTO 2 TO 3 INFORMATIVE INFOGRAPHIC BEATS:
Scene Number: {scene_number}
Duration: {duration} seconds
Narration: "{narration}"

CRITICAL EDUCATIONAL & ANIMATION RULES:
1. NO GENERIC LANDSCAPE WALLPAPERS. Every visual beat MUST be an INFORMATIVE INFOGRAPHIC, ANNOTATED CHART, TIMELINE, OR NARRATIVE COMIC SCENE that visually explains the exact narration words. The viewer MUST learn information by reading the image!
2. The sum of the beat durations MUST equal exactly {duration} seconds.
3. Keep the visual pacing lively and varied:
   - "shot_type": "WIDE_NARRATOR_LEFT", "WIDE_NARRATOR_RIGHT", "CLOSEUP_NARRATOR_CENTER", "FULL_CINEMATIC_NO_NARRATOR", or "SPLIT_SCREEN_DIAGRAM"
   - "narrator_pose": "EXPLAINING", "POINTING", "SHRUGGING_QUESTIONING", "THINKING", or "HIDDEN"
   - "narrator_position": "bottom-left", "bottom-right", "center", or null
   - "infographic_concept": 1-2 sentence description of the diagram, chart, or narrative scene
   - "background_prompt": FLUX prompt starting with "Clean 2D cel-shaded comic infographic illustration, 4px crisp black linework, vibrant flat colors, graphic novel aesthetic". Describe labeled diagrams, scale bars, charts, or character interactions.
   - "text_overlay": High-impact 2-4 word UPPERCASE bold title card (e.g. "2 TRILLION GALAXIES", "WHERE IS EVERYBODY?", "THE GREAT FILTER")
   - "graphic_annotation": Optional on-screen graphic overlay: "RED_CROSSOUT", "ARROW_POINTING", "QUESTION_MARKS", or null
   - "camera_motion": "SLOW_ZOOM_IN", "SLOW_ZOOM_OUT", "PAN_LEFT_TO_RIGHT", "PUNCH_IN", or "STATIC"

OUTPUT STRICT RAW JSON ARRAY ONLY matching this structure:
[
  {{
    "beat_index": 1,
    "start_time": 0.0,
    "end_time": 7.5,
    "duration": 7.5,
    "shot_type": "WIDE_NARRATOR_LEFT",
    "narrator_pose": "POINTING",
    "narrator_position": "bottom-left",
    "infographic_concept": "...",
    "background_prompt": "Clean 2D cel-shaded comic infographic illustration...",
    "text_overlay": "...",
    "graphic_annotation": "...",
    "camera_motion": "SLOW_ZOOM_IN"
  }}
]
DO NOT WRAP IN MARKDOWN CODEBLOCKS.
"""

def direct_scene(scene_data: dict) -> list[VisualBeat]:
    scene_num = scene_data["scene_number"]
    duration = scene_data.get("estimated_duration_seconds", 15.0)
    narration = scene_data["narration"]

    prompt = DIRECTOR_PROMPT_TEMPLATE.format(
        scene_number=scene_num,
        duration=duration,
        narration=narration
    )

    print(f"   [Director]: Planning Scene {scene_num} ({duration}s)...")
    res = client.chat.completions.create(
        model=DIRECTOR_MODEL,
        messages=[
            {"role": "system", "content": "You are an elite YouTube educational director. Output strict raw JSON array only."},
            {"role": "user", "content": prompt}
        ],
        temperature=0.3,
        max_tokens=1500
    )

    raw_text = res.choices[0].message.content.strip()
    if raw_text.startswith("```json"): raw_text = raw_text[7:]
    if raw_text.startswith("```"): raw_text = raw_text[3:]
    if raw_text.endswith("```"): raw_text = raw_text[:-3]

    raw_beats = json.loads(raw_text.strip())
    valid_poses = ["EXPLAINING", "POINTING", "AWE_HANDS_UP", "SHRUGGING_QUESTIONING", "DEADPAN_NEUTRAL", "SHOCKED", "HIDDEN", "THINKING"]

    beats = []
    current_time = 0.0
    for i, b in enumerate(raw_beats):
        b_dur = b.get("duration", duration / len(raw_beats))
        b["start_time"] = round(current_time, 2)
        b["end_time"] = round(current_time + b_dur, 2)
        current_time += b_dur

        # Normalize pose
        p = b.get("narrator_pose", "EXPLAINING")
        if p == "SHRUGGING": p = "SHRUGGING_QUESTIONING"
        if p not in valid_poses: p = "EXPLAINING"
        b["narrator_pose"] = p

        beats.append(VisualBeat(**b))

    return beats

def direct_storyboard(script_path: Path, output_path: Path, max_scenes: int = None) -> VideoProductionPlan:
    print("=========================================")
    print(f"🎬 Visual Director Agent: Infographic Planning ({DIRECTOR_MODEL})")
    print("=========================================")

    if not script_path.exists():
        raise FileNotFoundError(f"Script file not found at {script_path}")

    with open(script_path, "r") as f:
        script_data = json.load(f)

    title = script_data.get("title", "The Cosmic Silence")
    topic = script_data.get("topic", "Fermi Paradox")
    scenes = script_data.get("scenes", [])
    if max_scenes:
        scenes = scenes[:max_scenes]

    directed_scenes = []

    for s in scenes:
        scene_num = s["scene_number"]
        duration = s.get("estimated_duration_seconds", 15.0)
        audio_file = f"scene_{scene_num}.mp3"

        beats = direct_scene(s)
        directed_scenes.append(SceneStoryboard(
            scene_number=scene_num,
            audio_file=audio_file,
            duration=duration,
            beats=beats
        ))

    plan = VideoProductionPlan(
        title=title,
        topic=topic,
        art_style_prompt="Clean 2D cel-shaded comic infographic illustration, 4px crisp black linework, vibrant flat colors, graphic novel aesthetic",
        narrator_concept="Chibi astronomer skeleton professor wearing tweed jacket and bowtie",
        typography_theme="Heavy bold comic sans-serif font, vibrant yellow fill, 4px solid black border",
        scenes=directed_scenes
    )

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w") as f:
        f.write(plan.model_dump_json(indent=4))

    print(f"\n✅ Infographic storyboard complete! Saved to {output_path}")
    print(f"Directed {len(plan.scenes)} scenes with {sum(len(s.beats) for s in plan.scenes)} total beats.")
    return plan

if __name__ == "__main__":
    script_file = Path("data/script_with_audio.json")
    out_file = Path("data/storyboard.json")
    direct_storyboard(script_file, out_file, max_scenes=3)
