import os
import sys
import json
import re
import time
import subprocess
from pathlib import Path
from openai import OpenAI
from dotenv import load_dotenv

# Add project root to python path
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))))
from src.models.storyboard import VideoProductionPlan, SceneStoryboard, VisualBeat

load_dotenv()

DIRECTOR_MODEL = os.getenv("DIRECTOR_MODEL", "Qwen/Qwen2.5-72B-Instruct")

DIRECTOR_SYSTEM_PROMPT = """You are an elite YouTube Animation & Educational Visual Director (specializing in the visual style of OverSimplified and top-tier animated explainers).

Your mission is to transform a spoken script scene into 2 to 4 dynamic, highly engaging, and context-dense visual beats that strictly synchronize with the spoken audio.

CRITICAL DIRECTING PRINCIPLES:
1. STRICT SENTENCE-LEVEL AUDIO ALIGNMENT:
   - Divide the scene's narration into 2 to 4 distinct sentences or logical thoughts.
   - Every beat MUST specify the exact `narration_chunk` (the spoken words during this visual beat).
   - The visual beat MUST illustrate EXACTLY what is being spoken at that moment!

2. VISUAL STYLE STANDARD (BENCHMARK REFERENCE):
   - OverSimplified 2D cartoon explainer style: simple stick/cartoon characters with round white faces and expressive eyes, thin stick limbs, bold black outlines, cel-shaded flat colors.
   - Clear educational visual metaphors with diagram arrows, cutaways, cross-sections, and props.
   - Characters have standard anatomy: exactly two arms, two legs, clean normal hands. Zero third arms, zero extra limbs.
   - Realistic physical interactions: no limbs clipping through sealed glass canopies, solid walls, or tables.

3. TEXT HYGIENE & DISCIPLINE:
   - DEFAULT TO ZERO TEXT: Rely on visual diagrams, arrows, cutaways, and color-coded elements.
   - If a specific text word/title IS strictly required: specify the exact phrase in `expected_text` (e.g. 'HABITABLE ZONE' or 'ARE WE ALONE?').
   - Keep text to a single short, clean phrase. Absolutely NO pseudo-words, scribbles, or multiple confusing labels.

4. OUTPUT STRICT RAW JSON:
   Return ONLY a valid JSON array of beats. No markdown, no commentary.
"""

DIRECTOR_USER_PROMPT = """Direct Scene {scene_number} into 2 to 4 synchronized visual beats:
Total Spoken Duration: {duration} seconds
Full Scene Narration:
"{narration}"

Format for each beat in the JSON array:
[
  {{
    "beat_index": 1,
    "narration_chunk": "Exact segment of narration spoken during this beat...",
    "duration": 7.5,
    "concept": "1-2 sentence description of the educational infographic / comic scenario...",
    "expected_text": "EXACT TITLE (or null if no text needed)",
    "visual_prompt": "OverSimplified YouTube 2D cartoon explainer style, clean bold black outlines, cel-shaded flat colors, simple cartoon stick characters with round white faces and expressive eyes, thin stick limbs... (detailed visual description with props, diagrams, setting, clean anatomy, 16:9 widescreen)",
    "camera_motion": "SLOW_ZOOM_IN"
  }}
]

Available camera motions: "SLOW_ZOOM_IN", "SLOW_ZOOM_OUT", "PAN_LEFT_TO_RIGHT", "PAN_RIGHT_TO_LEFT", "PUNCH_IN", "STATIC".
The sum of the beat durations MUST equal exactly {duration} seconds.
"""

def call_director_llm(messages: list, temperature: float = 0.2, max_tokens: int = 1200) -> str:
    """Resilient curl execution with retry loop to bypass network latency and TLS connection resets."""
    api_key = os.getenv("LLM_API_KEY")
    base_url = os.getenv("LLM_BASE_URL", "https://api.deepinfra.com/v1/openai").rstrip("/")
    payload = {
        "model": DIRECTOR_MODEL,
        "messages": messages,
        "temperature": temperature,
        "max_tokens": max_tokens
    }
    import tempfile
    with tempfile.NamedTemporaryFile("w", suffix=".json", delete=False) as tf:
        json.dump(payload, tf)
        tmp_p = tf.name

    try:
        for attempt in range(1, 4):
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
                        return data["choices"][0]["message"]["content"].strip()
                    elif "error" in data:
                        print(f"      [Director LLM Error (Attempt {attempt})]: {data['error']}")
            except Exception as e:
                print(f"      [Director LLM Exception (Attempt {attempt})]: {e}")
                time.sleep(1.0)
    finally:
        if os.path.exists(tmp_p):
            try: os.unlink(tmp_p)
            except Exception: pass
    return ""

def direct_scene(scene_data: dict) -> list[VisualBeat]:
    scene_num = scene_data["scene_number"]
    total_duration = round(float(scene_data.get("estimated_duration_seconds", 15.0)), 2)
    narration = scene_data.get("narration", "").strip()

    prompt = DIRECTOR_USER_PROMPT.format(
        scene_number=scene_num,
        duration=total_duration,
        narration=narration
    )

    print(f"   🎬 [Visual Director]: Directing Scene {scene_num} ({total_duration}s)...")

    messages = [
        {"role": "system", "content": DIRECTOR_SYSTEM_PROMPT},
        {"role": "user", "content": prompt}
    ]

    raw_beats = None
    for parse_attempt in range(1, 3):
        raw_text = call_director_llm(messages, temperature=0.2, max_tokens=1200)
        clean_text = raw_text.strip()
        if clean_text.startswith("```json"): clean_text = clean_text[7:]
        if clean_text.startswith("```"): clean_text = clean_text[3:]
        if clean_text.endswith("```"): clean_text = clean_text[:-3]

        try:
            raw_beats = json.loads(clean_text.strip())
            if isinstance(raw_beats, list) and len(raw_beats) > 0:
                break
        except Exception as e:
            print(f"      ⚠️ Parse error on attempt {parse_attempt}: {e}. Retrying LLM...")

    if not raw_beats or not isinstance(raw_beats, list):
        # Fallback to sentence split
        sentences = [s.strip() for s in re.split(r'(?<=[.?!])\s+', narration) if s.strip()]
        if not sentences:
            sentences = [narration]
        raw_beats = []
        for s_idx, s_text in enumerate(sentences[:3]):
            raw_beats.append({
                "beat_index": s_idx + 1,
                "narration_chunk": s_text,
                "concept": f"Visual explainer depicting: {s_text[:70]}",
                "visual_prompt": f"OverSimplified YouTube 2D cartoon explainer style, clean bold black outlines, cel-shaded flat colors, chibi characters, scene depicting {s_text}, exactly two arms, two legs, normal hands, clean composition, no pseudo-text, 16:9 widescreen",
                "camera_motion": "SLOW_ZOOM_IN" if s_idx % 2 == 0 else "PAN_LEFT_TO_RIGHT"
            })

    # Normalize durations and timing to ensure 100% mathematical precision with audio
    num_beats = len(raw_beats)

    # Calculate word count weighting for exact audio pacing
    word_counts = []
    for b in raw_beats:
        chunk = b.get("narration_chunk", "")
        words = len(chunk.split()) if chunk else 1
        word_counts.append(max(1, words))

    total_words = sum(word_counts)
    proportional_durations = [round((w / total_words) * total_duration, 2) for w in word_counts]

    # Adjust rounding on the last beat so sum == total_duration
    sum_dur = sum(proportional_durations[:-1])
    proportional_durations[-1] = round(total_duration - sum_dur, 2)

    valid_cameras = ["STATIC", "SLOW_ZOOM_IN", "SLOW_ZOOM_OUT", "PAN_LEFT_TO_RIGHT", "PAN_RIGHT_TO_LEFT", "PUNCH_IN"]

    beats = []
    current_time = 0.0

    for idx, (b, beat_dur) in enumerate(zip(raw_beats, proportional_durations)):
        b_idx = idx + 1
        start_t = round(current_time, 2)
        end_t = round(current_time + beat_dur, 2)
        current_time = end_t

        cam = b.get("camera_motion", "SLOW_ZOOM_IN").upper()
        if cam not in valid_cameras:
            cam = "SLOW_ZOOM_IN"

        raw_prompt = b.get("visual_prompt", "")
        # Enforce quality anchors on prompt
        if "OverSimplified" not in raw_prompt:
            raw_prompt = f"OverSimplified YouTube 2D cartoon explainer style, clean bold black outlines, cel-shaded flat colors, {raw_prompt}"
        if "two arms" not in raw_prompt:
            raw_prompt = f"{raw_prompt}, exactly two arms, two legs, clean anatomy, physically plausible, no clipping through objects, no pseudo-words or gibberish, 16:9 widescreen"

        beat = VisualBeat(
            beat_index=b_idx,
            start_time=start_t,
            end_time=end_t,
            duration=beat_dur,
            narration_chunk=b.get("narration_chunk", ""),
            concept=b.get("concept", ""),
            visual_prompt=raw_prompt,
            image=f"shot_s{scene_num}_b{b_idx}.jpg",
            camera_motion=cam,
            expected_text=b.get("expected_text")
        )
        beats.append(beat)
        print(f"      Beat {b_idx}: [{start_t}s - {end_t}s ({beat_dur}s)] -> {beat.image} | Cam: {cam}")
        print(f"         Narration: \"{beat.narration_chunk[:60]}...\"")

    return beats

def direct_storyboard(
    script_path: Path,
    output_path: Path,
    max_scenes: int = None
) -> VideoProductionPlan:
    print("=========================================================================")
    print(f"🎬 Visual Director Agent: Intelligent Storyboard Engine ({DIRECTOR_MODEL})")
    print("=========================================================================")

    if not script_path.exists():
        raise FileNotFoundError(f"Script file not found at {script_path}")

    with open(script_path, "r") as f:
        script_data = json.load(f)

    title = script_data.get("title", "Educational Explainer")
    topic = script_data.get("topic", "General Topic")
    scenes = script_data.get("scenes", [])
    if max_scenes:
        scenes = scenes[:max_scenes]

    total_spoken_time = sum(float(s.get("estimated_duration_seconds", 0.0)) for s in scenes)
    print(f"Directing {len(scenes)} scenes | Total spoken runtime: {total_spoken_time:.2f}s (~{total_spoken_time/60:.1f} mins)\n")

    directed_scenes = []
    total_beats = 0

    for s in scenes:
        scene_num = s["scene_number"]
        duration = round(float(s.get("estimated_duration_seconds", 15.0)), 2)
        audio_file = f"scene_{scene_num}.mp3"
        narration = s.get("narration", "")

        beats = direct_scene(s)
        total_beats += len(beats)

        directed_scenes.append(SceneStoryboard(
            scene_number=scene_num,
            title=f"Scene {scene_num}",
            audio_file=audio_file,
            duration=duration,
            narration=narration,
            beats=beats
        ))

    plan = VideoProductionPlan(
        title=title,
        topic=topic,
        target_platform="YouTube Longform (5 to 8 Minutes)",
        art_style="OverSimplified YouTube 2D cartoon explainer style, clean bold black outlines, cel-shaded flat colors, chibi characters embedded in scene, exactly two arms, two legs, no pseudo-text",
        approach="Pure scene illustration and voiceover with exact sentence-level audio sync",
        scenes=directed_scenes
    )

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w") as f:
        f.write(plan.model_dump_json(indent=2))

    print(f"\n=========================================================================")
    print(f"✅ Master Storyboard Directing Complete!")
    print(f"📁 Saved to: {output_path}")
    print(f"🎬 Scenes: {len(directed_scenes)} | Total Visual Beats: {total_beats}")
    print(f"⏱️ Total Video Runtime: {total_spoken_time:.2f}s ({total_spoken_time/60:.2f} mins)")
    print("=========================================================================")
    return plan

if __name__ == "__main__":
    script_file = Path(sys.argv[1]) if len(sys.argv) > 1 else Path("data/script_with_audio.json")
    out_file = Path(sys.argv[2]) if len(sys.argv) > 2 else Path("data/storyboard_longform.json")
    direct_storyboard(script_file, out_file)
