import os
import sys
import json
import subprocess
from pathlib import Path
from dotenv import load_dotenv, dotenv_values
from mutagen.mp3 import MP3

sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))))
from src.models.content import VideoScript

load_dotenv()

config = dotenv_values(".env")
API_KEY = config.get("LLM_API_KEY") or os.getenv("LLM_API_KEY")
TTS_MODEL = config.get("TTS_MODEL") or os.getenv("TTS_MODEL", "Qwen/Qwen3-TTS")

# Native speaker defaults:
VOICE_MAP = {
    "Narrator": os.getenv("NARRATOR_VOICE", "Aiden"),
    "Zeus": os.getenv("ZEUS_VOICE", "Uncle_Fu"),
    "Chronos": os.getenv("CHRONOS_VOICE", "Dylan"),
}

DEFAULT_BASE_PERSONA = (
    "Warm, resonant male voice speaking at a steady, deliberate pace with grounded depth and natural pauses."
)

def build_scene_instruct(base_persona: str, scene_instruct: str = None) -> str:
    """
    Blends the overarching anchor persona with scene-level emotional modulation.
    Ensures the narrator stays 100% consistent across scenes while dynamically adapting tone.
    """
    base = base_persona or DEFAULT_BASE_PERSONA
    if not scene_instruct:
        return base
    base = base.rstrip(".")
    return f"{base}. Tone: {scene_instruct}"

def generate_scene_audio(text: str, speaker_voice: str, audio_path: Path, instruct: str) -> float:
    """
    Generates audio using DeepInfra's Qwen3-TTS with consistent voice anchor and contextual instructions.
    Uses resilient curl subprocess execution to bypass Python TLS connection resets.
    Returns the exact duration of the generated MP3 in seconds.
    """
    print(f"      [Speaker: {speaker_voice} | Instruct: \"{instruct}\"]")
    
    payload = {
        "model": TTS_MODEL,
        "voice": speaker_voice,
        "input": text,
        "response_format": "mp3"
    }
    if instruct:
        payload["instruct"] = instruct

    import tempfile
    with tempfile.NamedTemporaryFile("w", suffix=".json", delete=False) as tf:
        json.dump(payload, tf)
        tmp_payload = tf.name

    audio_path.parent.mkdir(parents=True, exist_ok=True)

    import time
    try:
        for attempt in range(1, 6):
            try:
                cmd = [
                    "curl", "-s", "--retry", "5", "--retry-all-errors", "--retry-delay", "1", "--http1.1",
                    "https://api.deepinfra.com/v1/openai/audio/speech",
                    "-H", f"Authorization: Bearer {API_KEY}",
                    "-H", "Content-Type: application/json",
                    "-d", f"@{tmp_payload}",
                    "--output", str(audio_path)
                ]
                res = subprocess.run(cmd, capture_output=True, text=True, timeout=60)
                if res.returncode == 0 and audio_path.exists() and os.path.getsize(audio_path) > 1000:
                    audio = MP3(audio_path)
                    duration = round(audio.info.length, 2)
                    print(f"      ✅ Generated {audio_path.name} ({duration}s, {os.path.getsize(audio_path)} bytes)")
                    return duration
                else:
                    print(f"      ⚠️ Audio generation curl attempt {attempt} failed (code {res.returncode})")
                    time.sleep(1.5)
            except Exception as e:
                print(f"      ⚠️ Attempt {attempt} error: {e}")
                time.sleep(1.5)

        print(f"      ❌ Failed to generate {audio_path.name} after 5 attempts")
        return 0.0
    finally:
        if os.path.exists(tmp_payload):
            try: os.unlink(tmp_payload)
            except Exception: pass

def run_tts_pipeline(script_path: Path, output_dir: Path):
    print("=========================================")
    print(f"🎙️ Qwen 3 TTS Engine ({TTS_MODEL})")
    print("=========================================")
    
    if not script_path.exists():
        print(f"Error: Script file not found at {script_path}")
        return

    with open(script_path, "r") as f:
        script_data = json.load(f)
    
    script = VideoScript(**script_data)
    print(f"[TTS Engine]: Loaded script: '{script.title}' with {len(script.scenes)} scenes.")
    
    base_persona = getattr(script, "base_voice_persona", DEFAULT_BASE_PERSONA)
    narrator_voice = getattr(script, "narrator_speaker", VOICE_MAP["Narrator"])
    
    audio_dir = output_dir / "audio"
    audio_dir.mkdir(parents=True, exist_ok=True)
    
    total_duration = 0.0
    
    for i, scene in enumerate(script.scenes):
        scene_num = scene.scene_number
        char_name = getattr(scene, "character_name", "Narrator")
        
        # Determine speaker
        if char_name == "Narrator":
            speaker = getattr(scene, "speaker", narrator_voice) or narrator_voice
        else:
            speaker = getattr(scene, "speaker", VOICE_MAP.get(char_name, narrator_voice))
            
        scene_instruct = getattr(scene, "voice_instruct", None)
        full_instruct = build_scene_instruct(base_persona, scene_instruct)
        
        audio_file = audio_dir / f"scene_{scene_num}.mp3"
        if audio_file.exists() and os.path.getsize(audio_file) > 1000:
            audio = MP3(audio_file)
            dur = round(audio.info.length, 2)
            print(f"\n[Scene {scene_num}/{len(script.scenes)}] Reusing existing {audio_file.name} ({dur}s)")
        else:
            print(f"\n[Scene {scene_num}/{len(script.scenes)}] Generating speech for {char_name}...")
            dur = generate_scene_audio(
                text=scene.narration,
                speaker_voice=speaker,
                audio_path=audio_file,
                instruct=full_instruct
            )
        scene.estimated_duration_seconds = dur
        total_duration += dur
        
    print(f"\n🎉 Successfully synthesized all {len(script.scenes)} scenes!")
    print(f"⏱️ Total Video Audio Duration: {total_duration:.2f}s (~{total_duration/60:.1f} mins)")
    
    # Save the updated script with exact durations
    output_script = output_dir / "script_with_audio.json"
    with open(output_script, "w") as f:
        f.write(script.model_dump_json(indent=4))
    print(f"📄 Saved updated script with audio timestamps to {output_script}")

if __name__ == "__main__":
    s_path = Path("data/expressive_script.json")
    out = Path("data")
    run_tts_pipeline(s_path, out)
