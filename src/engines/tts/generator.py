import os
import sys
import json
import wave
from pathlib import Path
from openai import OpenAI
from dotenv import load_dotenv

# Add project root to python path
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))))
from src.models.content import VideoScript

load_dotenv()

# Initialize OpenAI client pointing to DeepInfra
client = OpenAI(
    base_url="https://api.deepinfra.com/v1/openai",
    api_key=os.getenv("LLM_API_KEY")
)

# DeepInfra Qwen3-TTS models:
# - "Qwen/Qwen3-TTS" (Preset voices + instructions + cloning)
# - "Qwen/Qwen3-TTS-VoiceDesign" (Natural language voice description)
TTS_MODEL = os.getenv("TTS_MODEL", "Qwen/Qwen3-TTS")

# Voices available: Ryan, Eric, Aiden, Dylan, Uncle_Fu, Vivian, Serena, Ono_Anna, Sohee
VOICE_MAP = {
    "Narrator": os.getenv("NARRATOR_VOICE", "Ryan"),
    "Zeus": os.getenv("ZEUS_VOICE", "Uncle_Fu"),
    "Chronos": os.getenv("CHRONOS_VOICE", "Dylan"),
}

STYLE_INSTRUCTIONS = (
    "Speak with dramatic storytelling energy, lively inflections, dynamic pauses, "
    "and expressive animated cartoon narration cadence."
)

def get_audio_duration_seconds(file_path: Path) -> float:
    """Calculates exact duration of WAV file using the wave module."""
    try:
        with wave.open(str(file_path), "rb") as wf:
            frames = wf.getnframes()
            rate = wf.getframerate()
            return round(frames / float(rate), 2)
    except Exception as e:
        print(f"Warning: Could not compute audio duration: {e}")
        return 0.0

def generate_scene_audio(text: str, character_name: str, audio_path: Path) -> float:
    """
    Generates audio using DeepInfra's Qwen3-TTS model with natural style instructions.
    """
    voice = VOICE_MAP.get(character_name, VOICE_MAP["Narrator"])
    print(f"      [Qwen3-TTS ({TTS_MODEL}) for {character_name} | Voice: {voice}]")
    
    try:
        response = client.audio.speech.create(
            model=TTS_MODEL,
            voice=voice,
            input=text,
            extra_body={
                "instructions": STYLE_INSTRUCTIONS
            }
        )
        
        response.stream_to_file(audio_path)
        duration = get_audio_duration_seconds(audio_path)
        return duration
        
    except Exception as e:
        print(f"      [Error generating audio with Qwen3-TTS]: {e}")
        return 0.0

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
    
    audio_dir = output_dir / "audio"
    audio_dir.mkdir(parents=True, exist_ok=True)
    
    total_duration = 0.0
    
    for i, scene in enumerate(script.scenes):
        scene_num = scene.scene_number
        char_name = getattr(scene, "character_name", "Narrator")
        print(f"\n[TTS Engine]: Processing Scene {scene_num}/{len(script.scenes)} ({char_name} speaking)...")
        
        audio_file = audio_dir / f"scene_{scene_num}.wav"
        duration = generate_scene_audio(scene.narration, char_name, audio_file)
        scene.estimated_duration_seconds = duration
        total_duration += duration
        
        print(f"  -> Generated {audio_file.name} (Duration: {duration}s)")
        
    print(f"\n[TTS Engine]: All audio generated! Total video duration: {round(total_duration, 2)} seconds.")
    
    updated_script_path = output_dir / "script_with_audio.json"
    with open(updated_script_path, "w") as f:
        f.write(script.model_dump_json(indent=4))
        
    print(f"[TTS Engine]: Updated JSON with exact durations saved to {updated_script_path}")

if __name__ == "__main__":
    script_file = Path("data/expressive_script.json")
    out_dir = Path("data")
    run_tts_pipeline(script_file, out_dir)
