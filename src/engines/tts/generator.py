import os
import sys
import json
from pathlib import Path
from openai import OpenAI
from dotenv import load_dotenv

sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))))
from src.models.content import VideoScript

load_dotenv()

client = OpenAI(
    base_url="https://api.deepinfra.com/v1/openai",
    api_key=os.getenv("LLM_API_KEY")
)

TTS_MODEL = os.getenv("TTS_MODEL", "Qwen/Qwen3-TTS")

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
    # Clean and combine smoothly
    base = base.rstrip(".")
    return f"{base}. Tone: {scene_instruct}"

def generate_scene_audio(text: str, speaker_voice: str, audio_path: Path, instruct: str) -> None:
    """
    Generates audio using DeepInfra's Qwen3-TTS with consistent voice anchor and contextual instructions.
    """
    print(f"      [Speaker: {speaker_voice} | Instruct: \"{instruct}\"]")
    
    try:
        response = client.audio.speech.create(
            model=TTS_MODEL,
            voice=speaker_voice,
            input=text,
            response_format="mp3",
            extra_body={
                "instruct": instruct
            }
        )
        response.stream_to_file(audio_path)
    except Exception as e:
        print(f"      [Error generating audio with Qwen3-TTS]: {e}")

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
    
    for scene in script.scenes:
        scene_num = scene.scene_number
        char_name = getattr(scene, "character_name", "Narrator")
        
        # Determine speaker
        if char_name == "Narrator":
            speaker = getattr(scene, "speaker", narrator_voice) or narrator_voice
        else:
            speaker = getattr(scene, "speaker", None) or VOICE_MAP.get(char_name, "Uncle_Fu")
            
        # Determine contextual instruction
        scene_mod = getattr(scene, "voice_instruct", None)
        full_instruct = build_scene_instruct(base_persona, scene_mod)
        
        print(f"\n[TTS Engine]: Processing Scene {scene_num}/{len(script.scenes)} ({char_name} speaking)...")
        audio_file = audio_dir / f"scene_{scene_num}.mp3"
        generate_scene_audio(scene.narration, speaker, audio_file, full_instruct)
        print(f"  -> Generated {audio_file.name}")
        
    updated_script_path = output_dir / "script_with_audio.json"
    with open(updated_script_path, "w") as f:
        f.write(script.model_dump_json(indent=4))
        
    print(f"\n[TTS Engine]: Complete! Audio files saved to {audio_dir}")

if __name__ == "__main__":
    script_file = Path("data/expressive_script.json")
    out_dir = Path("data")
    run_tts_pipeline(script_file, out_dir)
