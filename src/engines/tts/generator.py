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

# DeepInfra Qwen3-TTS model
TTS_MODEL = os.getenv("TTS_MODEL", "Qwen/Qwen3-TTS")

# Native speakers from Qwen3-TTS:
# - Ryan: Dynamic male voice with strong rhythmic drive (English)
# - Aiden: Sunny American male voice with a clear midrange (English)
# - Uncle_Fu: Seasoned male voice with low, mellow timbre (Chinese / deep characters)
VOICE_MAP = {
    "Narrator": os.getenv("NARRATOR_VOICE", "Ryan"),
    "Zeus": os.getenv("ZEUS_VOICE", "Uncle_Fu"),
    "Chronos": os.getenv("CHRONOS_VOICE", "Dylan"),
}

# Focused natural-language instructions following the official Qwen3-TTS spec:
DEFAULT_INSTRUCT = (
    "Awe-struck and dramatic storytelling tone with suspenseful pauses and intense curiosity."
)

def generate_scene_audio(text: str, character_name: str, audio_path: Path, scene_instruct: str = None) -> None:
    """
    Generates audio using DeepInfra's Qwen3-TTS with precise natural tone/emotion instructions.
    """
    voice = VOICE_MAP.get(character_name, VOICE_MAP["Narrator"])
    instruct = scene_instruct or DEFAULT_INSTRUCT
    print(f"      [Qwen3-TTS | Voice: {voice} | Instruct: \"{instruct}\"]")
    
    try:
        response = client.audio.speech.create(
            model=TTS_MODEL,
            voice=voice,
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
    
    audio_dir = output_dir / "audio"
    audio_dir.mkdir(parents=True, exist_ok=True)
    
    for scene in script.scenes:
        scene_num = scene.scene_number
        char_name = getattr(scene, "character_name", "Narrator")
        print(f"\n[TTS Engine]: Processing Scene {scene_num}/{len(script.scenes)} ({char_name} speaking)...")
        
        audio_file = audio_dir / f"scene_{scene_num}.mp3"
        generate_scene_audio(scene.narration, char_name, audio_file)
        print(f"  -> Generated {audio_file.name}")
        
    updated_script_path = output_dir / "script_with_audio.json"
    with open(updated_script_path, "w") as f:
        f.write(script.model_dump_json(indent=4))
        
    print(f"\n[TTS Engine]: Complete! Audio files saved to {audio_dir}")

if __name__ == "__main__":
    script_file = Path("data/expressive_script.json")
    out_dir = Path("data")
    run_tts_pipeline(script_file, out_dir)
