import os
import sys
import json
import asyncio
from pathlib import Path
from openai import OpenAI
from dotenv import load_dotenv

# Add project root to python path
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))))
from src.models.content import VideoScript

load_dotenv()

# Initialize OpenAI client with DeepInfra endpoint
client = OpenAI(
    base_url="https://api.deepinfra.com/v1/openai",
    api_key=os.getenv("LLM_API_KEY")
)

# You will map character names from the script to their respective DeepInfra voice IDs here.
# To get these IDs, use the curl command: curl -X POST "https://api.deepinfra.com/v1/voices/add" ...
VOICE_MAP = {
    "Narrator": os.getenv("NARRATOR_VOICE_ID", "default_narrator_voice_id_here"),
    "Zeus": os.getenv("ZEUS_VOICE_ID", "default_zeus_voice_id_here"),
    # Add more characters as needed
}

def generate_scene_audio(text: str, character_name: str, audio_path: Path) -> float:
    """
    Generates audio using DeepInfra's tts-1 model.
    """
    voice_id = VOICE_MAP.get(character_name, VOICE_MAP["Narrator"])
    
    print(f"      [Requesting TTS for {character_name} using voice: {voice_id}]")
    
    try:
        response = client.audio.speech.create(
            model="tts-1",
            voice=voice_id,
            input=text
        )
        
        # Save the audio file
        response.stream_to_file(audio_path)
        
        # Note: Standard OpenAI TTS endpoints do not return word-level timestamps natively in the same payload.
        # DeepInfra XTTS may require a different endpoint for word boundaries if needed.
        # For now, we estimate duration based on file size or fallback to length.
        # An exact duration calculation will require loading the mp3 (e.g. using mutagen).
        return len(text) / 15.0 # Rough estimate: 15 characters per second
        
    except Exception as e:
        print(f"      [Error generating audio]: {e}")
        return 0.0

def run_tts_pipeline(script_path: Path, output_dir: Path):
    print("=========================================")
    print("🎙️ Welcome to the DeepInfra TTS Engine")
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
        
        audio_file = audio_dir / f"scene_{scene_num}.mp3"
        
        # Generate audio using DeepInfra
        duration = generate_scene_audio(scene.narration, char_name, audio_file)
        scene.estimated_duration_seconds = round(duration, 2)
        total_duration += duration
        
        print(f"  -> Generated {audio_file.name}")
        
    print(f"\n[TTS Engine]: All audio generated! Total video duration estimate: {round(total_duration, 2)} seconds.")
    
    updated_script_path = output_dir / "script_with_audio.json"
    with open(updated_script_path, "w") as f:
        f.write(script.model_dump_json(indent=4))
        
    print(f"[TTS Engine]: Updated JSON with durations saved to {updated_script_path}")

if __name__ == "__main__":
    script_file = Path("data/latest_script.json")
    out_dir = Path("data")
    run_tts_pipeline(script_file, out_dir)
