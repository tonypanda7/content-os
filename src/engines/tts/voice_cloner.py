import os
import requests
from pathlib import Path
from dotenv import load_dotenv, set_key

def clone_voice(audio_file_path: str, voice_name: str, description: str = "") -> str:
    """
    Programmatically creates a cloned voice on DeepInfra using an audio sample.
    Saves the returned voice_id to the .env file.
    """
    load_dotenv()
    api_key = os.getenv("LLM_API_KEY")
    
    if not api_key:
        raise ValueError("LLM_API_KEY is not set in the .env file.")

    url = "https://api.deepinfra.com/v1/voices/add"
    headers = {
        "Authorization": f"Bearer {api_key}"
    }

    if not os.path.exists(audio_file_path):
        raise FileNotFoundError(f"Audio file not found at {audio_file_path}")

    print(f"Uploading '{audio_file_path}' to DeepInfra to create voice '{voice_name}'...")

    with open(audio_file_path, "rb") as audio_file:
        files = {
            "audio": (os.path.basename(audio_file_path), audio_file, "audio/wav")
        }
        data = {
            "name": voice_name,
            "description": description
        }
        
        response = requests.post(url, headers=headers, files=files, data=data)

    if response.status_code == 200:
        response_data = response.json()
        voice_id = response_data.get("voice_id")
        print(f"✅ Voice successfully created! Voice ID: {voice_id}")
        
        # Save the Voice ID to the .env file so the TTS generator automatically uses it
        env_path = Path(".env")
        env_key = f"{voice_name.upper().replace(' ', '_')}_VOICE_ID"
        set_key(env_path, env_key, voice_id)
        print(f"Saved {env_key}={voice_id} to .env")
        
        return voice_id
    else:
        print(f"❌ Failed to create voice. Status Code: {response.status_code}")
        print(response.text)
        raise Exception("Voice creation failed.")

if __name__ == "__main__":
    # Example usage: Automatically cloning the Greek Narrator voice
    sample_path = "data/narrator_sample.wav"
    try:
        clone_voice(sample_path, "Narrator", "Animated stickman narrator voice")
    except Exception as e:
        print(e)
