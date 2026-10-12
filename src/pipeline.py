import os
import sys
import json
from pathlib import Path
from dotenv import load_dotenv

# Ensure project root is in path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.engines.content.researcher import gather_research
from src.engines.content.generator import generate_hooks, generate_script
from src.engines.tts.generator import run_tts_pipeline
from src.engines.video.director import direct_storyboard
from src.engines.video.image_generator import generate_storyboard_images
from src.engines.video.compositor import assemble_video

load_dotenv()

def run_pipeline(
    topic: str = "The Fermi Paradox",
    data_dir: Path = Path("data"),
    scene_limit: int = 3,
    skip_research_if_exists: bool = True
):
    print("=================================================================")
    print("🚀 Content-OS: End-to-End Autonomous Video Production Engine")
    print("=================================================================")
    print(f"Topic: {topic}")
    print(f"Target: First {scene_limit} scenes (~1 minute animatic demo)")
    print("=================================================================\n")

    data_dir.mkdir(parents=True, exist_ok=True)
    script_path = data_dir / "expressive_script.json"
    audio_script_path = data_dir / "script_with_audio.json"
    storyboard_path = data_dir / "storyboard.json"
    audio_dir = data_dir / "audio"
    images_dir = data_dir / "images"
    sprite_path = data_dir / "character_sprite.png"
    output_dir = data_dir / "output"
    output_dir.mkdir(parents=True, exist_ok=True)
    final_video = output_dir / "fermi_paradox_1min.mp4"

    # Step 1 & 2: Script Generation (if not already cached)
    if not script_path.exists() or not skip_research_if_exists:
        print("\n--- STEP 1: Research & Scriptwriting Agent ---")
        brief = gather_research(topic)
        hooks = generate_hooks(topic, brief)
        chosen_hook = hooks[0] if hooks else "Why is the universe so silent?"
        script = generate_script(topic, brief, chosen_hook)
        with open(script_path, "w") as f:
            f.write(script.model_dump_json(indent=4))
    else:
        print(f"⏩ [Script Agent]: Using existing script at {script_path}")

    # Step 3: TTS Generation
    print("\n--- STEP 2: Qwen 3 TTS Audio Generation ---")
    if not (audio_dir / "scene_1.mp3").exists():
        run_tts_pipeline(script_path, data_dir)
    else:
        print(f"⏩ [TTS Engine]: Audio files already generated in {audio_dir}")

    # Step 4: Visual Director & Storyboarding
    print("\n--- STEP 3: Visual Director Agent ---")
    if not storyboard_path.exists():
        direct_storyboard(audio_script_path, storyboard_path)
    else:
        print(f"⏩ [Director Agent]: Using existing storyboard at {storyboard_path}")

    # Step 5: Visual Image Generation
    print("\n--- STEP 4: FLUX Image Generation Engine ---")
    required_images = [images_dir / f"scene_{s}_beat_{b}.jpg" for s in range(1, scene_limit + 1) for b in range(1, 4)]
    all_exist = all(p.exists() for p in required_images)
    if not all_exist:
        generate_storyboard_images(storyboard_path, images_dir, max_images=scene_limit * 3)
    else:
        print(f"⏩ [Image Engine]: All {len(required_images)} background images already generated in {images_dir}")

    # Step 6: Timeline Compositor & Video Assembly
    print("\n--- STEP 5: Timeline Compositor & Animatic Assembler ---")
    success = assemble_video(
        storyboard_path=storyboard_path,
        audio_dir=audio_dir,
        images_dir=images_dir,
        sprite_path=sprite_path,
        output_path=final_video,
        scene_limit=scene_limit
    )

    if success:
        print(f"\n=================================================================")
        print(f"🎉 PRODUCTION COMPLETE! Final Video: {final_video}")
        print(f"=================================================================")
    else:
        print("\n❌ Pipeline failed during video composition.")

    return success

if __name__ == "__main__":
    run_pipeline()
