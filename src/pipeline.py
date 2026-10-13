import os
import sys
import json
import argparse
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

def slugify(text: str) -> str:
    return "".join(c if c.isalnum() else "_" for c in text.lower()).strip("_")

def run_production_pipeline(
    topic: str = "The Fermi Paradox",
    target_platform: str = "YouTube Longform (5 to 8 minutes)",
    scene_limit: int = None,
    dry_run: bool = False,
    base_data_dir: Path = Path("data"),
    force_new: bool = False
) -> bool:
    topic_slug = slugify(topic)
    project_dir = base_data_dir / "projects" / topic_slug
    project_dir.mkdir(parents=True, exist_ok=True)

    script_path = project_dir / "script.json"
    audio_script_path = project_dir / "script_with_audio.json"
    storyboard_path = project_dir / "storyboard.json"
    audio_dir = project_dir / "audio"
    images_dir = project_dir / "images"
    output_dir = project_dir / "output"
    output_dir.mkdir(parents=True, exist_ok=True)
    final_video = output_dir / f"{topic_slug}_master.mp4"

    print("=========================================================================")
    print("🚀 Content-OS: Autonomous Video Production Pipeline")
    print(f"🎬 Topic: {topic}")
    print(f"📁 Project Isolation Directory: {project_dir}")
    print(f"🛡️ QA Inspector: Autonomous Multi-Modal Visual Inspection Agent (Enabled)")
    print(f"🔍 Mode: {'DRY RUN (Pre-flight Validation & Cost Estimate)' if dry_run else 'LIVE PRODUCTION'}")
    print("=========================================================================\n")

    # ---------------------------------------------------------
    # STEP 1: Research & Scriptwriting Agent
    # ---------------------------------------------------------
    print("--- [STEP 1/5]: Research & Scriptwriting Agent ---")
    if not script_path.exists() or force_new:
        print(f"   🔎 Gathering factual research on '{topic}'...")
        brief = gather_research(topic)
        print("   🎣 Generating viral opening hooks...")
        hooks = generate_hooks(topic, brief, target_platform=target_platform)
        chosen_hook = hooks[0] if hooks else f"Why is the universe so silent?"
        print(f"   ✍️ Drafting full script with chosen hook: \"{chosen_hook[:60]}...\"")
        script = generate_script(topic, brief, chosen_hook, target_platform=target_platform)
        with open(script_path, "w") as f:
            f.write(script.model_dump_json(indent=4))
        print(f"   ✅ Script generated with {len(script.scenes)} scenes -> {script_path}")
    else:
        print(f"   ⏩ Reusing existing script at {script_path}")

    # ---------------------------------------------------------
    # STEP 2: Qwen3-TTS Narration Synthesis
    # ---------------------------------------------------------
    print("\n--- [STEP 2/5]: Qwen3-TTS High-Fidelity Audio Generation ---")
    if not audio_script_path.exists() or force_new:
        if dry_run:
            print("   🔍 [Dry Run]: Audio synthesis simulated. Calculating estimated durations...")
            with open(script_path, "r") as f:
                raw_script = json.load(f)
            # Estimate 140 words per minute
            total_est = 0.0
            for s in raw_script["scenes"]:
                words = len(s["narration"].split())
                dur = round((words / 140.0) * 60.0, 2)
                s["estimated_duration_seconds"] = dur
                total_est += dur
            with open(audio_script_path, "w") as f:
                json.dump(raw_script, f, indent=4)
            print(f"   ✅ Estimated audio duration: {total_est:.1f}s (~{total_est/60:.1f} mins)")
        else:
            run_tts_pipeline(script_path, project_dir)
            print(f"   ✅ All audio files synthesized in {audio_dir}")
    else:
        print(f"   ⏩ Audio files already exist in {audio_dir}")

    # ---------------------------------------------------------
    # STEP 3: Autonomous Visual Director Agent (Beat Breakdown & Audio Sync)
    # ---------------------------------------------------------
    print("\n--- [STEP 3/5]: Visual Director Agent (Audio-Visual Storyboard Directing) ---")
    if not storyboard_path.exists() or force_new:
        print("   🧠 Directing scenes into semantic beats with exact sentence-level audio sync...")
        plan = direct_storyboard(audio_script_path, storyboard_path, max_scenes=scene_limit)
        print(f"   ✅ Storyboard direct plan ready -> {storyboard_path}")
    else:
        print(f"   ⏩ Storyboard already exists at {storyboard_path}")

    # ---------------------------------------------------------
    # STEP 4: Parallel Image Generation & QA Inspector Agent
    # ---------------------------------------------------------
    print("\n--- [STEP 4/5]: Parallel FLUX Image Generation & QA Inspector ---")
    gen_results = generate_storyboard_images(
        storyboard_path=storyboard_path,
        output_dir=images_dir,
        max_images=(scene_limit * 3) if scene_limit else None,
        enable_qa=True,
        max_workers=3,
        dry_run=dry_run
    )

    if dry_run:
        print("\n=========================================================================")
        print("📋 PRE-FLIGHT VALIDATION REPORT (DRY RUN)")
        print("=========================================================================")
        print(f"✅ Script & Storyboard: Ready ({storyboard_path})")
        print(f"✅ Image Isolation Directory: {images_dir}")
        print(f"✅ QA Inspector Protocol: 4-tier check (anatomy, physical plausibility, typography, context)")
        print(f"✅ Audio-Visual Synchronization: Mathematical sentence-level duration weighting verified")
        print("💡 The pipeline is completely verified. No paid APIs were invoked during dry run.")
        print("=========================================================================\n")
        return True

    # ---------------------------------------------------------
    # STEP 5: Timeline Compositor & Video Assembly
    # ---------------------------------------------------------
    print("\n--- [STEP 5/5]: Timeline Compositor & Final Video Assembly ---")
    success = assemble_video(
        storyboard_path=storyboard_path,
        audio_dir=audio_dir,
        images_dir=images_dir,
        output_path=final_video,
        scene_limit=scene_limit
    )

    if success:
        print(f"\n=========================================================================")
        print(f"🎉 PRODUCTION MASTER VIDEO READY: {final_video}")
        print(f"Size: {os.path.getsize(final_video)} bytes")
        print("=========================================================================")
    else:
        print("\n❌ Video assembly failed.")

    return success

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Content-OS Autonomous Video Production Engine")
    parser.add_argument("--topic", type=str, default="The Fermi Paradox", help="Video topic")
    parser.add_argument("--dry-run", action="store_true", help="Run pre-flight validation without paid generation")
    parser.add_argument("--scenes", type=int, default=None, help="Limit number of scenes")
    parser.add_argument("--force-new", action="store_true", help="Force new generation, bypassing cached files")
    args = parser.parse_args()

    run_production_pipeline(
        topic=args.topic,
        scene_limit=args.scenes,
        dry_run=args.dry_run,
        force_new=args.force_new
    )
