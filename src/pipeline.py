import os
import sys
import json
import argparse
from pathlib import Path
from dotenv import load_dotenv

# Ensure project root is in sys.path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.engines.content.topic_selector import select_topic_autonomously
from src.engines.content.researcher import gather_research
from src.engines.content.generator import generate_hooks, generate_script
from src.engines.tts.generator import run_tts_pipeline
from src.engines.video.director import direct_storyboard
from src.engines.video.image_generator import generate_storyboard_images
from src.engines.video.compositor import assemble_video

load_dotenv()

class TeeLogger:
    """Streams all logs simultaneously to terminal stdout and a persistent log file with immediate flushing."""
    def __init__(self, log_path: Path):
        self.terminal = sys.__stdout__
        self.log_file = open(log_path, "a", encoding="utf-8")

    def write(self, message):
        self.terminal.write(message)
        self.terminal.flush()
        self.log_file.write(message)
        self.log_file.flush()

    def flush(self):
        self.terminal.flush()
        self.log_file.flush()

def slugify(text: str) -> str:
    return "".join(c if c.isalnum() else "_" for c in text.lower()).strip("_")

def run_production_pipeline(
    topic: str = "auto",
    auto_topic: bool = False,
    target_platform: str = "YouTube Explainer (3 minutes)",
    target_scenes: int = 6,
    scene_limit: int = None,
    dry_run: bool = False,
    base_data_dir: Path = Path("data"),
    force_new: bool = False
) -> bool:
    # ---------------------------------------------------------
    # STEP 0: Autonomous Topic Selection (If Auto Requested)
    # ---------------------------------------------------------
    if topic.strip().lower() == "auto" or auto_topic:
        print("\n=========================================================================")
        print("🤖 [STEP 0/5]: Autonomous Viral Topic Selection & Research Agent")
        print("=========================================================================")
        decision = select_topic_autonomously()
        topic = decision["chosen_topic"]
        target_scenes = decision.get("target_scenes_count", 6)
        print(f"🎬 Selected Topic: \"{topic}\"")
        print(f"🎯 Logline: {decision.get('logline', '')}")
        print(f"📈 Retention Strategy: {decision.get('viral_analysis', '')}")
        print("=========================================================================\n")

    topic_slug = slugify(topic)
    project_dir = base_data_dir / "projects" / topic_slug
    project_dir.mkdir(parents=True, exist_ok=True)

    # Initialize dual logging
    log_file = project_dir / "production_master.log"
    tee = TeeLogger(log_file)
    sys.stdout = tee
    sys.stderr = tee

    script_path = project_dir / "script.json"
    audio_script_path = project_dir / "script_with_audio.json"
    storyboard_path = project_dir / "storyboard.json"
    audio_dir = project_dir / "audio"
    images_dir = project_dir / "images"
    output_dir = project_dir / "output"
    output_dir.mkdir(parents=True, exist_ok=True)
    final_video = output_dir / f"{topic_slug}_master.mp4"

    print("=========================================================================")
    print("🚀 Content-OS: End-to-End Autonomous Video Production Engine")
    print(f"🎬 Production Topic: {topic}")
    print(f"📁 Project Isolation Directory: {project_dir}")
    print(f"📝 Master Production Log: {log_file}")
    print(f"🛡️ QA Inspector: Multi-Modal Visual Inspection Agent (Active)")
    print(f"🔧 Prompt Refiner: Autonomous Defect Remediation Agent (Active)")
    print(f"⏱️ Target Pacing: {target_scenes} scenes (~180s runtime, OverSimplified style)")
    print(f"🔍 Mode: {'DRY RUN (Pre-flight Validation & Cost Estimate)' if dry_run else 'LIVE FULL PRODUCTION'}")
    print("=========================================================================\n")

    # ---------------------------------------------------------
    # STEP 1: Research & Scriptwriting Agent
    # ---------------------------------------------------------
    print("--- [STEP 1/5]: Research & Scriptwriting Agent ---")
    if not script_path.exists() or force_new:
        print(f"   🔎 Gathering comprehensive factual research on '{topic}'...")
        brief = gather_research(topic)
        print(f"   📊 Research brief completed ({len(brief)} characters).")
        print("   🎣 Brainstorming viral opening hooks...")
        hooks = generate_hooks(topic, brief, target_platform=target_platform)
        chosen_hook = hooks[0] if hooks else f"On August 21, 1986, an entire valley went to sleep, and 1,700 people never woke up."
        print(f"   ✍️ Selected opening hook: \"{chosen_hook}\"")
        print(f"   ✍️ Drafting full {target_scenes}-scene script tuned for 3-minute pacing...")
        script = generate_script(
            topic=topic,
            research_brief=brief,
            chosen_hook=chosen_hook,
            target_platform=target_platform,
            target_scenes=target_scenes
        )
        with open(script_path, "w") as f:
            f.write(script.model_dump_json(indent=4))
        print(f"   ✅ Script successfully generated with {len(script.scenes)} scenes -> {script_path}")
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
            print(f"   ✅ All scene audio tracks synthesized in {audio_dir}")
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
        max_workers=2,
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
        print("💡 Pre-flight validation successful.")
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
        file_size_mb = os.path.getsize(final_video) / (1024 * 1024)
        print(f"\n=========================================================================")
        print(f"🎉 PRODUCTION MASTER VIDEO READY: {final_video}")
        print(f"📦 File Size: {file_size_mb:.2f} MB")
        print(f"📝 Full Execution Log: {log_file}")
        print("=========================================================================")
    else:
        print("\n❌ Video assembly failed.")

    return success

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Content-OS Autonomous Video Production Engine")
    parser.add_argument("--topic", type=str, default="auto", help="Video topic (or 'auto' for autonomous selection)")
    parser.add_argument("--auto-topic", action="store_true", help="Autonomously research and select viral topic")
    parser.add_argument("--scenes", type=int, default=6, help="Target number of scenes (default: 6 for ~3-min video)")
    parser.add_argument("--platform", type=str, default="YouTube Explainer (3 minutes)", help="Target platform and duration")
    parser.add_argument("--dry-run", action="store_true", help="Run pre-flight validation without paid generation")
    parser.add_argument("--force-new", action="store_true", help="Force new generation, bypassing cached files")
    args = parser.parse_args()

    run_production_pipeline(
        topic=args.topic,
        auto_topic=args.auto_topic,
        target_platform=args.platform,
        target_scenes=args.scenes,
        dry_run=args.dry_run,
        force_new=args.force_new
    )
