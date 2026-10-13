import os
import sys
import json
import math
import subprocess
import tempfile
from pathlib import Path
from PIL import Image

# Add project root to python path
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))))

WIDTH = 1280
HEIGHT = 720
DEFAULT_FPS = 30

def smoothstep(p: float) -> float:
    """Smooth cubic ease-in-out curve for cinematic camera motion."""
    p = max(0.0, min(1.0, p))
    return p * p * (3.0 - 2.0 * p)

def apply_camera_motion(img: Image.Image, motion: str, progress: float) -> Image.Image:
    """
    Applies cinematic Ken Burns camera pans and zooms to the illustration.
    """
    p = max(0.0, min(1.0, progress))
    sp = smoothstep(p)

    motion_upper = (motion or "STATIC").upper()

    if motion_upper == "SLOW_ZOOM_IN":
        # Smooth zoom from 1.00 to 1.12
        z = 1.00 + 0.12 * sp
        w, h = int(WIDTH * z), int(HEIGHT * z)
        scaled = img.resize((w, h), Image.Resampling.BILINEAR)
        crop_x = (w - WIDTH) // 2
        crop_y = (h - HEIGHT) // 2
        return scaled.crop((crop_x, crop_y, crop_x + WIDTH, crop_y + HEIGHT))

    elif motion_upper == "SLOW_ZOOM_OUT":
        # Smooth zoom from 1.12 to 1.00
        z = 1.12 - 0.12 * sp
        w, h = int(WIDTH * z), int(HEIGHT * z)
        scaled = img.resize((w, h), Image.Resampling.BILINEAR)
        crop_x = (w - WIDTH) // 2
        crop_y = (h - HEIGHT) // 2
        return scaled.crop((crop_x, crop_y, crop_x + WIDTH, crop_y + HEIGHT))

    elif motion_upper == "PAN_LEFT_TO_RIGHT":
        # Slight zoom with horizontal pan across the wide scene
        z = 1.14
        w, h = int(WIDTH * z), int(HEIGHT * z)
        scaled = img.resize((w, h), Image.Resampling.BILINEAR)
        max_pan = w - WIDTH
        crop_x = int(max_pan * sp)
        crop_y = (h - HEIGHT) // 2
        return scaled.crop((crop_x, crop_y, crop_x + WIDTH, crop_y + HEIGHT))

    elif motion_upper == "PAN_RIGHT_TO_LEFT":
        z = 1.14
        w, h = int(WIDTH * z), int(HEIGHT * z)
        scaled = img.resize((w, h), Image.Resampling.BILINEAR)
        max_pan = w - WIDTH
        crop_x = int(max_pan * (1.0 - sp))
        crop_y = (h - HEIGHT) // 2
        return scaled.crop((crop_x, crop_y, crop_x + WIDTH, crop_y + HEIGHT))

    elif motion_upper == "PUNCH_IN":
        # Faster punch-in zoom for dramatic impact
        z = 1.02 + 0.14 * (p ** 0.7)
        w, h = int(WIDTH * z), int(HEIGHT * z)
        scaled = img.resize((w, h), Image.Resampling.BILINEAR)
        crop_x = (w - WIDTH) // 2
        crop_y = (h - HEIGHT) // 2
        return scaled.crop((crop_x, crop_y, crop_x + WIDTH, crop_y + HEIGHT))

    else: # STATIC with subtle high-quality framing
        z = 1.02
        w, h = int(WIDTH * z), int(HEIGHT * z)
        scaled = img.resize((w, h), Image.Resampling.BILINEAR)
        crop_x = (w - WIDTH) // 2
        crop_y = (h - HEIGHT) // 2
        return scaled.crop((crop_x, crop_y, crop_x + WIDTH, crop_y + HEIGHT))

def resolve_beat_image(beat: dict, scene_num: int, beat_idx: int, images_dir: Path) -> Path:
    """
    Finds the strictly matching illustration file for a given beat.
    Never falls back to random images from other scenes.
    """
    candidates = []

    # 1. Direct explicit image field
    if beat.get("image"):
        candidates.append(images_dir / beat["image"])
    if beat.get("background_image"):
        candidates.append(images_dir / beat["background_image"])

    # 2. Convention naming
    candidates.append(images_dir / f"shot_s{scene_num}_b{beat_idx}.jpg")
    candidates.append(images_dir / f"shot_{scene_num}_{beat_idx}.jpg")
    candidates.append(images_dir / f"scene_{scene_num}_beat_{beat_idx}.jpg")

    for c in candidates:
        if c.exists() and c.stat().st_size > 1000:
            return c

    raise FileNotFoundError(
        f"❌ Strict Alignment Error: No valid image found for Scene {scene_num} Beat {beat_idx} in {images_dir}.\n"
        f"Checked candidates: {[c.name for c in candidates]}. Refusing to substitute random images."
    )

def render_scene(
    scene_dict: dict,
    images_dir: Path,
    audio_path: Path,
    output_mp4: Path,
    fps: int = DEFAULT_FPS
) -> bool:
    scene_num = scene_dict.get("scene_number", 1)
    duration = scene_dict.get("duration", 15.0)
    beats = scene_dict.get("beats", [])

    if not beats:
        print(f"❌ Scene {scene_num} has no visual beats!")
        return False

    total_frames = int(round(duration * fps))

    # Calculate frame counts per beat
    beat_frames_list = []
    accumulated = 0
    for i, beat in enumerate(beats):
        if i == len(beats) - 1:
            frames = max(1, total_frames - accumulated)
        else:
            frames = int(round(beat.get("duration", duration / len(beats)) * fps))
            accumulated += frames
        beat_frames_list.append(frames)

    print(f"\n🎬 Rendering Scene {scene_num} ({duration}s, {total_frames} frames @ {fps}fps across {len(beats)} beats)...")

    ffmpeg_cmd = [
        "ffmpeg", "-y",
        "-f", "rawvideo",
        "-vcodec", "rawvideo",
        "-s", f"{WIDTH}x{HEIGHT}",
        "-pix_fmt", "rgba",
        "-r", str(fps),
        "-i", "-",
        "-i", str(audio_path),
        "-c:v", "libx264",
        "-preset", "fast",
        "-crf", "18",
        "-pix_fmt", "yuv420p",
        "-c:a", "aac",
        "-b:a", "192k",
        "-shortest",
        str(output_mp4)
    ]

    proc = subprocess.Popen(
        ffmpeg_cmd,
        stdin=subprocess.PIPE,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.PIPE
    )

    for b_idx, (beat, num_frames) in enumerate(zip(beats, beat_frames_list)):
        beat_num = beat.get("beat_index", b_idx + 1)
        beat_duration = beat.get("duration", num_frames / float(fps))
        camera_motion = beat.get("camera_motion", "SLOW_ZOOM_IN")
        narration = beat.get("narration", "")

        img_path = resolve_beat_image(beat, scene_num, beat_num, images_dir)
        img = Image.open(img_path).convert("RGBA").resize((WIDTH, HEIGHT))

        print(f"   Beat {beat_num}/{len(beats)}: {num_frames} frames ({beat_duration:.1f}s) | Cam: {camera_motion} | Img: {img_path.name}")
        if narration:
            print(f"     Narration: \"{narration[:70]}...\"")

        for f_idx in range(num_frames):
            progress_in_beat = f_idx / float(num_frames) if num_frames > 1 else 0.0
            frame = apply_camera_motion(img, camera_motion, progress_in_beat)
            proc.stdin.write(frame.tobytes())

    proc.stdin.close()
    stderr_output = proc.stderr.read()
    proc.wait()

    if proc.returncode != 0:
        print(f"❌ FFmpeg error in Scene {scene_num}: {stderr_output.decode('utf-8', errors='ignore')[:300]}")
        return False

    print(f"   ✅ Scene {scene_num} successfully encoded -> {output_mp4} ({os.path.getsize(output_mp4)} bytes)")
    return True

def concatenate_scenes(scene_mp4s: list[Path], output_video: Path) -> bool:
    print(f"\n📦 Merging {len(scene_mp4s)} scene segments into final video...")
    output_video.parent.mkdir(parents=True, exist_ok=True)

    with tempfile.NamedTemporaryFile("w", suffix=".txt", delete=False) as f:
        list_file = Path(f.name)
        for mp4 in scene_mp4s:
            f.write(f"file '{mp4.resolve()}'\n")

    cmd = [
        "ffmpeg", "-y",
        "-f", "concat",
        "-safe", "0",
        "-i", str(list_file),
        "-c", "copy",
        str(output_video)
    ]

    result = subprocess.run(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.PIPE)
    list_file.unlink(missing_ok=True)

    if result.returncode != 0:
        # Fallback to filter_complex re-encoding concat
        inputs = []
        for mp4 in scene_mp4s:
            inputs.extend(["-i", str(mp4)])
        filter_str = "".join(f"[{i}:v][{i}:a]" for i in range(len(scene_mp4s))) + f"concat=n={len(scene_mp4s)}:v=1:a=1[v][a]"
        fallback_cmd = [
            "ffmpeg", "-y",
            *inputs,
            "-filter_complex", filter_str,
            "-map", "[v]",
            "-map", "[a]",
            "-c:v", "libx264",
            "-crf", "18",
            "-c:a", "aac",
            str(output_video)
        ]
        fb_res = subprocess.run(fallback_cmd, stdout=subprocess.DEVNULL, stderr=subprocess.PIPE)
        if fb_res.returncode != 0:
            print(f"❌ Concat failed: {fb_res.stderr.decode()[:300]}")
            return False

    print(f"🎉 FINAL VIDEO READY: {output_video} ({os.path.getsize(output_video)} bytes)")
    return True

def assemble_video(
    storyboard_path: Path,
    audio_dir: Path,
    images_dir: Path,
    sprites_dir: Path = None,
    output_path: Path = None,
    scene_limit: int = None
) -> bool:
    print("=========================================================================")
    print("🎞️ Timeline Compositor Engine (OverSimplified Pure Storyboard Edition)")
    print("=========================================================================")

    if not storyboard_path.exists():
        raise FileNotFoundError(f"Storyboard not found at {storyboard_path}")

    with open(storyboard_path, "r") as f:
        board = json.load(f)

    scenes = board.get("scenes", [])
    if scene_limit:
        scenes = scenes[:scene_limit]

    temp_dir = (output_path.parent / "temp_scenes") if output_path else Path("data/temp_scenes")
    temp_dir.mkdir(parents=True, exist_ok=True)

    rendered_scenes = []

    for s in scenes:
        scene_num = s.get("scene_number", 1)
        audio_file = audio_dir / s.get("audio_file", f"scene_{scene_num}.mp3")
        scene_mp4 = temp_dir / f"scene_{scene_num}.mp4"

        success = render_scene(
            scene_dict=s,
            images_dir=images_dir,
            audio_path=audio_file,
            output_mp4=scene_mp4,
            fps=DEFAULT_FPS
        )

        if not success:
            print(f"❌ Failed to render scene {scene_num}")
            return False

        rendered_scenes.append(scene_mp4)

    output_path = output_path or Path("data/output/fermi_paradox_1min.mp4")
    success = concatenate_scenes(rendered_scenes, output_path)
    return success

if __name__ == "__main__":
    storyboard = Path(sys.argv[1]) if len(sys.argv) > 1 else Path("data/storyboard_longform.json")
    if not storyboard.exists():
        storyboard = Path("data/storyboard.json")

    audio_dir = Path("data/audio")
    images_dir = Path("data/images")
    final_output = Path(sys.argv[2]) if len(sys.argv) > 2 else Path("data/output/fermi_paradox_longform.mp4")

    assemble_video(storyboard, audio_dir, images_dir, output_path=final_output)
