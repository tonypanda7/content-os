import os
import sys
import json
import math
import subprocess
import tempfile
from pathlib import Path
from PIL import Image, ImageDraw, ImageFont

# Add project root to python path
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))))

FONT_PATH = "/usr/share/fonts/noto/NotoSans-Black.ttf"
DEFAULT_FPS = 24
WIDTH = 1280
HEIGHT = 720

def get_font(size: int):
    try:
        return ImageFont.truetype(FONT_PATH, size)
    except Exception:
        return ImageFont.load_default()

def render_text_badge(canvas_w: int, canvas_h: int, text: str, progress_sec: float, total_sec: float) -> Image.Image:
    """
    Renders an animated cel-shaded comic title card with pop-in bounce,
    golden-yellow fill, bold black stroke, drop shadow, and dark backdrop pill.
    """
    if progress_sec > 4.5:
        fade = max(0.0, 1.0 - (progress_sec - 4.5) / 0.4)
        if fade <= 0.0:
            return None
    else:
        fade = 1.0

    # Pop-in bounce curve for the first 0.35s
    if progress_sec < 0.22:
        scale = 0.65 + 0.45 * (progress_sec / 0.22)
    elif progress_sec < 0.35:
        scale = 1.10 - 0.10 * ((progress_sec - 0.22) / 0.13)
    else:
        scale = 1.0

    base_font_size = 56
    font_size = max(20, int(base_font_size * scale))
    font = get_font(font_size)

    # Measure text
    dummy_img = Image.new("RGBA", (10, 10))
    dummy_draw = ImageDraw.Draw(dummy_img)
    bbox = dummy_draw.textbbox((0, 0), text, font=font)
    t_w = bbox[2] - bbox[0]
    t_h = bbox[3] - bbox[1]

    # Shrink font if too wide for 1280 canvas
    max_text_w = canvas_w - 200
    if t_w > max_text_w:
        font_size = int(font_size * (max_text_w / t_w))
        font = get_font(font_size)
        bbox = dummy_draw.textbbox((0, 0), text, font=font)
        t_w = bbox[2] - bbox[0]
        t_h = bbox[3] - bbox[1]

    center_x = canvas_w // 2
    center_y = 105

    pad_x = int(36 * scale)
    pad_y = int(16 * scale)
    rect = [
        center_x - t_w // 2 - pad_x,
        center_y - t_h // 2 - pad_y,
        center_x + t_w // 2 + pad_x,
        center_y + t_h // 2 + pad_y
    ]

    badge = Image.new("RGBA", (canvas_w, canvas_h), (0, 0, 0, 0))
    bdraw = ImageDraw.Draw(badge)

    alpha_pill = int(220 * fade)
    alpha_border = int(255 * fade)

    # Pill background with gold border
    bdraw.rounded_rectangle(
        rect,
        radius=int(18 * scale),
        fill=(12, 16, 26, alpha_pill),
        outline=(255, 215, 0, alpha_border),
        width=max(2, int(4 * scale))
    )

    draw_x = center_x - t_w // 2 - bbox[0]
    draw_y = center_y - t_h // 2 - bbox[1]

    # Drop shadow
    bdraw.text(
        (draw_x + 3, draw_y + 3),
        text,
        font=font,
        fill=(0, 0, 0, int(240 * fade))
    )

    # Punchy yellow text with solid black outline
    stroke_w = max(2, int(4 * scale))
    bdraw.text(
        (draw_x, draw_y),
        text,
        font=font,
        fill=(255, 215, 0, int(255 * fade)),
        stroke_width=stroke_w,
        stroke_fill=(0, 0, 0, int(255 * fade))
    )

    return badge

def render_graphic_annotation(canvas_w: int, canvas_h: int, annotation_type: str, progress_sec: float) -> Image.Image:
    """
    Renders on-screen comic annotations like RED_CROSSOUT ('X'), QUESTION_MARKS ('???'), or ARROW_POINTING.
    """
    if not annotation_type:
        return None

    overlay = Image.new("RGBA", (canvas_w, canvas_h), (0, 0, 0, 0))
    draw = ImageDraw.Draw(overlay)

    # Animate entrance pop
    scale = min(1.0, progress_sec / 0.25) if progress_sec < 0.25 else 1.0

    if annotation_type == "RED_CROSSOUT":
        # Big dramatic red X over right-center area
        cx, cy, sz = 850, 340, int(150 * scale)
        if sz > 10:
            # Drop shadow
            draw.line([(cx - sz + 4, cy - sz + 4), (cx + sz + 4, cy + sz + 4)], fill=(0, 0, 0, 180), width=16)
            draw.line([(cx - sz + 4, cy + sz + 4), (cx + sz + 4, cy - sz + 4)], fill=(0, 0, 0, 180), width=16)
            # Red stroke
            draw.line([(cx - sz, cy - sz), (cx + sz, cy + sz)], fill=(225, 30, 30, 245), width=14)
            draw.line([(cx - sz, cy + sz), (cx + sz, cy - sz)], fill=(225, 30, 30, 245), width=14)

    elif annotation_type == "QUESTION_MARKS":
        font_size = max(20, int(64 * scale))
        font = get_font(font_size)
        # Floating ??? near narrator
        bob = int(math.sin(progress_sec * 4.0) * 8)
        draw.text((260, 150 + bob), "???", font=font, fill="#FFD700", stroke_width=5, stroke_fill="black")

    elif annotation_type == "ARROW_POINTING":
        # Comic yellow arrow pointing towards diagram center
        arrow_x = int(500 + math.sin(progress_sec * 5.0) * 12)
        arrow_y = 300
        font = get_font(max(20, int(54 * scale)))
        draw.text((arrow_x, arrow_y), "➔", font=font, fill="#FFD700", stroke_width=4, stroke_fill="black")

    return overlay

def apply_camera_motion(bg: Image.Image, motion: str, progress: float) -> Image.Image:
    """
    Applies Ken Burns zoom and pan effects onto a 1280x720 canvas.
    """
    p = max(0.0, min(1.0, progress))

    if motion == "SLOW_ZOOM_IN":
        z = 1.05 + 0.13 * p
        w, h = int(WIDTH * z), int(HEIGHT * z)
        scaled = bg.resize((w, h), Image.Resampling.BILINEAR)
        crop_x = (w - WIDTH) // 2
        crop_y = (h - HEIGHT) // 2
        return scaled.crop((crop_x, crop_y, crop_x + WIDTH, crop_y + HEIGHT))

    elif motion == "SLOW_ZOOM_OUT":
        z = 1.18 - 0.13 * p
        w, h = int(WIDTH * z), int(HEIGHT * z)
        scaled = bg.resize((w, h), Image.Resampling.BILINEAR)
        crop_x = (w - WIDTH) // 2
        crop_y = (h - HEIGHT) // 2
        return scaled.crop((crop_x, crop_y, crop_x + WIDTH, crop_y + HEIGHT))

    elif motion == "PAN_LEFT_TO_RIGHT":
        z = 1.16
        w, h = int(WIDTH * z), int(HEIGHT * z)
        scaled = bg.resize((w, h), Image.Resampling.BILINEAR)
        max_pan = w - WIDTH
        crop_x = int(max_pan * p)
        crop_y = (h - HEIGHT) // 2
        return scaled.crop((crop_x, crop_y, crop_x + WIDTH, crop_y + HEIGHT))

    elif motion == "PAN_RIGHT_TO_LEFT":
        z = 1.16
        w, h = int(WIDTH * z), int(HEIGHT * z)
        scaled = bg.resize((w, h), Image.Resampling.BILINEAR)
        max_pan = w - WIDTH
        crop_x = int(max_pan * (1.0 - p))
        crop_y = (h - HEIGHT) // 2
        return scaled.crop((crop_x, crop_y, crop_x + WIDTH, crop_y + HEIGHT))

    elif motion == "PUNCH_IN":
        z = 1.08 + 0.16 * (p ** 0.65)
        w, h = int(WIDTH * z), int(HEIGHT * z)
        scaled = bg.resize((w, h), Image.Resampling.BILINEAR)
        crop_x = (w - WIDTH) // 2
        crop_y = (h - HEIGHT) // 2
        return scaled.crop((crop_x, crop_y, crop_x + WIDTH, crop_y + HEIGHT))

    else: # STATIC
        z = 1.05
        w, h = int(WIDTH * z), int(HEIGHT * z)
        scaled = bg.resize((w, h), Image.Resampling.BILINEAR)
        crop_x = (w - WIDTH) // 2
        crop_y = (h - HEIGHT) // 2
        return scaled.crop((crop_x, crop_y, crop_x + WIDTH, crop_y + HEIGHT))

def load_pose_sprite(sprites_dir: Path, pose_name: str) -> Image.Image:
    """
    Loads the appropriate character sprite based on the directed pose.
    Falls back gracefully if a specific pose sprite is not yet generated.
    """
    pose_map = {
        "POINTING": "narrator_pointing.png",
        "SHRUGGING": "narrator_shrugging.png",
        "SHRUGGING_QUESTIONING": "narrator_shrugging.png",
        "THINKING": "narrator_thinking.png",
        "EXPLAINING": "narrator_explaining.png",
        "AWE_HANDS_UP": "narrator_pointing.png"
    }
    filename = pose_map.get(pose_name, "narrator_explaining.png")
    path = sprites_dir / filename
    if path.exists():
        img = Image.open(path).convert("RGBA")
        bbox = img.getbbox()
        return img.crop(bbox) if bbox else img

    # Fallback
    fallback = sprites_dir / "narrator_explaining.png"
    if fallback.exists():
        img = Image.open(fallback).convert("RGBA")
        bbox = img.getbbox()
        return img.crop(bbox) if bbox else img
    return None

def render_scene(
    scene_dict: dict,
    images_dir: Path,
    audio_path: Path,
    sprites_dir: Path,
    output_mp4: Path,
    fps: int = DEFAULT_FPS
) -> bool:
    """
    Renders a complete scene with multiple visual beats, dynamic pose swaps,
    talking body pulse, Ken Burns camera motion, infographic annotations, and pop-up typography.
    """
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
        "-crf", "19",
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

    elapsed_scene_sec = 0.0

    for b_idx, (beat, num_frames) in enumerate(zip(beats, beat_frames_list)):
        beat_num = beat.get("beat_index", b_idx + 1)
        camera_motion = beat.get("camera_motion", "STATIC")
        shot_type = beat.get("shot_type", "WIDE_NARRATOR_LEFT")
        pose_name = beat.get("narrator_pose", "EXPLAINING")
        narrator_pos = beat.get("narrator_position", "bottom-left")
        text_overlay = beat.get("text_overlay")
        graphic_ann = beat.get("graphic_annotation")

        # Load background image
        bg_path = images_dir / f"scene_{scene_num}_beat_{beat_num}.jpg"
        if not bg_path.exists():
            fallback = images_dir / f"scene_{scene_num}_beat_1.jpg"
            bg_path = fallback if fallback.exists() else None

        if bg_path and bg_path.exists():
            bg_img = Image.open(bg_path).convert("RGBA")
        else:
            bg_img = Image.new("RGBA", (WIDTH, HEIGHT), (20, 24, 35, 255))

        # Dynamic pose loading from pose library
        has_narrator = (
            shot_type != "FULL_CINEMATIC_NO_NARRATOR" and
            narrator_pos is not None and
            pose_name != "HIDDEN"
        )

        sprite = load_pose_sprite(sprites_dir, pose_name) if has_narrator else None
        scaled_sprite = None
        base_sprite_x = 0
        base_sprite_y = 0

        if sprite and has_narrator:
            # Determine size by shot type
            if "CLOSEUP" in shot_type or narrator_pos == "center":
                target_h = 510
            else:
                target_h = 390

            target_w = int(sprite.width * (target_h / sprite.height))
            scaled_sprite = sprite.resize((target_w, target_h), Image.Resampling.BILINEAR)

            if narrator_pos == "bottom-right":
                scaled_sprite = scaled_sprite.transpose(Image.Transpose.FLIP_LEFT_RIGHT)
                base_sprite_x = WIDTH - target_w - 60
                base_sprite_y = HEIGHT - target_h
            elif narrator_pos == "center":
                base_sprite_x = (WIDTH - target_w) // 2
                base_sprite_y = HEIGHT - target_h
            else: # bottom-left
                base_sprite_x = 60
                base_sprite_y = HEIGHT - target_h

        print(f"   Beat {beat_num}/{len(beats)}: {num_frames} frames | Pose: {pose_name} | Cam: {camera_motion} | Annotation: {graphic_ann or 'None'} | Overlay: {text_overlay or 'None'}")

        for f_idx in range(num_frames):
            progress_in_beat = f_idx / float(num_frames) if num_frames > 1 else 0.0
            beat_sec = f_idx / float(fps)
            scene_sec = elapsed_scene_sec + beat_sec

            # 1. Background layer with camera motion
            frame = apply_camera_motion(bg_img, camera_motion, progress_in_beat)

            # 2. Anchored Narrator sprite layer with breathing & talking pulse
            if has_narrator and scaled_sprite:
                # Talking pulse / subtle body bounce
                pulse_y = int(math.sin(scene_sec * 12.0) * 3.0)
                cur_y = base_sprite_y + pulse_y
                frame.paste(scaled_sprite, (base_sprite_x, cur_y), scaled_sprite)

            # 3. Graphic Annotation layer (Red 'X', arrows, ???)
            if graphic_ann:
                ann_badge = render_graphic_annotation(WIDTH, HEIGHT, graphic_ann, beat_sec)
                if ann_badge:
                    frame = Image.alpha_composite(frame, ann_badge)

            # 4. Typography Title Card overlay
            if text_overlay:
                badge = render_text_badge(WIDTH, HEIGHT, text_overlay, beat_sec, num_frames / float(fps))
                if badge:
                    frame = Image.alpha_composite(frame, badge)

            proc.stdin.write(frame.tobytes())

        elapsed_scene_sec += num_frames / float(fps)

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
            "-crf", "19",
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
    sprites_dir: Path,
    output_path: Path,
    scene_limit: int = 3
) -> bool:
    print("=========================================")
    print("🎞️ Timeline Compositor Engine (Interactive Multi-Pose Edition)")
    print("=========================================")

    if not storyboard_path.exists():
        raise FileNotFoundError(f"Storyboard not found at {storyboard_path}")

    with open(storyboard_path, "r") as f:
        board = json.load(f)

    scenes = board.get("scenes", [])
    if scene_limit:
        scenes = scenes[:scene_limit]

    temp_dir = Path("data/temp_scenes")
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
            sprites_dir=sprites_dir,
            output_mp4=scene_mp4,
            fps=DEFAULT_FPS
        )

        if not success:
            print(f"❌ Failed to render scene {scene_num}")
            return False

        rendered_scenes.append(scene_mp4)

    success = concatenate_scenes(rendered_scenes, output_path)
    return success

if __name__ == "__main__":
    storyboard = Path("data/storyboard.json")
    audio_dir = Path("data/audio")
    images_dir = Path("data/images")
    sprites = Path("data/sprites")
    final_output = Path("data/output/fermi_paradox_1min.mp4")

    assemble_video(storyboard, audio_dir, images_dir, sprites, final_output, scene_limit=3)
