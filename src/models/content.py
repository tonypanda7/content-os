from typing import List, Optional
from pydantic import BaseModel, Field

class Scene(BaseModel):
    """
    Represents a single atomic scene in the video.
    """
    scene_number: int
    character_name: str = Field(default="Narrator", description="The name of the character speaking (e.g., 'Narrator', 'Zeus', 'Chronos').")
    speaker: str = Field(default="Aiden", description="The Qwen3-TTS speaker anchor (e.g., 'Aiden', 'Ryan', 'Uncle_Fu').")
    narration: str = Field(..., description="The exact spoken text for this scene to be passed to the TTS engine.")
    voice_instruct: Optional[str] = Field(
        default=None,
        description="Scene-specific emotional/tonal modulation that transitions smoothly from past scenes to upcoming ones."
    )
    visual_prompt: str = Field(..., description="The prompt used to generate the image/video for this scene.")
    estimated_duration_seconds: Optional[float] = Field(default=None, description="Estimated duration in seconds. Filled by TTS.")

class VideoScript(BaseModel):
    """
    The final output of the Content Generation Engine.
    """
    title: str
    topic: str
    target_platform: str = Field(default="YouTube Longform (5 minutes)", description="Target platform/format.")
    narrator_speaker: str = Field(default="Aiden", description="The consistent anchor speaker voice across the entire video.")
    base_voice_persona: str = Field(
        default="Warm, resonant male voice speaking at a steady, deliberate pace with grounded depth and natural pauses.",
        description="The overarching narrator persona that ensures uniform vocal identity across all scenes."
    )
    scenes: List[Scene]
