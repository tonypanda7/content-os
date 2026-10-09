from typing import List, Optional
from pydantic import BaseModel, Field

class Scene(BaseModel):
    """
    Represents a single atomic scene in the video.
    """
    scene_number: int
    character_name: str = Field(default="Narrator", description="The name of the character speaking (e.g., 'Narrator', 'Zeus', 'Chronos').")
    narration: str = Field(..., description="The exact spoken text for this scene to be passed to the TTS engine.")
    visual_prompt: str = Field(..., description="The prompt used to generate the image/video for this scene.")
    estimated_duration_seconds: Optional[float] = Field(default=None, description="Estimated duration in seconds. Can be filled by TTS later.")

class VideoScript(BaseModel):
    """
    The final output of the Content Generation Engine.
    """
    title: str
    topic: str
    target_platform: str = Field(default="YouTube Shorts", description="e.g., YouTube Shorts, TikTok, YouTube Longform")
    scenes: List[Scene]
