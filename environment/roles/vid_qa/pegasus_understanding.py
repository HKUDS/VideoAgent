import logging
import os
import time
from typing import Dict, List

from pydantic import BaseModel, Field
from tenacity import (
    retry,
    retry_if_exception_type,
    stop_after_attempt,
    wait_exponential,
)

from environment.agents.base import BaseTool
from environment.config.llm import pegasus, twelvelabs_client

# Configure logging
logging.basicConfig(level=logging.INFO,
                    format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


class PegasusVideoUnderstanding(BaseTool):
    """
    Agent that answers questions about videos using TwelveLabs Pegasus.
    Unlike the transcript-based Q&A agent, Pegasus understands the video
    natively (visuals + audio), so it works without local transcription and
    can reason about on-screen actions, scenes, and objects, not just speech.
    This is an opt-in backend: it only runs when a TwelveLabs API key is set
    in environment/config/config.yml (free key at https://twelvelabs.io).
    """

    def __init__(self, max_tokens: int = 2048):
        super().__init__()
        self.max_tokens = max_tokens
        # Video extensions Pegasus can ingest directly.
        self.video_extensions = {'.mp4', '.avi', '.mov', '.mkv', '.flv', '.wmv', '.webm', '.m4v', '.3gp'}

    class InputSchema(BaseTool.BaseInputSchema):
        video_path: str = Field(
            ...,
            description="Path to a video file, a directory of videos, or a public video URL to analyze"
        )
        prompt: str = Field(
            ...,
            description="The question or instruction to ask Pegasus about the video(s)"
        )
        model: str = Field(
            "pegasus1.5",
            description="TwelveLabs Pegasus model name (e.g. pegasus1.5)"
        )

    class OutputSchema(BaseModel):
        answers: Dict[str, str] = Field(
            ...,
            description="Mapping of each processed video (filename or URL) to Pegasus's answer"
        )
        processed_videos: List[str] = Field(
            ...,
            description="List of videos that were analyzed"
        )
        status: str = Field(
            ...,
            description="Overall status of the analysis ('success' or 'error')"
        )

    def _get_video_files(self, video_path: str) -> List[str]:
        """Resolve the input into a list of analyzable video sources."""
        # A URL is passed straight through to Pegasus (analyzed server-side).
        if video_path.startswith(("http://", "https://")):
            return [video_path]

        if not os.path.exists(video_path):
            raise FileNotFoundError(f"Video path not found: {video_path}")

        if os.path.isfile(video_path):
            return [video_path]

        video_files = []
        for filename in os.listdir(video_path):
            file_path = os.path.join(video_path, filename)
            if os.path.isfile(file_path):
                _, ext = os.path.splitext(filename.lower())
                if ext in self.video_extensions:
                    video_files.append(file_path)
        video_files.sort()
        logger.info(f"Found {len(video_files)} video files in directory: {video_path}")
        return video_files

    def _upload_asset(self, file_path: str):
        """Upload a local video file as a TwelveLabs asset for analysis."""
        logger.info(f"Uploading local video as TwelveLabs asset: {file_path}")
        client = twelvelabs_client()
        with open(file_path, "rb") as f:
            return client.assets.create(method="direct", file=f, filename=os.path.basename(file_path))

    @retry(
        retry=retry_if_exception_type((Exception)),
        wait=wait_exponential(multiplier=1, min=4, max=60),
        stop=stop_after_attempt(3),
        reraise=True
    )
    def _analyze_one(self, source: str, prompt: str, model: str) -> str:
        """Send a single video to Pegasus and return the generated text."""
        # URLs are passed directly; local files are uploaded as a TwelveLabs
        # asset first and referenced by id.
        if source.startswith(("http://", "https://")):
            video = {"type": "url", "url": source}
        else:
            asset = self._upload_asset(source)
            video = {"type": "asset_id", "asset_id": asset.id}

        logger.info(f"Analyzing with Pegasus ({model}): {source}")
        start_time = time.time()
        response = pegasus(video=video, prompt=prompt, model=model, max_tokens=self.max_tokens)
        logger.info(f"Pegasus analysis completed in {time.time() - start_time:.2f}s")
        return response.data or ""

    def execute(self, **kwargs):
        """Analyze the given video(s) with Pegasus and return per-video answers."""
        params = self.InputSchema(**kwargs)

        print("\n=== PEGASUS VIDEO UNDERSTANDING ===")
        print(f"Source: {params.video_path}")
        print(f"Prompt: {params.prompt}")

        try:
            sources = self._get_video_files(params.video_path)
            if not sources:
                raise ValueError(f"No video files found at: {params.video_path}")

            answers: Dict[str, str] = {}
            processed: List[str] = []
            for i, source in enumerate(sources, 1):
                key = source if source.startswith(("http://", "https://")) else os.path.basename(source)
                print(f"\nAnalyzing {i}/{len(sources)}: {key}")
                try:
                    answers[key] = self._analyze_one(source, params.prompt, params.model)
                    processed.append(source)
                except Exception as e:
                    logger.error(f"Failed to analyze {source}: {e}")
                    answers[key] = f"[Error analyzing {key}: {e}]"

            print("\n=== ANALYSIS COMPLETED ===")
            return {
                "answers": answers,
                "processed_videos": [
                    s if s.startswith(("http://", "https://")) else os.path.basename(s)
                    for s in processed
                ],
                "status": "success",
            }

        except Exception as e:
            error_msg = f"Error in Pegasus video understanding: {e}"
            logger.error(error_msg)
            print(f"\nError: {error_msg}")
            return {
                "answers": {},
                "processed_videos": [],
                "status": "error",
            }
