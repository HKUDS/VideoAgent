import json
import os
import math
import time
import re
import logging
import sys
from typing import Dict, List, Any, Optional
from tenacity import retry, stop_after_attempt, wait_exponential, retry_if_exception_type
from environment.config.llm import gpt
from pydantic import BaseModel, Field
from environment.agents.base import BaseTool
import torch
from transformers import AutoModelForSpeechSeq2Seq, AutoProcessor, pipeline

# Configure logging
logging.basicConfig(level=logging.INFO,
                    format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


class VideoSummarizationGenerator(BaseTool):
    """
    Agent that generates summarization content based on user ideas and reference materials,
    with specialized formatting for video presentations. Loads Whisper model and processes video directories.
    """

    def __init__(self):
        super().__init__()
        self.max_tokens = 15000
        self.timeout = 45
        self.transcript_pipe = None
        # Supported video extensions
        self.video_extensions = {'.mp4', '.avi', '.mov', '.mkv', '.flv', '.wmv', '.webm', '.m4v', '.3gp', '.mp3', '.wav', '.m4a', '.aac', '.ogg'}

    class InputSchema(BaseTool.BaseInputSchema):
        user_idea: str = Field(
            ...,
            description="User's idea for the video summarization including word count requirements"
        )
        video_dir: str = Field(
            ...,
            description="Path to the directory containing video files or path to transcript file"
        )
        present_style_path: str = Field(
            ...,
            description="Path to the video summarization presentation style file or direct style content as string"
        )
        output_path: str = Field(
            ...,
            description="Output file path for saving content (must be a file path, not directory)"
        )

    class OutputSchema(BaseModel):
        content_output: Dict[str, str] = Field(
            ...,
            description="Generated content including user idea, summarization content"
        )
        status: str = Field(
            ...,
            description="Status of the content generation process"
        )
        output_path: str = Field(
            ...,
            description="Path where the content was saved"
        )
        processed_videos: Optional[List[str]] = Field(
            default=None,
            description="List of video files that were processed (if video_dir contains videos)"
        )
        transcript_source: str = Field(
            ...,
            description="Source of the transcript (transcribed videos or loaded text file)"
        )

    @retry(
        retry=retry_if_exception_type((Exception)),
        wait=wait_exponential(multiplier=1, min=4, max=60),
        stop=stop_after_attempt(3),
        reraise=True
    )
    def _initialize_transcription(self):
        """Initialize Whisper model for transcription with retry logic"""
        if self.transcript_pipe is None:
            try:
                logger.info("Initializing Whisper large-v3-turbo model...")
                
                parent_root = os.getcwd()
                
                # Path to the Whisper model
                model_path = os.path.join(parent_root, "tools", "whisper-large-v3-turbo")
                
                if not os.path.exists(model_path):
                    raise FileNotFoundError(f"Whisper model not found at: {model_path}")
                
                device = "cuda:0" if torch.cuda.is_available() else "cpu"
                torch_dtype = torch.float16 if torch.cuda.is_available() else torch.float32
                
                logger.info(f"Loading model from: {model_path}")
                logger.info(f"Using device: {device}, dtype: {torch_dtype}")
                
                model = AutoModelForSpeechSeq2Seq.from_pretrained(
                    model_path,
                    torch_dtype=torch_dtype,
                    low_cpu_mem_usage=True,
                    use_safetensors=True
                )
                model.to(device)
                
                processor = AutoProcessor.from_pretrained(model_path)
                
                self.transcript_pipe = pipeline(
                    "automatic-speech-recognition",
                    model=model,
                    tokenizer=processor.tokenizer,
                    feature_extractor=processor.feature_extractor,
                    max_new_tokens=128,
                    chunk_length_s=30,
                    batch_size=16,
                    return_timestamps=True,
                    torch_dtype=torch_dtype,
                    device=device,
                    generate_kwargs={"language": "en", "task": "transcribe"}
                )
                
                logger.info("Whisper model initialized successfully")
                
            except Exception as e:
                logger.error(f"Error initializing Whisper model: {e}")
                raise e

    def _get_video_files(self, video_dir: str) -> List[str]:
        """Get all video files from the directory"""
        try:
            if not os.path.exists(video_dir):
                raise FileNotFoundError(f"Video directory not found: {video_dir}")
            
            if not os.path.isdir(video_dir):
                # If it's a file, check if it's a video file
                _, ext = os.path.splitext(video_dir.lower())
                if ext in self.video_extensions:
                    return [video_dir]
                else:
                    return []  # Not a video file, will be treated as text file
            
            video_files = []
            for filename in os.listdir(video_dir):
                file_path = os.path.join(video_dir, filename)
                if os.path.isfile(file_path):
                    _, ext = os.path.splitext(filename.lower())
                    if ext in self.video_extensions:
                        video_files.append(file_path)
            
            video_files.sort()  # Sort for consistent processing order
            logger.info(f"Found {len(video_files)} video files in directory: {video_dir}")
            
            return video_files
            
        except Exception as e:
            logger.error(f"Error getting video files from directory: {e}")
            raise e

    @retry(
        retry=retry_if_exception_type((Exception)),
        wait=wait_exponential(multiplier=1, min=2, max=30),
        stop=stop_after_attempt(3),
        reraise=True
    )
    def _transcribe_video(self, video_path: str) -> str:
        """Transcribe a single video file using Whisper with retry logic"""
        try:
            logger.info(f"Starting transcription of video: {video_path}")
            
            if not os.path.exists(video_path):
                raise FileNotFoundError(f"Video file not found: {video_path}")
            
            # Initialize Whisper if not already done
            self._initialize_transcription()
            
            # Transcribe the video with proper language settings
            result = self.transcript_pipe(
                video_path,
                generate_kwargs={"language": "en", "task": "transcribe"}
            )
            transcript_text = result["text"]
            
            logger.info(f"Transcription completed for: {os.path.basename(video_path)}")
            logger.info(f"Transcript length: {len(transcript_text)} characters")
            
            return transcript_text
            
        except Exception as e:
            logger.error(f"Error during transcription of {video_path}: {e}")
            raise e

    def _transcribe_all_videos(self, video_files: List[str]) -> tuple[str, List[str]]:
        """Transcribe all videos in the directory and combine transcripts"""
        try:
            combined_transcript = ""
            processed_videos = []
            
            print(f"\n=== STARTING TRANSCRIPTION ===")
            print(f"Found {len(video_files)} video files to process")
            
            for i, video_file in enumerate(video_files, 1):
                print(f"\nTranscribing video {i}/{len(video_files)}: {os.path.basename(video_file)}")
                logger.info(f"Processing video {i}/{len(video_files)}: {os.path.basename(video_file)}")
                
                # Transcribe individual video with retry logic
                try:
                    transcript = self._transcribe_video(video_file)
                    
                    # Add to combined transcript with video identification
                    video_name = os.path.basename(video_file)
                    combined_transcript += f"\n\n=== VIDEO: {video_name} ===\n"
                    combined_transcript += transcript
                    
                    processed_videos.append(video_file)
                    
                except Exception as e:
                    logger.error(f"Failed to transcribe {video_file} after retries: {e}")
                    # Add error message to transcript
                    video_name = os.path.basename(video_file)
                    combined_transcript += f"\n\n=== VIDEO: {video_name} ===\n"
                    combined_transcript += f"[Error transcribing {video_name}: {str(e)}]"
                
                # Brief pause between videos to prevent overheating
                if i < len(video_files):
                    time.sleep(1)
            
            print(f"\n=== TRANSCRIPTION COMPLETED ===")
            print(f"Successfully processed {len(processed_videos)}/{len(video_files)} videos")
            logger.info(f"Total transcript length: {len(combined_transcript)} characters")
            
            return combined_transcript.strip(), processed_videos
            
        except Exception as e:
            logger.error(f"Error during batch transcription: {e}")
            raise e

    @retry(
        retry=retry_if_exception_type((Exception)),
        wait=wait_exponential(multiplier=1, min=2, max=30),
        stop=stop_after_attempt(5),
        reraise=True
    )
    def _load_text(self, txt_path: str) -> str:
        """Load content from a text file with retry logic"""
        encodings_to_try = ['utf-8', 'gb18030', 'gbk', 'gb2312', 'cp1252', 'iso-8859-1']

        for encoding in encodings_to_try:
            try:
                logger.info(f"Trying to read file with encoding: {encoding}")
                with open(txt_path, 'r', encoding=encoding, errors='replace') as file:
                    content = file.read()
                logger.info(f"Successfully read file with encoding: {encoding}")
                return content
            except UnicodeDecodeError:
                continue

        try:
            logger.info("Trying binary reading approach")
            with open(txt_path, 'rb') as file:
                content = file.read().decode('utf-8', errors='replace')
            return content
        except Exception as e:
            logger.error(f"Error loading text file: {e}")
            raise e

    @retry(
        retry=retry_if_exception_type((Exception)),
        wait=wait_exponential(multiplier=1, min=4, max=60),
        stop=stop_after_attempt(5),
        reraise=True
    )
    def _make_api_call(self, system_message, user_message, temperature=0.7, timeout=None):
        """Make an API call with retries and exponential backoff using tenacity"""
        if timeout is None:
            timeout = self.timeout

        try:
            logger.info(f"Making API call with system and user messages, timeout={timeout}s")
            start_time = time.time()


            response = gpt(system=system_message, user=user_message)

            elapsed_time = time.time() - start_time
            logger.info(f"API call completed in {elapsed_time:.2f}s")
            return response.choices[0].message.content
        except Exception as e:
            logger.warning(f"API call failed with error: {e}, retrying...")
            raise e 

    @retry(
        retry=retry_if_exception_type((Exception)),
        wait=wait_exponential(multiplier=1, min=2, max=30),
        stop=stop_after_attempt(3),
        reraise=True
    )
    def _presenter_agent(self, user_idea: str, content: str, present_content: str) -> str:
        """Process content and adapt to user's idea with retry logic"""

        prompt = f"""
        Create a content summarization, strictly following the user's ideas and presentation methods, answer using the user's idea language (English/中文).

        User's idea:
        "{user_idea}"

        Grounded text content:
        {content}

        Follow this presentation method, read it and apply it carefully:
        {present_content}

        Requirements:
        1. Format and Structure:
        - Less point forms

        2. Content Guidelines:
        - Strictly abide by the user's words/字数 count requirements
        - Use only original key dialogues (no fabricated dialogues)
        - Remove unnecessary environmental descriptions
        - Focus on plot-advancing elements

        3. Language and Style:
        - Third-person perspective
        - Process in text language (English/中文)
        - Maintain clear narrative flow
        - Remove duplicated sentences

        Create a single, polished version that meets all these requirements.
        """

        try:
            system_message = "You are an experienced expert in writing transcripts summarization. Pay special attention to user's words/字数 count requirements."

            logger.info("Starting presenter agent processing")
            result = self._make_api_call(
                system_message=system_message,
                user_message=prompt,
                temperature=0.7,
                timeout=120  # Increased timeout for processing larger content
            )

            logger.info("Completed summarization generation")
            return result

        except Exception as e:
            logger.error(f"Error in presenter_agent: {e}")
            raise e

    def _get_content_from_video_dir(self, video_dir: str) -> tuple[str, Optional[List[str]], str]:
        """Get content from video_dir - either by transcribing videos or loading text file"""
        

        print("Initializing transcription model...")
        self._initialize_transcription()
        
        # Check if video_dir is a file or directory
        if os.path.isfile(video_dir):
            # Check if it's a video file
            _, ext = os.path.splitext(video_dir.lower())
            if ext in self.video_extensions:
                # Single video file
                print(f"Processing single video file: {os.path.basename(video_dir)}")
                transcript = self._transcribe_video(video_dir)
                return transcript, [video_dir], "transcribed_single_video"
            else:
                # Text file
                print(f"Loading text content from file: {os.path.basename(video_dir)}")
                content = self._load_text(video_dir)
                return content, None, "loaded_text_file"
        
        elif os.path.isdir(video_dir):

            video_files = self._get_video_files(video_dir)
            
            if video_files:

                print(f"Processing video directory with {len(video_files)} videos")
                combined_transcript, processed_videos = self._transcribe_all_videos(video_files)
                return combined_transcript, processed_videos, "transcribed_video_directory"
            else:

                default_transcript = os.path.join(video_dir, "audio_transcript.txt")
                if os.path.exists(default_transcript):
                    print(f"No videos found, loading default transcript: {default_transcript}")
                    content = self._load_text(default_transcript)
                    return content, None, "loaded_default_transcript"
                else:
                    raise ValueError(f"No video files or transcript found in directory: {video_dir}")
        else:
            raise FileNotFoundError(f"Path does not exist: {video_dir}")

    def _process_pipeline(self, user_idea: str, video_dir: str, present_style_path: str) -> tuple[str, Optional[List[str]], str]:
        """Main pipeline process - get content and process with presenter agent"""
        
        # Check if present_style_path is a file path or direct content
        if os.path.exists(present_style_path):
            present_content = self._load_text(present_style_path)
            logger.info("Loaded presentation method from file")
        else:
            present_content = present_style_path  # Use the string directly
            logger.info("Using provided presentation method string")

        # Get content from video_dir (transcribe videos or load text)
        content, processed_videos, source_type = self._get_content_from_video_dir(video_dir)

        word_count = len(content.split())
        char_count = len(content)
        logger.info(f"Text data statistics: {word_count} words, {char_count} characters")
        logger.info(f"Content source: {source_type}")

        try:

            presenter_output = self._presenter_agent(user_idea, content, present_content)
            logger.info("Successfully generated presentation")
        except Exception as e:
            logger.error(f"Error in presenter_agent: {e}")

            presenter_output = content[:15000]
            logger.info("Used truncated content as fallback")

        return presenter_output, processed_videos, source_type

    def _create_content(self, user_idea: str, video_dir: str, present_style_path: str) -> tuple[Dict[str, str], Optional[List[str]], str]:
        """Generate content incorporating user ideas and reference materials."""
        try:
            formatted_content, processed_videos, source_type = self._process_pipeline(user_idea, video_dir, present_style_path)
            return {"general": formatted_content}, processed_videos, source_type
        except Exception as e:
            return {"error": f"Error: {e}"}, None, "error"

    @retry(
        retry=retry_if_exception_type((Exception)),
        wait=wait_exponential(multiplier=1, min=2, max=30),
        stop=stop_after_attempt(3),
        reraise=True
    )
    def execute(self, **kwargs):
        """Generate summarization content based on user ideas and reference materials with retry logic."""
        # Validate input parameters
        params = self.InputSchema(**kwargs)
        user_idea = params.user_idea

        parent_root = os.getcwd()

        # Use video_dir parameter
        video_dir_path = params.video_dir
        if not os.path.isabs(video_dir_path):
            # If relative path, make it relative to project root
            video_dir_path = os.path.join(parent_root, video_dir_path)

        # Handle output path - must be a file path, not directory
        output_path = params.output_path
        if not os.path.isabs(output_path):
            output_path = os.path.join(parent_root, output_path)
        
        # Check if output_path is a directory and create a default filename
        if os.path.isdir(output_path):
            output_path = os.path.join(output_path, "video_summarization.txt")
        
        # Ensure the directory exists
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        logger.info(f"Content will be saved to: {output_path}")

        # Get presentation style from present_style_path
        present_style = params.present_style_path
        if not os.path.isabs(present_style) and os.path.exists(os.path.join(parent_root, present_style)):
            present_style = os.path.join(parent_root, present_style)

        print("\n=== CREATING VIDEO SUMMARIZATION CONTENT ===")
        print(f"\nUsing idea: {user_idea}")
        print(f"Using video/content source: {video_dir_path}")
        print(f"Output will be saved to: {output_path}")

        try:
            # Generate content
            content_result, processed_videos, source_type = self._create_content(user_idea, video_dir_path, present_style)

            if "error" in content_result:
                raise RuntimeError(f"Failed to create content: {content_result['error']}")

            content_output = {
                "user_idea": user_idea,
                "content_created": content_result.get("general", "")
            }

            # Save content to the specified output path as plain text
            with open(output_path, 'w', encoding='utf-8') as f:
                f.write(content_result.get("general", ""))
            print(f"\nContent saved to {output_path}")

            return {
                "content_output": content_output,
                "status": "success",
                "output_path": output_path,
                "processed_videos": [os.path.basename(v) for v in processed_videos] if processed_videos else None,
                "transcript_source": source_type
            }

        except Exception as e:
            logger.error(f"Error in execute method: {e}")
            raise e
