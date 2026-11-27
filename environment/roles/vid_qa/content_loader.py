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


class VideoContentQA(BaseTool):
    """
    Agent that transcribes all videos in a directory and provides interactive Q&A session.
    First transcribes all videos, then opens Q&A window for user questions.
    """

    def __init__(self, max_tokens: int = 10000):
        super().__init__()
        self.max_tokens = max_tokens
        self.timeout = 45
        self.transcript_pipe = None
        # Supported video extensions
        self.video_extensions = {'.mp4', '.avi', '.mov', '.mkv', '.flv', '.wmv', '.webm', '.m4v', '.3gp', '.mp3', '.wav', '.m4a', '.aac', '.ogg'}
        self.combined_transcript = ""
        self.qa_history = []

    class InputSchema(BaseTool.BaseInputSchema):
        video_dir: str = Field(
            ...,
            description="Path to the directory containing video files to be transcribed"
        )
        output_path: str = Field(
            None,
            description="Custom output path for saving conversation history (not used, kept for compatibility)"
        )
        save_history: bool = Field(
            True,
            description="Whether to save the Q&A to conversation history (True/False)"
        )

    class OutputSchema(BaseModel):
        transcript_path: str = Field(
            ...,
            description="Path where the combined transcript was saved"
        )
        processed_videos: List[str] = Field(
            ...,
            description="List of video files that were processed"
        )
        total_videos_found: int = Field(
            ...,
            description="Total number of video files found in the directory"
        )
        qa_session_completed: bool = Field(
            ...,
            description="Whether the Q&A session was completed"
        )
        total_questions_asked: int = Field(
            ...,
            description="Total number of questions asked during the session"
        )
        history_saved: bool = Field(
            ...,
            description="Whether the conversation history was saved"
        )
        history_path: Optional[str] = Field(
            default=None,
            description="Path where conversation history was saved"
        )
        status: str = Field(
            ...,
            description="Overall status of the transcription and Q&A process"
        )

    def _get_video_files(self, video_dir: str) -> List[str]:
        """Get all video files from the directory"""
        try:
            if not os.path.exists(video_dir):
                raise FileNotFoundError(f"Video directory not found: {video_dir}")
            
            if not os.path.isdir(video_dir):
                raise ValueError(f"Path is not a directory: {video_dir}")
            
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

    def _initialize_transcription(self):
        """Initialize model for transcription"""
        if self.transcript_pipe is None:
            try:
                logger.info("Initializing transcription model...")
                
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
                )
                
                logger.info("Whisper model initialized successfully")
                
            except Exception as e:
                logger.error(f"Error initializing Whisper model: {e}")
                raise e

    def _transcribe_video(self, video_path: str) -> str:
        """Transcribe a single video file using Whisper"""
        try:
            logger.info(f"Starting transcription of video: {video_path}")
            
            if not os.path.exists(video_path):
                raise FileNotFoundError(f"Video file not found: {video_path}")
            

            self._initialize_transcription()
            
            # Transcribe the video (Whisper can handle video files directly)
            result = self.transcript_pipe(video_path)
            transcript_text = result["text"]
            
            logger.info(f"Transcription completed for: {os.path.basename(video_path)}")
            logger.info(f"Transcript length: {len(transcript_text)} characters")
            
            return transcript_text
            
        except Exception as e:
            logger.error(f"Error during transcription of {video_path}: {e}")
            return f"[Error transcribing {os.path.basename(video_path)}: {str(e)}]"

    def _transcribe_all_videos(self, video_files: List[str], transcript_path: str) -> tuple[str, List[str]]:
        """Transcribe all videos in the directory and combine transcripts"""
        try:
            combined_transcript = ""
            processed_videos = []
            
            print(f"\n=== STARTING TRANSCRIPTION ===")
            print(f"Found {len(video_files)} video files to process")
            
            for i, video_file in enumerate(video_files, 1):
                print(f"\nTranscribing video {i}/{len(video_files)}: {os.path.basename(video_file)}")
                logger.info(f"Processing video {i}/{len(video_files)}: {os.path.basename(video_file)}")
                
                # Transcribe individual video
                transcript = self._transcribe_video(video_file)
                
                # Add to combined transcript with video identification
                video_name = os.path.basename(video_file)
                combined_transcript += f"\n\n=== VIDEO: {video_name} ===\n"
                combined_transcript += transcript
                
                processed_videos.append(video_file)
                
                # Brief pause between videos to prevent overheating
                if i < len(video_files):
                    time.sleep(1)
            
            # Save combined transcript
            os.makedirs(os.path.dirname(transcript_path), exist_ok=True)
            with open(transcript_path, 'w', encoding='utf-8') as f:
                f.write(combined_transcript.strip())
            
            print(f"\n=== TRANSCRIPTION COMPLETED ===")
            print(f"All {len(processed_videos)} videos transcribed successfully")
            print(f"Combined transcript saved to: {transcript_path}")
            logger.info(f"Combined transcript saved to: {transcript_path}")
            logger.info(f"Total transcript length: {len(combined_transcript)} characters")
            
            return combined_transcript.strip(), processed_videos
            
        except Exception as e:
            logger.error(f"Error during batch transcription: {e}")
            raise e

    def _load_text(self, txt_path: str) -> str:
        """Load content from a text file"""
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
            return ""

    @retry(
        retry=retry_if_exception_type((Exception)),
        wait=wait_exponential(multiplier=1, min=4, max=60),
        stop=stop_after_attempt(5),
        reraise=True
    )
    def _make_api_call(self, system_message, user_message, temperature=0.7, max_tokens=None, timeout=None):
        """Make an API call with retries and exponential backoff using tenacity"""
        if max_tokens is None:
            max_tokens = min(2000, self.max_tokens)
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
            raise e  # Re-raise to trigger tenacity retry

    def _qa_agent(self, user_question: str, content: str) -> str:
        """Process user question against video content"""

        prompt = f"""
        Answer the user's question carefully based only on the information contained in the video transcripts.
        The transcript contains content from multiple videos, each marked with "=== VIDEO: filename ===".
        If the answer cannot be found in the transcripts, state that clearly.

        User's question:
        "{user_question}"

        Video transcripts content:
        {content}

        Requirements:
        1. Be concise and direct in your answer
        2. Only use information from the transcripts
        3. If the answer is not in the transcripts, say "I don't have enough information from the videos to answer this question"
        4. When referencing information, mention which video file it came from if relevant
        5. Don't make up information not present in the transcripts
        """

        try:
            system_message = "You are a helpful assistant who answers questions about video content based strictly on the provided transcripts from multiple videos."

            logger.info("Starting QA agent processing")
            result = self._make_api_call(
                system_message=system_message,
                user_message=prompt,
                temperature=0.3,  
                timeout=120 
            )

            logger.info("Completed answer generation")
            return result

        except Exception as e:
            logger.error(f"Error in qa_agent: {e}")
            return "I'm sorry, I encountered an error processing your question. Please try again."

    def _run_qa_session(self) -> int:
        """Run interactive Q&A session until user types 'quit'"""
        question_count = 0
        
        print(f"\n" + "="*60)
        print("INTERACTIVE Q&A SESSION")
        print("="*60)
        print("Ask questions about the video content.")
        print("Type 'quit', 'exit', or 'q' to end the session.")
        print("="*60)
        
        # Limit content length if necessary for API calls
        content_for_qa = self.combined_transcript
        if len(content_for_qa) > 50000:
            logger.warning(f"Content too large ({len(content_for_qa)} chars), truncating to 50K for Q&A")
            content_for_qa = content_for_qa[:50000]
        
        while True:
            try:
                # Get user question
                print(f"\nQuestion #{question_count + 1}:")
                user_question = input("Your question: ").strip()
                
                # Check for quit commands
                if user_question.lower() in ['quit', 'exit', 'q']:
                    print("\nEnding Q&A session...")
                    break
                
                # Skip empty questions
                if not user_question:
                    print("Please enter a question or type 'quit' to exit.")
                    continue
                
                print("\nProcessing your question...")
                
                # Get answer from QA agent
                answer = self._qa_agent(user_question, content_for_qa)
                
                # Display answer
                print(f"\nAnswer: {answer}")
                print("-" * 60)
                
                # Save Q&A to session history
                timestamp = time.strftime("%Y-%m-%d %H:%M:%S")
                qa_entry = {
                    "question_number": question_count + 1,
                    "question": user_question,
                    "answer": answer,
                    "timestamp": timestamp
                }
                self.qa_history.append(qa_entry)
                question_count += 1
                
            except KeyboardInterrupt:
                print("\n\nSession interrupted by user. Ending Q&A session...")
                break
            except Exception as e:
                print(f"\nError processing question: {e}")
                print("Please try again or type 'quit' to exit.")
        
        print(f"\nQ&A session ended. Total questions asked: {question_count}")
        return question_count

    def _save_to_history(self, qa_data: Dict[str, Any], history_path: str) -> bool:
        """Save Q&A session to conversation history"""
        try:
            # Load existing conversation history or create new one
            conversation_history = []
            if os.path.exists(history_path):
                try:
                    with open(history_path, 'r', encoding='utf-8') as f:
                        conversation_history = json.load(f)
                        logger.info(f"Loaded existing conversation history with {len(conversation_history)} entries")
                except Exception as e:
                    logger.error(f"Error loading conversation history: {e}")
                    conversation_history = []

            # Add new Q&A session to history
            conversation_history.append(qa_data)

            # Save updated history
            with open(history_path, 'w', encoding='utf-8') as f:
                json.dump(conversation_history, f, indent=2, ensure_ascii=False)

            logger.info(f"Updated conversation history saved to {history_path}")
            return True

        except Exception as e:
            logger.error(f"Error saving conversation history: {e}")
            return False

    def execute(self, **kwargs):
        """Transcribe all videos in directory and run interactive Q&A session."""
        # Validate input parameters
        params = self.InputSchema(**kwargs)
        video_dir = params.video_dir
        
        # Convert save_history string to boolean if needed
        save_history = params.save_history
        if isinstance(save_history, str):
            save_history = save_history.lower() in ('true', '1', 'yes', 'on')

        parent_root = os.getcwd()

        # Define base directory paths
        dataset_dir = os.path.join(parent_root, 'dataset')
        video_edit_dir = os.path.join(dataset_dir, 'video_edit')
        os.makedirs(video_edit_dir, exist_ok=True)

        # Ensure required directories exist
        writing_data_dir = os.path.join(video_edit_dir, 'writing_data')
        os.makedirs(writing_data_dir, exist_ok=True)

        # Define transcript path - always save to the specified location
        transcript_path = os.path.join(writing_data_dir, "audio_transcript.txt")

        # Fixed history file path
        history_file_path = os.path.join(dataset_dir, "user_qa_history", "video_content_qa.json")
        os.makedirs(os.path.dirname(history_file_path), exist_ok=True)
        logger.info(f"Conversation history will be saved to: {history_file_path}")

        print(f"\n=== VIDEO CONTENT Q&A SYSTEM ===")
        print(f"Video directory: {video_dir}")
        print(f"Transcript will be saved to: {transcript_path}")

        try:
            # Step 1: Get all video files
            video_files = self._get_video_files(video_dir)
            total_videos_found = len(video_files)
            
            if total_videos_found == 0:
                raise ValueError(f"No video files found in directory: {video_dir}")

            # Step 2: Transcribe all videos
            try:
                self.combined_transcript, processed_videos = self._transcribe_all_videos(video_files, transcript_path)
            except Exception as e:
                logger.error(f"Batch transcription failed: {e}")
                # If transcription fails, try to load existing transcript
                if os.path.exists(transcript_path):
                    print("Transcription failed, loading existing transcript file...")
                    self.combined_transcript = self._load_text(transcript_path)
                    processed_videos = video_files  # Assume all were processed previously
                else:
                    raise e

            word_count = len(self.combined_transcript.split())
            char_count = len(self.combined_transcript)
            logger.info(f"Combined text data statistics: {word_count} words, {char_count} characters")

            # Step 3: Run interactive Q&A session
            total_questions_asked = self._run_qa_session()
            
            # Step 4: Save Q&A session to history if requested
            history_saved = False
            if save_history and self.qa_history:
                session_data = {
                    "session_timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
                    "video_dir": video_dir,
                    "transcript_path": transcript_path,
                    "processed_videos": [os.path.basename(v) for v in processed_videos],
                    "total_videos_found": total_videos_found,
                    "total_questions_asked": total_questions_asked,
                    "qa_history": self.qa_history
                }
                history_saved = self._save_to_history(session_data, history_file_path)

            print(f"\n=== SESSION SUMMARY ===")
            print(f"Videos processed: {len(processed_videos)}/{total_videos_found}")
            print(f"Questions asked: {total_questions_asked}")
            print(f"History saved: {'Yes' if history_saved else 'No'}")

            return {
                "transcript_path": transcript_path,
                "processed_videos": [os.path.basename(v) for v in processed_videos],
                "total_videos_found": total_videos_found,
                "qa_session_completed": True,
                "total_questions_asked": total_questions_asked,
                "history_saved": history_saved,
                "history_path": history_file_path if history_saved else None,
                "status": "success"
            }

        except Exception as e:
            error_msg = f"Error in video content Q&A system: {str(e)}"
            logger.error(error_msg)
            print(f"\nError: {error_msg}")

            return {
                "transcript_path": transcript_path,
                "processed_videos": [],
                "total_videos_found": 0,
                "qa_session_completed": False,
                "total_questions_asked": 0,
                "history_saved": False,
                "history_path": None,
                "status": "error"
            }
