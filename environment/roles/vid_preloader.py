import os
import logging
import warnings
import multiprocessing
import sys
from typing import List
from pydantic import BaseModel, Field
from environment.agents.base import BaseTool

class VideoPreloader(BaseTool):
    """
    Application scenario: Add video footage to the audio
    Initialize environment and preprocesses video files.
    Note: VideoPreloader, VideoSearcher, and VideoEditor need to be called together.
    """

    def __init__(self):
        super().__init__()
        # Setup multiprocessing
        try:
            if multiprocessing.get_start_method(allow_none=True) != 'spawn':
                multiprocessing.set_start_method('spawn')
        except RuntimeError:
            # This handles the case where the start method has already been set
            pass
        
        # Configure logging and warnings
        warnings.filterwarnings("ignore")
        logging.getLogger("httpx").setLevel(logging.WARNING)

        self.project_root = os.getcwd()


        # Add the tools directory to the path
        tools_dir = os.path.abspath('tools')
        sys.path.append(tools_dir)
        
        # Initialize directories
        self._initialize_directories()
        
        # VideoRAG objects will be initialized when needed
        self.VideoRAG = None
        self.QueryParam = None

    class InputSchema(BaseTool.BaseInputSchema):
        video_dir: str = Field(
            ...,
            description="Directory containing the source MP4 video files to be processed"
        )

    class OutputSchema(BaseModel):
        status: str = Field(
            ...,
            description="Execution status (success/error)"
        )

    def _initialize_directories(self):
        """Initialize all necessary directories for the video processing pipeline"""
        # Define the path to the dataset directory and ensure it exists
        current_dir = os.getcwd()
        self.dataset_dir = os.path.join(current_dir, 'dataset')
        os.makedirs(self.dataset_dir, exist_ok=True)
        
        # Define and create video_edit directory
        self.video_edit_dir = os.path.join(self.dataset_dir, 'video_edit')
        os.makedirs(self.video_edit_dir, exist_ok=True)
        
        # Define and create all subdirectories except video_source
        subdirectories = [
            'audio_analysis',
            'scene_output',
            'videosource-workdir',
            'writing_data',
            'video_output'
        ]
        
        # Create all subdirectories
        for subdir in subdirectories:
            dir_path = os.path.join(self.video_edit_dir, subdir)
            os.makedirs(dir_path, exist_ok=True)
        
        # Set working directory path
        self.working_dir = os.path.join(self.video_edit_dir, 'videosource-workdir')

    def _load_videorag(self):
        """Load VideoRAG modules if not already loaded"""
        if self.VideoRAG is None:
            try:
                from videorag.videoragcontent import VideoRAG, QueryParam
                self.VideoRAG = VideoRAG
                self.QueryParam = QueryParam
                print("VideoRAG modules loaded successfully")
                return True
            except ImportError as e:
                print(f"Error loading VideoRAG modules: {e}")
                return False
        return True

    def _preload_videos(self, video_source_dir: str) -> List[str]:
        """Process videos with VideoRAG from the specified directory"""
        # Check if directory exists without creating it
        if not os.path.exists(video_source_dir):
            print(f"Error: Video source directory does not exist: {video_source_dir}")
            return []
            
        # Check if there are any videos in the source directory
        if not os.listdir(video_source_dir):
            print(f"Warning: No files found in {video_source_dir}")
            print("Please add your MP4 video files to this directory before running.")
            return []
        
        # Get all MP4 files from the directory
        video_paths = [os.path.join(video_source_dir, f) for f in os.listdir(video_source_dir) 
                      if f.endswith('.mp4')]
        
        if not video_paths:
            print("No MP4 video files found. Please add some videos to process.")
            return []
        
        print(f"Found {len(video_paths)} video files to process:")
        for video in video_paths:
            print(f" - {os.path.basename(video)}")

        # Make sure VideoRAG is loaded
        if not self._load_videorag():
            print("Failed to load VideoRAG modules. Cannot continue.")
            return []

        # Initialize and process videos with VideoRAG
        try:
            videoragcontent = self.VideoRAG(working_dir=self.working_dir)
            videoragcontent.insert_video(video_path_list=video_paths)
            print("Video preprocessing completed successfully")
            return [os.path.basename(path) for path in video_paths]
        except Exception as e:
            print(f"Error during video preprocessing: {e}")
            return []

    def execute(self, **kwargs):
        """Execute the video preprocessing pipeline"""
        # Validate input parameters
        params = self.InputSchema(**kwargs)
        video_source_dir = params.video_dir
        
        print(f"Starting video preprocessing from directory: {video_source_dir}")
        
        # Process the videos
        processed_videos = self._preload_videos(video_source_dir)
        
        # Return results
        if processed_videos:
            return {
                "status": "success"
            }
        else:
            return {
                "status": "error"
            }
