"""Tests for the TwelveLabs Pegasus video-understanding backend.

The no-network tests run anywhere. The live test is skipped unless
TWELVELABS_API_KEY is set (free key at https://twelvelabs.io).
"""
import os
import sys
from unittest import mock

import pytest

# Make the project root importable when running pytest from the repo root.
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from environment.roles.vid_qa.pegasus_understanding import PegasusVideoUnderstanding


def test_url_is_passed_through_without_filesystem_access():
    """A public URL should be analyzed directly, with no disk lookup."""
    tool = PegasusVideoUnderstanding()
    sources = tool._get_video_files("https://example.com/clip.mp4")
    assert sources == ["https://example.com/clip.mp4"]


def test_execute_routes_url_to_pegasus():
    """execute() should call the pegasus() helper and surface its text."""
    tool = PegasusVideoUnderstanding()
    fake_response = mock.Mock(data="A rabbit wakes up in a meadow.")

    with mock.patch(
        "environment.roles.vid_qa.pegasus_understanding.pegasus",
        return_value=fake_response,
    ) as mock_pegasus:
        result = tool.execute(
            video_path="https://example.com/clip.mp4",
            prompt="Describe this video.",
        )

    assert result["status"] == "success"
    assert result["answers"]["https://example.com/clip.mp4"] == "A rabbit wakes up in a meadow."
    # The URL is forwarded to Pegasus untouched.
    _, kwargs = mock_pegasus.call_args
    assert kwargs["video"] == {"type": "url", "url": "https://example.com/clip.mp4"}
    assert kwargs["prompt"] == "Describe this video."


def test_missing_path_reports_error():
    tool = PegasusVideoUnderstanding()
    result = tool.execute(video_path="/no/such/video.mp4", prompt="What is this?")
    assert result["status"] == "error"


@pytest.mark.skipif(
    not os.environ.get("TWELVELABS_API_KEY"),
    reason="requires TWELVELABS_API_KEY",
)
def test_twelvelabs_credentials_and_sdk_live():
    """Live check: the SDK + key reach the TwelveLabs backend.

    Uses a fast Marengo text embedding (512-dim) to confirm wiring without
    waiting on a full Pegasus video analysis.
    """
    from twelvelabs import TwelveLabs

    client = TwelveLabs(api_key=os.environ["TWELVELABS_API_KEY"])
    resp = client.embed.create(model_name="marengo3.0", text="a person walking a dog")
    assert len(resp.text_embedding.segments[0].float_) == 512
