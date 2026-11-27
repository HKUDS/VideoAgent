import logging
import os
import sys

for tool_dir in ("CosyVoice", "CosyVoice/third_party/Matcha-TTS", "fish-speech", "ImageBind"):
    sys.path.append(os.path.abspath(os.path.join("tools", tool_dir)))

logging.basicConfig(level=logging.WARNING)
logging.getLogger("modelscope").setLevel(logging.WARNING)
from environment.agents.multi import MultiAgent


def print_banner():
    """Display VideoAgent banner and welcome message"""
    banner = """
╔═════════════════════════════════════════════════════════════════════════════════════╗
║                                                                                     ║
║  ██╗   ██╗██╗██████╗ ███████╗ ██████╗  █████╗  ██████╗ ███████╗███╗   ██╗████████╗  ║
║  ██║   ██║██║██╔══██╗██╔════╝██╔═══██╗██╔══██╗██╔════╝ ██╔════╝████╗  ██║╚══██╔══╝  ║
║  ██║   ██║██║██║  ██║█████╗  ██║   ██║███████║██║  ███╗█████╗  ██╔██╗ ██║   ██║     ║
║  ╚██╗ ██╔╝██║██║  ██║██╔══╝  ██║   ██║██╔══██║██║   ██║██╔══╝  ██║╚██╗██║   ██║     ║
║   ╚████╔╝ ██║██████╔╝███████╗╚██████╔╝██║  ██║╚██████╔╝███████╗██║ ╚████║   ██║     ║
║    ╚═══╝  ╚═╝╚═════╝ ╚══════╝ ╚═════╝ ╚═╝  ╚═╝ ╚═════╝ ╚══════╝╚═╝  ╚═══╝   ╚═╝     ║
║                                                                                     ║
║                         🎬 Open Agentic Video Intelligence 🤖                       ║
║                                                                                     ║
╚═════════════════════════════════════════════════════════════════════════════════════╝
"""
    print(banner)

def print_welcome_message():
    """Display welcome message"""
    welcome_msg = """
🎉 Welcome to VideoAgent - Open Agentic Video Intelligence!

VideoAgent is your intelligent companion for comprehensive video processing and creation.
Our multi-modal agentic framework transforms how you interact with video content through
natural language conversations.

🔥 Core Capabilities:

   🧠 Understanding Video Content
      Enable in-depth analysis, summarization, and insight extraction from video media
      with advanced multi-modal intelligence capabilities.

   ✂️ Editing Video Clips
      Provide intuitive tools for assembling, clipping, and reconfiguring content with
      seamless workflow integration.

   🎨 Remaking Creative Videos
      Utilize generative technologies to produce new, imaginative video content through
      AI-powered creative assistance.

   🔧 Multi-Modal Agentic Framework
      Deliver comprehensive video intelligence through an integrated framework that
      combines multiple AI modalities for enhanced performance.

   🚀 Seamless Natural Language Experience
      Transform video interaction and creation through pure conversational AI - no
      complex interfaces or technical expertise required, just natural dialogue with
      VideoAgent.

📖 Getting Started:
   Simply start conversing with VideoAgent! Describe what you want to do with your
   videos, and our intelligent agents will coordinate to fulfill your requests.

"""
    print(welcome_msg)

def main():
    print_banner()
    print_welcome_message()
    
    multi_agent = MultiAgent()
    multi_agent.run()

if __name__ == "__main__":
    try:
        main()
    except EOFError:
        pass
