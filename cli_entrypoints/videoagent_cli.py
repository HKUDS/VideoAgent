"""Workspace and runtime entry point for the VideoAgent distribution."""

import argparse
import os
import subprocess
import sys
from importlib.metadata import version
from pathlib import Path


ROOT = Path(__file__).resolve().parent
APP = ROOT / "videoagent_package" / "app"
if not APP.is_dir():
    APP = ROOT.parent

CONFIG_DIR = Path("environment/config")
CONFIG = CONFIG_DIR / "config.yml"
RESOURCES = {
    CONFIG: CONFIG_DIR / "config.example.yml",
    CONFIG_DIR / "graph.txt": CONFIG_DIR / "graph.txt",
    CONFIG_DIR / "intents.yml": CONFIG_DIR / "intents.yml",
    CONFIG_DIR / "registry.json": CONFIG_DIR / "registry.json",
    Path("requirements-runtime.txt"): Path("requirements-runtime.txt"),
}


def init_workspace(workspace: Path) -> None:
    for relative_path, source in RESOURCES.items():
        destination = workspace / relative_path
        destination.parent.mkdir(parents=True, exist_ok=True)
        data = (APP / source).read_bytes()
        try:
            fd = os.open(destination, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
        except FileExistsError:
            continue
        with os.fdopen(fd, "wb") as stream:
            stream.write(data)
        print(f"Created {relative_path}")


def main() -> int:
    parser = argparse.ArgumentParser(prog="videoagent")
    parser.add_argument("--version", action="version", version=f"VideoAgent {version('videoagent')}")
    commands = parser.add_subparsers(dest="command", required=True)
    commands.add_parser("init", help="Prepare the current directory as a VideoAgent workspace")
    commands.add_parser("run", help="Start the VideoAgent conversation")
    args = parser.parse_args()

    try:
        if args.command == "init":
            init_workspace(Path.cwd())
            return 0
        missing = [str(path) for path in RESOURCES if path != Path("requirements-runtime.txt") and not path.is_file()]
        if missing:
            parser.exit(1, f"videoagent: missing {', '.join(missing)}. Run 'videoagent init' first.\n")
        return subprocess.call([sys.executable, str(APP / "main.py")])
    except KeyboardInterrupt:
        return 130
    except OSError as exc:
        parser.exit(1, f"videoagent: {exc}\n")
