from pathlib import Path

# Root path for ingestion
DEFAULT_PROJECT_PATH = Path.cwd() / "example_project"

# Directories to ignore
IGNORE_DIRS = {".git", ".venv", "__pycache__", "node_modules", "dist"}

# Allowed file extensions
ALLOWED_EXTS = {".py", ".js", ".ts"}
