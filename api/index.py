import sys
from pathlib import Path

# Add root folder to sys.path so we can import modules under 'backend'
root_dir = Path(__file__).resolve().parent.parent
sys.path.append(str(root_dir))

# Import the FastAPI instance
from backend.app.main import app
