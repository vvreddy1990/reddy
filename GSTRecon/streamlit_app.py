import sys
import os

# Ensure the current directory is in sys.path for module imports
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

# Import and run the main app
import app

# If app.py uses __name__ == "__main__" for initialization, call it explicitly
if hasattr(app, 'initialize_session_state'):
    app.initialize_session_state() 