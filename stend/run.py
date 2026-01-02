import os
import sys
import uvicorn

# Add project root (where stend.py lives) to path
# This allows 'import stend.core...' to work
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from stend.core.api_server import app

if __name__ == "__main__":
    print("Launching Stend Platform (Advanced API Mode)...")
    uvicorn.run("stend.core.api_server:app", host="0.0.0.0", port=5001, reload=True)
