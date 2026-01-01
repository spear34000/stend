import uvicorn
from core.api_server import app

if __name__ == "__main__":
    print("Launching Stend Platform (Advanced API Mode)...")
    uvicorn.run(app, host="0.0.0.0", port=5001)
