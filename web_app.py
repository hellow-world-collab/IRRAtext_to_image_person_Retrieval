# filename: web_app.py (Corrected Import)
# description: This version fixes the incorrect import statement for dashboard_api.

import socket, uvicorn, tempfile
# These imports are correct
import yolo_api, search_api, clip_api

# ==================== The Fix ====================
# Correctly import the dashboard_api module
import dashboard_api
# =================================================

from fastapi.staticfiles import StaticFiles
from pathlib import Path

app = yolo_api.app

# Register all functional routes
app.include_router(search_api.router)
app.include_router(clip_api.router)

# Now this line will work correctly
app.include_router(dashboard_api.router)

# Mount a static directory for accessing temporary files
temp_dir = tempfile.gettempdir()
app.mount("/temp", StaticFiles(directory=temp_dir), name="temp_files")

# Mount the main frontend application
app.mount("/", StaticFiles(directory="web", html=True), name="static")

def lan_ip():
    # This function remains unchanged
    s = socket.socket(socket.AF_INET,socket.SOCK_DGRAM)
    try:
        s.connect(("8.8.8.8",80)); ip=s.getsockname()[0]
    except: ip="127.0.0.1"
    finally: s.close()
    return ip

if __name__ == "__main__":
    PORT = 8000
    print("═"*60)
    print(f"🚀 Access: http://127.0.0.1:{PORT}  or  http://{lan_ip()}:{PORT}")
    print("═"*60)
    # It's helpful to enable reload for easier debugging
    uvicorn.run("web_app:app", host="0.0.0.0", port=PORT, log_level="info", reload=True)
