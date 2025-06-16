# web_app.py
import socket, uvicorn, yolo_api, search_api
from fastapi.staticfiles import StaticFiles

app = yolo_api.app
app.include_router(search_api.router)
app.mount("/", StaticFiles(directory="web", html=True), name="static")

def lan_ip():
    s = socket.socket(socket.AF_INET,socket.SOCK_DGRAM)
    try:
        s.connect(("8.8.8.8",80)); ip=s.getsockname()[0]
    except: ip="127.0.0.1"
    finally: s.close()
    return ip

if __name__ == "__main__":
    PORT = 8000
    print("═"*60)
    print(f"🚀 访问: http://127.0.0.1:{PORT}  或  http://{lan_ip()}:{PORT}")
    print("═"*60)
    uvicorn.run("web_app:app", host="0.0.0.0", port=PORT, log_level="info")
