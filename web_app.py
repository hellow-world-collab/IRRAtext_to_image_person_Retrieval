# 文件名: web_app.py (已更新)
# 描述: 注册所有API路由，并为临时文件提供静态访问路径

import socket, uvicorn, tempfile
import yolo_api, search_api, clip_api # 导入新的 clip_api
from fastapi.staticfiles import StaticFiles
from pathlib import Path

app = yolo_api.app

# 注册所有功能的路由
app.include_router(search_api.router)
app.include_router(clip_api.router)

# 【重要】新增：挂载一个用于访问临时文件的静态目录
# 这允许前端通过 /temp/filename.mp4 的URL访问到服务器上的临时文件
temp_dir = tempfile.gettempdir()
app.mount("/temp", StaticFiles(directory=temp_dir), name="temp_files")

# 挂载主前端应用
app.mount("/", StaticFiles(directory="web", html=True), name="static")

def lan_ip():
    # ... (此函数内容保持不变) ...
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
