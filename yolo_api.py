"""
yolo_api.py — YOLOv8 视频检测 API
--------------------------------
· POST /detect : 上传视频 → 返回带框 (H.264 MP4 | MJPEG AVI)
· GET  /ping   : 健康检查
· POST /reload : 热切换权重
依赖：ultralytics fastapi uvicorn[standard] opencv-python-headless tqdm
零 FFmpeg
"""

import os, cv2, sys, asyncio, tempfile, uvicorn
from functools import lru_cache
from pathlib import Path
from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.responses import FileResponse
from pydantic import BaseModel
from ultralytics import YOLO
from tqdm import tqdm

# —— Windows：用 SelectorEventLoop 避免 Proactor 报错 ——
if sys.platform.startswith("win"):
    asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())

# —— FastAPI 实例 (供 web_app.py 导入) ——
app = FastAPI(title="YOLOv8 API (H264→MJPEG)", version="3.1")

# —— YOLO 配置 ——————————————————————————
WEIGHTS_PATH = "yolov8n.pt"   # 可替换为 best.pt
CONF, CLASSES, IMG_SIZE = 0.25, [0], 640            # 只检测行人
# —————————————————————————————————————————

# —— 通用安全头 ——
@app.middleware("http")
async def add_headers(request, call_next):
    resp = await call_next(request)
    resp.headers["x-content-type-options"] = "nosniff"
    resp.headers["cache-control"] = "no-store"
    return resp

# —— 缓存模型加载 ——
@lru_cache(maxsize=1)
def get_model(path: str = WEIGHTS_PATH) -> YOLO:
    if not os.path.exists(path):
        raise FileNotFoundError(path)
    m = YOLO(path); m.fuse(); return m

# —— 尝试初始化 VideoWriter：H.264→MJPEG ——
def _init_writer(out_path: str, fps: float, size: tuple[int, int]):
    w, h = size
    # 1) 先尝试 H.264 (Media Foundation 支持)
    for fourcc_str in ("H264", "avc1"):
        writer = cv2.VideoWriter(
            out_path, cv2.VideoWriter_fourcc(*fourcc_str), fps, (w, h)
        )
        if writer.isOpened():
            return writer, "H264"
    # 2) 回退 MJPEG AVI
    avi_path = Path(out_path).with_suffix(".avi")
    writer = cv2.VideoWriter(
        str(avi_path), cv2.VideoWriter_fourcc(*"MJPG"), fps, (w, h)
    )
    if writer.isOpened():
        return writer, "MJPEG"
    raise RuntimeError("OpenCV 无法初始化 H.264 或 MJPEG 编码器")

# —— 推理主函数 ——
def infer(video_in: str, model: YOLO) -> tuple[str, str]:
    cap = cv2.VideoCapture(video_in)
    if not cap.isOpened():
        raise RuntimeError("无法打开视频文件")

    fps = cap.get(cv2.CAP_PROP_FPS) or 25
    w, h = int(cap.get(3)), int(cap.get(4))
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    Path("outputs").mkdir(exist_ok=True)
    base_mp4 = Path("outputs") / (Path(video_in).stem + "_det.mp4")
    writer, codec = _init_writer(str(base_mp4), fps, (w, h))
    out_path = str(base_mp4) if codec == "H264" else writer.filename

    for _ in tqdm(range(total), desc=f"detect({codec})", unit="frame"):
        ok, frame = cap.read()
        if not ok: break
        res = model(frame, imgsz=IMG_SIZE, conf=CONF, classes=CLASSES, verbose=False)
        writer.write(res[0].plot())

    cap.release(); writer.release()
    return out_path, codec

# —— 路由 ——————————————————
@app.get("/ping")
async def ping():
    return {"status": "ok"}

@app.post("/detect")
async def detect(file: UploadFile = File(...)):
    if not file.filename.lower().endswith((".mp4", ".avi", ".mov", ".mkv")):
        raise HTTPException(400, "仅支持视频文件")

    fd, tmp = tempfile.mkstemp(suffix=Path(file.filename).suffix); os.close(fd)
    with open(tmp, "wb") as f:
        f.write(await file.read())

    try:
        out_path, codec = infer(tmp, get_model())
    except Exception as e:
        raise HTTPException(500, f"推理失败: {e}")

    mime = "video/mp4" if out_path.endswith(".mp4") else "video/x-msvideo"
    headers = {"x-video-codec": codec}
    return FileResponse(out_path, media_type=mime,
                        headers=headers,
                        filename=Path(out_path).name)

# —— 热切换权重 ——
class ReloadReq(BaseModel):
    weight_path: str

@app.post("/reload")
async def reload_model(req: ReloadReq):
    get_model.cache_clear()
    get_model(req.weight_path)
    return {"msg": f"已切换权重：{req.weight_path}"}

# —— 独立启动 (可选) ——
if __name__ == "__main__":
    uvicorn.run("yolo_api:app", host="0.0.0.0", port=8000)
