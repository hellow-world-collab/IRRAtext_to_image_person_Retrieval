"""
yolo_api.py — YOLOv8 视频检测 API
--------------------------------
· POST /detect : 上传视频 → 返回带框 (H.264 MP4 | MJPEG AVI)
· GET  /ping   : 健康检查
· POST /reload : 热切换权重
依赖：ultralytics fastapi uvicorn[standard] opencv-python-headless tqdm
零 FFmpeg
"""

import os, cv2, sys, asyncio, tempfile, uvicorn, logging
from functools import lru_cache
from pathlib import Path
# 导入 BackgroundTasks 用于后台清理任务
from fastapi import FastAPI, UploadFile, File, HTTPException, BackgroundTasks
from fastapi.responses import FileResponse
from pydantic import BaseModel
from ultralytics import YOLO
from tqdm import tqdm

# —— 配置日志记录 ——
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# —— Windows：用 SelectorEventLoop 避免 Proactor 报错 ——
if sys.platform.startswith("win"):
    asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())

# —— FastAPI 实例 (供 web_app.py 导入) ——
app = FastAPI(title="YOLOv8 API (H264→MJPEG)", version="3.2")

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
        logging.error(f"权重文件未找到: {path}")
        raise FileNotFoundError(path)
    logging.info(f"正在加载模型: {path}")
    m = YOLO(path); m.fuse(); return m

# —— [优化] 尝试初始化 VideoWriter：H.264→MJPEG，并增加日志 ——
def _init_writer(out_path: str, fps: float, size: tuple[int, int]):
    w, h = size
    # 1) 先尝试 H.264
    for fourcc_str in ("H264", "avc1"):
        writer = cv2.VideoWriter(
            out_path, cv2.VideoWriter_fourcc(*fourcc_str), fps, (w, h)
        )
        if writer.isOpened():
            logging.info(f"成功使用 {fourcc_str} (H.264) 编码器。")
            return writer, "H264"

    # 如果 H.264 失败，打印警告
    logging.warning("H.264 编码器初始化失败。这通常是因为环境中缺少 'openh264' 库。")
    logging.warning("正在回退到 MJPEG (.avi) 编码器...")

    # 2) 回退 MJPEG AVI
    avi_path = Path(out_path).with_suffix(".avi")
    writer = cv2.VideoWriter(
        str(avi_path), cv2.VideoWriter_fourcc(*"MJPG"), fps, (w, h)
    )
    if writer.isOpened():
        logging.info("成功使用 MJPEG 编码器。")
        return writer, "MJPEG"

    raise RuntimeError("OpenCV 无法初始化 H.264 或 MJPEG 编码器")

# —— 推理主函数 ——
def infer(video_in: str, model: YOLO) -> tuple[str, str]:
    cap = cv2.VideoCapture(video_in)
    if not cap.isOpened():
        raise RuntimeError(f"无法打开视频文件: {video_in}")

    fps = cap.get(cv2.CAP_PROP_FPS) or 25
    w, h = int(cap.get(3)), int(cap.get(4))
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    Path("outputs").mkdir(exist_ok=True)
    base_mp4 = Path("outputs") / (Path(video_in).stem + "_det.mp4")
    writer, codec = _init_writer(str(base_mp4), fps, (w, h))
    out_path = str(base_mp4) if codec == "H264" else writer.filename

    # 使用 TQDM 显示处理进度
    for _ in tqdm(range(total), desc=f"detect({codec})", unit="frame", file=sys.stdout):
        ok, frame = cap.read()
        if not ok: break
        res = model(frame, imgsz=IMG_SIZE, conf=CONF, classes=CLASSES, verbose=False)
        writer.write(res[0].plot())

    cap.release(); writer.release()
    logging.info(f"推理完成，输出文件: {out_path}")
    return out_path, codec

# —— [新增] 清理文件的后台任务 ——
def cleanup_files(paths: list[str]):
    for p in paths:
        try:
            os.remove(p)
            logging.info(f"已清理临时文件: {p}")
        except OSError as e:
            logging.warning(f"清理文件失败: {p}, 错误: {e}")

# —— 路由 ——————————————————
@app.get("/ping")
async def ping():
    return {"status": "ok"}

@app.post("/detect")
async def detect(file: UploadFile, background_tasks: BackgroundTasks):
    if not file.filename.lower().endswith((".mp4", ".avi", ".mov", ".mkv")):
        raise HTTPException(400, "仅支持视频文件")

    # 创建临时文件来保存上传的视频
    fd, tmp_in = tempfile.mkstemp(suffix=Path(file.filename).suffix)
    os.close(fd)
    try:
        with open(tmp_in, "wb") as f:
            f.write(await file.read())

        # 执行推理
        model = get_model()
        out_path, codec = infer(tmp_in, model)

        # [优化] 添加后台任务，在响应发送后清理输入和输出文件
        background_tasks.add_task(cleanup_files, [tmp_in, out_path])

        # 准备并返回文件响应
        mime = "video/mp4" if out_path.endswith(".mp4") else "video/x-msvideo"
        headers = {"x-video-codec": codec}
        return FileResponse(out_path, media_type=mime,
                            headers=headers,
                            filename=Path(out_path).name)
    except Exception as e:
        # 如果发生任何错误，也要确保清理了临时输入文件
        cleanup_files([tmp_in])
        logging.error(f"推理过程中发生错误: {e}", exc_info=True)
        raise HTTPException(500, f"推理失败: {e}")

# —— 热切换权重 ——
class ReloadReq(BaseModel):
    weight_path: str

@app.post("/reload")
async def reload_model(req: ReloadReq):
    try:
        get_model.cache_clear()
        get_model(req.weight_path)
        return {"msg": f"已成功切换权重: {req.weight_path}"}
    except Exception as e:
        raise HTTPException(404, f"加载权重失败: {e}")

# —— 独立启动 (可选) ——
if __name__ == "__main__":
    uvicorn.run("yolo_api:app", host="0.0.0.0", port=8000)
