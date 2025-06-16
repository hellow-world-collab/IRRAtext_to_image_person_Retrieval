import sys, os, cv2, uuid, tempfile, subprocess, asyncio, threading
from pathlib import Path
from fastapi import APIRouter, UploadFile, File, Form, HTTPException
from fastapi.responses import FileResponse, StreamingResponse

# --------- 路径配置 ----------
YOLO_WEIGHTS = "yolov8n.pt"
IRRA_CONFIG  = "logs/CUHK-PEDES/configs.yaml"  # best.pth 位于 logs/IRR/best.pth
# -----------------------------

if sys.platform.startswith("win"):
    asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())

router   = APIRouter()
progress = {}   # tid -> int 或 ("DONE",路径) / ("ERR",msg)

# ---------- 工具函数 ----------
def mp4_to_h264(src: str) -> str:
    """将 mp4v 编码重新封装为 H.264；若系统缺编码器则返回原文件"""
    cap = cv2.VideoCapture(src)
    if not cap.isOpened():
        return src
    fps, w, h = cap.get(5) or 25, int(cap.get(3)), int(cap.get(4))
    p   = Path(src)
    dst = p.with_name(p.stem + "_h264.mp4")         # 合法文件名
    writer = cv2.VideoWriter(str(dst),
                             cv2.VideoWriter_fourcc(*"avc1"),
                             fps, (w, h))
    if not writer.isOpened():                       # 无 H.264 编码器
        cap.release()
        return src
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        writer.write(frame)
    cap.release(); writer.release()
    return str(dst)

def run_test_cli(video: str, query: str, out_mp4: str):
    cmd = [
        sys.executable, "testforpth.py",
        "--config_file", IRRA_CONFIG,
        "--yolo_model_path", YOLO_WEIGHTS,
        "--video_path", video,
        "--output_video_path", out_mp4,
        "--text_query", query,
        "--similarity_threshold", "0.25",
        "--process_every_n_frames", "5"
    ]
    print(">>", " ".join(cmd), flush=True)
    return subprocess.run(cmd, stdout=subprocess.PIPE,
                          stderr=subprocess.STDOUT, text=True)

# ---------- SSE 进度 ----------
@router.get("/progress/{tid}")
async def progress_sse(tid: str):
    async def events():
        while True:
            val = progress.get(tid, 0)
            yield f"data:{val}\n\n"
            if isinstance(val, tuple):
                break
            await asyncio.sleep(0.5)
    return StreamingResponse(events(), media_type="text/event-stream")

# ---------- /search ----------
@router.post("/search")
async def search(
    video: UploadFile = File(...),
    query: str = Form(...)
):
    if not video.filename.lower().endswith((".mp4", ".avi", ".mov", ".mkv")):
        raise HTTPException(400, "仅支持视频文件")

    tid = uuid.uuid4().hex[:8]
    progress[tid] = 0

    # 保存上传文件
    fd, tmp_vid = tempfile.mkstemp(suffix=Path(video.filename).suffix)
    os.close(fd)
    with open(tmp_vid, "wb") as f:
        f.write(await video.read())

    out_mp4 = Path(tempfile.mkdtemp()) / (Path(video.filename).stem + "_det.mp4")

    def worker():
        try:
            progress[tid] = 10
            res = run_test_cli(tmp_vid, query, str(out_mp4))
            if res.returncode != 0:
                progress[tid] = ("ERR", res.stdout[:400])
                return
            progress[tid] = 90
            final = mp4_to_h264(str(out_mp4))
            progress[tid] = ("DONE", final)
        except Exception as e:
            progress[tid] = ("ERR", str(e))

    threading.Thread(target=worker, daemon=True).start()
    return {"task_id": tid}

# ---------- 下载结果 ----------
@router.get("/result/{tid}")
def fetch_result(tid: str):
    val = progress.get(tid)
    if not isinstance(val, tuple):
        raise HTTPException(202, "Processing")
    state, payload = val
    if state == "ERR":
        raise HTTPException(500, payload)
    return FileResponse(payload, media_type="video/mp4",
                        filename=Path(payload).name)
