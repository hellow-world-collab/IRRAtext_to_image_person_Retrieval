# 文件名: search_api.py (完整代码)

import sys, os, cv2, uuid, tempfile, subprocess, asyncio, threading, shutil
from pathlib import Path
from fastapi import APIRouter, UploadFile, File, Form, HTTPException
from fastapi.responses import FileResponse, StreamingResponse

# --------- 路径配置 ----------
PROJECT_ROOT = Path(__file__).resolve().parent

YOLO_WEIGHTS = str(PROJECT_ROOT / "yolov8n.pt")
IRRA_CONFIG = str(PROJECT_ROOT / "logs" / "CUHK-PEDES" / "configs.yaml")
# -----------------------------

if sys.platform.startswith("win"):
    asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())

router = APIRouter()
progress = {}


# ---------- 工具函数 (视频处理) ----------
def mp4_to_h264(src: str) -> str:
    cap = cv2.VideoCapture(src)
    if not cap.isOpened(): return src
    fps, w, h = cap.get(5) or 25, int(cap.get(3)), int(cap.get(4))
    p = Path(src)
    dst = p.with_name(p.stem + "_h264.mp4")
    writer = cv2.VideoWriter(str(dst), cv2.VideoWriter_fourcc(*"avc1"), fps, (w, h))
    if not writer.isOpened():
        cap.release()
        return src
    while True:
        ok, frame = cap.read()
        if not ok: break
        writer.write(frame)
    cap.release();
    writer.release()
    return str(dst)


def run_test_cli_for_video(video: str, query: str, out_mp4: str):
    cmd = [
        sys.executable, "testforpth.py",
        "--config_file", IRRA_CONFIG,
        "--yolo_model_path", YOLO_WEIGHTS,
        "--video_path", video,
        "--output_video_path", out_mp4,
        "--text_query", query,
        "--similarity_threshold", "0.25",  # 视频的阈值可以保持不变
        "--process_every_n_frames", "5"
    ]
    print(">>", " ".join(cmd), flush=True)
    return subprocess.run(cmd, stdout=subprocess.PIPE,
                          stderr=subprocess.STDOUT, text=True, encoding='utf-8')


# --- 最终版：同时接收阈值参数的图片检索工具函数 ---
def run_test_cli_for_image(image: str, query: str, out_img: str, threshold: float):
    cmd = [
        sys.executable, "testforpth.py",
        "--config_file", IRRA_CONFIG,
        "--yolo_model_path", YOLO_WEIGHTS,
        "--input_image_path", image,
        "--output_image_path", out_img,
        "--text_query", query,
        "--similarity_threshold", str(threshold),  # <--- 使用从网页传入的阈值
    ]
    print(">>", " ".join(cmd), flush=True)
    return subprocess.run(cmd, stdout=subprocess.PIPE,
                          stderr=subprocess.STDOUT, text=True, encoding='utf-8')


# ---------- SSE 进度 (视频) ----------
@router.get("/progress/{tid}")
async def progress_sse(tid: str):
    async def events():
        while True:
            val = progress.get(tid, 0)
            yield f"data:{val}\n\n"
            if isinstance(val, tuple): break
            await asyncio.sleep(0.5)

    return StreamingResponse(events(), media_type="text/event-stream")


# ---------- /search (视频) ----------
@router.post("/search")
async def search(video: UploadFile = File(...), query: str = Form(...)):
    if not video.filename.lower().endswith((".mp4", ".avi", ".mov", ".mkv")):
        raise HTTPException(400, "仅支持视频文件")
    tid = uuid.uuid4().hex[:8]
    progress[tid] = 0
    fd, tmp_vid = tempfile.mkstemp(suffix=Path(video.filename).suffix)
    os.close(fd)
    with open(tmp_vid, "wb") as f:
        f.write(await video.read())
    out_mp4 = Path(tempfile.gettempdir()) / (uuid.uuid4().hex[:8] + "_det.mp4")

    def worker():
        try:
            progress[tid] = 10
            res = run_test_cli_for_video(tmp_vid, query, str(out_mp4))
            if res.returncode != 0:
                progress[tid] = ("ERR", res.stdout[:400])
                return
            progress[tid] = 90
            final = mp4_to_h264(str(out_mp4))
            progress[tid] = ("DONE", final)
        except Exception as e:
            progress[tid] = ("ERR", str(e))
        finally:
            if os.path.exists(tmp_vid): os.remove(tmp_vid)

    threading.Thread(target=worker, daemon=True).start()
    return {"task_id": tid}


# ---------- 下载视频结果 ----------
@router.get("/result/{tid}")
def fetch_result(tid: str):
    val = progress.get(tid)
    if not isinstance(val, tuple): raise HTTPException(202, "Processing")
    state, payload = val
    if state == "ERR": raise HTTPException(500, payload)
    return FileResponse(payload, media_type="video/mp4", filename=Path(payload).name)


# --- 最终版: 图片检索路由 (整合所有修复) ---
@router.post("/search_image")
async def search_image(
        image: UploadFile = File(...),
        query: str = Form(...),
        threshold: float = Form(...)
):
    # 1. 验证文件类型
    allowed_image_types = (".jpg", ".jpeg", ".png", ".bmp", ".webp")
    original_filename = image.filename
    if not original_filename.lower().endswith(allowed_image_types):
        raise HTTPException(400, "仅支持图片文件 (jpg, png, bmp, webp)")

    # 2. 使用 UUID 生成安全路径
    original_suffix = Path(original_filename).suffix
    tmp_img = tempfile.NamedTemporaryFile(delete=False, suffix=original_suffix)
    out_img = tempfile.NamedTemporaryFile(delete=False, suffix=original_suffix)

    tmp_img_path = Path(tmp_img.name)
    out_img_path = Path(out_img.name)
    tmp_img.close()
    out_img.close()

    # 3. 保存上传图片到临时文件
    with open(tmp_img_path, "wb") as f:
        shutil.copyfileobj(image.file, f)

    # 4. 调用测试脚本，传入路径与阈值
    res = run_test_cli_for_image(str(tmp_img_path), query, str(out_img_path), threshold)

    # 5. 检查执行结果
    if res.returncode != 0 or not out_img_path.exists():
        error_output = res.stdout if res.stdout else "No output from script."
        print("--- Subprocess Error ---")
        print(error_output)
        print("--- End Subprocess Error ---")
        raise HTTPException(500, detail=f"推理脚本执行失败:\n{error_output}")

    # 6. 返回图片响应，注意: 临时文件不会马上删除
    return FileResponse(str(out_img_path), media_type=image.content_type,
                        filename="result_" + original_filename)