# 文件名: search_api.py (修改后)

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


# ---------- 工具函数 (视频处理) - 修改处 ----------
def mp4_to_h264(src: str) -> str:
    """
    使用 FFmpeg 将视频文件转换为浏览器兼容的 H.264 格式。

    Args:
        src (str): 输入的视频文件路径。

    Returns:
        str: 转换后的 H.264 视频文件路径。如果转换失败，则返回原始路径。
    """
    if not os.path.exists(src):
        print(f"Error: Source file for conversion does not exist: {src}")
        return src

    p = Path(src)
    # 创建一个新的文件名给转码后的视频，避免覆盖
    dst = p.with_name(f"{p.stem}_h264.mp4")

    # 构建 FFmpeg 命令
    # -i: 输入文件
    # -c:v libx264: 使用 H.264 编码器
    # -pix_fmt yuv420p: 确保最大的浏览器兼容性
    # -preset veryfast: 在速度和质量之间取得良好平衡
    # -y: 如果输出文件已存在则覆盖
    command = [
        "ffmpeg",
        "-i", str(p),
        "-c:v", "libx264",
        "-pix_fmt", "yuv420p",
        "-preset", "veryfast",
        "-y",
        str(dst)
    ]

    print(f"INFO: Running FFmpeg command: {' '.join(command)}")
    try:
        # 执行命令，并隐藏FFmpeg自身的日志输出
        result = subprocess.run(command, check=True, capture_output=True, text=True)
        print(f"INFO: FFmpeg conversion successful. Output: {dst}")
        # 转换成功，删除旧文件，返回新文件路径
        if os.path.exists(str(p)):
            os.remove(str(p))
        return str(dst)
    except FileNotFoundError:
        print("ERROR: 'ffmpeg' command not found. Please ensure FFmpeg is installed and in your system's PATH.")
        # FFmpeg未找到，无法转换，返回原文件
        return str(p)
    except subprocess.CalledProcessError as e:
        # FFmpeg 执行出错
        print(f"ERROR: FFmpeg conversion failed. Return code: {e.returncode}")
        print(f"FFmpeg stderr: {e.stderr}")
        # 转换失败，返回原文件
        return str(p)


def run_test_cli_for_video(video: str, query: str, out_mp4: str, threshold: float):
    """
    修改：增加了 threshold 参数
    """
    cmd = [
        sys.executable, "testforpth.py",
        "--config_file", IRRA_CONFIG,
        "--yolo_model_path", YOLO_WEIGHTS,
        "--video_path", video,
        "--output_video_path", out_mp4,
        "--text_query", query,
        "--similarity_threshold", str(threshold),  # <-- 使用传入的阈值
        "--process_every_n_frames", "10"
    ]
    print(">>", " ".join(cmd), flush=True)
    return subprocess.run(cmd, stdout=subprocess.PIPE,
                          stderr=subprocess.STDOUT, text=True, encoding='utf-8')


# --- 图片检索工具函数 (保持不变) ---
def run_test_cli_for_image(image: str, query: str, out_img: str, threshold: float):
    cmd = [
        sys.executable, "testforpth.py",
        "--config_file", IRRA_CONFIG,
        "--yolo_model_path", YOLO_WEIGHTS,
        "--input_image_path", image,
        "--output_image_path", out_img,
        "--text_query", query,
        "--similarity_threshold", str(threshold),
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


# ---------- /search (视频) - 修改处 ----------
@router.post("/search")
async def search(
    video: UploadFile = File(...),
    query: str = Form(...),
    threshold: float = Form(...)  # <-- 接收阈值
):
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
            # 将阈值传递给处理函数
            res = run_test_cli_for_video(tmp_vid, query, str(out_mp4), threshold)
            if res.returncode != 0:
                progress[tid] = ("ERR", res.stdout[:400])
                return
            progress[tid] = 90
            # 【确认这一行是激活的】
            # 它会调用我们上面修改好的基于FFmpeg的转码函数
            final = mp4_to_h264(str(out_mp4))
            progress[tid] = ("DONE", final)
        except Exception as e:
            progress[tid] = ("ERR", str(e))
        finally:
            # 清理最初上传的临时文件
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


# --- 图片检索路由 (保持不变) ---
@router.post("/search_image")
async def search_image(
        image: UploadFile = File(...),
        query: str = Form(...),
        threshold: float = Form(...)
):
    allowed_image_types = (".jpg", ".jpeg", ".png", ".bmp", ".webp")
    original_filename = image.filename
    if not original_filename.lower().endswith(allowed_image_types):
        raise HTTPException(400, "仅支持图片文件 (jpg, png, bmp, webp)")

    original_suffix = Path(original_filename).suffix
    tmp_img = tempfile.NamedTemporaryFile(delete=False, suffix=original_suffix)
    out_img = tempfile.NamedTemporaryFile(delete=False, suffix=original_suffix)

    tmp_img_path = Path(tmp_img.name)
    out_img_path = Path(out_img.name)
    tmp_img.close()
    out_img.close()

    with open(tmp_img_path, "wb") as f:
        shutil.copyfileobj(image.file, f)

    res = run_test_cli_for_image(str(tmp_img_path), query, str(out_img_path), threshold)

    if res.returncode != 0 or not out_img_path.exists():
        error_output = res.stdout if res.stdout else "No output from script."
        print("--- Subprocess Error ---")
        print(error_output)
        print("--- End Subprocess Error ---")
        raise HTTPException(500, detail=f"推理脚本执行失败:\n{error_output}")

    return FileResponse(str(out_img_path), media_type=image.content_type,
                        filename="result_" + original_filename)