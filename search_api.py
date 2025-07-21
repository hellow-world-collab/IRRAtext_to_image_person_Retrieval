# 文件名: search_api.py (完整最终版)
# 描述: 包含了所有必需的路由和逻辑，修正了因省略代码导致的错误。

import sys, os, uuid, tempfile, subprocess, asyncio, threading, shutil, logging
from pathlib import Path
from fastapi import APIRouter, UploadFile, File, Form, HTTPException, BackgroundTasks
from fastapi.responses import FileResponse, StreamingResponse
from datetime import datetime

# --- 从主应用导入所需内容 ---
from yolo_api import MODELS, HISTORY_LOG, cleanup_files


def mp4_to_h264(src: str) -> str:
    """使用 FFmpeg 将视频文件转换为浏览器兼容的 H.264 格式。"""
    if not os.path.exists(src):
        logging.error(f"转换错误: 源文件不存在 {src}")
        return src
    p = Path(src)
    dst = p.with_name(f"{p.stem}_h264.mp4")
    command = ["ffmpeg", "-i", str(p), "-c:v", "libx264", "-pix_fmt", "yuv420p", "-preset", "veryfast", "-y", str(dst)]
    try:
        subprocess.run(command, check=True, stdout=subprocess.DEVNULL, stderr=subprocess.PIPE)
        logging.info(f"FFmpeg 转码成功. 输出: {dst}")
        cleanup_files([str(p)])
        return str(dst)
    except (FileNotFoundError, subprocess.CalledProcessError) as e:
        logging.error(f"FFmpeg 转码失败: {e}")
        return str(p)


if sys.platform.startswith("win"):
    asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())

router = APIRouter()
progress = {}


# ==================== 【补全】进度查询路由 ====================
@router.get("/progress/{tid}")
async def progress_sse(tid: str):
    """
    为前端提供服务器发送事件(SSE)，用于实时更新进度条。
    """

    async def events():
        while True:
            val = progress.get(tid, 0)
            if isinstance(val, tuple):
                state, payload = val
                # 发送一个带有事件名称的特殊消息
                yield f"event: {state.lower()}\ndata: {payload}\n\n"
                break
            else:
                # 发送普通的进度数据
                yield f"data: {val}\n\n"
            await asyncio.sleep(0.5)

    return StreamingResponse(events(), media_type="text/event-stream")


# ==============================================================

@router.post("/search")
async def search(video: UploadFile = File(...), query: str = Form(...), threshold: float = Form(...)):
    searcher = MODELS.get("person_searcher")
    if not searcher: raise HTTPException(503, "服务不可用。")

    tid = uuid.uuid4().hex[:8]
    progress[tid] = 1

    # 【补全】在worker函数外部定义临时文件路径
    fd, tmp_vid_in = tempfile.mkstemp(suffix=Path(video.filename).suffix)
    os.close(fd)
    with open(tmp_vid_in, "wb") as f:
        f.write(await video.read())

    out_mp4 = str(Path(tempfile.gettempdir()) / f"{tid}_searched.mp4")

    def worker():
        try:
            progress[tid] = 10
            searcher.search_in_video(video_path=tmp_vid_in, text_query=query, similarity_threshold=threshold,
                                     output_path=out_mp4)
            progress[tid] = 90
            final_path = mp4_to_h264(out_mp4)

            HISTORY_LOG.insert(0, {
                "type": "全身行人检索", "query": query,
                "result_url": f"/result/{tid}", "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            })
            progress[tid] = ("DONE", tid)
        except Exception as e:
            logging.error(f"检索工作线程出错: {e}", exc_info=True)
            progress[tid] = ("ERR", str(e))
        finally:
            cleanup_files([tmp_vid_in])

    threading.Thread(target=worker, daemon=True).start()
    return {"task_id": tid}


@router.post("/search_image")
async def search_image(image: UploadFile = File(...), query: str = Form(...), threshold: float = Form(...),
                       background_tasks: BackgroundTasks = BackgroundTasks()):
    searcher = MODELS.get("person_searcher")
    if not searcher: raise HTTPException(503, "服务不可用。")

    with tempfile.NamedTemporaryFile(delete=False, suffix=Path(image.filename).suffix) as tmp_in:
        shutil.copyfileobj(image.file, tmp_in)
        tmp_in_path = tmp_in.name

    tmp_out_path = str(Path(tempfile.gettempdir()) / (f"{uuid.uuid4().hex[:8]}_searched{Path(image.filename).suffix}"))

    try:
        success = searcher.search_in_image(image_path=tmp_in_path, text_query=query, similarity_threshold=threshold,
                                           output_path=tmp_out_path)
        if not success: raise HTTPException(500, "图像处理失败。")

        HISTORY_LOG.insert(0, {
            "type": "图片内容检索", "query": query,
            "result_url": f"/temp/{Path(tmp_out_path).name}", "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        })

        background_tasks.add_task(cleanup_files, [tmp_in_path, tmp_out_path])
        return FileResponse(tmp_out_path, media_type=image.content_type, background=background_tasks)
    except Exception as e:
        cleanup_files([tmp_in_path])
        raise HTTPException(500, f"图像检索失败: {e}")


# ==================== 【补全】结果下载路由 ====================
@router.get("/result/{tid}")
def fetch_result(tid: str, background_tasks: BackgroundTasks = BackgroundTasks()):
    """
    根据任务ID，返回处理完成的视频文件。
    """
    val = progress.get(tid)
    if not isinstance(val, tuple) or val[0] != "DONE":
        raise HTTPException(404, "结果未找到或未准备好。")

    # 重新构造最终文件路径
    final_path = str(Path(tempfile.gettempdir()) / f"{tid}_searched_h264.mp4")
    if not os.path.exists(final_path):
        final_path = str(Path(tempfile.gettempdir()) / f"{tid}_searched.mp4")
        if not os.path.exists(final_path):
            raise HTTPException(404, "结果文件不存在。")

    # 返回文件后，再启动后台任务清理它
    background_tasks.add_task(cleanup_files, [final_path])
    return FileResponse(final_path, media_type="video/mp4", filename=Path(final_path).name, background=background_tasks)
# ==============================================================
