# 文件名: search_api.py (OSS集成最终版)
# 描述: 将处理结果上传到OSS并存入数据库

import os, uuid, tempfile, subprocess, asyncio, threading, shutil, logging
from pathlib import Path
from fastapi import APIRouter, UploadFile, File, Form, HTTPException
from fastapi.responses import StreamingResponse, JSONResponse
from datetime import datetime

# --- 从主应用导入所需内容 ---
from yolo_api import MODELS, cleanup_files
from config_mysql.database import add_history_record
from config_mysql.oss_utils import upload_to_oss


def mp4_to_h264(src: str) -> str:
    if not os.path.exists(src): return src
    p = Path(src)
    dst = p.with_name(f"{p.stem}_h264.mp4")
    command = ["ffmpeg", "-i", str(p), "-c:v", "libx264", "-pix_fmt", "yuv420p", "-preset", "veryfast", "-y", str(dst)]
    try:
        subprocess.run(command, check=True, stdout=subprocess.DEVNULL, stderr=subprocess.PIPE)
        cleanup_files([str(p)])
        return str(dst)
    except Exception:
        return str(p)


router = APIRouter()
progress = {}


@router.get("/progress/{tid}")
async def progress_sse(tid: str):
    async def events():
        while True:
            val = progress.get(tid, 0)
            if isinstance(val, tuple):
                state, payload = val
                yield f"event: {state.lower()}\ndata: {payload}\n\n"
                break
            else:
                yield f"data: {val}\n\n"
            await asyncio.sleep(0.5)

    return StreamingResponse(events(), media_type="text/event-stream")


@router.post("/search")
async def search(video: UploadFile = File(...), query: str = Form(...), threshold: float = Form(...)):
    searcher = MODELS.get("person_searcher")
    if not searcher: raise HTTPException(503, "服务不可用。")
    tid = uuid.uuid4().hex[:8]
    progress[tid] = 1

    fd, tmp_vid_in = tempfile.mkstemp(suffix=Path(video.filename).suffix)
    os.close(fd)
    with open(tmp_vid_in, "wb") as f:
        f.write(await video.read())

    out_mp4 = str(Path(tempfile.gettempdir()) / f"{tid}_searched.mp4")

    def worker():
        local_files_to_clean = [tmp_vid_in, out_mp4]
        try:
            progress[tid] = 10
            searcher.search_in_video(video_path=tmp_vid_in, text_query=query, similarity_threshold=threshold,
                                     output_path=out_mp4)
            progress[tid] = 90
            final_path = mp4_to_h264(out_mp4)
            local_files_to_clean.append(final_path)

            # 【核心修改】上传到OSS并存入数据库
            oss_url = upload_to_oss(final_path)
            if oss_url:
                add_history_record(
                    operation_type="全身行人检索", query_text=query,
                    result_url=oss_url, timestamp=datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                )
                progress[tid] = ("DONE", oss_url)  # 将永久URL发给前端
            else:
                progress[tid] = ("ERR", "文件上传到OSS失败。")

        except Exception as e:
            logging.error(f"检索工作线程出错: {e}", exc_info=True)
            progress[tid] = ("ERR", str(e))
        finally:
            cleanup_files(list(set(local_files_to_clean)))  # 清理所有本地临时文件

    threading.Thread(target=worker, daemon=True).start()
    return {"task_id": tid}


@router.post("/search_image")
async def search_image(image: UploadFile = File(...), query: str = Form(...), threshold: float = Form(...)):
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

        # 【核心修改】上传到OSS并存入数据库
        oss_url = upload_to_oss(tmp_out_path)
        if oss_url:
            add_history_record(
                operation_type="图片内容检索", query_text=query,
                result_url=oss_url, timestamp=datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            )
            # 对于图片，直接返回包含永久URL的JSON
            return JSONResponse(content={"result_url": oss_url})
        else:
            raise HTTPException(500, "文件上传到OSS失败。")

    except Exception as e:
        raise HTTPException(500, f"图像检索失败: {e}")
    finally:
        cleanup_files([tmp_in_path, tmp_out_path])

