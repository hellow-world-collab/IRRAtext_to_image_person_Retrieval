# 文件名: clip_api.py (已更新)
# 描述: 增加了接收 similarity_threshold 参数的功能

import os
import tempfile
import logging
from pathlib import Path
from fastapi import APIRouter, UploadFile, File, Form, HTTPException, BackgroundTasks

# 从主应用导入模型字典和历史记录列表
from yolo_api import MODELS, HISTORY_LOG, cleanup_files
from datetime import datetime

router = APIRouter()


@router.post("/clip/search")
async def clip_search(
        background_tasks: BackgroundTasks,
        video: UploadFile = File(...),
        query: str = Form(...),
        top_n: int = Form(1),
        threshold: float = Form(0.2)  # 【新增】接收相似度阈值
):
    retriever = MODELS.get("video_retriever")
    if not retriever:
        raise HTTPException(503, "视频片段检索服务当前不可用。")

    fd, tmp_vid_in = tempfile.mkstemp(suffix=Path(video.filename).suffix)
    os.close(fd)
    try:
        with open(tmp_vid_in, "wb") as f:
            f.write(await video.read())

        # 【修改点】将 threshold 传递给核心函数
        results = retriever.search_and_save_top_segments(
            video_path=tmp_vid_in,
            text_query=query,
            top_n=top_n,
            similarity_threshold=threshold
        )

        if not results:
            raise HTTPException(404, "未能找到任何高于设定阈值的视频片段。")

        history_entry = {
            "type": "视频片段检索",
            "query": query,
            "top_n": top_n,
            "results": results,
            "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        }
        HISTORY_LOG.insert(0, history_entry)

        background_tasks.add_task(cleanup_files, [tmp_vid_in])

        return {"results": results}

    except Exception as e:
        logging.error(f"Clip search failed: {e}", exc_info=True)
        # 确保即使发生错误，上传的临时文件也会被尝试删除
        cleanup_files([tmp_vid_in])
        raise HTTPException(500, f"处理视频时发生错误: {e}")

