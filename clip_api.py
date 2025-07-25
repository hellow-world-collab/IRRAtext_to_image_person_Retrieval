# 文件名: clip_api.py (完整版)
# 描述: 正确处理从 video_retriever 返回的本地路径并上传到OSS

import os, tempfile, logging, json
from pathlib import Path
from fastapi import APIRouter, UploadFile, File, Form, HTTPException
from datetime import datetime

from yolo_api import MODELS, cleanup_files
from config_mysql.database import add_history_record
from config_mysql.oss_utils import upload_to_oss

router = APIRouter()


@router.post("/clip/search")
async def clip_search(
        video: UploadFile = File(...),
        query: str = Form(...),
        top_n: int = Form(1),
        threshold: float = Form(0.2)
):
    retriever = MODELS.get("video_retriever")
    if not retriever:
        raise HTTPException(503, "视频片段检索服务当前不可用。")

    fd, tmp_vid_in = tempfile.mkstemp(suffix=Path(video.filename).suffix)
    os.close(fd)
    local_files_to_clean = [tmp_vid_in]

    try:
        with open(tmp_vid_in, "wb") as f:
            f.write(await video.read())

        local_results = retriever.search_and_save_top_segments(
            video_path=tmp_vid_in, text_query=query,
            top_n=top_n, similarity_threshold=threshold
        )

        if not local_results:
            raise HTTPException(404, "未能找到任何高于设定阈值的视频片段。")

        oss_results = []
        for res in local_results:
            local_clip_path = res.pop("video_path", None)
            if local_clip_path:
                local_files_to_clean.append(local_clip_path)
                oss_url = upload_to_oss(local_clip_path)
                if oss_url:
                    res["video_url"] = oss_url
                    oss_results.append(res)

        if not oss_results:
            raise HTTPException(500, "片段已生成，但上传到OSS时失败。")

        add_history_record(
            operation_type="视频片段检索", query_text=query,
            result_url=oss_results[0]['video_url'],
            timestamp=datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            details=json.dumps({"results": oss_results})
        )

        return {"results": oss_results}

    except Exception as e:
        logging.error(f"Clip search failed: {e}", exc_info=True)
        raise HTTPException(500, f"处理视频时发生错误: {e}")
    finally:
        cleanup_files(list(set(local_files_to_clean)))
