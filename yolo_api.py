
import os, sys, asyncio, uvicorn, logging, math
from contextlib import asynccontextmanager
from typing import List

from fastapi import FastAPI, Query, Body
from datetime import datetime


from person_searcher import PersonSearcher
from video_retriever import VideoRetriever
from config_mysql.database import get_history_paginated, clear_all_history,delete_history_ids

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
if sys.platform.startswith("win"):
    asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())

YOLO_WEIGHTS = "yolov8l.pt"
IRRA_CONFIG = "logs/CUHK-PEDES/configs.yaml"
CLIP_MODEL = "Searchium-ai/clip4clip-webvid150k"

MODELS = {}


def cleanup_files(paths: list[str]):
    for p in paths:
        if p and os.path.exists(p):
            try:
                os.remove(p)
                logging.info(f"已成功清理临时文件: {p}")
            except OSError as e:
                logging.warning(f"清理文件失败: {p}, 错误: {e}")


@asynccontextmanager
async def lifespan(app: FastAPI):
    logging.info("Web App starting up...")
    try:
        MODELS["person_searcher"] = PersonSearcher(irra_config_file=IRRA_CONFIG, yolo_model_path=YOLO_WEIGHTS)
        logging.info("PersonSearcher (IRRA+YOLO) preloaded successfully.")
    except Exception as e:
        logging.error(f"Failed to preload PersonSearcher: {e}", exc_info=True)
    try:
        MODELS["video_retriever"] = VideoRetriever(model_name=CLIP_MODEL)
        logging.info("VideoRetriever (clip4clip) preloaded successfully.")
    except Exception as e:
        logging.error(f"Failed to preload VideoRetriever: {e}", exc_info=True)
    yield
    logging.info("Web App shutting down...")
    MODELS.clear()


app = FastAPI(
    title="Unified Vision API",
    version="7.3-FinalFix",
    lifespan=lifespan
)


@app.get("/history")
async def get_history_paginated_api(page: int = Query(1, gt=0), limit: int = Query(10, gt=0)):
    """
    获取分页后的历史记录API端点。
    """
    # 调用新的数据库分页函数
    db_result = get_history_paginated(page=page, limit=limit)

    paginated_items = db_result["items"]
    total_items = db_result["total"]

    total_pages = math.ceil(total_items / limit)

    # 格式化时间戳
    for record in paginated_items:
        if isinstance(record.get('timestamp'), datetime):
            record['timestamp'] = record['timestamp'].strftime("%Y-%m-%d %H:%M:%S")

    return {
        "items": paginated_items,
        "total": total_items,
        "page": page,
        "limit": limit,
        "pages": total_pages
    }
# =================================================================

@app.delete("/history")
async def clear_history_from_db():
    clear_all_history()
    return {"message": "历史记录已清空"}

@app.delete("/history/selected")
async def delete_selected_history(ids: List[int] = Body(..., embed=True)):
    """
    根据ID列表删除指定的历史记录。
    前端应发送格式为 {"ids": [1, 2, 3]} 的JSON。
    """
    if not ids:
        return {"message": "未提供任何ID。"}
    success = delete_history_ids(ids)
    if success:
        return {"message": f"成功删除 {len(ids)} 条记录。"}
    else:
        return {"error": "删除操作失败。"}, 500

