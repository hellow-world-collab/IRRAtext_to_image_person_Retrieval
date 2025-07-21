# 文件名: yolo_api.py (已更新)
# 描述: 作为主App，统一管理所有模型，并新增历史记录功能

import os, cv2, sys, asyncio, tempfile, uvicorn, logging
from contextlib import asynccontextmanager
from functools import lru_cache
from pathlib import Path
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from ultralytics import YOLO
from tqdm import tqdm

# --- 导入新的模型封装 ---
from person_searcher import PersonSearcher
from video_retriever import VideoRetriever

# —— 配置 ——
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
if sys.platform.startswith("win"):
    asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())

# —— 模型和配置路径 (集中管理) ——
YOLO_WEIGHTS = "yolov8n.pt"
IRRA_CONFIG = "logs/CUHK-PEDES/configs.yaml"
CLIP_MODEL = "Searchium-ai/clip4clip-webvid150k"

# —— 全局存储 ——
MODELS = {}
HISTORY_LOG = []  # 新增：用于存储操作历史的列表


# —— 工具函数 ——
def cleanup_files(paths: list[str]):
    for p in paths:
        if p and os.path.exists(p):
            try:
                os.remove(p)
                logging.info(f"已成功清理临时文件: {p}")
            except OSError as e:
                logging.warning(f"清理文件失败: {p}, 错误: {e}")


# —— 模型加载函数 ——
@lru_cache(maxsize=1)
def get_yolo_detector(path: str = YOLO_WEIGHTS) -> YOLO:
    # ... (此函数内容保持不变) ...
    if not os.path.exists(path): raise FileNotFoundError(path)
    m = YOLO(path);
    m.fuse();
    return m


# —— Lifespan 管理器 (统一加载所有模型) ——
@asynccontextmanager
async def lifespan(app: FastAPI):
    logging.info("Web App starting up...")

    # 1. 预加载 PersonSearcher (IRRA+YOLO)
    try:
        MODELS["person_searcher"] = PersonSearcher(irra_config_file=IRRA_CONFIG, yolo_model_path=YOLO_WEIGHTS)
        logging.info("PersonSearcher (IRRA+YOLO) preloaded successfully.")
    except Exception as e:
        logging.error(f"Failed to preload PersonSearcher: {e}", exc_info=True)

    # 2. 预加载 VideoRetriever (clip4clip)
    try:
        MODELS["video_retriever"] = VideoRetriever(model_name=CLIP_MODEL)
        logging.info("VideoRetriever (clip4clip) preloaded successfully.")
    except Exception as e:
        logging.error(f"Failed to preload VideoRetriever: {e}", exc_info=True)

    yield

    logging.info("Web App shutting down...")
    MODELS.clear()
    HISTORY_LOG.clear()


# —— FastAPI 实例 ——
app = FastAPI(
    title="Unified Vision API",
    version="5.0",
    lifespan=lifespan
)


# ... (旧的 /detect 相关路由和函数可以保留或删除，这里为简洁省略) ...

# ==================== 新增：历史记录 API ====================
@app.get("/history")
async def get_history():
    """返回存储在内存中的操作历史记录。"""
    return {"history": HISTORY_LOG}


@app.delete("/history")
async def clear_history():
    """清空历史记录。"""
    HISTORY_LOG.clear()
    return {"message": "历史记录已清空"}
# ==========================================================
