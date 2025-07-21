# 文件名: video_retriever.py (已更新)
# 描述: 增加了相似度阈值过滤，并使用 ffmpeg 转码视频以便在浏览器中播放

import os
import cv2
import torch
import numpy as np
import subprocess
import logging
from PIL import Image
from transformers import CLIPTokenizer, CLIPTextModelWithProjection, CLIPVisionModelWithProjection
from torchvision.transforms import Compose, Resize, CenterCrop, ToTensor, Normalize, InterpolationMode


class VideoRetriever:
    """
    一个封装了 clip4clip 模型用于视频片段检索的类。
    """

    def __init__(self, model_name: str = "Searchium-ai/clip4clip-webvid150k"):
        self.logger = logging.getLogger(self.__class__.__name__)
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.logger.info(f"Initializing VideoRetriever on device: {self.device}")

        try:
            self.text_model = CLIPTextModelWithProjection.from_pretrained(model_name)
            self.vision_model = CLIPVisionModelWithProjection.from_pretrained(model_name)
            self.tokenizer = CLIPTokenizer.from_pretrained(model_name)

            self.text_model.to(self.device)
            self.vision_model.to(self.device)
            self.vision_model.eval()
            self.logger.info(f"Successfully loaded model '{model_name}' to {self.device}.")
        except Exception as e:
            self.logger.error(f"Error loading model from HuggingFace: {e}")
            raise

        self.preprocessor = Compose([
            Resize(224, interpolation=InterpolationMode.BICUBIC),
            CenterCrop(224),
            ToTensor(),
            Normalize((0.48145466, 0.4578275, 0.40821073), (0.26862954, 0.26130258, 0.27577711)),
        ])

    def _get_text_embedding(self, text: str) -> torch.Tensor:
        with torch.no_grad():
            inputs = self.tokenizer(text=text, return_tensors="pt")
            inputs = {k: v.to(self.device) for k, v in inputs.items()}
            text_outputs = self.text_model(**inputs)
            text_embedding = text_outputs.text_embeds
            text_embedding /= text_embedding.norm(dim=-1, keepdim=True)
        return text_embedding

    def _get_video_segment_embedding(self, frames: list) -> torch.Tensor:
        if not frames: return torch.empty(0)
        sample_indices = np.linspace(0, len(frames) - 1, num=16, dtype=int)
        sampled_frames = [frames[i] for i in sample_indices]
        processed_frames = torch.stack(
            [self.preprocessor(Image.fromarray(frame).convert("RGB")) for frame in sampled_frames]
        ).to(self.device)
        with torch.no_grad():
            visual_outputs = self.vision_model(pixel_values=processed_frames)
            video_embedding_segment = visual_outputs.image_embeds
        video_embedding_segment /= video_embedding_segment.norm(dim=-1, keepdim=True)
        video_embedding_segment = torch.mean(video_embedding_segment, dim=0)
        video_embedding_segment /= video_embedding_segment.norm(dim=-1, keepdim=True)
        return video_embedding_segment

    def _save_segment(self, video_path: str, output_path: str, start_frame: int, end_frame: int, fps: float):
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            self.logger.error(f"Save Error: Cannot open video {video_path}")
            return False
        w, h = int(cap.get(3)), int(cap.get(4))
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter(output_path, fourcc, fps, (w, h))
        cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)
        for _ in range(end_frame - start_frame):
            ret, frame = cap.read()
            if not ret: break
            out.write(frame)
        cap.release()
        out.release()
        self.logger.info(f"Successfully saved segment to: {output_path}")
        return True

    # ==================== 【新增】视频转码函数 ====================
    def _transcode_to_h264(self, src_path: str) -> str:
        """使用ffmpeg将视频转码为浏览器兼容的H.264格式。"""
        from pathlib import Path
        p = Path(src_path)
        dst_path = p.with_name(f"{p.stem}_h264.mp4")
        command = ["ffmpeg", "-i", str(p), "-c:v", "libx264", "-pix_fmt", "yuv420p", "-preset", "veryfast", "-y",
                   str(dst_path)]
        try:
            subprocess.run(command, check=True, stdout=subprocess.DEVNULL, stderr=subprocess.PIPE)
            self.logger.info(f"Transcoding successful. Output: {dst_path}")
            # 删除未转码的原始片段
            if os.path.exists(src_path):
                os.remove(src_path)
            return str(dst_path)
        except (FileNotFoundError, subprocess.CalledProcessError) as e:
            self.logger.error(f"FFmpeg transcoding failed: {e}. Returning original file.")
            return src_path  # 转码失败则返回原始文件

    # =============================================================

    def search_and_save_top_segments(self, video_path: str, text_query: str, top_n: int = 1,
                                     segment_duration: float = 5.0, similarity_threshold: float = 0.2) -> list:
        text_embedding = self._get_text_embedding(text_query)
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            self.logger.error(f"Search Error: Cannot open video {video_path}")
            return []

        fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        frames_per_segment = int(fps * segment_duration)
        self.logger.info(f"Processing video: FPS={fps:.2f}, Total Frames={total_frames}")

        results = []
        for start_frame in range(0, total_frames, frames_per_segment):
            end_frame = min(start_frame + frames_per_segment, total_frames)
            current_segment_frames = []
            cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)
            for _ in range(end_frame - start_frame):
                ret, frame = cap.read()
                if not ret: break
                current_segment_frames.append(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
            if not current_segment_frames: continue
            video_embedding = self._get_video_segment_embedding(current_segment_frames)
            similarity = torch.matmul(text_embedding, video_embedding.T).item()
            results.append({"start_frame": start_frame, "end_frame": end_frame, "similarity": similarity})
        cap.release()

        if not results: return []

        # ==================== 【修改点】根据阈值过滤并排序 ====================
        filtered_results = [r for r in results if r['similarity'] >= similarity_threshold]
        if not filtered_results:
            self.logger.warning(f"No segments found above similarity threshold of {similarity_threshold}")
            return []

        sorted_results = sorted(filtered_results, key=lambda x: x['similarity'], reverse=True)
        top_matches = sorted_results[:top_n]
        # ====================================================================

        output_data = []
        for i, match in enumerate(top_matches):
            from uuid import uuid4
            from pathlib import Path
            import tempfile

            temp_filename = f"clip_result_{uuid4().hex[:8]}.mp4"
            temp_path = str(Path(tempfile.gettempdir()) / temp_filename)

            if self._save_segment(video_path, temp_path, match['start_frame'], match['end_frame'], fps):
                # 【修改点】对保存的片段进行转码
                final_path = self._transcode_to_h264(temp_path)
                final_filename = Path(final_path).name

                output_data.append({
                    "rank": i + 1,
                    "similarity": match['similarity'],
                    "start_time": match['start_frame'] / fps,
                    "end_time": match['end_frame'] / fps,
                    "video_url": f"/temp/{final_filename}"
                })
        return output_data
