# 文件名: person_searcher.py (方案一最终版)
# 描述: 禁用了重捕获奖励，在目标丢失后总是依据原始文本描述进行搜索，以提高精度。

import os

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

import cv2
import torch
from PIL import Image
from ultralytics import YOLO
from utils.iotools import load_train_configs
from utils.logger import setup_logger
from model import build_model as build_irra_model
from utils.checkpoint import Checkpointer
from datasets.bases import tokenize
from utils.simple_tokenizer import SimpleTokenizer
from datasets.build import build_transforms


class PersonSearcher:
    """
    一个封装了YOLO和IRRA模型用于图文检索的类。
    """

    def __init__(self, irra_config_file: str, yolo_model_path: str):
        self.logger = setup_logger('IRRA_Search_Service', save_dir="logs", if_train=False)
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.logger.info(f"PersonSearcher is using device: {self.device}")
        self.irra_args = load_train_configs(irra_config_file)
        self.irra_args.training = False

        try:
            self.yolo_model = YOLO(yolo_model_path)
            self.yolo_model.to(self.device)
            self.logger.info(f"Loaded YOLO model from {yolo_model_path}")
        except Exception as e:
            self.logger.error(f"Error loading YOLO model: {e}")
            raise

        try:
            num_classes = self.irra_args.get('num_classes', 3701)
            self.irra_model = build_irra_model(self.irra_args, num_classes=num_classes)
            model_path = os.path.join(self.irra_args.output_dir, 'best.pth')
            if not os.path.exists(model_path):
                raise FileNotFoundError(f"IRRA model checkpoint not found at {model_path}.")
            checkpointer = Checkpointer(self.irra_model)
            checkpointer.load(f=model_path)
            self.irra_model.to(self.device)
            self.irra_model.eval()
            self.logger.info(f"Loaded IRRA model from {model_path}")
        except Exception as e:
            self.logger.error(f"Error loading IRRA model: {e}")
            raise

        self.irra_tokenizer = SimpleTokenizer()
        img_h = self.irra_args.get('img_h', 384)
        img_w = self.irra_args.get('img_w', 128)
        self.irra_img_transforms = build_transforms(img_size=(img_h, img_w), is_train=False)
        self.logger.info("PersonSearcher initialized successfully.")

    def _get_text_features(self, text_query: str):
        text_length = self.irra_args.get('text_length', 77)
        tokenized_query = tokenize(text_query, self.irra_tokenizer, text_length=text_length, truncate=True).unsqueeze(
            0).to(self.device)
        with torch.no_grad(), torch.amp.autocast(device_type="cuda" if self.device == "cuda" else "cpu"):
            text_features = self.irra_model.encode_text(tokenized_query)
        text_features /= text_features.norm(dim=-1, keepdim=True)
        return text_features

    def search_in_image(self, image_path: str, text_query: str, similarity_threshold: float, output_path: str):
        # search_in_image 方法保持不变
        self.logger.info(f"Starting search in image: {image_path}")
        frame = cv2.imread(image_path)
        if frame is None:
            self.logger.error(f"Could not read image: {image_path}")
            return False

        text_features = self._get_text_features(text_query)
        yolo_results = self.yolo_model(frame, verbose=False, conf=0.5)

        detected_persons_crops, detected_persons_boxes = [], []
        for result in yolo_results:
            for box in result.boxes:
                if self.yolo_model.names[int(box.cls[0])].lower() == 'person':
                    x1, y1, x2, y2 = map(int, box.xyxy[0])
                    if x1 >= x2 or y1 >= y2: continue
                    crop_pil = Image.fromarray(cv2.cvtColor(frame[y1:y2, x1:x2], cv2.COLOR_BGR2RGB))
                    detected_persons_crops.append(crop_pil)
                    detected_persons_boxes.append((x1, y1, x2, y2))

        annotated_frame = frame.copy()
        if detected_persons_crops:
            crops_tensor = torch.stack([self.irra_img_transforms(p) for p in detected_persons_crops]).to(self.device)
            with torch.no_grad(), torch.amp.autocast(device_type="cuda" if self.device == "cuda" else "cpu"):
                batch_img_feats = self.irra_model.encode_image(crops_tensor)

            batch_img_feats /= batch_img_feats.norm(dim=-1, keepdim=True)
            similarities = (text_features @ batch_img_feats.T).squeeze(0)

            if similarities.numel() > 0:
                best_score, best_idx = torch.max(similarities, dim=0)
                if best_score.item() >= similarity_threshold:
                    x1, y1, x2, y2 = detected_persons_boxes[best_idx.item()]
                    cv2.rectangle(annotated_frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                    label = f"Match: {best_score.item():.2f}"
                    cv2.putText(annotated_frame, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        cv2.imwrite(output_path, annotated_frame)
        self.logger.info(f"Image search complete. Output saved to {output_path}")
        return True

    def search_in_video(self, video_path: str, text_query: str, similarity_threshold: float, output_path: str,
                        process_every_n_frames: int = 10):
        self.logger.info(f"Starting search in video: {video_path}")
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            self.logger.error(f"Cannot open video: {video_path}")
            return False

        w, h = int(cap.get(3)), int(cap.get(4))
        fps = cap.get(cv2.CAP_PROP_FPS) or 25
        out_writer = cv2.VideoWriter(output_path, cv2.VideoWriter_fourcc(*'mp4v'), fps, (w, h))

        text_features = self._get_text_features(text_query)
        target_track_id = None
        identified_track_ids = set()

        last_known_target_features = None
        # RE_ACQUISITION_BONUS = 0.15 # 这个值不再被使用

        frame_idx = 0
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret: break

            annotated_frame = frame.copy()
            frame_idx += 1
            if frame_idx % process_every_n_frames == 0:
                yolo_results = self.yolo_model.track(frame, persist=True, classes=[0], conf=0.5, verbose=False)

                if target_track_id is not None and yolo_results[0].boxes.id is not None:
                    found_target = False
                    for box in yolo_results[0].boxes:
                        if box.id is not None and box.id.int().item() == target_track_id:
                            x1, y1, x2, y2 = map(int, box.xyxy[0])
                            cv2.rectangle(annotated_frame, (x1, y1), (x2, y2), (0, 255, 0), 3)
                            cv2.putText(annotated_frame, "Target Found", (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.8,
                                        (0, 255, 0), 2)
                            found_target = True
                            break
                    if not found_target:
                        self.logger.warning(f"Target with track ID {target_track_id} lost. Resuming search.")
                        target_track_id = None

                if target_track_id is None and yolo_results[0].boxes.id is not None:
                    crops, boxes, track_ids = [], [], []
                    for box in yolo_results[0].boxes:
                        track_id = box.id.int().item()
                        if track_id in identified_track_ids: continue
                        x1, y1, x2, y2 = map(int, box.xyxy[0])
                        if x1 >= x2 or y1 >= y2: continue
                        crop_pil = Image.fromarray(cv2.cvtColor(frame[y1:y2, x1:x2], cv2.COLOR_BGR2RGB))
                        crops.append(crop_pil)
                        boxes.append((x1, y1, x2, y2))
                        track_ids.append(track_id)

                    if crops:
                        crops_tensor = torch.stack([self.irra_img_transforms(p) for p in crops]).to(self.device)
                        with torch.no_grad(), torch.amp.autocast(
                                device_type="cuda" if self.device == "cuda" else "cpu"):
                            img_feats = self.irra_model.encode_image(crops_tensor)

                        img_feats /= img_feats.norm(dim=-1, keepdim=True)
                        text_similarities = (text_features @ img_feats.T).squeeze(0)

                        # ==================== 【核心修改：方案一】 ====================
                        # 直接使用文本相似度作为最终分数，不再添加视觉重捕获奖励，以避免错误传播。
                        final_similarities = text_similarities

                        if last_known_target_features is not None:
                            self.logger.info("Re-acquisition mode is active, but visual bonus is DISABLED.")
                        # ==========================================================

                        identified_track_ids.update(track_ids)

                        if final_similarities.numel() > 0:
                            best_score, best_idx = torch.max(final_similarities, dim=0)
                            if best_score.item() >= similarity_threshold:
                                target_track_id = track_ids[best_idx.item()]
                                # 仍然记录下新目标的特征，以备未来可能的逻辑调整
                                last_known_target_features = img_feats[best_idx]
                                self.logger.info(
                                    f"Found target! Track ID: {target_track_id}, Score: {best_score.item():.2f}")
                                x1, y1, x2, y2 = boxes[best_idx.item()]
                                cv2.rectangle(annotated_frame, (x1, y1), (x2, y2), (0, 255, 0), 3)
                                cv2.putText(annotated_frame, "Target Found", (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX,
                                            0.8, (0, 255, 0), 2)

            out_writer.write(annotated_frame)

        cap.release()
        out_writer.release()
        self.logger.info(f"Video search complete. Output saved to {output_path}")
        return True