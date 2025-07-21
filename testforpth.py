import argparse
import cv2
import torch
import os
import os.path as op
from PIL import Image
from ultralytics import YOLO
from utils.iotools import load_train_configs
from utils.logger import setup_logger
from model import build_model as build_irra_model
from utils.checkpoint import Checkpointer
from datasets.bases import tokenize
from utils.simple_tokenizer import SimpleTokenizer
from datasets.build import build_transforms


def process_single_image(frame, yolo_model, irra_model, text_features, irra_img_transforms, cli_args, logger):
    """
    处理单帧图像的函数, 供图片和视频模式复用
    """
    device = text_features.device
    logger.info(f"Processing a single image/frame...")
    yolo_results = yolo_model(frame, verbose=False)

    detected_persons_crops_pil = []
    detected_persons_boxes = []

    for result in yolo_results:
        for box in result.boxes:
            cls_id = int(box.cls[0])
            if yolo_model.names[cls_id].lower() == 'person':
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                if x1 >= x2 or y1 >= y2: continue
                crop_np = frame[y1:y2, x1:x2]
                if crop_np.size == 0: continue
                crop_pil = Image.fromarray(cv2.cvtColor(crop_np, cv2.COLOR_BGR2RGB))
                detected_persons_crops_pil.append(crop_pil)
                detected_persons_boxes.append((x1, y1, x2, y2))

    annotated_frame = frame.copy()

    if detected_persons_crops_pil:
        transformed_crops_batch = []
        for crop_pil_img in detected_persons_crops_pil:
            transformed_img = irra_img_transforms(crop_pil_img)
            transformed_crops_batch.append(transformed_img)

        if transformed_crops_batch:
            crops_tensor_batch = torch.stack(transformed_crops_batch).to(device)
            with torch.no_grad():
                batch_img_feats = irra_model.encode_image(crops_tensor_batch)
            batch_img_feats = batch_img_feats / batch_img_feats.norm(dim=-1, keepdim=True)
            similarities = text_features @ batch_img_feats.T
            similarities = similarities.squeeze(0)

            if similarities.numel() > 0:
                best_similarity_score, best_idx = torch.max(similarities, dim=0)
                best_similarity_score = best_similarity_score.item()
                best_idx = best_idx.item()

                if best_similarity_score >= cli_args.similarity_threshold:
                    x1, y1, x2, y2 = detected_persons_boxes[best_idx]
                    cv2.rectangle(annotated_frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                    text_to_display = f"Best Match: {best_similarity_score:.2f}"
                    text_y_pos = y1 - 10 if y1 > 20 else y1 + 15
                    cv2.putText(annotated_frame, text_to_display, (x1, text_y_pos),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
                    logger.debug(
                        f"Best match person at {detected_persons_boxes[best_idx]} with similarity {best_similarity_score:.2f}")
                else:
                    logger.debug(
                        f"Best match similarity {best_similarity_score:.2f} (below threshold {cli_args.similarity_threshold})")
            else:
                logger.debug("No persons processed for similarity calculation.")

    return annotated_frame


def main(cli_args):
    # --- 模型加载 (与之前相同) ---
    args = load_train_configs(cli_args.config_file)
    args.training = False
    logger = setup_logger('IRRA_Search', save_dir=args.output_dir, if_train=args.training)
    logger.info("Loaded IRRA Configs: %s", args)
    logger.info("CLI Args: %s", cli_args)
    device = "cuda" if torch.cuda.is_available() else "cpu"

    try:
        num_classes_for_model = args.num_classes if hasattr(args, 'num_classes') else 3701
        irra_model = build_irra_model(args, num_classes=num_classes_for_model)
    except AttributeError as e:
        logger.warning(f"AttributeError during model build: {e}. Trying with default num_classes=1.")
        irra_model = build_irra_model(args, num_classes=1)

    checkpointer = Checkpointer(irra_model)
    model_path = op.join(args.output_dir, 'best.pth')
    if not op.exists(model_path):
        logger.error(f"IRRA model checkpoint not found at {model_path}.")
        return
    checkpointer.load(f=model_path)
    irra_model.to(device)
    irra_model.eval()
    logger.info(f"Loaded IRRA model from {model_path}")

    try:
        yolo_model = YOLO(cli_args.yolo_model_path)
        yolo_model.to(device)
        logger.info(f"Loaded YOLO model from {cli_args.yolo_model_path}")
    except Exception as e:
        logger.error(f"Error loading YOLO model: {e}")
        return

    # --- 文本特征提取 (与之前相同) ---
    text_length = args.text_length if hasattr(args, 'text_length') else 77
    irra_tokenizer = SimpleTokenizer()
    tokenized_query = tokenize(cli_args.text_query, irra_tokenizer, text_length=text_length, truncate=True).unsqueeze(
        0).to(device)
    with torch.no_grad():
        text_features = irra_model.encode_text(tokenized_query)
    text_features = text_features / text_features.norm(dim=-1, keepdim=True)
    logger.info(f"Generated text features for query: '{cli_args.text_query}'")

    img_h = args.img_size[0] if hasattr(args, 'img_size') and args.img_size else 384
    img_w = args.img_size[1] if hasattr(args, 'img_size') and args.img_size else 128
    irra_img_transforms = build_transforms(img_size=(img_h, img_w), is_train=False)

    if cli_args.video_path:
        cap = cv2.VideoCapture(cli_args.video_path)
        if not cap.isOpened():
            logger.error(f"Error: Could not open video {cli_args.video_path}")
            return

        frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fps = int(cap.get(cv2.CAP_PROP_FPS)) or 25

        output_path = cli_args.output_video_path
        output_dir_path = os.path.dirname(output_path)
        if output_dir_path and not os.path.exists(output_dir_path):
            os.makedirs(output_dir_path)

        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out_writer = cv2.VideoWriter(output_path, fourcc, fps, (frame_width, frame_height))
        logger.info(f"Processing video: {cli_args.video_path}. Output will be saved to: {output_path}")

        frame_idx = 0
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret: break
            frame_idx += 1
            if frame_idx % cli_args.process_every_n_frames != 0:
                out_writer.write(frame)
                continue

            annotated_frame = process_single_image(frame, yolo_model, irra_model, text_features, irra_img_transforms,
                                                   cli_args, logger)
            out_writer.write(annotated_frame)

        cap.release()
        out_writer.release()
        logger.info(f"Finished processing video. Output saved to {output_path}")

    elif cli_args.input_image_path:
        # --- 图片处理逻辑 ---
        img = cv2.imread(cli_args.input_image_path)
        if img is None:
            logger.error(f"Error: Could not read image {cli_args.input_image_path}")
            return

        output_path = cli_args.output_image_path
        output_dir_path = os.path.dirname(output_path)
        if output_dir_path and not os.path.exists(output_dir_path):
            os.makedirs(output_dir_path)

        logger.info(f"Processing image: {cli_args.input_image_path}. Output will be saved to: {output_path}")
        annotated_image = process_single_image(img, yolo_model, irra_model, text_features, irra_img_transforms,
                                               cli_args, logger)
        cv2.imwrite(output_path, annotated_image)
        logger.info(f"Finished processing image. Output saved to {output_path}")

    else:
        logger.error("No input specified. Please provide either --video_path or --input_image_path.")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="IRRA-YOLO Video/Image Person Search")
    # 修改--video_path为非必需
    parser.add_argument("--video_path", type=str, default=None,
                        help="Path to the input video file.")
    parser.add_argument("--output_video_path", type=str, default="output_searched_video.mp4",
                        help="Path to save the processed video with annotations.")
    # 新增图片输入/输出参数
    parser.add_argument("--input_image_path", type=str, default=None,
                        help="Path to the input image file.")
    parser.add_argument("--output_image_path", type=str, default="output_searched_image.jpg",
                        help="Path to save the processed image with annotations.")

    # 保持其他参数不变
    parser.add_argument("--config_file", type=str, required=True,
                        help="Path to IRRA model's training config YAML file.")
    parser.add_argument("--yolo_model_path", type=str, default="yolov8n.pt",
                        help="Path to YOLO model weights (.pt file).")
    parser.add_argument("--text_query", type=str, required=True,
                        help="Text description of the person to search for.")
    parser.add_argument("--similarity_threshold", type=float, default=0.15,
                        help="Similarity threshold for marking a person as a match.")
    parser.add_argument("--process_every_n_frames", type=int, default=1,
                        help="Process every Nth frame to speed up (for video).")

    cli_args = parser.parse_args()
    main(cli_args)