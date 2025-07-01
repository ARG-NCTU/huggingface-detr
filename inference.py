import argparse
import os
import torch
import cv2
from transformers import AutoImageProcessor, AutoModelForObjectDetection
from PIL import Image, ImageDraw, ImageFont
from tqdm import tqdm
import numpy as np
from huggingface_hub import login

def parse_args():
    parser = argparse.ArgumentParser(description='Run inference with DETR model on an image or video.')
    
    # Model parameters
    parser.add_argument('--hub_id', type=str, default='ARG-NCTU', help='Hugging Face Hub ID')
    parser.add_argument('--repo_id', type=str, default='detr-resnet-50-finetuned-600-epochs-Kaohsiung-Port-dataset', help='Model repository ID')
    parser.add_argument('--input_path', type=str, required=True, help='Path to the input image or video')
    parser.add_argument('--output_path', type=str, default=None, help='Path to save the output image or video')
    parser.add_argument('--confidence_threshold', type=float, default=0.5, help='Confidence threshold for object detection')
    parser.add_argument('--device', type=str, default='cuda' if torch.cuda.is_available() else 'cpu', help='Device (default: cuda if available)')
    
    return parser.parse_args()

def load_model(hub_id, repo_id, device):
    """Load the DETR model and processor."""
    image_processor = AutoImageProcessor.from_pretrained(f"{hub_id}/{repo_id}")
    model = AutoModelForObjectDetection.from_pretrained(f"{hub_id}/{repo_id}").to(device)
    return model, image_processor

def detect_objects(model, image_processor, image, device, confidence_threshold):
    """Perform object detection on the input image."""
    inputs = image_processor(images=image, return_tensors="pt")
    inputs = {name: tensor.to(device) for name, tensor in inputs.items()}
    outputs = model(**inputs)
    target_sizes = torch.tensor([image.size[::-1]], device=device)
    results = image_processor.post_process_object_detection(
        outputs, threshold=confidence_threshold, target_sizes=target_sizes
    )[0]
    return results

def draw_detections(image, detections, model):
    """Draw bounding boxes and labels on the image."""
    draw = ImageDraw.Draw(image)
    try:
        # Load a custom font for better visualization
        font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", 20)
    except IOError:
        font = ImageFont.load_default()
        print("Warning: Custom font not found, using default font.")
    
    # Colors for bounding boxes
    colors = ["blue", "green", "red", "yellow", "purple", "orange", "cyan", "magenta",
                    "lime", "pink", "teal", "lavender", "brown", "beige", "maroon", "mint",
                    "olive", "apricot", "navy", "grey", "white", "black"]
    class_list = model.config.id2label.values()
    class_colors = {class_name: colors[i % len(colors)] for i, class_name in enumerate(class_list)}
    
    # Draw each detection
    for score, label, box in zip(detections["scores"], detections["labels"], detections["boxes"]):
        class_name = model.config.id2label[label.item()]
        box_color = class_colors.get(class_name, "white")
        x, y, x2, y2 = [round(i, 2) for i in box.tolist()]
        draw.rectangle((x, y, x2, y2), outline=box_color, width=2)
        bbox_area = int((x2 - x) * (y2 - y))
        draw.text((x, y), f"class: {class_name}, conf: {score:.2f}, area: {bbox_area}", fill=box_color, font=font)
    
    return image

def process_image(input_path, output_path, model, image_processor, device, confidence_threshold):
    """Process a single image for inference."""
    image = Image.open(input_path).convert("RGB")
    detections = detect_objects(model, image_processor, image, device, confidence_threshold)
    result_image = draw_detections(image, detections, model)
    result_image.save(output_path)
    print(f"Inference completed. Output saved at {output_path}")

def process_video(input_path, output_path, model, image_processor, device, confidence_threshold):
    """Process a video file for inference."""
    cap = cv2.VideoCapture(input_path)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
    
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        
        image = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
        detections = detect_objects(model, image_processor, image, device, confidence_threshold)
        result_image = draw_detections(image, detections, model)
        
        result_frame = cv2.cvtColor(np.array(result_image), cv2.COLOR_RGB2BGR)
        out.write(result_frame)
    
    cap.release()
    out.release()
    print(f"Video processing completed. Output saved at {output_path}")

def main():
    args = parse_args()
    model, image_processor = load_model(args.hub_id, args.repo_id, args.device)
    
    if args.input_path.lower().endswith(('.png', '.jpg', '.jpeg')):
        # Handle image input
        if args.output_path is None:
            args.output_path = os.path.join(os.path.dirname(args.input_path), "output.png")
        os.makedirs(os.path.dirname(args.output_path), exist_ok=True)
        process_image(args.input_path, args.output_path, model, image_processor, args.device, args.confidence_threshold)
    elif args.input_path.lower().endswith(('.mp4', '.avi', '.mov')):
        # Handle video input
        if args.output_path is None:
            args.output_path = os.path.join(os.path.dirname(args.input_path), "output.mp4")
        os.makedirs(os.path.dirname(args.output_path), exist_ok=True)
        process_video(args.input_path, args.output_path, model, image_processor, args.device, args.confidence_threshold)
    else:
        print("Unsupported file format. Please provide an image or video file.")

if __name__ == '__main__':
    login(token=os.environ["HUGGINGFACE_TOKEN"])
    main()

# Usage:
# python3 inference.py --hub_id ARG-NCTU --repo_id detr-resnet-50-finetuned-600-epochs-Kaohsiung-Port-dataset --input_path images/output_video_path1_1.png --output_path output.png --confidence_threshold 0.5

# python3 inference.py --hub_id ARG-NCTU --repo_id detr-resnet-50-finetuned-600-epochs-KS-Buoy-dataset --input_path source_videos/2025-03-14-19-18-00_stitched.mp4 --output_path output.mp4 --confidence_threshold 0.8

# python3 inference.py --hub_id ARG-NCTU --repo_id detr-resnet-50-finetuned-20-epochs-Boat-dataset-0314 --input_path source_videos/Multi_Boat.mp4 --output_path output_videos/Multi_Boat.mp4 --confidence_threshold 0.5