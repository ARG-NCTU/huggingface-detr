import argparse
import os
import json
import torch
import torchvision
from transformers import AutoImageProcessor, AutoModelForObjectDetection
from PIL import Image
from tqdm import tqdm
import evaluate
from dataloader import DETRDataLoader
import os
from huggingface_hub import login

def parse_args():
    parser = argparse.ArgumentParser(description='Evaluate DETR model with a custom dataset.')
    
    # Model save/load parameters
    parser.add_argument('--hub_id', type=str, default='ARG-NCTU', help='Hugging Face Hub ID')
    parser.add_argument('--repo_id', type=str, default='detr-resnet-50-finetuned-600-epochs-Kaohsiung-Port-dataset', help='Model repository ID')
    
    # Dataset parameters
    parser.add_argument('--dataset_hub_id', type=str, default='ARG-NCTU', help='Dataset Hugging Face Hub ID')
    parser.add_argument('--dataset_repo_id', type=str, default='Kaohsiung_Port_dataset_2024', help='Dataset Hugging Face repository ID')
    parser.add_argument('--dataset_format', type=str, choices=['jsonl', 'parquet'], default='parquet', help='Dataset format')
    
    # Other parameters
    parser.add_argument('--classes_path', type=str, default='data/Kaohsiung_Port_classes.txt', help='Path to class labels file')
    parser.add_argument('--image_height', type=int, default=480, help='Image height')
    parser.add_argument('--image_width', type=int, default=1920, help='Image width')
    parser.add_argument('--batch_size', type=int, default=8, help='Batch size for evaluation')
    parser.add_argument('--num_workers', type=int, default=4, help='Number of workers for DataLoader')
    parser.add_argument('--device', type=str, default='cuda' if torch.cuda.is_available() else 'cpu', help='Device (default: cuda if available)')
    
    return parser.parse_args()

# format annotations the same as for training, no need for data augmentation
def val_formatted_anns(image_id, objects):
    annotations = []
    for i in range(0, len(objects["id"])):
        new_ann = {
            "id": objects["id"][i],
            "category_id": objects["category"][i],
            "iscrowd": 0,   # Assume no crowd annotations
            "image_id": image_id,
            "area": objects["area"][i],
            "bbox": objects["bbox"][i],
        }
        annotations.append(new_ann)

    return annotations
    
def save_annotation_file_images(dataset, id2label, mode="val"):
    output_json = {}
    path_output = f"{os.getcwd()}/output/"

    # Create output directory if it doesn't exist
    if not os.path.exists(path_output):
        os.makedirs(path_output)

    # Define annotation file path
    path_anno = os.path.join(path_output, "boat_ann_val.json" if mode == "val" else "boat_ann_val_real.json")
    categories_json = [{"supercategory": "none", "id": id, "name": id2label[id]} for id in id2label]
    output_json["images"] = []
    output_json["annotations"] = []
    
    #Process each example in the dataset
    for example in dataset:
        ann = val_formatted_anns(example["image_id"], example["objects"])
        if not os.path.exists(example["image_path"]):
            continue
        image_example = Image.open(example["image_path"])
        output_json["images"].append(
            {
                "id": example["image_id"],
                "width": image_example.width,
                "height": image_example.height,
                "file_name": f"{example['image_id']}.png",
            }
        )
        output_json["annotations"].extend(ann)
    output_json["categories"] = categories_json

    # Save annotations to JSON file
    with open(path_anno, "w") as file:
        json.dump(output_json, file, ensure_ascii=False, indent=4)

    # Save images to the output directory
    for image_path, img_id in zip(dataset["image_path"], dataset["image_id"]):
        if not os.path.exists(image_path):
            continue
        im = Image.open(image_path)
        path_img = os.path.join(path_output, f"{img_id}.png")
        im.save(path_img)

    return path_output, path_anno

class CocoDetection(torchvision.datasets.CocoDetection):
    def __init__(self, img_folder, feature_extractor, ann_file):
        super().__init__(img_folder, ann_file)
        self.feature_extractor = feature_extractor

    def __getitem__(self, idx):
        img, target = super(CocoDetection, self).__getitem__(idx)
        image_id = self.ids[idx]
        target = {"image_id": image_id, "annotations": target}
        encoding = self.feature_extractor(images=img, annotations=target, return_tensors="pt")
        pixel_values = encoding["pixel_values"].squeeze()
        target = encoding["labels"][0]
        return {"pixel_values": pixel_values, "labels": target}

def main():
    args = parse_args()
    dataloader = DETRDataLoader(
        dataset_format=args.dataset_format,
        image_height=args.image_height,
        image_width=args.image_width,
        dataset_hub_id=args.dataset_hub_id,
        dataset_repo_id=args.dataset_repo_id,
        classes_path=args.classes_path,
    )
    eval_dataset = dataloader.dataset["validation"]
    collate_fn = dataloader.collate_fn
    im_processor = AutoImageProcessor.from_pretrained(f"{args.hub_id}/{args.repo_id}")
    path_output, path_anno = save_annotation_file_images(eval_dataset, dataloader.id2label)
    test_ds_coco_format = CocoDetection(path_output, im_processor, path_anno)
    model = AutoModelForObjectDetection.from_pretrained(f"{args.hub_id}/{args.repo_id}").to(args.device)
    module = evaluate.load("ybelkada/cocoevaluate", coco=test_ds_coco_format.coco)
    val_dataloader = torch.utils.data.DataLoader(
        test_ds_coco_format, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers, collate_fn=collate_fn
    )
    # Perform evaluation
    with torch.no_grad():
        for idx, batch in enumerate(tqdm(val_dataloader)):
            batch = {k: v.to(args.device) if isinstance(v, torch.Tensor) else v for k, v in batch.items()}
            pixel_values = batch["pixel_values"]
            pixel_mask = batch["pixel_mask"]
            labels = batch["labels"]
            
            # forward pass
            outputs = model(pixel_values=pixel_values, pixel_mask=pixel_mask)
    
            orig_target_sizes = torch.stack([target["orig_size"] for target in labels], dim=0)
            results = im_processor.post_process(outputs, orig_target_sizes)  # convert outputs of model to COCO api
    
            module.add(prediction=results, reference=labels)
            del batch
    # Compute and print evaluation results
    results = module.compute()
    print(results)
    # Save results to a txt file
    with open("results.txt", "w") as f:
        f.write(str(results))

if __name__ == '__main__':
    login(token=os.environ["HUGGINGFACE_TOKEN"])
    main()

# Usage:
# python3 eval.py --hub_id ARG-NCTU --repo_id detr-resnet-50-finetuned-600-epochs-Kaohsiung-Port-dataset --dataset_hub_id ARG-NCTU --dataset_repo_id Kaohsiung_Port_dataset_2024 --dataset_format parquet --classes_path data/Kaohsiung_Port_classes.txt --image_height 480 --image_width 1920 --batch_size 8 --num_workers 4 --device cuda

# python3 eval.py --hub_id ARG-NCTU --repo_id detr-resnet-50-finetuned-600-epochs-KS-Buoy-dataset --dataset_hub_id ARG-NCTU --dataset_repo_id KS_Buoy_dataset_2025 --dataset_format parquet --classes_path data/KS_Buoy_classes.txt --image_height 480 --image_width 1920 --batch_size 8 --num_workers 4 --device cuda

# python3 eval.py --hub_id ARG-NCTU --repo_id detr-resnet-50-finetuned-20-epochs-Boat-dataset-0314 --dataset_hub_id ARG-NCTU --dataset_repo_id Boat_dataset_2024 --dataset_format jsonl --classes_path data/boat_classes.txt --image_height 480 --image_width 640 --batch_size 8 --num_workers 4 --device cuda
