import argparse
import torch
from transformers import AutoModelForObjectDetection, TrainingArguments, Trainer, AutoImageProcessor, TrainerCallback, TrainingArguments
from dataloader import DETRDataLoader
import os
import shutil
import threading
import time
import evaluate
from tqdm import tqdm
from torchvision.datasets import CocoDetection
import torchvision
from huggingface_hub import login
from PIL import Image
import json
import matplotlib.pyplot as plt

def parse_args():
    parser = argparse.ArgumentParser(description='Train DETR model with a custom dataset.')
    
    # Model save/load parameters
    parser.add_argument('--save_model_hub_id', type=str, default='ARG-NCTU', help='Save model to Hugging Face Hub ID')
    parser.add_argument('--save_model_repo_id', type=str, default='detr-resnet-50-finetuned-600-epochs-Kaohsiung-Port-dataset', help='Save model to Hugging Face repository ID')
    parser.add_argument('--load_model_hub_id', type=str, default='facebook', help='Load model from Hugging Face Hub ID')
    parser.add_argument('--load_model_repo_id', type=str, default='detr-resnet-50', help='Load model from Hugging Face repository ID')

    # Dataset parameters
    parser.add_argument('--dataset_hub_id', type=str, default='ARG-NCTU', help='Dataset Hugging Face Hub ID')
    parser.add_argument('--dataset_repo_id', type=str, default='Kaohsiung_Port_dataset_2024', help='Dataset Hugging Face repository ID')
    parser.add_argument('--dataset_format', type=str, choices=['jsonl', 'parquet'], default='parquet', help='Dataset format')

    # Training parameters
    parser.add_argument('--epoch', type=int, default=600, help='Number of training epochs')
    parser.add_argument('--batch_size', type=int, default=8, help='Batch size for training')
    parser.add_argument('--learning_rate', type=float, default=1e-5, help='Learning rate')
    parser.add_argument('--weight_decay', type=float, default=1e-4, help='Weight decay')
    parser.add_argument('--logging_steps', type=int, default=50, help='Logging steps interval')
    parser.add_argument('--save_total_limit', type=int, default=100, help='Total limit for model checkpoints')

    # Other parameters
    parser.add_argument('--classes_path', type=str, default='data/Kaohsiung_Port_classes.txt', help='Path to class labels file')
    parser.add_argument('--image_height', type=int, default=480, help='Image height')
    parser.add_argument('--image_width', type=int, default=1920, help='Image width')

    parser.add_argument('--device', type=str, default='cuda' if torch.cuda.is_available() else 'cpu', help='Device (default: cuda if available)')

    parser.add_argument('--push_every_n_epochs', type=int, default=1, help='Number of epochs after which to push the model to the hub')
    parser.add_argument('--validate_every_n_epochs', type=int, default=5, help='Number of epochs after which to perform validation')

    return parser.parse_args()

# Custom Trainer class to handle custom push logic
class CustomTrainer(Trainer):
    def __init__(self, *args, val_coco_dataset=None, image_processor=None, id2label=None, **kwargs):
        self.val_coco_dataset = val_coco_dataset
        self.image_processor = image_processor
        self.id2label = id2label
        self.validation_history = {
            "iou_bbox": {}
        }
        super().__init__(*args, **kwargs)

    def perform_validation(self):
        module = evaluate.load("ybelkada/cocoevaluate", coco=self.val_coco_dataset.coco)

        # Create DataLoader for validation dataset
        val_dataloader = torch.utils.data.DataLoader(
            self.val_coco_dataset,
            batch_size=self.args.per_device_eval_batch_size,
            shuffle=False,
            num_workers=self.args.dataloader_num_workers,
            collate_fn=self.data_collator,
        )

        # Perform evaluation
        self.model.eval()
        with torch.no_grad():
            for batch in tqdm(val_dataloader):
                batch = {k: v.to(self.args.device) if isinstance(v, torch.Tensor) else v for k, v in batch.items()}
                outputs = self.model(pixel_values=batch["pixel_values"], pixel_mask=batch["pixel_mask"])
                orig_target_sizes = torch.stack([target["orig_size"] for target in batch["labels"]], dim=0)
                results = self.image_processor.post_process(outputs, orig_target_sizes)
                module.add(prediction=results, reference=batch["labels"])
        self.model.train()  # Set model back to training mode

        # Compute and print evaluation results
        results = module.compute()
        print(f"Validation results at epoch {int(self.state.epoch)}: {results}")

        for metric, value in results["iou_bbox"].items():
            if metric not in self.validation_history["iou_bbox"]:
                self.validation_history["iou_bbox"][metric] = []
            self.validation_history["iou_bbox"][metric].append(value)

        with open("validation_history.json", "w") as f:
            json.dump(self.validation_history, f, indent=4)
            
class EpochActionsCallback(TrainerCallback):
    """
    - every_push: push the model to the hub every epoch
    - every_validation: perform validation every 5 epochs
    """
    def __init__(self, trainer, every_push=1, every_val=5):
        super().__init__()
        self.trainer = trainer
        self.every_push = every_push
        self.every_val = every_val

    def on_epoch_end(self, args, state, control, **kwargs):
        epoch = int(state.epoch + 1e-6)  # Convert to integer epoch number

        if epoch % self.every_push == 0 and self.trainer.is_world_process_zero():
            tqdm.write(f"Pushing model to the hub at epoch {epoch}...")
            self.trainer.push_to_hub(commit_message=f"Checkpoint at epoch {epoch}")

        if epoch % self.every_val == 0:
            tqdm.write(f"Performing validation at epoch {epoch}...")
            self.trainer.perform_validation()

# Function to find the latest checkpoint
def get_latest_checkpoint(output_dir):
    checkpoints = [d for d in os.listdir(output_dir) if d.startswith("checkpoint-")]
    if checkpoints:
        # Sort checkpoints based on the epoch number and return the latest one
        checkpoints = sorted(checkpoints, key=lambda x: int(x.split("-")[-1]))
        latest_checkpoint = checkpoints[-1]
        print(f"Resuming from the latest checkpoint: {latest_checkpoint}")
        return os.path.join(output_dir, latest_checkpoint)
    else:
        return None

# Function to clean checkpints
def start_checkpoint_cleaner(output_dir, save_total_limit, interval_sec=300):
    def cleaner_loop():
        while True:
            try:
                checkpoints = [d for d in os.listdir(output_dir) if d.startswith("checkpoint-")]
                checkpoints = sorted(checkpoints, key=lambda x: int(x.split("-")[-1]))
                if len(checkpoints) > save_total_limit:
                    for ckpt_to_remove in checkpoints[:-save_total_limit]:
                        full_path = os.path.join(output_dir, ckpt_to_remove)
                        print(f"[CheckpointCleaner] Removing old checkpoint: {full_path}")
                        shutil.rmtree(full_path, ignore_errors=True)
                time.sleep(interval_sec)
            except Exception as e:
                print(f"[CheckpointCleaner] Error: {e}")
                time.sleep(interval_sec)

    cleaner_thread = threading.Thread(target=cleaner_loop, daemon=True)
    cleaner_thread.start()

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

def plot_validation_history(history_file = "validation_history.json"):
    # Load validation history from JSON file
    with open(history_file, "r") as f:
        history = json.load(f)

    # Plot each metric
    for metric, values in history["iou_bbox"].items():
        plt.plot(range(1, len(values) + 1), values, label=metric, marker='o')

    plt.xlabel("Epoch")
    plt.ylabel("Score")
    plt.title("Validation IoU Metrics Over Epochs")
    plt.legend()
    plt.grid(True)
    plt.savefig("validation_iou_metrics.png")
    plt.show()

def main():
    args = parse_args()

    # Initialize dataset using the `DataLoader` class
    dataloader = DETRDataLoader(
        dataset_format=args.dataset_format,
        image_height=args.image_height,
        image_width=args.image_width,
        load_model_hub_id=args.load_model_hub_id,
        load_model_repo_id=args.load_model_repo_id,
        dataset_hub_id=args.dataset_hub_id,
        dataset_repo_id=args.dataset_repo_id,
        classes_path=args.classes_path,
    )

    # Get dataset, collate function, and image processor
    train_dataset = dataloader.dataset["train"]
    val_dataset_hf = dataloader.dataset["validation"]
    collate_fn = dataloader.collate_fn
    image_processor = dataloader.image_processor
    # print(dataloader.id2label)

    print("Preparing validation dataset in COCO format (one-time operation)...")
    val_coco_output_path, val_coco_anno_path = save_annotation_file_images(val_dataset_hf, dataloader.id2label)
    val_coco_dataset = CocoDetection(val_coco_output_path, image_processor, val_coco_anno_path)
    print("Validation dataset prepared.")

    # Load model using AutoModelForObjectDetection
    model = AutoModelForObjectDetection.from_pretrained(
        f"{args.load_model_hub_id}/{args.load_model_repo_id}",
        ignore_mismatched_sizes=True,
        id2label=dataloader.id2label,
        label2id=dataloader.label2id,
    )

    # Set correct image size for DETR
    model.config.image_size = (args.image_height, args.image_width)  

    # Training arguments
    training_args = TrainingArguments(
        output_dir=args.save_model_repo_id,
        per_device_train_batch_size=args.batch_size,
        num_train_epochs=args.epoch,
        fp16=False,
        save_steps=len(train_dataset) // args.batch_size,
        logging_steps=args.logging_steps,
        learning_rate=args.learning_rate,
        weight_decay=args.weight_decay,
        save_strategy="no",
        evaluation_strategy="no",
        save_total_limit=1000,
        remove_unused_columns=False,
        push_to_hub=True,
        hub_model_id=f"{args.save_model_hub_id}/{args.save_model_repo_id}",
    )

    start_checkpoint_cleaner(training_args.output_dir, args.save_total_limit, interval_sec=300)

    # Initialize Trainer with collate function, etc.
    trainer = CustomTrainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        val_coco_dataset=val_coco_dataset,
        image_processor=image_processor,
        data_collator=collate_fn,
        tokenizer=image_processor,
        id2label=dataloader.id2label,
    )

    trainer.add_callback(EpochActionsCallback(trainer, every_push=args.push_every_n_epochs, every_val=args.validate_every_n_epochs))

    # Check if any checkpoint exists in the output directory
    latest_checkpoint = get_latest_checkpoint(training_args.output_dir)

    # Resume training from the latest checkpoint or start from scratch
    if latest_checkpoint:
        print("Resuming from the latest checkpoint...")
        trainer.train(resume_from_checkpoint=latest_checkpoint)
    else:
        print("Starting training from scratch...")
        trainer.train()

    trainer.push_to_hub(commit_message=f"{args.save_model_repo_id} trained for {args.epoch} epochs")


if __name__ == '__main__':
    login(token=os.environ["HUGGINGFACE_TOKEN"])
    main()

    plot_validation_history()

# Usage:
# python3 train.py --save_model_hub_id ARG-NCTU --save_model_repo_id detr-resnet-50-finetuned-600-epochs-Kaohsiung-Port-dataset --load_model_hub_id facebook --load_model_repo_id detr-resnet-50 --dataset_hub_id ARG-NCTU --dataset_repo_id Kaohsiung_Port_dataset_2024 --dataset_format parquet --epoch 600 --batch_size 8 --learning_rate 1e-5 --weight_decay 1e-4 --logging_steps 50 --save_total_limit 100 --classes_path data/Kaohsiung_Port_classes.txt --image_height 480 --image_width 1920 --device cuda

# python3 train.py --save_model_hub_id ARG-NCTU --save_model_repo_id detr-resnet-50-finetuned-600-epochs-KS-Buoy-dataset --load_model_hub_id facebook --load_model_repo_id detr-resnet-50 --dataset_hub_id ARG-NCTU --dataset_repo_id KS_Buoy_dataset_2025 --dataset_format parquet --epoch 600 --batch_size 8 --learning_rate 1e-5 --weight_decay 1e-4 --logging_steps 50 --save_total_limit 100 --classes_path data/KS_Buoy_classes.txt --image_height 480 --image_width 1920 --device cuda

# python3 train.py --save_model_hub_id ARG-NCTU --save_model_repo_id detr-resnet-50-finetuned-20-epochs-Boat-dataset-0314 --load_model_hub_id facebook --load_model_repo_id detr-resnet-50 --dataset_hub_id ARG-NCTU --dataset_repo_id Boat_dataset_2024 --dataset_format jsonl --epoch 20 --batch_size 8 --learning_rate 1e-5 --weight_decay 1e-4 --logging_steps 50 --save_total_limit 100 --classes_path data/boat_classes.txt --image_height 480 --image_width 640 --device cuda

# python3 train.py --save_model_hub_id ARG-NCTU --save_model_repo_id detr-resnet-50-finetuned-600-epochs-GuardBoat-dataset --load_model_hub_id ARG-NCTU --load_model_repo_id detr-resnet-50-finetuned-20-epochs-boat-dataset --dataset_hub_id ARG-NCTU --dataset_repo_id GuardBoat_dataset_2025 --dataset_format parquet --epoch 600 --batch_size 8 --learning_rate 1e-5 --weight_decay 1e-4 --logging_steps 50 --save_total_limit 100 --classes_path data/GuardBoat_classes.txt --image_height 480 --image_width 1920 --device cuda
