import argparse
import torch
from transformers import AutoModelForObjectDetection, TrainingArguments, Trainer, AutoImageProcessor, TrainerCallback, TrainingArguments
from transformers import YolosConfig, YolosForObjectDetection
from dataloader import DETRDataLoader, YolosDataLoader
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
from eval import save_annotation_file_images, val_formatted_anns, CocoDetection

def parse_args():
    parser = argparse.ArgumentParser(description='Train DETR model with a custom dataset.')
    
    # Model Selection
    parser.add_argument('--model_type', type=str, default='detr', help='Model type to use for training (e.g., detr, yolos, etc.)')

    # Model save/load parameters
    parser.add_argument('--save_model_hub_id', type=str, default='ARG-NCTU', help='Save model to Hugging Face Hub ID')
    parser.add_argument('--save_model_repo_id', type=str, default='detr-resnet-50-finetuned-600-epochs-TW-Marine-5cls-dataset', help='Save model to Hugging Face repository ID')
    parser.add_argument('--load_model_hub_id', type=str, default='facebook', help='Load model from Hugging Face Hub ID')
    parser.add_argument('--load_model_repo_id', type=str, default='detr-resnet-50', help='Load model from Hugging Face repository ID')

    # Dataset parameters
    parser.add_argument('--dataset_hub_id', type=str, default='ARG-NCTU', help='Dataset Hugging Face Hub ID')
    parser.add_argument('--dataset_repo_id', type=str, default='TW_Marine_5cls_dataset', help='Dataset Hugging Face repository ID')
    parser.add_argument('--dataset_format', type=str, choices=['jsonl', 'parquet'], default='parquet', help='Dataset format')
    
    # Training parameters
    parser.add_argument('--epoch', type=int, default=600, help='Number of training epochs')
    parser.add_argument('--batch_size', type=int, default=2, help='Batch size for training')
    parser.add_argument('--learning_rate', type=float, default=1e-5, help='Learning rate')
    parser.add_argument('--weight_decay', type=float, default=1e-4, help='Weight decay')
    parser.add_argument('--logging_steps', type=int, default=50, help='Logging steps interval')
    parser.add_argument('--save_total_limit', type=int, default=100, help='Total limit for model checkpoints')

    # Other parameters
    parser.add_argument('--classes_path', type=str, default='data/TW_Marine_5cls_classes.txt', help='Path to class labels file')
    parser.add_argument('--image_height', type=int, default=480, help='Image height')
    parser.add_argument('--image_width', type=int, default=1920, help='Image width')

    parser.add_argument('--device', type=str, default='cuda' if torch.cuda.is_available() else 'cpu', help='Device (default: cuda if available)')

    parser.add_argument('--push_every_n_epochs', type=int, default=1, help='Number of epochs after which to push the model to the hub')
    parser.add_argument('--validate_every_n_epochs', type=int, default=5, help='Number of epochs after which to perform validation')

    return parser.parse_args()

# Custom Trainer class to handle custom push logic
class CustomTrainer(Trainer):
    def __init__(self, *args, model_type='detr', val_coco_dataset=None, image_processor=None, id2label=None, output_dir=None, save_total_limit=None, **kwargs):
        self.model_type = model_type
        self.val_coco_dataset = val_coco_dataset
        self.image_processor = image_processor
        self.id2label = id2label
        self.output_dir = output_dir
        self.save_total_limit = save_total_limit
        self.validation_history = {
            "iou_bbox": {}
        }
        self.validation_history_file_path = os.path.join(self.output_dir, "validation_history.json")
        if os.path.exists(self.validation_history_file_path):
            try:
                with open(self.validation_history_file_path, "r") as f:
                    self.validation_history = json.load(f)
                print("[CustomTrainer] Loaded existing validation_history.json.")
            except Exception as e:
                print(f"[CustomTrainer] Failed to load validation_history.json: {e}")
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
                if self.model_type == 'detr':
                    outputs = self.model(pixel_values=batch["pixel_values"], pixel_mask=batch["pixel_mask"])
                elif self.model_type == 'yolos':
                    outputs = self.model(pixel_values=batch["pixel_values"])
                orig_target_sizes = torch.stack([target["orig_size"] for target in batch["labels"]], dim=0)
                results = self.image_processor.post_process(outputs, orig_target_sizes)
                module.add(prediction=results, reference=batch["labels"])
        self.model.train()  # Set model back to training mode

        # Compute and print evaluation results
        results = module.compute()
        print(f"Validation results at epoch {int(self.state.epoch)}: {results}")

        current_epoch = int(self.state.epoch)
        print(f"Validation results at epoch {current_epoch}: {results}")

        if "epoch" not in self.validation_history:
            self.validation_history["epoch"] = []

        if current_epoch in self.validation_history["epoch"]:
            val_idx = self.validation_history["epoch"].index(current_epoch)
        else:
            val_idx = len(self.validation_history["epoch"])
            self.validation_history["epoch"].append(current_epoch)

        last_log = self.state.log_history[-1] if self.state.log_history else {}
        training_loss = last_log.get("loss", None)

        if "training_loss" not in self.validation_history:
            self.validation_history["training_loss"] = []
        
        if val_idx < len(self.validation_history["training_loss"]):
            self.validation_history["training_loss"][val_idx] = training_loss
        else:
            self.validation_history["training_loss"].append(training_loss)

        for metric, value in results["iou_bbox"].items():
            if metric not in self.validation_history["iou_bbox"]:
                self.validation_history["iou_bbox"][metric] = []
            
            if val_idx < len(self.validation_history["iou_bbox"][metric]):
                self.validation_history["iou_bbox"][metric][val_idx] = value
            else:
                self.validation_history["iou_bbox"][metric].append(value)

        with open(self.validation_history_file_path, "w") as f:
            json.dump(self.validation_history, f, indent=4)


    # Function to clean checkpints
    def start_checkpoint_cleaner(self, interval_sec=3):
        try:
            checkpoints = [d for d in os.listdir(self.output_dir) if d.startswith("checkpoint-")]
            checkpoints = sorted(checkpoints, key=lambda x: int(x.split("-")[-1]))
            if len(checkpoints) > self.save_total_limit:
                for ckpt_to_remove in checkpoints[:-self.save_total_limit]:
                    full_path = os.path.join(self.output_dir, ckpt_to_remove)
                    print(f"[CheckpointCleaner] Removing old checkpoint: {full_path}")
                    shutil.rmtree(full_path, ignore_errors=True)
            time.sleep(interval_sec)
        except Exception as e:
            print(f"[CheckpointCleaner] Error: {e}")
            time.sleep(interval_sec)
            
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

        if epoch % self.every_val == 0 and epoch > 0:
            tqdm.write(f"Performing validation at epoch {epoch}...")
            self.trainer.perform_validation()
            self.trainer.start_checkpoint_cleaner(interval_sec=3)

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

def plot_single_metric(epochs, values, metric_name, save_path, ylabel="Value"):
    plt.figure()
    plt.plot(epochs, values, marker='o')
    plt.xlabel("Epoch")
    plt.ylabel(ylabel)
    plt.title(f"{metric_name} over Epochs")
    plt.grid(True)
    plt.savefig(save_path)
    plt.close()


def plot_validation_history(save_model_dir):
    # Load validation history
    with open(os.path.join(save_model_dir, "validation_history.json"), "r") as f:
        history = json.load(f)

    epochs = history.get("epoch", list(range(1, len(next(iter(history["iou_bbox"].values()))) + 1)))

    # Plot each AP metric
    iou_dir = os.path.join(save_model_dir, "validation_plots")
    os.makedirs(iou_dir, exist_ok=True)

    for metric, values in history["iou_bbox"].items():
        save_path = os.path.join(iou_dir, f"{metric.replace('/', '_')}.png")
        plot_single_metric(epochs, values, metric, save_path)

    # Plot training loss
    if "training_loss" in history:
        save_path = os.path.join(save_model_dir, "training_loss.png")
        plot_single_metric(epochs, history["training_loss"], "Training Loss", save_path, ylabel="Loss")

    print(f"Validation plots saved to {save_model_dir}")



def main():
    args = parse_args()

    # Initialize dataset using the appropriate DataLoader based on the model type
    if args.model_type == 'detr':
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
    elif args.model_type == 'yolos':
        dataloader = YolosDataLoader(
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
    if args.model_type == 'detr':
        model.config.image_size = (args.image_height, args.image_width)
    elif args.model_type == 'yolos':
        model.config.image_size = (800, 1333)

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
        save_strategy="epoch",
        evaluation_strategy="no",
        save_total_limit=1000,
        remove_unused_columns=False,
        push_to_hub=True,
        hub_model_id=f"{args.save_model_hub_id}/{args.save_model_repo_id}",
    )

    # Initialize Trainer with collate function, etc.
    trainer = CustomTrainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        model_type=args.model_type,
        val_coco_dataset=val_coco_dataset,
        image_processor=image_processor,
        data_collator=collate_fn,
        tokenizer=image_processor,
        id2label=dataloader.id2label,
        output_dir=args.save_model_repo_id,
        save_total_limit=args.save_total_limit,
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

    plot_validation_history(args.save_model_repo_id)


if __name__ == '__main__':
    login(token=os.environ["HUGGINGFACE_TOKEN"])
    main()

