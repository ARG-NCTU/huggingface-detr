import argparse
import torch
from transformers import AutoModelForObjectDetection, TrainingArguments, Trainer
from dataloader import DETRDataLoader
import os
import shutil
import threading
import time

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

    return parser.parse_args()

# Custom Trainer class to handle custom push logic
class CustomTrainer(Trainer):
    def on_epoch_end(self):
        super().on_epoch_end()
        # Push the model to the hub every epoch
        if self.state.epoch % 1 == 0:
            print(f"Pushing model to the hub at epoch {self.state.epoch}...")
            self.push_to_hub(commit_message=f"Checkpoint at epoch {int(self.state.epoch)}")

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
    collate_fn = dataloader.collate_fn
    image_processor = dataloader.image_processor

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
        data_collator=collate_fn,
        tokenizer=image_processor,
    )

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
    main()

# Usage:
# python3 train.py --save_model_hub_id ARG-NCTU --save_model_repo_id detr-resnet-50-finetuned-600-epochs-Kaohsiung-Port-dataset --load_model_hub_id facebook --load_model_repo_id detr-resnet-50 --dataset_hub_id ARG-NCTU --dataset_repo_id Kaohsiung_Port_dataset_2024 --dataset_format parquet --epoch 600 --batch_size 8 --learning_rate 1e-5 --weight_decay 1e-4 --logging_steps 50 --save_total_limit 100 --classes_path data/Kaohsiung_Port_classes.txt --image_height 480 --image_width 1920 --device cuda

# python3 train.py --save_model_hub_id ARG-NCTU --save_model_repo_id detr-resnet-50-finetuned-600-epochs-KS-Buoy-dataset --load_model_hub_id facebook --load_model_repo_id detr-resnet-50 --dataset_hub_id ARG-NCTU --dataset_repo_id KS_Buoy_dataset_2025 --dataset_format parquet --epoch 600 --batch_size 8 --learning_rate 1e-5 --weight_decay 1e-4 --logging_steps 50 --save_total_limit 100 --classes_path data/KS_Buoy_classes.txt --image_height 480 --image_width 1920 --device cuda

# python3 train.py --save_model_hub_id ARG-NCTU --save_model_repo_id detr-resnet-50-finetuned-20-epochs-Boat-dataset-0314 --load_model_hub_id facebook --load_model_repo_id detr-resnet-50 --dataset_hub_id ARG-NCTU --dataset_repo_id Boat_dataset_2024 --dataset_format jsonl --epoch 20 --batch_size 8 --learning_rate 1e-5 --weight_decay 1e-4 --logging_steps 50 --save_total_limit 100 --classes_path data/boat_classes.txt --image_height 480 --image_width 640 --device cuda

# python3 train.py --save_model_hub_id ARG-NCTU --save_model_repo_id detr-resnet-50-finetuned-600-epochs-GuardBoat-dataset --load_model_hub_id ARG-NCTU --load_model_repo_id detr-resnet-50-finetuned-20-epochs-boat-dataset --dataset_hub_id ARG-NCTU --dataset_repo_id GuardBoat_dataset_2025 --dataset_format parquet --epoch 600 --batch_size 8 --learning_rate 1e-5 --weight_decay 1e-4 --logging_steps 50 --save_total_limit 100 --classes_path data/GuardBoat_classes.txt --image_height 480 --image_width 1920 --device cuda
