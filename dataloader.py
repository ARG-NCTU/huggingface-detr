import torch
from torch.utils.data import DataLoader
from datasets import load_dataset, Features, Value, Sequence
from transformers import AutoImageProcessor
from PIL import Image
import os
import numpy as np
import albumentations as A
import matplotlib.pyplot as plt
import argparse
import torch
import cv2


class BaseObjectDetectionDataLoader:
    def __init__(self, dataset_format, image_height, image_width, load_model_hub_id, 
                 load_model_repo_id, dataset_hub_id, 
                 dataset_repo_id, classes_path):
        self.dataset_format = dataset_format
        self.image_height = image_height
        self.image_width = image_width
        self.load_model_hub_id = load_model_hub_id
        self.load_model_repo_id = load_model_repo_id
        self.dataset_hub_id = dataset_hub_id
        self.dataset_repo_id = dataset_repo_id
        self.classes_path = classes_path

        # Load class mappings
        self.id2label = self.get_id2label()
        self.label2id = self.get_label2id()

        # Initialize image processor and transforms
        self.image_processor = self.get_image_processor()
        self.transform = self.get_transform()
        self.collate_fn = self.get_collate_fn()

        # Load dataset
        self.dataset = self.get_dataset()

    def load_classes(self):
        with open(self.classes_path, "r") as f:
            return [cname.strip() for cname in f.readlines()]

    def get_id2label(self):
        return {index: x for index, x in enumerate(self.load_classes(), start=0)}

    def get_label2id(self):
        return {v: k for k, v in self.get_id2label().items()}
    
    def load_image(self, image_path):
        """Load an image from JSONL or Parquet dataset."""
        try:
            return np.array(Image.open(image_path).convert("RGB"))[:, :, ::-1]
        except Exception:
            print(f"Warning: {image_path} not found, using black placeholder.")
            return np.zeros((self.image_height, self.image_width, 3), dtype=np.uint8)

    def get_dataset(self):
        if self.dataset_format == "jsonl":
            features = Features({
                'image_id': Value('int32'),
                'image_path': Value('string'),
                'width': Value('int32'),
                'height': Value('int32'),
                'objects': {
                    'id': Sequence(Value('int32')),
                    'area': Sequence(Value('float32')),
                    'bbox': Sequence(Sequence(Value('float32'), length=4)),
                    'category': Sequence(Value('int32'))
                }
            })
            dataset = load_dataset(
                'json',
                data_files={'train': 'data/instances_train2024_rvrr.jsonl'},
                features=features
            )
        else:
            dataset = load_dataset(f"{self.dataset_hub_id}/{self.dataset_repo_id}")

        dataset["train"] = dataset["train"].with_transform(lambda x: self.transform_aug_ann(x))
        return dataset

    def get_collate_fn(self):
        raise NotImplementedError("Subclasses must implement this method.")
    
    def get_transform(self):
        raise NotImplementedError("Subclasses must implement this method.")
    
    def formatted_anns(self, image_id, category, area, bbox):
        raise NotImplementedError("Subclasses must implement this method.")
    
    def get_image_processor(self):
        raise NotImplementedError("Subclasses must implement this method.")
    
    def transform_aug_ann(self, examples):
        raise NotImplementedError("Subclasses must implement this method.")
    

class DETRDataLoader(BaseObjectDetectionDataLoader):
    def __init__(self, dataset_format, image_height=480, image_width=1920, load_model_hub_id="facebook", 
                 load_model_repo_id="detr-resnet-50", dataset_hub_id="ARG-NCTU", 
                 dataset_repo_id="TW_Marine_5cls_dataset", classes_path="data/TW_Marine_5cls_classes.txt"):
        
        super().__init__(dataset_format, image_height, image_width, load_model_hub_id, 
                 load_model_repo_id, dataset_hub_id, 
                 dataset_repo_id, classes_path)

    def get_collate_fn(self):
        def collate_fn(batch):
            pixel_values = [item["pixel_values"] for item in batch]
            encoding = self.image_processor.pad(pixel_values, return_tensors="pt")
            labels = [item["labels"] for item in batch]
            batch = {}
            batch["pixel_values"] = encoding["pixel_values"]
            batch["pixel_mask"] = encoding["pixel_mask"]
            batch["labels"] = labels
            return batch
        return collate_fn

    def formatted_anns(self, image_id, category, area, bbox):
        return [
            {"image_id": image_id, "category_id": category[i], "isCrowd": 0, "area": area[i], "bbox": list(bbox[i])}
            for i in range(len(category))
        ]
    
    def get_transform(self):
        return A.Compose(
            [
                A.HorizontalFlip(p=0.5),
                A.RandomBrightnessContrast(p=0.5),
            ],
            bbox_params=A.BboxParams(format="coco", label_fields=["category"]),
        )
    
    def get_image_processor(self):
        processor = AutoImageProcessor.from_pretrained(f"{self.load_model_hub_id}/{self.load_model_repo_id}")
        processor.size = {"height": self.image_height, "width": self.image_width}
        return processor
    
    def transform_aug_ann(self, examples):
        image_ids = examples["image_id"]
        images, bboxes, areas, categories = [], [], [], []

        for image_path, objects in zip(examples["image_path"], examples["objects"]):
            image = self.load_image(image_path)

            try:
                transformed = self.transform(image=image, bboxes=objects["bbox"], category=objects["category"])
                image, bboxes_trans, categories_trans = transformed["image"], transformed["bboxes"], transformed["category"]
            except Exception as e:
                print(f"Transform error: {e}. Using default bbox.")
                image = np.zeros((self.image_height, self.image_width, 3), dtype=np.uint8)
                bboxes_trans, categories_trans = [[0.4, 0.4, 0.2, 0.2]], [0]

            images.append(image)
            bboxes.append(bboxes_trans)
            areas.append(objects["area"])
            categories.append(categories_trans)

        targets = [
            {"image_id": id_, "annotations": self.formatted_anns(id_, cat_, ar_, box_)}
            for id_, cat_, ar_, box_ in zip(image_ids, categories, areas, bboxes)
        ]

        return self.image_processor(images=images, annotations=targets, return_tensors="pt")

class YolosDataLoader(BaseObjectDetectionDataLoader):
    def __init__(self, dataset_format, image_height=480, image_width=1920, load_model_hub_id="hustvl", 
                 load_model_repo_id="yolos-tiny", dataset_hub_id="ARG-NCTU", 
                 dataset_repo_id="TW_Marine_5cls_dataset", classes_path="data/TW_Marine_5cls_classes.txt"):
        
        super().__init__(dataset_format, image_height, image_width, load_model_hub_id, 
                 load_model_repo_id, dataset_hub_id, 
                 dataset_repo_id, classes_path)

    def get_collate_fn(self):
        def collate_fn(batch):
            pixel_values = [item["pixel_values"] for item in batch]
            labels = [item["labels"] for item in batch]
            batch = {
                "pixel_values": torch.stack(pixel_values),
                "labels": labels
            }
            return batch
        return collate_fn
    
    def formatted_anns(self, image_id, category, area, bbox, obj_id):
        return [
            {"image_id": image_id, "object_id": obj_id[i], "category_id": category[i], "isCrowd": 0, "area": area[i], "bbox": list(bbox[i])}
            for i in range(len(category))
        ]
    
    def get_transform(self):
        return A.Compose(
            [
                A.SmallestMaxSize(max_size=800),
                A.PadIfNeeded(min_height=800, min_width=1333, border_mode=0, value=(0,0,0)),
                # A.Resize(800, 1333, interpolation=cv2.INTER_LINEAR),
                A.HorizontalFlip(p=0.5),
                A.RandomBrightnessContrast(p=0.5),
            ],
            bbox_params=A.BboxParams(format="coco", label_fields=["category"]),
        )
    
    def get_image_processor(self):
        processor = AutoImageProcessor.from_pretrained(f"{self.load_model_hub_id}/{self.load_model_repo_id}")
        processor.size = {"height": 800, "width": 1333}
        return processor

    def transform_aug_ann(self, examples):
        image_ids = examples["image_id"]
        images, bboxes, areas, categories, obj_ids = [], [], [], [], []

        for image_path, objects in zip(examples["image_path"], examples["objects"]):
            image = self.load_image(image_path)
            try:
                transformed = self.transform(image=image, bboxes=objects["bbox"], category=objects["category"])
                image, bboxes_trans, categories_trans = transformed["image"], transformed["bboxes"], transformed["category"]
            except Exception as e:
                print(f"Transform error: {e}. Using default bbox.")
                image = np.zeros((self.image_height, self.image_width, 3), dtype=np.uint8)
                bboxes_trans, categories_trans = [[0.4, 0.4, 0.2, 0.2]], [0]
            images.append(Image.fromarray(image[..., ::-1]))
            bboxes.append(bboxes_trans)
            areas.append(objects["area"])
            categories.append(categories_trans)
            obj_ids.append(objects["id"])

        targets = [
            {
                "image_id": id_,
                "annotations": self.formatted_anns(id_, cat_, ar_, box_, oid_)
            }
            for id_, cat_, ar_, box_, oid_ in zip(image_ids, categories, areas, bboxes, obj_ids)
        ]

        return self.image_processor(images=images, annotations=targets, return_tensors="pt")

def parse_args():
    parser = argparse.ArgumentParser(description='DETR DataLoader Augmentation Visualization')

    # Dataset parameters
    parser.add_argument('--dataset_hub_id', type=str, default='ARG-NCTU', help='Dataset Hugging Face Hub ID')
    parser.add_argument('--dataset_repo_id', type=str, default='GuardBoat_dataset_2025', help='Dataset Hugging Face repository ID')
    parser.add_argument('--dataset_format', type=str, choices=['jsonl', 'parquet'], default='parquet', help='Dataset format')

    # Other parameters
    parser.add_argument('--classes_path', type=str, default='data/GuardBoat_classes.txt', help='Path to class labels file')
    parser.add_argument('--image_height', type=int, default=480, help='Image height')
    parser.add_argument('--image_width', type=int, default=1920, help='Image width')

    parser.add_argument('--device', type=str, default='cuda' if torch.cuda.is_available() else 'cpu', help='Device (default: cuda if available)')

    # Output
    parser.add_argument('--output_aug_path', type=str, default='visualize_aug.png', help='Output image file path')
    parser.add_argument('--output_pad_mask_path', type=str, default='visualize_pad_mask.png', help='Output image file path')

    return parser.parse_args()


if __name__ == "__main__":

    args = parse_args()

    loader = DETRDataLoader(
        dataset_format=args.dataset_format,
        image_height=args.image_height,
        image_width=args.image_width,
        load_model_hub_id="facebook",                   # default model hub
        load_model_repo_id="detr-resnet-50",            # default model repo
        dataset_hub_id=args.dataset_hub_id,
        dataset_repo_id=args.dataset_repo_id,
        classes_path=args.classes_path
    )

    # 🔷 Load raw dataset without transform
    raw_ds = load_dataset(
        f"{args.dataset_hub_id}/{args.dataset_repo_id}"
    )["train"]

    # 🔷 Get the first example
    first_raw = raw_ds[0]
    print("=" * 50)
    print(f"First data example: {first_raw}")
    print("=" * 50)
    img_pil = first_raw["image"]                     # PIL.Image
    img_raw = np.array(img_pil)[:, :, ::-1]         # Convert to HWC, BGR (for OpenCV)
    bbox = first_raw["objects"]["bbox"]             # COCO format [x, y, w, h]
    category = first_raw["objects"]["category"]     # Category IDs

    # 🔷 Define four types of augmentations
    normal_transform = A.Compose(
        [], bbox_params=A.BboxParams(format="coco", label_fields=["category"])
    )
    hflip_transform = A.Compose(
        [A.HorizontalFlip(p=1.0)],
        bbox_params=A.BboxParams(format="coco", label_fields=["category"])
    )
    
    brightness_contrast_aug = A.RandomBrightnessContrast(
        brightness_limit=(0.1, 0.11),  # brightness = 1 ± 0.9
        contrast_limit=(0.1, 0.11),
        p=1.0
    )
    enhanced_transform = A.Compose(
        [brightness_contrast_aug],
        bbox_params=A.BboxParams(format="coco", label_fields=["category"])
    )
    hflip_enhanced_transform = A.Compose(
        [brightness_contrast_aug, A.HorizontalFlip(p=1.0)],
        bbox_params=A.BboxParams(format="coco", label_fields=["category"])
    )

    # 🔷 Prepare transform list
    transforms = [
        ("Normal", normal_transform),
        ("Horizontal Flip", hflip_transform),
        ("Enhanced Brightness & Contrast", enhanced_transform),
        ("Horizontal Flip + Enhanced Brightness & Contrast", hflip_enhanced_transform),
    ]

    fig, axes = plt.subplots(len(transforms), 1, figsize=(12, 20), constrained_layout=True)

    def draw_bboxes(image, bboxes, categories, id2label, color=(0, 0, 255)):
        """
        Draw bounding boxes and category IDs on the image.
        """
        img_copy = image.copy()
        for box, cls in zip(bboxes, categories):
            x, y, w, h = box
            x1, y1, x2, y2 = int(x), int(y), int(x + w), int(y + h)
            cv2.rectangle(img_copy, (x1, y1), (x2, y2), color, 2)
            cls_name = id2label.get(cls, str(cls))
            cv2.putText(
                img_copy, cls_name, (x1, y1 - 20),
                cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2
            )
        return img_copy

    # 🔷 Apply each transform, draw bounding boxes, and plot
    for ax, (title, tfm) in zip(axes, transforms):
        result = tfm(image=img_raw, bboxes=bbox, category=category)
        img_aug = result["image"]
        bboxes_aug = result["bboxes"]
        categories_aug = result["category"]

        img_vis = draw_bboxes(img_aug, bboxes_aug, categories_aug, loader.id2label)
        ax.imshow(cv2.cvtColor(img_vis, cv2.COLOR_BGR2RGB))
        ax.set_title(title)
        ax.axis("off")

    # plt.tight_layout()
    plt.savefig(args.output_aug_path)
    print(f"✅ Augmentation visualization saved at {args.output_aug_path}")
    # plt.show()

    # ======================
    # 🔷 Draw Original + Padded + Mask
    # ======================

    processed_sample = loader.dataset["train"][0]

    pixel_values = processed_sample["pixel_values"]  # (3,H,W)
    pixel_mask   = processed_sample["pixel_mask"]    # (H,W)

    # Convert padded image
    padded_img = pixel_values.permute(1, 2, 0).cpu().numpy()
    padded_img = (padded_img - padded_img.min()) / (padded_img.max() - padded_img.min()) * 255
    padded_img = padded_img.astype(np.uint8)

    mask_img = pixel_mask.cpu().numpy().astype(np.uint8) * 255  # 0/1 → 0/255

    # Add black border (width=5 pixels)
    border_size = 5
    mask_with_border = cv2.copyMakeBorder(
        mask_img,
        top=border_size,
        bottom=border_size,
        left=border_size,
        right=border_size,
        borderType=cv2.BORDER_CONSTANT,
        value=0  # black
    )

    # Original image: already loaded before as img_raw
    # convert to RGB
    orig_img_rgb = cv2.cvtColor(img_raw, cv2.COLOR_BGR2RGB)

    # Plot all three
    fig2, axes2 = plt.subplots(3, 1, figsize=(12, 15), constrained_layout=True)

    axes2[0].imshow(orig_img_rgb)
    axes2[0].set_title("Original Image")
    axes2[0].axis("off")

    axes2[1].imshow(padded_img)
    axes2[1].set_title("Augmented + Padded Image")
    axes2[1].axis("off")

    axes2[2].imshow(mask_with_border, cmap="gray", vmin=0, vmax=255)
    axes2[2].set_title("Pixel Mask + Black Border")
    axes2[2].axis("off")

    # plt.tight_layout()
    plt.savefig(args.output_pad_mask_path)
    # plt.show()

    print(f"✅ Saved {args.output_pad_mask_path}")

