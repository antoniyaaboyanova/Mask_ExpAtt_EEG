#!/usr/bin/env python3
import os
import argparse
from typing import Any, List

import torch
import numpy as np

from thingsvision import get_extractor
from thingsvision.utils.storing import save_features
from thingsvision.utils.data import ImageDataset, DataLoader

def extract_features(
    extractor: Any,
    module_name: str,
    image_path: str,
    out_path: str,
    batch_size: int,
    flatten_activations: bool,
    apply_center_crop: bool,
    model_name: str,
    class_names: List[str] = None,
    file_names: List[str] = None,
) -> np.ndarray:
    """Extract features for a single layer."""

    dataset = ImageDataset(
        root=image_path,
        out_path=out_path,
        backend=extractor.get_backend(),
        transforms=extractor.get_transformations(
            resize_dim=256,
            crop_dim=224 if apply_center_crop else None,
        ),
        class_names=class_names,
        file_names=file_names,
    )

    batches = DataLoader(
        dataset=dataset,
        batch_size=batch_size,
        backend=extractor.get_backend(),
    )

    features = extractor.extract_features(
        batches=batches,
        module_name=module_name,
        flatten_acts=flatten_activations,
    )

    print("Features extracted!")
    print("-" * 20)

    # Save
    save_features(
        features,
        out_path=f"{out_path}/raw_{model_name}_{module_name}",
        file_format="npy",
    )

    return features


def main(category: str):
    # ---------------- paths ----------------
    base_dir = "/projects/archiv/DataStore_Boyanova/ExpAtt_EEG/Image_dataset"

    full_image_path = f"{base_dir}/Images_{category}"
    full_output_path = f"{base_dir}/features_{category}"
    os.makedirs(full_output_path, exist_ok=True)

    # ---------------- config ----------------
    pretrained = True
    model_path = None
    batch_size = 32
    apply_center_crop = True
    flatten_activations = True
    class_names = None
    file_names = None

    device = "cuda" if torch.cuda.is_available() else "cpu"

    model_name = "cornet-s"
    source = "custom"

    # ---------------- extractor ----------------
    extractor = get_extractor(
        model_name=model_name,
        pretrained=pretrained,
        model_path=model_path,
        device=device,
        source=source,
    )

    # ---------------- layers ----------------
    module_names = [
        name
        for name, layer in extractor.model.named_modules()
        if isinstance(layer, torch.nn.Conv2d)
    ]
    
    print(module_names)
    if model_name == "cornet-s":
        module_names = ['V1.conv1',  'IT.conv3']

    # ---------------- extraction loop ----------------
    for idx, module_name in enumerate(module_names):
        print(module_name)
        print(f"Layer {idx + 1} / {len(module_names)}")
        print("-" * 20)

        extract_features(
            extractor=extractor,
            module_name=module_name,
            image_path=full_image_path,
            out_path=full_output_path,
            batch_size=batch_size,
            flatten_activations=flatten_activations,
            apply_center_crop=apply_center_crop,
            model_name=model_name,
            class_names=class_names,
            file_names=file_names,
        )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Extract CNN features with thingsvision"
    )
    parser.add_argument(
        "--category",
        required=True,
        choices=["plants", "objects", "animals", "food"],
        help="Image category to process",
    )

    args = parser.parse_args()
    main(args.category)
