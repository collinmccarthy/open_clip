"""Manually created from content of:
    https://colab.research.google.com/github/mlfoundations/open_clip/blob/master/docs/Interacting_with_open_clip.ipynb
"""

import os
import skimage
import numpy as np
import torch
from PIL import Image
from torch import nn
from torchvision import datasets
from torchvision import transforms

from open_clip import create_model_and_transforms, tokenizer


def get_model_and_inputs(
    model_name: str = "convnext_base_w", pretrained: str = "laion2b_s13b_b82k_augreg"
) -> tuple[
    nn.Module,
    transforms.Compose,
    torch.Tensor,
    torch.Tensor,
    list[Image.Image],
    dict[str, str],
]:
    # Load pre-trained model and pre-processors
    model: nn.Module
    preprocess_val: transforms.Compose

    model, _preprocess_train, preprocess_val = (
        create_model_and_transforms(  # pyright: ignore[reportAssignmentType]
            model_name=model_name,
            pretrained=pretrained,
        )
    )

    # Get images from skimage and some captions
    unordered_descriptions = {
        "page": "a page of text about segmentation",
        "chelsea": "a facial photo of a tabby cat",
        "astronaut": "a portrait of an astronaut with the American flag",
        "rocket": "a rocket standing on a launchpad",
        "motorcycle_right": "a red motorcycle standing in a garage",
        "camera": "a person looking at a camera on a tripod",
        "horse": "a black-and-white silhouette of a horse",
        "coffee": "a cup of coffee on a saucer",
    }

    data_dir: str = skimage.data_dir  # pyright: ignore[reportAttributeAccessIssue]
    filenames: list[str] = [
        filename
        for filename in os.listdir(data_dir)
        if filename.endswith(".png") or filename.endswith(".jpg")
    ]

    original_images: list[Image.Image] = []
    images: list[torch.Tensor] = []
    texts: list[str] = []
    descriptions: dict[str, str] = {}
    for filename in filenames:
        name = os.path.splitext(filename)[0]
        if name not in unordered_descriptions:
            continue

        image = Image.open(os.path.join(data_dir, filename)).convert("RGB")
        original_images.append(image)
        images.append(
            preprocess_val(image)  # pyright: ignore[reportCallIssue,reportArgumentType]
        )
        texts.append(unordered_descriptions[name])
        descriptions[name] = unordered_descriptions[name]

    image_input = torch.tensor(np.stack(images))
    text_tokens = tokenizer.tokenize(["This is " + desc for desc in texts])

    return (
        model,
        preprocess_val,
        image_input,
        text_tokens,
        original_images,
        descriptions,
    )


def inference_test(
    model: nn.Module,
    image_input: torch.Tensor,
    text_tokens: torch.Tensor,
    descriptions: dict[str, str],
) -> None:
    model.eval()

    # Forward pass to get embeddings (same as zero_shot_test())
    with torch.no_grad():
        image_features = model.encode_image(image_input).float()
        text_features = model.encode_text(text_tokens).float()

        # Normalize
        image_features /= image_features.norm(dim=-1, keepdim=True)
        text_features /= text_features.norm(dim=-1, keepdim=True)

    # Compute similarity between N images and K descriptions with C dims (here N=8 and K=8)
    similarity = image_features @ text_features.T  # NxC @ CxK -> NxK

    # Take argmax over K descriptions to get best matches
    best_matches = similarity.argmax(dim=-1).cpu().tolist()

    print("-" * 80)
    print(f"Inference test")
    print("- " * 40)
    print(f"Image features shape: {image_features.shape}")
    print(f"Text features shape: {text_features.shape}")
    print(f"Similarity shape: {similarity.shape}")
    print("- " * 40)
    print(f"Results:")
    description_keys = list(descriptions.keys())
    for key, match_idx in zip(description_keys, best_matches):
        print(f"  {key} -> {descriptions[description_keys[match_idx]]}")
    print("-" * 80)
    return  # So we can set a breakpoint here


def zero_shot_test(
    model: nn.Module,
    preprocess_val: transforms.Compose,
    image_input: torch.Tensor,
    original_images: list[Image.Image],
    descriptions: dict[str, str],
) -> None:
    model.eval()

    # Download dataset
    cifar100 = datasets.CIFAR100(
        os.path.expanduser("~/.cache"), transform=preprocess_val, download=True
    )

    # Create basic captions
    text_descriptions = [f"A photo of a {label}" for label in cifar100.classes]
    text_tokens = tokenizer.tokenize(text_descriptions)

    # Forward pass to get embeddings (same as inference_test())
    with torch.no_grad():
        image_features = model.encode_image(image_input).float()
        text_features = model.encode_text(text_tokens).float()

        # Normalize
        image_features /= image_features.norm(dim=-1, keepdim=True)
        text_features /= text_features.norm(dim=-1, keepdim=True)

    # Compute similarity between N images and K descriptions with C dims (here N=8 and K=100)
    similarity = image_features @ text_features.T  # NxC @ CxK -> NxK

    # Compute softmax (with 100.0 factor) to get probabilities
    top_k = 5
    text_probs = (100.0 * similarity).softmax(dim=-1)
    top_probs, top_labels = text_probs.cpu().topk(top_k, dim=-1)

    print("-" * 80)
    print(f"Zero shot test")
    print("- " * 40)
    print(f"Image features shape: {image_features.shape}")
    print(f"Text features shape: {text_features.shape}")
    print(f"Similarity shape: {similarity.shape}")
    print("- " * 40)
    print(f"Results:")
    description_keys = list(descriptions.keys())
    for i, key in enumerate(description_keys):
        orig_image = original_images[i]
        curr_labels = top_labels[i].cpu().numpy()
        curr_probs = top_probs[i].cpu().numpy()
        label_probs = [
            (cifar100.classes[curr_labels[i]], f"{100 * curr_probs[i]:.2f}%")
            for i in range(top_k)
        ]
        print(f"  {key} -> {label_probs}")
    print("-" * 80)
    return  # So we can set a breakpoint here


if __name__ == "__main__":
    model, preprocess_val, image_input, text_tokens, original_images, descriptions = (
        get_model_and_inputs()
    )
    inference_test(
        model=model,
        image_input=image_input,
        text_tokens=text_tokens,
        descriptions=descriptions,
    )
    zero_shot_test(
        model=model,
        preprocess_val=preprocess_val,
        image_input=image_input,
        original_images=original_images,
        descriptions=descriptions,
    )
