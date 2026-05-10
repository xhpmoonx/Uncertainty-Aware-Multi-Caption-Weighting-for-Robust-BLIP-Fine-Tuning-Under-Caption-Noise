#################################################################
# This dataset class is designed to load and preprocess the COCO 
# dataset for training a captioning model with weighted captions. 
# It reads the annotation JSON file, processes the images and their 
# corresponding candidate captions, and extracts features for each 
# caption that can be used to predict confidence weights. The dataset 
# returns a dictionary containing the processed image, a list of caption
#  texts (prefixed with a prompt), a tensor of features for each caption, 
# and metadata such as image ID and path. The features are based on 
# alignment, fluency, and agreement scores, which can be used as 
# input to a confidence MLP to predict weights for each caption 
# during training. The dataset also includes a mechanism to limit 
# the number of candidate captions per image to reduce noise and 
# manage memory usage.
#################################################################   

import json
from pathlib import Path

import torch
from torch.utils.data import Dataset
from PIL import Image

from torchvision import transforms
from torchvision.transforms.functional import InterpolationMode


def _clean_text(x):
    return " ".join(str(x).split()).strip()


def _zscore(vals):
    vals = torch.tensor(vals, dtype=torch.float32)
    if len(vals) <= 1:
        return torch.zeros_like(vals)
    std = vals.std(unbiased=False)
    if std.item() < 1e-8:
        return torch.zeros_like(vals)
    return (vals - vals.mean()) / std


class WeightedCaptionTrainDataset(Dataset):
    def __init__(self, ann_path, image_size=384, prompt="", max_candidates=3):
        self.ann_path = ann_path
        self.prompt = prompt
        self.max_candidates = max_candidates

        with open(ann_path, "r") as f:
            self.data = json.load(f)

        normalize = transforms.Normalize(
            (0.48145466, 0.4578275, 0.40821073),
            (0.26862954, 0.26130258, 0.27577711),
        )

        self.transform = transforms.Compose([
            transforms.Resize((image_size, image_size), interpolation=InterpolationMode.BICUBIC),
            transforms.ToTensor(),
            normalize,
        ])

    def __len__(self):
        return len(self.data)

    def _extract_caption_text(self, cap):
        if isinstance(cap, str):
            return _clean_text(cap)
        if isinstance(cap, dict):
            if "text" in cap:
                return _clean_text(cap["text"])
            if "caption" in cap:
                return _clean_text(cap["caption"])
        return ""
    
    #####################################################################
    # The _extract_features method is responsible for extracting the alignment, fluency, and agreement features from the candidate captions. It first checks if the candidates already have z-scored features (indicated by keys like "z_alignment"). If they do, it directly constructs a tensor of these features. If not, it collects the raw alignment, fluency, and agreement scores for all candidates, applies z-score normalization to each feature across the candidates for that image, and then stacks them into a tensor. This ensures that the features are on a comparable scale for the confidence MLP to process.
    # The __getitem__ method retrieves a sample from the dataset, processes the image, extracts the top-K candidate captions and their features, and returns a dictionary containing the processed image, the list of caption texts (prefixed with a prompt), the corresponding features tensor, and metadata like image ID and path. If no valid captions are found for an image, it provides a fallback caption and zero features to ensure the model can still learn from that sample.
    # Note that the features extracted here are intended to be used as input to a confidence MLP, which will predict weights for each caption based on these features during training.
    # Overall, this dataset class is designed to facilitate training a captioning model with weighted captions, where the weights are derived from the confidence MLP's predictions based on the extracted features of the candidate captions.
    # The use of z-score normalization helps to ensure that the features are on a similar scale, which can improve the training stability and performance of the confidence MLP. By limiting the number of candidate captions to a maximum (max_candidates), it also helps to reduce noise from low-quality captions and manage memory usage during training.
    # The prompt can be used to provide additional context or instructions to the model when processing the captions, which can be beneficial for guiding the model's learning process.
    # In summary, this dataset class is a crucial component for training a captioning model that leverages weighted captions, allowing the model to learn to assign different importance to different candidate captions based on their features.
    # The confidence MLP will learn to predict weights for each caption based on the extracted features, which can help the model focus on higher-quality captions during training and improve overall captioning performance.
    # The design of this dataset class allows for flexibility in how the candidate captions and their features are structured, making it adaptable to different datasets and feature extraction methods. It also provides a clear interface for retrieving the necessary data for training the captioning model with weighted captions.
    # The use of the PIL library for image processing and torchvision transforms ensures that the images are properly preprocessed and normalized for input into the model, which can help improve training efficiency and performance.
    # Overall, this dataset class is a key component in the training pipeline for a captioning model that utilizes weighted captions, providing the necessary data and features for the confidence MLP to learn to assign appropriate weights to candidate captions during training.
    #######################################################################
    def _extract_features(self, candidates):
        # Prefer z_* if already present; otherwise z-score raw features per image.
        use_z = all(
            isinstance(c, dict) and
            ("z_alignment" in c) and
            ("z_fluency" in c) and
            ("z_agreement" in c)
            for c in candidates
        )

        if use_z:
            feats = []
            for c in candidates:
                feats.append([
                    float(c.get("z_alignment", 0.0)),
                    float(c.get("z_fluency", 0.0)),
                    float(c.get("z_agreement", 0.0)),
                ])
            return torch.tensor(feats, dtype=torch.float32)

        align = []
        flu = []
        agree = []

        for c in candidates:
            if not isinstance(c, dict):
                align.append(0.0)
                flu.append(0.0)
                agree.append(0.0)
                continue

            align.append(float(c.get("alignment", 0.0)))
            flu.append(float(c.get("fluency", 0.0)))
            agree.append(float(c.get("agreement", 0.0)))

        z_align = _zscore(align)
        z_flu = _zscore(flu)
        z_agree = _zscore(agree)

        return torch.stack([z_align, z_flu, z_agree], dim=1)

    def __getitem__(self, index):
        item = self.data[index]

        image_path = item["image_path"]
        image = Image.open(image_path).convert("RGB")
        image = self.transform(image)

        # Only keep top-K candidates per image to avoid noisy captions and reduce memory usage.
        candidates = item.get("generated_captions", [])[: self.max_candidates]
        # Extract caption texts and features, while filtering out candidates without valid text.
        captions = []
        kept_candidates = []
        for c in candidates:
            text = self._extract_caption_text(c)
            if text:
                captions.append(self.prompt + text)
                kept_candidates.append(c)

        if len(captions) == 0:
            # safe fallback
            captions = [self.prompt + "a picture of something"]
            # features will be all zeros, which should be safe for the model to learn to ignore
            kept_candidates = [{"alignment": 0.0, "fluency": 0.0, "agreement": 0.0}]

        features = self._extract_features(kept_candidates)

        return {
            "image": image,
            "captions": captions,
            "features": features,
            "image_id": int(item.get("image_id", index)),
            "image_path": image_path,
        }