from PIL import Image
import numpy as np
from pathlib import Path
from typing import List, Dict
import time
from tqdm import tqdm
import torch
import torch.nn as nn
from torchvision import models, transforms
from sklearn.metrics.pairwise import cosine_similarity


class TransformerEmbedder:
    def __init__(self, image_size: tuple, model_name: str, device: str = "cuda"):
        self.device = device if torch.cuda.is_available() else "cpu"
        self.image_size = image_size
        self.model = self._load_model(model_name).to(self.device)
        self.transform = self._get_transform()

    def _load_model(self, model_name: str) -> nn.Module:
        if model_name == "resnet50":
            model = models.resnet50(weights="IMAGENET1K_V2")
        if model_name == "efficientnet_b4":
            model = models.efficientnet_b4(weights="IMAGENET1K_V1")
        if model_name == "convnext_tiny":
            model = models.convnext_tiny(weights="IMAGENET1K_V1")
        model = nn.Sequential(*list(model.children())[:-1])  
        model.eval()
        return model

    def _get_transform(self):
        return transforms.Compose(
            [
                transforms.Resize(self.image_size),
                transforms.ToTensor(),
                transforms.Normalize(
                    mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
                ),
            ]
        )

    def compute_embeddings(self, image_paths: List[Path]) -> Dict[str, np.ndarray]:
        embeddings = {}
        for path in tqdm(image_paths, desc="Вычисление эмбеддингов"):
            img = Image.open(path).convert("RGB")
            img_tensor = self.transform(img).unsqueeze(0).to(self.device)
            with torch.no_grad():
                embedding = self.model(img_tensor).squeeze().cpu().numpy()
            if embedding is not None:
                embeddings[str(path)] = embedding
        return embeddings

    def get_image_comparison(
        self,
        train_embeddings: Dict[str, np.ndarray],
        test_embeddings: Dict[str, np.ndarray],
        test_paths: List[Path],
        threshold: float = 0.95,
    ) -> List[str]:

        unwanted_images = []

        train_embs = list(train_embeddings.values())

        for item, test_item in enumerate(
            tqdm(
                test_embeddings.items(),
                total=len(test_embeddings),
                desc="Поиск совпадений",
            )
        ):
            sims = cosine_similarity([test_item[1]], train_embs)[0]
            if np.max(sims) > threshold:
                unwanted_images.append(test_paths[item])

        print(
            f"Количество лишних изображений согласно предсказаниям: {len(unwanted_images)}"
        )

        return unwanted_images