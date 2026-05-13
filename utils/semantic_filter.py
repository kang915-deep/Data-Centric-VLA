import torch
import open_clip
from PIL import Image
from typing import List, Union
import numpy as np

class SemanticFilter:
    """
    Utility to filter frames based on Vision-Language alignment (CLIP score).
    """
    
    def __init__(self, model_name: str = "ViT-B-32", pretrained: str = "openai", device: str = "cuda"):
        self.device = device if torch.cuda.is_available() else "cpu"
        self.model, _, self.preprocess = open_clip.create_model_and_transforms(
            model_name, pretrained=pretrained, device=self.device
        )
        self.tokenizer = open_clip.get_tokenizer(model_name)
        self.model.eval()

    @torch.no_grad()
    def get_scores(self, images: List[Image.Image], text: str) -> np.ndarray:
        """
        Compute similarity scores between a list of images and a single text instruction.
        """
        text_tokens = self.tokenizer([text]).to(self.device)
        text_features = self.model.encode_text(text_tokens)
        text_features /= text_features.norm(dim=-1, keepdim=True)

        scores = []
        for img in images:
            image_input = self.preprocess(img).unsqueeze(0).to(self.device)
            image_features = self.model.encode_image(image_input)
            image_features /= image_features.norm(dim=-1, keepdim=True)
            
            similarity = (image_features @ text_features.T).item()
            scores.append(similarity)
            
        return np.array(scores)

    def filter_by_threshold(self, images: List[Image.Image], text: str, threshold: float = 0.2) -> List[int]:
        """
        Returns indices of images that pass the threshold.
        """
        scores = self.get_scores(images, text)
        return np.where(scores >= threshold)[0].tolist()

if __name__ == "__main__":
    # Mock test
    print("Initializing SemanticFilter...")
    # Using cpu for mock test if no gpu
    filter = SemanticFilter(device="cpu")
    
    # Create a dummy image
    dummy_img = Image.fromarray(np.uint8(np.random.rand(224, 224, 3) * 255))
    instruction = "pick up the red cube"
    
    scores = filter.get_scores([dummy_img], instruction)
    print(f"Alignment Score: {scores}")
