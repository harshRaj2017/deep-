from __future__ import absolute_import
import numpy as np
import torch
import torchvision.transforms as T
from PIL import Image

from ..utils import (
    check_isfile, load_pretrained_weights, compute_model_complexity
)
from ..models import build_model

class FeatureExtractor(object):
    def __init__(
        self,
        model_name='',
        model_path='',
        image_size=(256, 128),
        pixel_mean=[0.485, 0.456, 0.406],
        pixel_std=[0.229, 0.224, 0.225],
        pixel_norm=True,
        device='cuda',
        verbose=True  # Changed to True for debugging
    ):
        print("\n=== Initializing ReID Feature Extractor ===")
        print(f"Model: {model_name}")
        print(f"Image size: {image_size}")
        
        # Build model
        model = build_model(
            model_name,
            num_classes=1,
            pretrained=not (model_path and check_isfile(model_path)),
            use_gpu=device.startswith(device)
        )
        model.eval()

        if verbose:
            num_params, flops = compute_model_complexity(
                model, (1, 3, image_size[0], image_size[1])
            )
            print('Model parameters: {:,}'.format(num_params))
            print('Model FLOPs: {:,}'.format(flops))

        if model_path and check_isfile(model_path):
            print(f"Loading weights from: {model_path}")
            load_pretrained_weights(model, model_path)
        else:
            print("Using ImageNet pretrained weights")

        # Build transform functions
        transforms = []
        transforms += [T.Resize(image_size)]
        transforms += [T.ToTensor()]
        if pixel_norm:
            transforms += [T.Normalize(mean=pixel_mean, std=pixel_std)]
        preprocess = T.Compose(transforms)

        to_pil = T.ToPILImage()

        device = torch.device(device)
        model.to(device)

        self.model = model
        self.preprocess = preprocess
        self.to_pil = to_pil
        self.device = device
        self.verbose = verbose

    def __call__(self, input):
        if isinstance(input, list):
            images = []
            if self.verbose:
                print(f"\nProcessing batch of {len(input)} images")

            for i, element in enumerate(input):
                if isinstance(element, np.ndarray):
                    if self.verbose:
                        print(f"Image {i+1} shape: {element.shape}")
                        if element.size == 0:
                            print(f"Warning: Empty image at index {i}")
                        if element.max() > 1.0 or element.min() < 0:
                            print(f"Note: Image {i+1} range: [{element.min():.2f}, {element.max():.2f}]")
                    
                    image = self.to_pil(element)
                    image = self.preprocess(image)
                    images.append(image)

            images = torch.stack(images, dim=0)
            images = images.to(self.device)

            if self.verbose:
                print(f"Batch tensor shape: {images.shape}")

        else:
            raise ValueError("Expected a list of numpy arrays")

        with torch.no_grad():
            features = self.model(images)
            
            if self.verbose:
                print(f"Features shape: {features.shape}")
                norms = torch.norm(features, dim=1)
                print(f"Feature norms - min: {norms.min().item():.2f}, max: {norms.max().item():.2f}")
        
        return features