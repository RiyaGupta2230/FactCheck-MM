"""
Image Preprocessing for FactCheck-MM
Handles resizing, normalization, and augmentations for ViT models.
"""

import cv2
import numpy as np
import torch
import torchvision.transforms as transforms
from PIL import Image, ImageEnhance, ImageOps
from typing import Dict, List, Optional, Tuple, Union, Any
import albumentations as A
from transformers import ViTImageProcessor
from pathlib import Path

from ..utils import get_logger


class ImageProcessor:
    """
    Comprehensive image processor for multimodal analysis.
    Handles preprocessing for vision transformers and CNN models.
    """
    
    def __init__(
        self,
        model_name: str = "google/vit-large-patch16-224",
        image_size: int = 224,
        normalize: bool = True,
        augment_training: bool = True,
        augment_strength: float = 0.5,
        preserve_aspect_ratio: bool = False,
        cache_dir: Optional[str] = None
    ):
        """
        Initialize image processor.
        
        Args:
            model_name: ViT model name for processor
            image_size: Target image size
            normalize: Whether to normalize pixel values
            augment_training: Whether to apply augmentations for training
            augment_strength: Augmentation strength (0-1)
            preserve_aspect_ratio: Whether to preserve aspect ratio
            cache_dir: Cache directory
        """
        self.model_name = model_name
        self.image_size = image_size
        self.normalize = normalize
        self.augment_training = augment_training
        self.augment_strength = augment_strength
        self.preserve_aspect_ratio = preserve_aspect_ratio
        
        self.logger = get_logger("ImageProcessor")
        
        # Initialize ViT processor
        try:
            self.vit_processor = ViTImageProcessor.from_pretrained(
                model_name,
                cache_dir=cache_dir
            )
            self.logger.info(f"Loaded ViT processor: {model_name}")
        except Exception as e:
            self.logger.error(f"Failed to load ViT processor: {e}")
            raise
        
        # Initialize transforms
        self._setup_transforms()
        
        self.logger.info("Image processor initialized successfully")
    
    def _setup_transforms(self):
        """Setup image transforms for different scenarios."""
        
        # Basic preprocessing transforms
        self.basic_transforms = transforms.Compose([
            transforms.Resize(
                (self.image_size, self.image_size),
                interpolation=transforms.InterpolationMode.BICUBIC
            ),
            transforms.ToTensor(),
        ])
        
        # Normalization (will use ViT processor values)
        if hasattr(self.vit_processor, 'image_mean') and hasattr(self.vit_processor, 'image_std'):
            mean = self.vit_processor.image_mean
            std = self.vit_processor.image_std
        else:
            # Default ImageNet normalization
            mean = [0.485, 0.456, 0.406]
            std = [0.229, 0.224, 0.225]
        
        self.normalize_transform = transforms.Normalize(mean=mean, std=std)
        
        # Training augmentations using Albumentations
        augmentation_list = []
        
        if self.augment_training and self.augment_strength > 0:
            prob = min(self.augment_strength, 1.0)
            
            augmentation_list.extend([
                A.HorizontalFlip(p=0.5),
                A.RandomBrightnessContrast(
                    brightness_limit=0.2 * prob,
                    contrast_limit=0.2 * prob,
                    p=0.5
                ),
                A.HueSaturationValue(
                    hue_shift_limit=20 * prob,
                    sat_shift_limit=20 * prob,
                    val_shift_limit=20 * prob,
                    p=0.3
                ),
                A.RandomGamma(
                    gamma_limit=(80, 120),
                    p=0.3
                ),
                A.GaussianBlur(
                    blur_limit=(3, 7),
                    p=0.2
                ),
                A.GaussNoise(
                    var_limit=(10.0, 50.0 * prob),
                    p=0.2
                ),
                A.ShiftScaleRotate(
                    shift_limit=0.1 * prob,
                    scale_limit=0.1 * prob,
                    rotate_limit=15 * prob,
                    p=0.5
                ),
                A.CoarseDropout(
                    max_holes=8,
                    max_height=16,
                    max_width=16,
                    p=0.2
                )
            ])
        
        self.augmentation_pipeline = A.Compose(augmentation_list)
        
        # Test-time augmentations (TTA)
        self.tta_transforms = A.Compose([
            A.HorizontalFlip(p=1.0),
        ])
    
    def load_image(
        self,
        image_path: Union[str, Path],
        mode: str = "RGB"
    ) -> Image.Image:
        """
        Load image from file.
        
        Args:
            image_path: Path to image file
            mode: Color mode ('RGB', 'RGBA', 'L')
            
        Returns:
            PIL Image
        """
        try:
            image = Image.open(image_path).convert(mode)
            self.logger.debug(f"Loaded image: {image_path} (size: {image.size})")
            return image
        except Exception as e:
            self.logger.error(f"Failed to load image {image_path}: {e}")
            raise
    
    def resize_image(
        self,
        image: Union[Image.Image, np.ndarray],
        size: Optional[Tuple[int, int]] = None,
        maintain_aspect_ratio: Optional[bool] = None
    ) -> Image.Image:
        """
        Resize image to target size.
        
        Args:
            image: Input image
            size: Target size (width, height)
            maintain_aspect_ratio: Whether to maintain aspect ratio
            
        Returns:
            Resized image
        """
        if isinstance(image, np.ndarray):
            image = Image.fromarray(image)
        
        if size is None:
            size = (self.image_size, self.image_size)
        
        if maintain_aspect_ratio is None:
            maintain_aspect_ratio = self.preserve_aspect_ratio
        
        if maintain_aspect_ratio:
            # Resize maintaining aspect ratio and pad if necessary
            image.thumbnail(size, Image.LANCZOS)
            
            # Create new image with target size and paste resized image
            new_image = Image.new('RGB', size, (128, 128, 128))  # Gray background
            
            # Center the image
            x = (size[0] - image.width) // 2
            y = (size[1] - image.height) // 2
            new_image.paste(image, (x, y))
            
            image = new_image
        else:
            # Direct resize (may distort)
            image = image.resize(size, Image.LANCZOS)
        
        return image
    
    def apply_augmentations(
        self,
        image: Union[Image.Image, np.ndarray],
        training: bool = True
    ) -> np.ndarray:
        """
        Apply augmentations to image.
        
        Args:
            image: Input image
            training: Whether this is for training (applies stronger augmentations)
            
        Returns:
            Augmented image as numpy array
        """
        # Convert to numpy array if PIL Image
        if isinstance(image, Image.Image):
            image_array = np.array(image)
        else:
            image_array = image.copy()
        
        # Apply augmentations
        if training and self.augment_training:
            augmented = self.augmentation_pipeline(image=image_array)
            return augmented['image']
        else:
            return image_array
    
    def apply_tta(self, image: Union[Image.Image, np.ndarray]) -> List[np.ndarray]:
        """
        Apply test-time augmentations.
        
        Args:
            image: Input image
            
        Returns:
            List of augmented images
        """
        if isinstance(image, Image.Image):
            image_array = np.array(image)
        else:
            image_array = image.copy()
        
        tta_images = [image_array]  # Original image
        
        # Apply TTA transforms
        augmented = self.tta_transforms(image=image_array)
        tta_images.append(augmented['image'])
        
        return tta_images
    
    def enhance_image(
        self,
        image: Image.Image,
        enhance_contrast: bool = True,
        enhance_brightness: bool = True,
        enhance_color: bool = True,
        enhance_sharpness: bool = True,
        factor_range: Tuple[float, float] = (0.8, 1.2)
    ) -> Image.Image:
        """
        Apply image enhancements.
        
        Args:
            image: Input image
            enhance_contrast: Whether to enhance contrast
            enhance_brightness: Whether to enhance brightness
            enhance_color: Whether to enhance color
            enhance_sharpness: Whether to enhance sharpness
            factor_range: Range of enhancement factors
            
        Returns:
            Enhanced image
        """
        min_factor, max_factor = factor_range
        
        if enhance_contrast:
            factor = np.random.uniform(min_factor, max_factor)
            enhancer = ImageEnhance.Contrast(image)
            image = enhancer.enhance(factor)
        
        if enhance_brightness:
            factor = np.random.uniform(min_factor, max_factor)
            enhancer = ImageEnhance.Brightness(image)
            image = enhancer.enhance(factor)
        
        if enhance_color:
            factor = np.random.uniform(min_factor, max_factor)
            enhancer = ImageEnhance.Color(image)
            image = enhancer.enhance(factor)
        
        if enhance_sharpness:
            factor = np.random.uniform(min_factor, max_factor)
            enhancer = ImageEnhance.Sharpness(image)
            image = enhancer.enhance(factor)
        
        return image
    
    def extract_visual_features(self, image: np.ndarray) -> Dict[str, float]:
        """
        Extract basic visual features for analysis.
        
        Args:
            image: Input image array
            
        Returns:
            Dictionary of visual features
        """
        # Convert to grayscale for some features
        if len(image.shape) == 3:
            gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
        else:
            gray = image
        
        features = {}
        
        # Basic statistics
        features['mean_brightness'] = float(np.mean(image))
        features['std_brightness'] = float(np.std(image))
        features['mean_gray'] = float(np.mean(gray))
        features['std_gray'] = float(np.std(gray))
        
        # Color statistics (if color image)
        if len(image.shape) == 3:
            features['mean_red'] = float(np.mean(image[:,:,0]))
            features['mean_green'] = float(np.mean(image[:,:,1]))
            features['mean_blue'] = float(np.mean(image[:,:,2]))
        
        # Edge density (using Canny edge detection)
        edges = cv2.Canny(gray, 50, 150)
        features['edge_density'] = float(np.mean(edges > 0))
        
        # Texture features (using Laplacian variance)
        features['texture_variance'] = float(cv2.Laplacian(gray, cv2.CV_64F).var())
        
        # Contrast (RMS contrast)
        features['rms_contrast'] = float(np.sqrt(np.mean((gray - np.mean(gray)) ** 2)))
        
        return features
    
    def process_image(
        self,
        image: Union[str, Path, Image.Image, np.ndarray],
        training: bool = False,
        return_features: bool = False,
        apply_tta: bool = False
    ) -> Dict[str, Any]:
        """
        Complete image processing pipeline.
        
        Args:
            image: Image input (path, PIL Image, or numpy array)
            training: Whether this is for training
            return_features: Whether to extract visual features
            apply_tta: Whether to apply test-time augmentation
            
        Returns:
            Dictionary with processed image and metadata
        """
        # Load image if path provided
        if isinstance(image, (str, Path)):
            pil_image = self.load_image(image)
        elif isinstance(image, np.ndarray):
            pil_image = Image.fromarray(image)
        else:
            pil_image = image
        
        # Resize image
        resized_image = self.resize_image(pil_image)
        
        # Apply augmentations
        if apply_tta and not training:
            # Test-time augmentation
            image_arrays = self.apply_tta(resized_image)
            processed_images = []
            
            for img_array in image_arrays:
                # Convert to PIL for ViT processor
                pil_img = Image.fromarray(img_array)
                processed_images.append(pil_img)
        else:
            # Single image processing
            augmented_array = self.apply_augmentations(resized_image, training)
            processed_images = [Image.fromarray(augmented_array)]
        
        # Prepare result
        result = {
            'images': processed_images,
            'original_size': pil_image.size,
            'processed_size': (self.image_size, self.image_size),
            'num_images': len(processed_images)
        }
        
        # Extract features if requested
        if return_features:
            # Use first processed image for feature extraction
            img_array = np.array(processed_images[0])
            result['features'] = self.extract_visual_features(img_array)
        
        return result
    
    def process_for_vit(
        self,
        image: Union[str, Path, Image.Image, np.ndarray],
        training: bool = False
    ) -> Dict[str, torch.Tensor]:
        """
        Process image for ViT model input.
        
        Args:
            image: Image input
            training: Whether this is for training
            
        Returns:
            ViT processor output
        """
        # Process image
        processed = self.process_image(image, training=training)
        images = processed['images']
        
        # Use ViT processor (handles normalization)
        inputs = self.vit_processor(
            images=images[0],  # Use first image
            return_tensors="pt"
        )
        
        return inputs
    
    def process_batch(
        self,
        images: List[Union[str, Path, Image.Image, np.ndarray]],
        training: bool = False
    ) -> Dict[str, torch.Tensor]:
        """
        Process a batch of images.
        
        Args:
            images: List of image inputs
            training: Whether this is for training
            
        Returns:
            Batched processor outputs
        """
        processed_images = []
        
        for image in images:
            processed = self.process_image(image, training=training)
            processed_images.append(processed['images'][0])
        
        # Use ViT processor for batching
        inputs = self.vit_processor(
            images=processed_images,
            return_tensors="pt"
        )
        
        return inputs
    
    def save_image(
        self,
        image: Union[Image.Image, np.ndarray],
        output_path: Union[str, Path],
        quality: int = 95
    ) -> None:
        """
        Save processed image to file.
        
        Args:
            image: Image to save
            output_path: Output file path
            quality: JPEG quality (0-100)
        """
        if isinstance(image, np.ndarray):
            image = Image.fromarray(image)
        
        image.save(output_path, quality=quality, optimize=True)
        self.logger.info(f"Saved image to {output_path}")
    
    def __repr__(self) -> str:
        return (
            f"ImageProcessor(\n"
            f"  model_name='{self.model_name}',\n"
            f"  image_size={self.image_size},\n"
            f"  augmentation={self.augment_training},\n"
            f"  preserve_aspect_ratio={self.preserve_aspect_ratio}\n"
            f")"
        )
