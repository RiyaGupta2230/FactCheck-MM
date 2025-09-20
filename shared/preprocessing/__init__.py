"""
FactCheck-MM Preprocessing Modules
Unified preprocessing for text, audio, image, and video data.
"""

from .text_processor import TextProcessor
from .audio_processor import AudioProcessor
from .image_processor import ImageProcessor
from .video_processor import VideoProcessor

__all__ = [
    "TextProcessor",
    "AudioProcessor", 
    "ImageProcessor",
    "VideoProcessor"
]

def get_processor(modality: str, **kwargs):
    """
    Get processor for specific modality.
    
    Args:
        modality: Modality type ('text', 'audio', 'image', 'video')
        **kwargs: Processor-specific arguments
        
    Returns:
        Appropriate processor instance
    """
    processors = {
        "text": TextProcessor,
        "audio": AudioProcessor,
        "image": ImageProcessor,
        "video": VideoProcessor
    }
    
    if modality not in processors:
        raise ValueError(f"Unknown modality: {modality}. Available: {list(processors.keys())}")
    
    return processors[modality](**kwargs)
