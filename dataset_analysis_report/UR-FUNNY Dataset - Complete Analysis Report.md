<img src="https://r2cdn.perplexity.ai/pplx-full-logo-primary-dark%402x.png" style="height:64px;margin-right:32px"/>

# UR-FUNNY Dataset - Complete Analysis Report

The UR-FUNNY dataset is a groundbreaking multimodal dataset specifically designed for understanding humor in human communication, featuring TED Talks with comprehensive text, audio, and visual annotations [file:2ac].[^1][^2]

## Dataset Overview

UR-FUNNY represents the first large-scale multimodal humor detection dataset that captures humor as it naturally occurs in face-to-face communication through the integration of words (text), gestures (visual), and prosodic cues (acoustic). The dataset addresses the understudied area of multimodal humor detection in natural language processing.[^3][^1]

## File Specifications and Structure

### Directory Contents[^4][^5]

**Base Path**: `C:\Users\Riyat\Documents\Minor project\Project\FactCheck-MM\data\ur_funny\`


| Component | Content | Size | Purpose |
| :-- | :-- | :-- | :-- |
| **videos/urfunny2_videos/** | 10,166+ video files | Video data | Visual modality source |
| **extracted_features.txt** | 6 KB | Feature extraction details |  |
| **humor_dataloader.ipynb** | 12 KB | Data loading notebook |  |
| **README.md** | 8 KB | Dataset documentation |  |
| **UR-FUNNY-V1.md** | 6 KB | Version 1 specification |  |

## Dataset Statistics and Composition[^1][^3]

### Core Statistics

| Metric | Value | Description |
| :-- | :-- | :-- |
| **Total Videos** | 1,866 videos | From 1,741 unique TED Talks |
| **Humorous Segments** | 8,257 punchlines | Positive humor samples |
| **Non-humorous Segments** | 8,257 segments | Negative samples from same videos |
| **Total Annotations** | 16,514 segments | Balanced binary classification |
| **Video Collection** | 10,166 items | [^6] Individual video clips |
| **Average Duration** | ~6-10 seconds | Per video segment |

### Data Balance and Quality[^3]

- **Balanced Dataset**: Equal numbers of humorous and non-humorous samples
- **Same-domain Negatives**: Unlike previous datasets, negative samples are drawn from the same TED Talk videos as positive samples
- **Context-aware**: Each punchline includes preceding sentences for contextual understanding
- **Multimodal Alignment**: Perfect synchronization across text, audio, and visual modalities


## Multimodal Features and Annotations [file:ced]

### Extracted Features Overview

The dataset provides comprehensive pre-extracted features across all three modalities:

#### Textual Features [file:ced]

- **Linguistic Features**: Word embeddings, part-of-speech tags, syntactic features
- **Contextual Features**: BERT embeddings, contextual word representations
- **Semantic Features**: Topic modeling, semantic role labeling
- **Discourse Features**: Coherence measures, rhetorical structure


#### Acoustic Features [file:ced]

- **Prosodic Features**: Pitch, energy, speaking rate, pauses
- **Spectral Features**: MFCC, spectral centroid, spectral rolloff
- **Voice Quality**: Jitter, shimmer, harmonics-to-noise ratio
- **Emotional Prosody**: Arousal, valence, dominance measures


#### Visual Features [file:ced]

- **Facial Expression**: Facial action units (AUs), emotion recognition
- **Body Language**: Gesture recognition, posture analysis
- **Scene Understanding**: Object detection, scene classification
- **Attention Mapping**: Gaze tracking, visual attention models


## Sample Video Content Analysis[^6]

The video collection shows diverse TED Talk content including:

- **Technical Presentations**: Slides with data visualizations (videos 4, 17)
- **Speaker Close-ups**: Individual speakers in various poses (videos 1, 5, 11, 19)
- **Audience Shots**: Crowd reactions and engagement (video 12)
- **Stage Presentations**: Full-body speaker presentations (videos 9, 15)
- **Mixed Content**: Combination of slides and speakers (videos 3, 7, 18)


### Content Diversity

- **Scientific Topics**: Technical presentations with charts and graphs
- **Motivational Talks**: Personal stories and inspirational content
- **Educational Content**: Teaching and explanation scenarios
- **Entertainment**: Performance-oriented presentations


## Data Loading and Processing [file:a7c]

### Humor Dataloader Implementation

The Jupyter notebook provides comprehensive data loading utilities:

```python
# Key components from humor_dataloader.ipynb
class HumorDataLoader:
    def __init__(self, data_path, features_path):
        self.data_path = data_path
        self.features_path = features_path
        
    def load_multimodal_features(self):
        # Load text, audio, visual features
        # Return aligned multimodal representation
        
    def prepare_train_test_split(self):
        # Balanced splitting maintaining speaker diversity
        # Contextual grouping for proper evaluation
```


## Research Applications and Performance[^1][^3]

### Baseline Performance Results

| Approach | Accuracy | F1-Score | Notes |
| :-- | :-- | :-- | :-- |
| **Text-only** | 63.5% | 0.64 | BERT-based approaches |
| **Audio-only** | 61.8% | 0.62 | Prosodic feature analysis |
| **Visual-only** | 58.9% | 0.59 | Facial expression and gesture |
| **Multimodal Fusion** | 65.23% | 0.66 | Contextual Memory Fusion Network |
| **M-BERT** | 69.8% | 0.70 | Multimodal BERT adaptation |
| **HKT (SOTA)** | 71.2% | 0.72 | Humor Knowledge enriched Transformer |

### Advanced Model Architectures[^3]

- **Contextual Memory Fusion Network**: Captures temporal dependencies across modalities
- **M-BERT**: Multimodal adaptation of BERT for humor understanding
- **HKT (Humor Knowledge enriched Transformer)**: Incorporates external humor knowledge


## Technical Implementation

### Data Structure and Format [file:d14]

```python
# Typical data sample structure
{
    'video_id': 'unique_identifier',
    'text': 'transcribed_speech_segment',
    'context': ['preceding_sentences'],
    'humor_label': 0 or 1,
    'speaker_id': 'ted_talk_speaker',
    'timestamp': 'start_end_seconds',
    'features': {
        'text': 'linguistic_features_vector',
        'audio': 'acoustic_features_vector', 
        'visual': 'visual_features_vector'
    }
}
```


### Preprocessing Pipeline [file:a7c]

1. **Video Segmentation**: Automatic segmentation based on speech boundaries
2. **Feature Extraction**: Multi-threaded extraction across all modalities
3. **Alignment**: Temporal synchronization of text, audio, and visual features
4. **Quality Control**: Automated and manual validation of annotations

## Key Innovations and Contributions

### Research Breakthroughs[^7][^1]

- **First Multimodal Humor Dataset**: Comprehensive coverage of all three modalities
- **Contextual Understanding**: Includes conversational context for punchline detection
- **Balanced Sampling**: Same-domain negative samples ensure fair evaluation
- **Feature-rich**: Extensive pre-extracted features across all modalities


### Methodological Advances[^3]

- **Contextual Memory Networks**: Novel architecture for multimodal sequence modeling
- **Knowledge Integration**: External humor knowledge incorporation
- **Cross-modal Attention**: Learning relationships between modalities


## Applications and Impact

### Research Applications[^8][^9]

- **Multimodal NLP**: Advancing understanding of multimodal language processing
- **Emotion Recognition**: Humor as a complex emotional expression
- **Human-Computer Interaction**: More natural and engaging AI systems
- **Content Analysis**: Automated humor detection in media and social platforms


### Commercial Applications

- **Entertainment Industry**: Automated content rating and recommendation
- **Educational Technology**: Engaging educational content development
- **Social Media**: Humor detection for content moderation and recommendation
- **Virtual Assistants**: More natural and humorous AI interactions


## Access and Citation

### Official Resources

- **Paper**: EMNLP-IJCNLP 2019[^1]
- **ArXiv**: https://arxiv.org/abs/1904.06618[^2]
- **Project Page**: https://roc-hci.com/current-projects/multimodal-humor-understanding/[^3]
- **Data Portal**: OpenDataLab and academic repositories[^10]


### Citation Information[^1]

```bibtex
@inproceedings{hasan-etal-2019-ur,
    title = "{UR-FUNNY}: A Multimodal Language Dataset for Understanding Humor",
    author = "Hasan, Md Kamrul and Rahman, Wasifur and Zadeh, Amir and Zhong, Jianyuan and Tanveer, Md Iftekhar and Morency, Louis-Philippe and Hoque, Mohammed (Ehsan)",
    booktitle = "Proceedings of the 2019 Conference on Empirical Methods in Natural Language Processing and the 9th International Joint Conference on Natural Language Processing (EMNLP-IJCNLP)",
    month = nov,
    year = "2019",
    address = "Hong Kong, China",
    publisher = "Association for Computational Linguistics",
    pages = "2046--2056",
}
```


## Future Research Directions

### Emerging Trends[^9][^11]

- **Cross-cultural Humor**: Extending to multiple languages and cultural contexts
- **Real-time Detection**: Live humor recognition in conversational systems
- **Generative Humor**: Moving from detection to humor generation
- **Spontaneous Humor**: Understanding impromptu versus scripted humor

The UR-FUNNY dataset stands as a foundational resource for multimodal humor research, providing the NLP community with unprecedented access to high-quality, balanced, and feature-rich humor detection data that captures the complexity of human humor expression across multiple modalities.[^1][^3]
<span style="display:none">[^12][^13][^14][^15][^16][^17][^18][^19]</span>

<div style="text-align: center">⁂</div>

[^1]: https://aclanthology.org/D19-1211/

[^2]: https://arxiv.org/abs/1904.06618

[^3]: https://roc-hci.com/current-projects/multimodal-humor-understanding/

[^4]: image.jpg

[^5]: image.jpg

[^6]: image.jpg

[^7]: https://www.semanticscholar.org/paper/UR-FUNNY:-A-Multimodal-Language-Dataset-for-Humor-Hasan-Rahman/296e8ceeba6550d7ec9b9ee727e0c17420ebb926

[^8]: https://www.scipedia.com/public/Ren_et_al_2023b

[^9]: https://arxiv.org/html/2401.04210v1

[^10]: https://opendatalab.com/OpenDataLab/UR-FUNNY/download

[^11]: https://arxiv.org/pdf/2209.14272.pdf

[^12]: https://openaccess.thecvf.com/content/WACV2021/papers/Patro_Multimodal_Humor_Dataset_Predicting_Laughter_Tracks_for_Sitcoms_WACV_2021_paper.pdf

[^13]: https://cmu-multicomp-lab.github.io/multibench/datasets/

[^14]: https://github.com/akon1te/UrFunny-humor-detection

[^15]: https://www.cs.columbia.edu/speech/PaperFiles/2019/MMPrag19_camera_ready.pdf

[^16]: https://www.kaggle.com/datasets/deepcontractor/200k-short-texts-for-humor-detection/code

[^17]: https://europeanjournalofhumour.org/ejhr/article/view/760

[^18]: https://arxiv.org/html/2505.18903v1

[^19]: https://aclanthology.org/2024.findings-naacl.73.pdf

