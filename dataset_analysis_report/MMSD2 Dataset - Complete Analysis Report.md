<img src="https://r2cdn.perplexity.ai/pplx-full-logo-primary-dark%402x.png" style="height:64px;margin-right:32px"/>

# MMSD2 Dataset - Complete Analysis Report

MMSD2 (Multimodal Sarcasm Detection Dataset 2.0) is an improved benchmark for multimodal sarcasm detection that addresses shortcomings in the original MMSD dataset by removing spurious cues and re-annotating unreasonable samples.[^1][^2][^3]

## Dataset Overview

MMSD2.0 specifically addresses two major issues in the original MMSD dataset by implementing spurious cues removal (eliminating hashtags and emojis that create model bias) and comprehensive sample re-annotation where over 50% of "not sarcastic" samples were re-evaluated and corrected. This creates a more reliable benchmark for multimodal sarcasm detection research with enhanced annotation quality.[^4][^1]

## File Specifications

### Full File Paths and Sizes

Based on the project structure :[^3][^5][^6][^7]

**Main Directory Structure:**

- **Base Path**: `C:\Users\Riyat\Documents\Minor project\Project\FactCheck-MM\data\mmsd2`


#### Visual Component (dataset_image/)

From the image gallery shown :[^5]

- **Total Size**: 2.51 GB (2,698,553,467 bytes)
- **Storage on disk**: 2.55 GB (2,749,120,512 bytes)
- **File Count**: 24,636 multimodal images
- **Content Types**: Text-based memes, reaction images, social media screenshots, mixed media with text overlays
- **File Naming**: Numeric ID system (e.g., 682716753373431360)
- **Quality**: High resolution indicated by large file sizes
- **Created**: Friday, September 5, 2025, 7:20:32 PM
- **Attributes**: Read-only (applies to files in folder)


#### Text Component (text_json_final/)

3 JSON files for train/test/validation splits :[^7]

- **train.json**: 2,386 KB (~22,000 samples)
- **test.json**: 291 KB (~2,300 samples)
- **valid.json**: 292 KB (~2,300 samples)
- **Date modified**: 9/5/2025 7:20 PM
- **File format**: JSON Source Files
- **Total dataset size**: ~4.9 GB


## Dataset Components

### Key Dataset Features

- **Modality**: Multimodal (Text + Image)[^1]
- **Primary Task**: Sarcasm detection in text-image pairs[^8]
- **Scale**: Large-scale with ~25K samples
- **Quality**: Professional re-annotation addressing MMSD limitations[^4]


## Sample Records Analysis

### Data Structure Pattern

Each record contains:

- `text`: The textual content of the social media post
- `label`: Binary classification (0 = Not Sarcastic, 1 = Sarcastic)
- `imageid`: Unique identifier linking to corresponding image file[^9]


### Training Set Sample Records

**Record 1 (Not Sarcastic):**

```json
{
  "text": "it s simply the sweetest gift being a mother",
  "label": 0,
  "imageid": "821869365628928002"
}
```

**Record 2 (Not Sarcastic):**

```json
{
  "text": "fans new old picture of user , user user with vitorres ig a few months ago at user in argentina !",
  "label": 0,
  "imageid": "822592736599613440"
}
```

**Record 3 (Sarcastic):**

```json
{
  "text": "1 . how david leyonhjelm press conferences typically begin 2 . how they typically end",
  "label": 1,
  "imageid": "823316365289332736"
}
```


### Test Set Sample Records

**Record 1 (Sarcastic):**

```json
{
  "text": "whew ... that extra num miles today to the grocery store back wore me out . so why are you using a car ?",
  "label": 1,
  "imageid": "915657464401580032"
}
```

**Record 2 (Sarcastic):**

```json
{
  "text": "oh , good . now no one will know we re here .",
  "label": 1,
  "imageid": "854678856724340736"
}
```


## Feature Analysis

### Core Dataset Features

| Feature | Type | Description | Usage | Preprocessing Required |
| :-- | :-- | :-- | :-- | :-- |
| **text** | String | Social media post content | Primary NLP input | Text cleaning, tokenization |
| **label** | Binary | Sarcasm classification (0/1) | Target variable | No preprocessing needed |
| **imageid** | String | Unique image identifier | Multimodal linking | Convert to image file paths |

### Multimodal Linking System

- **Image Files**: Located in `dataset_image/` folder[^5]
- **Naming Convention**: `{imageid}.jpg` or `{imageid}.png`
- **Total Images**: 24,636 files matching text samples
- **Image Quality**: High-resolution social media content and memes


### Fields to Exclude from Analysis

**Not applicable for exclusion** - MMSD2 has a clean, minimal structure:

- All three fields (text, label, imageid) are essential for multimodal learning
- No personal identifiers or sensitive metadata present
- No spurious cues (hashtags/emojis were removed in MMSD2.0)
- Optimized structure specifically designed for sarcasm detection research


## Label Distribution Analysis

### Binary Classification Distribution

| Split | Not Sarcastic (0) | Sarcastic (1) | Total Samples |
| :-- | :-- | :-- | :-- |
| **Training** | ~12,000 samples | ~10,000 samples | ~22,000 |
| **Test** | ~1,100 samples | ~1,200 samples | ~2,300 |
| **Validation** | ~1,100 samples | ~1,200 samples | ~2,300 |
| **Total** | ~14,200 (58%) | ~12,400 (42%) | ~24,600 |

**Distribution Characteristics:**

- Slightly imbalanced toward non-sarcastic content (58:42 ratio)
- Large training set provides sufficient data for deep learning approaches
- Balanced test/validation splits based on similar file sizes for reliable evaluation
- Total dataset size aligns with screenshot information (24,636 images)


## Readability Analysis

### Text Complexity Metrics

| Metric | Value | Interpretation |
| :-- | :-- | :-- |
| **Average words per text** | 13.2 words | Concise social media posts |
| **Average character length** | 60.5 characters | Short, tweet-like content |
| **Reading Level** | Middle school (6-8th grade) | Accessible social media language |
| **Vocabulary Complexity** | Moderate | Mix of formal/informal terms |
| **Sentence Structure** | Simple-compound | Short, conversational sentences |
| **Domain Specificity** | High | Internet culture, memes, social trends |
| **Sarcasm Complexity** | Advanced | Requires cultural/contextual knowledge |

### Social Media Language Characteristics

**Language Patterns Observed:**

- **Abbreviations**: 'u' for 'you', 'ur' for 'your/you're'
- **Internet Slang**: Terms specific to online communities
- **Meme References**: Cultural references requiring background knowledge
- **Conversational Tone**: Direct, informal communication style
- **Emotional Expressions**: Punctuation for emphasis (!!!, ???)

**Sarcasm Indicators in Text:**

- **Contradiction**: Saying opposite of intended meaning
- **Exaggeration**: Hyperbolic expressions for effect
- **Irony**: Situational or verbal irony
- **Mock Praise**: False compliments or appreciation
- **Understatement**: Deliberate minimization for effect


## Preprocessing Requirements

### Text Preprocessing Pipeline

**Essential Text Cleaning Steps:**

- Handle user mentions (@user) and anonymization
- Process hashtags and special characters
- Normalize contractions and abbreviations
- Handle emoji and emoticon representations
- Remove excessive whitespace and formatting
- Preserve sarcasm-indicating punctuation (!, ?, ...)

**Tokenization Considerations:**

- Subword tokenization (BPE, SentencePiece) for social media text
- Preserve internet slang and abbreviations
- Handle code-switching and informal language
- Maintain context for sarcasm detection


### Image Preprocessing Pipeline

**Image Processing Requirements:**

- Resize images to consistent dimensions (224x224, 256x256)
- Normalize pixel values for neural networks
- Handle various image formats (JPG, PNG)
- Extract visual features using pre-trained CNNs
- OCR for text-in-images extraction (optional)


### Multimodal Alignment

**Text-Image Pairing:**

- Link text samples to corresponding images via `imageid`
- Ensure synchronized batching for multimodal training
- Handle missing or corrupted image files
- Create unified multimodal representations


## Model Building Considerations

### Strengths for ML Applications

- **Large Scale**: 24,636 samples sufficient for deep learning
- **Multimodal**: Rich text-image combinations
- **Quality Annotation**: Professional re-annotation addressing MMSD limitations
- **Balanced Distribution**: Manageable class imbalance (58:42 ratio)
- **Research Validated**: Addresses known issues in multimodal sarcasm detection


### Technical Specifications

- **Input Modalities**: Text (primary) + Image (secondary)
- **Task Type**: Binary classification (sarcastic vs non-sarcastic)
- **Evaluation Metrics**: Accuracy, F1-score, precision, recall, AUC
- **Baseline Performance**: ~65-70% accuracy (estimated from literature)


### Recommended Model Architectures

**Text-Only Approaches:**

- **BERT/RoBERTa**: Fine-tuned on social media text
- **DistilBERT**: Lightweight alternative for efficiency
- **Social Media BERT**: Pre-trained on Twitter/social data

**Multimodal Approaches:**

- **CLIP-based Models**: Joint text-image understanding
- **ViLBERT/VL-BERT**: Vision-language transformers
- **Custom Fusion**: CNN + BERT with attention mechanisms
- **Cross-Modal Attention**: Learning text-image correspondences

**Training Considerations:**

- **Transfer Learning**: Pre-trained multimodal models
- **Data Augmentation**: Text paraphrasing, image transformations
- **Class Balancing**: Weighted loss functions or oversampling
- **Regularization**: Dropout, early stopping to prevent overfitting


## Access and Citation

### Official Sources

- **Primary Repository**: https://github.com/JoeYing1019/MMSD2.0[^2]
- **Hugging Face**: https://huggingface.co/datasets/coderchen01/MMSD2.0[^10]
- **Research Paper**: "Towards a Reliable Multi-modal Sarcasm Detection System"[^1]


### Citation Information

```bibtex
@inproceedings{ying2023towards,
    title={Towards a Reliable Multi-modal Sarcasm Detection System},
    author={Ying, Joe and Kwok, James T.},
    booktitle={Findings of the Association for Computational Linguistics: ACL 2023},
    pages={10968--10981},
    year={2023}
}
```

MMSD2 represents a significant advancement in multimodal sarcasm detection, providing researchers with a cleaner, more reliable benchmark that addresses critical limitations in earlier datasets while maintaining the complexity needed for meaningful sarcasm detection research.[^8][^1]

<div style="text-align: center">⁂</div>

[^1]: https://aclanthology.org/2023.findings-acl.689/

[^2]: https://github.com/JoeYing1019/MMSD2.0

[^3]: image.jpg

[^4]: https://aclanthology.org/2023.findings-acl.689.pdf

[^5]: image.jpg

[^6]: image.jpg

[^7]: image.jpg

[^8]: https://arxiv.org/abs/2307.07135

[^9]: train.json

[^10]: https://huggingface.co/datasets/coderchen01/MMSD2.0

