<img src="https://r2cdn.perplexity.ai/pplx-full-logo-primary-dark%402x.png" style="height:64px;margin-right:32px"/>

# MRPC Dataset - Complete Analysis Report

The MRPC (Microsoft Research Paraphrase Corpus) is a corpus of sentence pairs automatically extracted from online news sources, with human annotations for whether the sentences in each pair are semantically equivalent (paraphrases) or not.[^1][^2]

## Dataset Overview

MRPC is part of the GLUE benchmark and contains 5,801 sentence pairs extracted from news sources on the web, each hand-labeled with a binary judgment as to whether the pair constitutes a paraphrase. The corpus was created using heuristic extraction techniques in conjunction with an SVM-based classifier, with human judges confirming that 67% were semantically equivalent.[^1]

## File Specifications

### Full File Paths and Sizes

Based on the project structure :[^3]

**Base Path**: `C:\Users\Riyat\Documents\Minor project\Project\FactCheck-MM\data\MRPC\`


| File | Size | Character Count | Purpose |
| :-- | :-- | :-- | :-- |
| **train.tsv** | 944 KB | 944,891 characters | Training data (~3,668 pairs) |
| **test.tsv** | 447 KB | 446,948 characters | Test data (~1,725 pairs) |
| **dev.tsv** | 106 KB | 105,984 characters | Development data (~408 pairs) |
| **Total** | ~1.5 MB | ~1.5M characters | 5,801 sentence pairs |

## Data Structure

### Core Features

Based on GLUE benchmark specifications, MRPC files contain tab-separated values with columns:


| Column | Type | Description | Usage | Preprocessing Required |
| :-- | :-- | :-- | :-- | :-- |
| **Quality** | Float/Binary | Data quality indicator | Optional filtering | Remove low-quality samples |
| \#1 ID | String | Unique identifier for sentence 1 | Reference tracking | No preprocessing needed |
| \#2 ID | String | Unique identifier for sentence 2 | Reference tracking | No preprocessing needed |
| \#1 String | Text | First sentence in the pair | Primary input feature | Text cleaning, tokenization |
| \#2 String | Text | Second sentence in the pair | Primary input feature | Text cleaning, tokenization |
| **Label** | Binary | Paraphrase label (0/1) | Target variable | No preprocessing needed |

### Alternative Column Naming

- **sentence1, sentence2** (instead of \#1 String, \#2 String)
- **text1, text2** (in some preprocessed versions)
- **is_paraphrase** (instead of Label)


## Sample Records Analysis

### Expected Sample Format

**Record 1 (Paraphrase - Label: 1):**

```
Quality: 1
#1 ID: 1234567
#2 ID: 1234568
#1 String: "Amrozi accused his brother, whom he called 'the witness', of deliberately distorting his evidence."
#2 String: "Referring to him as only 'the witness', Amrozi accused his brother of deliberately distorting his evidence."
Label: 1
```

**Record 2 (Non-Paraphrase - Label: 0):**

```
Quality: 1
#1 ID: 1234569
#2 ID: 1234570
#1 String: "Yucaipa owned Dominick's before selling the chain to Safeway in 1998 for $2.5 billion."
#2 String: "Yucaipa bought Dominick's in 1995 for $693 million and sold it to Safeway for $1.8 billion in 1998."
Label: 0
```


## Label Distribution Analysis

### Binary Classification Distribution

| Split | Paraphrases (1) | Non-Paraphrases (0) | Total Samples |
| :-- | :-- | :-- | :-- |
| **Training** | ~2,458 pairs (67%) | ~1,210 pairs (33%) | ~3,668 |
| **Development** | ~273 pairs (67%) | ~135 pairs (33%) | ~408 |
| **Test** | ~1,156 pairs (67%) | ~569 pairs (33%) | ~1,725 |
| **Total** | ~3,887 (67%) | ~1,914 (33%) | 5,801 |

**Distribution Characteristics:**

- **Imbalanced dataset** favoring paraphrases (67:33 ratio)
- Consistent distribution across all splits
- Sufficient samples for training transformer models
- Standard train/dev/test split ratios


## Readability Analysis

### Text Characteristics

| Metric | Value | Interpretation |
| :-- | :-- | :-- |
| **Average characters per sample** | 258 characters | Medium-length news sentences |
| **Domain** | News articles | Formal, journalistic language |
| **Reading Level** | College level | Complex sentence structures |
| **Vocabulary** | Advanced | News-specific terminology |
| **Sentence Structure** | Complex | Formal journalistic writing |

### Language Patterns

- **Formal tone**: Professional news writing style
- **Complex syntax**: Subordinate clauses, passive voice
- **Domain-specific vocabulary**: Business, politics, current events
- **Proper nouns**: Names, locations, organizations
- **Numerical data**: Dates, monetary amounts, statistics


## Preprocessing Requirements

### Text Preprocessing Pipeline

- **Sentence pair handling**: Process both sentences simultaneously
- **Punctuation normalization**: Handle quotes, apostrophes, hyphens
- **Named entity preservation**: Maintain proper nouns and dates
- **Case normalization**: Optional lowercasing
- **Tokenization**: Subword tokenization (BPE, WordPiece)


### Paraphrase-Specific Considerations

- **Semantic preservation**: Maintain meaning during preprocessing
- **Syntactic variation handling**: Account for structural differences
- **Word order sensitivity**: Preserve important ordering cues
- **Negation handling**: Critical for non-paraphrase detection


### Fields to Exclude from Model Training

- **Quality**: Metadata field, not input feature
- \#1 ID, \#2 ID: Reference identifiers, not linguistic content
- **Core modeling fields**: \#1 String, \#2 String, Label


## Model Building Considerations

### Strengths for ML Applications

- **GLUE benchmark standard**: Well-established evaluation protocol
- **High-quality annotation**: Professional human labeling
- **News domain consistency**: Coherent text style
- **Balanced complexity**: Neither too simple nor overly complex
- **Research validation**: Extensively studied in NLP research


### Technical Specifications

- **Task type**: Binary sequence pair classification
- **Input modality**: Text-only (sentence pairs)
- **Evaluation metrics**: Accuracy, F1-score (standard GLUE metrics)
- **Baseline performance**: ~65-70% accuracy for simple models, >85% for BERT-based models[^4][^5]


### Recommended Model Architectures

**Traditional Approaches:**

- **Siamese Networks**: Twin networks for sentence pair comparison
- **Feature-based**: Hand-crafted semantic similarity features
- **RNN/LSTM**: Sequential models with attention mechanisms

**Transformer-based Approaches:**

- **BERT**: Pre-trained language model fine-tuning[^4]
- **RoBERTa**: Robust optimization of BERT
- **DeBERTa**: Disentangled attention mechanisms
- **Sentence-BERT**: Specialized for sentence similarity tasks


## Access and Citation

### Official Sources

- **Microsoft Download**: https://www.microsoft.com/en-us/download/details.aspx?id=52398[^6]
- **GLUE Benchmark**: https://gluebenchmark.com/tasks[^7]
- **Hugging Face**: Multiple versions available[^8][^9]
- **Kaggle**: https://www.kaggle.com/datasets/doctri/microsoft-research-paraphrase-corpus[^10]


### Citation Information

```bibtex
@inproceedings{dolan2005automatically,
    title={Automatically constructing a corpus of sentential paraphrases},
    author={Dolan, William B and Brockett, Chris},
    booktitle={Proceedings of the Third International Workshop on Paraphrasing (IWP2005)},
    year={2005}
}
```

MRPC remains a fundamental benchmark for paraphrase detection research, providing a challenging yet manageable dataset that has driven significant advances in semantic similarity modeling and natural language understanding.[^11][^1]
<span style="display:none">[^12][^13][^14][^15][^16][^17][^18][^19][^20][^21]</span>

<div style="text-align: center">⁂</div>

[^1]: https://aclanthology.org/I05-5002.pdf

[^2]: https://www.unitxt.ai/en/1.12.1/catalog/catalog.cards.mrpc.html

[^3]: image.jpg

[^4]: https://huggingface.co/ParitKansal/BERT_Paraphrase_Detection_GLUE_MRPC

[^5]: https://www.promptlayer.com/models/glue-mrpc

[^6]: https://www.microsoft.com/en-us/download/details.aspx?id=52398

[^7]: https://gluebenchmark.com/tasks

[^8]: https://huggingface.co/datasets/SetFit/mrpc

[^9]: https://huggingface.co/datasets/nyu-mll/glue/tree/main/mrpc

[^10]: https://www.kaggle.com/datasets/doctri/microsoft-research-paraphrase-corpus

[^11]: https://openreview.net/pdf?id=rJ4km2R5t7

[^12]: https://huggingface.co/docs/datasets/v1.13.0/quickstart.html

[^13]: https://metatext.io/datasets/microsoft-research-paraphrase-corpus-(mrpc)

[^14]: https://www.tensorflow.org/datasets/catalog/glue

[^15]: https://live.european-language-grid.eu/catalogue/corpus/5038

[^16]: https://gist.github.com/W4ngatang/60c2bdb54d156a41194446737ce03e2e

[^17]: https://toolbox.google.com/datasetsearch/search?query=Paraphrase+Identification

[^18]: https://www.kaggle.com/datasets/thedevastator/nli-dataset-for-sentence-understanding

[^19]: https://zilliz.com/glossary/glue-benchmark

[^20]: https://docs.pytorch.org/text/0.16.0/_modules/torchtext/datasets/mrpc.html

[^21]: https://aclanthology.org/W18-5446.pdf

