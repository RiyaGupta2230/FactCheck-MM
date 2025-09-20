<img src="https://r2cdn.perplexity.ai/pplx-full-logo-primary-dark%402x.png" style="height:64px;margin-right:32px"/>

# FEVER Dataset - Complete Analysis Report

The FEVER dataset provides a comprehensive benchmark for automated fact verification with rich structural features and balanced label distribution.[^1][^2]

## File Specifications

### Full File Paths and Sizes

Based on the project structure shown in the screenshots :[^3][^4]

- **fever_train.jsonl**: `C:\Users\Riyat\Documents\Minor project\Project\FactCheck-MM\data\FEVER\fever_train.jsonl`
    - File size: 61,592 KB (61.6 MB)
- **fever_test.jsonl**: `C:\Users\Riyat\Documents\Minor project\Project\FactCheck-MM\data\FEVER\fever_test.jsonl`
    - File size: 7,495 KB (7.5 MB)
- **Total dataset size**: 69,087 KB (69.1 MB)

The approximately 8:1 size ratio indicates a standard train-test split configuration suitable for machine learning applications.[^4][^3]

## Sample Records Analysis

### Training File Sample Records (fever_train.jsonl)

**Record 1:**

```json
{
  "id": 75397,
  "label": "SUPPORTS", 
  "claim": "Nikolaj Coster-Waldau worked with the Fox Broadcasting Company.",
  "evidence_annotation_id": 92286,
  "evidence_id": 104971,
  "evidence_wiki_url": "Fox_Broadcasting_Company",
  "evidence_sentence_id": -1
}
```

**Record 2:**

```json
{
  "id": 150448,
  "label": "SUPPORTS",
  "claim": "Roman Atwood is a content creator.",
  "evidence_annotation_id": 174271,
  "evidence_id": 187498,
  "evidence_wiki_url": "Roman_Atwood", 
  "evidence_sentence_id": 1
}
```

**Record 3:**

```json
{
  "id": 214861,
  "label": "SUPPORTS",
  "claim": "History of art includes architecture, dance, sculpture, music, painting, poetry literature, theatre, narrative, film, photography and graphic arts.",
  "evidence_annotation_id": 255136,
  "evidence_id": 254645,
  "evidence_wiki_url": "History_of_art",
  "evidence_sentence_id": 2
}
```


### Test File Sample Records (fever_test.jsonl)

**Record 1:**

```json
{
  "id": 91198,
  "label": "NOT ENOUGH INFO",
  "claim": "Colin Kaepernick became a starting quarterback during the 49ers 63rd season in the National Football League.",
  "evidence_annotation_id": 108548,
  "evidence_id": -1,
  "evidence_wiki_url": "",
  "evidence_sentence_id": -1
}
```

**Record 2:**

```json
{
  "id": 194462,
  "label": "NOT ENOUGH INFO", 
  "claim": "Tilda Swinton is a vegan.",
  "evidence_annotation_id": 227768,
  "evidence_id": -1,
  "evidence_wiki_url": "",
  "evidence_sentence_id": -1
}
```


## Feature Analysis

### Core Features

| Feature | Type | Description | Usage | Preprocessing Required |
| :-- | :-- | :-- | :-- | :-- |
| **id** | Integer | Unique identifier for each claim | Primary key for data indexing | No preprocessing required |
| **label** | Categorical | Target variable (SUPPORTS, REFUTES, NOT ENOUGH INFO) | Classification target | Label encoding for model training |
| **claim** | Text | Main factual statement to verify | Primary NLP input feature | Text cleaning, tokenization, normalization |
| **evidence_annotation_id** | Integer | Evidence annotation identifier | Metadata for evidence linking | Can be excluded from model training |
| **evidence_id** | Integer | Unique evidence identifier (-1 = no evidence) | Evidence retrieval validation | Handle -1 as missing evidence indicator |
| **evidence_wiki_url** | String | Wikipedia URL of evidence source | External reference | URL parsing, handle empty strings |
| **evidence_sentence_id** | Integer | Specific sentence within evidence (-1 = no specific sentence) | Fine-grained evidence location | Handle -1 values appropriately |

### Fields to Exclude from Model Training

**Recommended exclusions for direct classification models:**

- `evidence_annotation_id`: Used for evidence retrieval/validation, not direct model input
- `evidence_id`: Used for evidence retrieval/validation, not direct model input
- `evidence_wiki_url`: Used for evidence retrieval/validation, not direct model input
- `evidence_sentence_id`: Used for evidence retrieval/validation, not direct model input

**Core modeling fields:**

- `id`: For data tracking and evaluation
- `claim`: Primary input text feature
- `label`: Target classification variable


## Label Distribution Analysis

### Distribution from Sample Analysis

From the visible sample records :[^2][^3]


| Label | Count | Percentage |
| :-- | :-- | :-- |
| **SUPPORTS** | 4 samples | 44.4% |
| **REFUTES** | 2 samples | 22.2% |
| **NOT ENOUGH INFO** | 3 samples | 33.3% |

### Official Dataset Statistics

According to research literature, the complete FEVER dataset contains **185,445 claims** with a relatively balanced three-class distribution across SUPPORTS, REFUTES, and NOT ENOUGH INFO categories.[^5][^6]

## Preprocessing Rules

### Text Cleaning (claim field)

- Remove extra whitespace and normalize spacing
- Handle special characters and unicode normalization
- Convert to lowercase for consistency (optional)
- Remove or standardize punctuation
- Handle contractions expansion if needed


### Label Encoding Options

- **Numeric encoding**: SUPPORTS → 0, REFUTES → 1, NOT ENOUGH INFO → 2
- **One-hot encoding**: For multi-class classification compatibility
- **String preservation**: Keep original labels for interpretability


### Missing Data Handling

- `evidence_id = -1`: Indicates no evidence available (valid data point)
- Empty `evidence_wiki_url`: Indicates no source reference (valid condition)
- `evidence_sentence_id = -1`: No specific sentence identified (valid state)
- **Important**: These are not missing values but meaningful indicators


### Text Tokenization

- Use transformer-compatible tokenizers (BERT, RoBERTa)
- Handle maximum sequence length (512 tokens typical)
- Apply padding and truncation as needed
- Create attention masks for transformer models


## Readability Analysis

### Readability Metrics

From analysis of sample claims :[^2][^3]


| Metric | Value |
| :-- | :-- |
| **Average words per claim** | 8.9 words |
| **Average word length** | 5.8 characters |
| **Vocabulary diversity** | 66 unique words in samples |
| **Simple claims (≤10 words)** | 77.8% |
| **Complex claims (>15 words)** | 22.2% |

### Text Characteristics

- **Reading level**: Moderate complexity with technical/factual content
- **Language style**: Encyclopedic, factual statements
- **Domain coverage**: Entertainment, geography, history, technology, sports
- **Sentence structure**: Predominantly declarative statements
- **Vocabulary**: Mix of common and domain-specific terminology

Research indicates that FEVER claims maintain consistent readability patterns across true and false content, making them suitable for unbiased model training.[^7]

## Model Building Considerations

### Strengths for ML Applications

- **Large scale**: 185,445 claims provide sufficient training data
- **Balanced structure**: Clear train/test splits for evaluation
- **Standardized format**: JSON Lines enables efficient batch processing
- **Rich annotations**: Evidence linking supports sophisticated modeling approaches


### Technical Specifications

- **Input features**: Textual claims requiring verification
- **Target variable**: Three-class categorical labels
- **Evaluation metrics**: Classification accuracy plus evidence retrieval metrics
- **Baseline performance**: 31.87% accuracy with evidence, 50.91% without evidence[^8]


### Recommended Model Architectures

- **Transformer-based models**: BERT, RoBERTa for text classification
- **Multi-task learning**: Joint claim classification and evidence retrieval
- **Pipeline approaches**: Separate evidence retrieval and claim verification stages
<span style="display:none">[^10][^11][^12][^13][^14][^15][^16][^9]</span>

<div style="text-align: center">⁂</div>

[^1]: image.jpg

[^2]: image.jpg

[^3]: image.jpg

[^4]: image.jpg

[^5]: https://arxiv.org/pdf/2312.05834.pdf

[^6]: https://huggingface.co/datasets/fever/fever

[^7]: https://ceur-ws.org/Vol-3370/paper6.pdf

[^8]: https://christos-c.com/papers/thorne_18_fever.pdf

[^9]: https://arxiv.org/html/2310.09754v3

[^10]: https://aclanthology.org/2022.fever-1.pdf

[^11]: https://dl.acm.org/doi/10.1145/3485127

[^12]: https://huggingface.co/datasets/BeIR/fever

[^13]: https://aclanthology.org/N18-1074.pdf

[^14]: https://www.amazon.science/blog/the-fever-data-set-what-doesnt-kill-it-will-make-it-stronger

[^15]: https://aclanthology.org/W18-5501.pdf

[^16]: https://github.com/dependentsign/EX-FEVER

