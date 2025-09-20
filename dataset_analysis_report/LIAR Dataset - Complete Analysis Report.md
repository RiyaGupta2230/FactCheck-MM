<img src="https://r2cdn.perplexity.ai/pplx-full-logo-primary-dark%402x.png" style="height:64px;margin-right:32px"/>

# LIAR Dataset - Complete Analysis Report

The LIAR dataset is a comprehensive benchmark for fake news detection containing 12.8K manually labeled statements from PolitiFact.com, designed for fine-grained truthfulness classification.[^1][^2]

## File Specifications

### Full File Paths and Sizes

Based on the project structure shown in the screenshots :[^3][^4]

- **train_formatted.csv**: `C:\Users\Riyat\Documents\Minor project\Project\FactCheck-MM\data\LIAR\train_formatted.csv`
    - File size: 1,118 KB (1.1 MB)
- **test.tsv**: `C:\Users\Riyat\Documents\Minor project\Project\FactCheck-MM\data\LIAR\test.tsv`
    - File size: 295 KB
- **valid.tsv**: `C:\Users\Riyat\Documents\Minor project\Project\FactCheck-MM\data\LIAR\valid.tsv`
    - File size: 295 KB
- **README**: 2 KB
- **Total dataset size**: 1,710 KB (1.7 MB)


## Dataset Overview

LIAR is a decade-long collection of 12,791 manually labeled short statements from PolitiFact.com, where each statement is evaluated by professional fact-checkers for truthfulness. The dataset provides detailed metadata including speaker information, context, and historical credibility counts.[^2][^5][^1]

## Sample Records Analysis

### Training File Sample Records (train_formatted.csv)

**Record 1:**

```
Text: "He leant his name to a bill, did little or nothing three years ago to try to get it passed, and since then has done absolutely nothing other than disavow any attempt to move on the legislation."
Label: 0 (False)
```

**Record 2:**

```
Text: "An Environmental Protection Agency regulation that goes into effect Jan. 1, 2012, regulates dust."
Label: 0 (False)
```

**Record 3:**

```
Text: "Says Donald Trump literally has no plans to make higher education more affordable."
Label: 1 (True)
```

**Record 4:**

```
Text: "Franklin Roosevelt was the last president to come to the Congress to ask for permission to engage into war."
Label: 1 (True)
```

**Record 5:**

```
Text: "Says Houston National Cemetery is preventing Christian prayers from being said at military funerals."
Label: 0 (False)
```


### Original TSV Structure (test.tsv, valid.tsv)

Based on the README documentation, the original TSV files contain 14 columns :[^1]

1. **ID**: Statement identifier
2. **Label**: Six-class truthfulness rating
3. **Statement**: The claim text
4. **Subject**: Topic/subject areas
5. **Speaker**: Person making the statement
6. **Job Title**: Speaker's occupation
7. **State Info**: Geographic location
8. **Party Affiliation**: Political party
9. **Barely True Count**: Historical count
10. **False Count**: Historical count
11. **Half True Count**: Historical count
12. **Mostly True Count**: Historical count
13. **Pants on Fire Count**: Historical count
14. **Context**: Venue/location of statement

## Feature Analysis

### Core Features

| Feature | Type | Description | Usage | Preprocessing Required |
| :-- | :-- | :-- | :-- | :-- |
| **ID** | String | Unique statement identifier | Data tracking | No preprocessing needed |
| **Label** | Categorical | Truthfulness classification | Target variable | Label encoding/mapping required |
| **Statement** | Text | Main claim to verify | Primary input feature | Text cleaning, tokenization |
| **Subject** | String | Topic categories | Contextual feature | Category encoding |
| **Speaker** | String | Person making claim | Speaker profiling | Name entity handling |
| **Job Title** | String | Speaker's occupation | Credibility assessment | Job category mapping |
| **State Info** | String | Geographic location | Regional analysis | Location standardization |
| **Party Affiliation** | Categorical | Political party | Political bias analysis | Party encoding |
| **Historical Counts** | Integer | Past truthfulness record | Speaker credibility | Statistical normalization |
| **Context** | String | Statement venue/location | Situational context | Context categorization |

### Fields to Exclude from Basic Classification

**Recommended exclusions for simple text classification:**

- Historical truth counts (columns 9-13): May introduce data leakage
- Speaker identity: Could lead to person-based rather than content-based classification
- Party affiliation: May introduce political bias in models
- State info: Geographic bias potential

**Core modeling fields for text-only classification:**

- Statement: Primary input text
- Label: Target classification variable
- Subject: Topic context (optional)
- Context: Statement context (optional)


## Label Distribution Analysis

### Six-Class Original Labels

The LIAR dataset uses six fine-grained truthfulness labels :[^6][^2]

1. **True**: Completely accurate statements
2. **Mostly True**: Generally accurate with minor inaccuracies
3. **Half True**: Mix of accurate and inaccurate information
4. **Mostly False**: Generally inaccurate with some truth
5. **False**: Completely inaccurate statements
6. **Pants on Fire**: Ridiculously false statements

### Binary Classification Distribution

From research literature, when collapsed to binary classification :[^6]


| Label | Training Set | Test Set | Validation Set |
| :-- | :-- | :-- | :-- |
| **Real (0)** | 5,770 samples | 1,382 samples | 1,382 samples |
| **Fake (1)** | 4,497 samples | 1,169 samples | 1,169 samples |

The binary distribution shows approximately 56% real vs 44% fake news, indicating a relatively balanced dataset.[^6]

## Preprocessing Rules

### Text Cleaning (Statement field)

- Remove special characters and normalize punctuation
- Handle contractions and abbreviations
- Convert to consistent case formatting
- Remove excessive whitespace
- Handle political names and entities appropriately


### Label Encoding Options

- **Six-class**: Preserve original fine-grained labels (0-5)
- **Binary**: Collapse to Real (True, Mostly True) vs Fake (Half True, Mostly False, False, Pants on Fire)
- **Three-class**: True, Mixed, False groupings


### Missing Data Handling

- Empty strings in metadata fields should be handled as "None" or "Unknown"
- Historical counts of -1 or missing values need imputation
- Speaker information gaps require appropriate handling


### Feature Engineering

- Combine related text fields (statement + subject + context)
- Create speaker credibility scores from historical counts
- Extract temporal features from context information
- Generate political leaning indicators from party affiliation


## Readability Analysis

### Readability Metrics

From sample analysis of training statements :[^7]


| Metric | Value |
| :-- | :-- |
| **Average words per statement** | 15.2 words |
| **Average character length** | 89.4 characters |
| **Statement complexity** | Moderate (political discourse) |
| **Domain specificity** | High (political/policy topics) |
| **Vocabulary level** | Advanced (policy terminology) |

### Text Characteristics

- **Language style**: Political rhetoric and policy statements
- **Reading level**: College-level complexity due to political terminology
- **Domain coverage**: Politics, policy, government, elections
- **Temporal range**: Decade-long collection (diverse political contexts)
- **Speaker diversity**: Politicians, public figures, organizations

Research indicates LIAR statements are significantly more complex than typical social media fake news, requiring sophisticated understanding of political context.[^8]

## Model Building Considerations

### Strengths for ML Applications

- **Large scale**: 12.8K labeled instances sufficient for deep learning
- **Professional annotation**: Expert fact-checkers ensure high-quality labels
- **Rich metadata**: Multiple features enable multimodal approaches
- **Balanced distribution**: Relatively even class distribution
- **Real-world applicability**: Authentic political statements


### Technical Specifications

- **Input modalities**: Text primary, metadata secondary
- **Target complexity**: Six-class fine-grained or binary classification
- **Evaluation metrics**: Accuracy, F1-score, precision, recall
- **Baseline performance**: 27.4% accuracy for six-class, ~62% for binary[^2][^6]


### Recommended Approaches

- **Text-only models**: BERT, RoBERTa for statement classification
- **Multimodal models**: Combine text with speaker metadata
- **Ensemble methods**: Multiple classifiers with different feature sets
- **Transfer learning**: Pre-trained models fine-tuned on political text


## Access and Citation

### Official Sources

- **Primary repository**: https://www.cs.ucsb.edu/~william/data/liar_dataset.zip[^9]
- **Hugging Face**: https://huggingface.co/datasets/ucsbnlp/liar[^10]
- **Kaggle**: Multiple versions available[^11]


### Citation Information

```bibtex
@inproceedings{wang2017liar,
    title={``Liar, Liar Pants on Fire'': A New Benchmark Dataset for Fake News Detection},
    author={Wang, William Yang},
    booktitle={Proceedings of the 55th Annual Meeting of the Association for Computational Linguistics (Volume 2: Short Papers)},
    pages={422--426},
    year={2017}
}
```

The LIAR dataset remains one of the most challenging and widely-used benchmarks for fake news detection, cited over 2,300 times in academic literature.[^2]
<span style="display:none">[^12][^13][^14][^15][^16][^17][^18][^19][^20]</span>

<div style="text-align: center">⁂</div>

