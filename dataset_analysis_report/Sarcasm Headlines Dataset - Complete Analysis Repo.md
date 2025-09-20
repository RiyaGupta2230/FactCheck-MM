<img src="https://r2cdn.perplexity.ai/pplx-full-logo-primary-dark%402x.png" style="height:64px;margin-right:32px"/>

# Sarcasm Headlines Dataset - Complete Analysis Report

The Sarcasm Headlines Dataset is a professionally curated collection of news headlines for sarcasm detection research, containing approximately 28,619 headlines from satirical and legitimate news sources.[^1][^2]

## Dataset Overview

This dataset addresses the limitations of Twitter-based sarcasm detection by using professionally written news headlines from two distinct sources: The Onion (satirical/sarcastic) and HuffPost (legitimate news). The dataset was created by Rishabh Misra to provide high-quality, formally written text for sarcasm detection research.[^3][^4]

## File Specifications and Structure

### Directory Contents[^5][^6]

**Base Path**: `C:\Users\Riyat\Documents\Minor project\Project\FactCheck-MM\data\Sarcasm Headlines\`


| File | Size | Purpose |
| :-- | :-- | :-- |
| **Sarcasm_Headlines_Dataset.json** | 5,916 KB (5.8 MB) | Main dataset file |

### Data Structure and Schema[^7]

**JSON Line Format:**
Each line contains a JSON object with three fields:

```json
{
  "is_sarcastic": 1,
  "headline": "thirtysomething scientists unveil doomsday clock of hair loss", 
  "article_link": "https://www.theonion.com/thirtysomething-scientists-unveil-doomsday-clock-of-hai-1819586205"
}
```

**Field Specifications:**

- `is_sarcastic`: Binary indicator (0 = non-sarcastic, 1 = sarcastic)
- `headline`: The news headline text
- `article_link`: URL to the original article for additional context


## Sample Records Analysis[^7]

### Sarcastic Headlines (The Onion)

**Record 1:**

```json
{
  "is_sarcastic": 1,
  "headline": "thirtysomething scientists unveil doomsday clock of hair loss",
  "article_link": "https://www.theonion.com/thirtysomething-scientists-unveil-doomsday-clock-of-hai-1819586205"
}
```

*Analysis: Satirical take on scientific announcements with absurd premise*

**Record 4:**

```json
{
  "is_sarcastic": 1,
  "headline": "inclement weather prevents liar from getting to work",
  "article_link": "https://local.theonion.com/inclement-weather-prevents-liar-from-getting-to-work-1819576031"
}
```

*Analysis: Ironic commentary on excuse-making behavior*

### Non-Sarcastic Headlines (HuffPost)

**Record 2:**

```json
{
  "is_sarcastic": 0,
  "headline": "dem rep. totally nails why congress is falling short on gender, racial equality",
  "article_link": "https://www.huffingtonpost.com/entry/donna-edwards-inequality_us_57455f7fe4b055bb1170b207"
}
```

*Analysis: Straightforward political news reporting*

**Record 6:**

```json
{
  "is_sarcastic": 0,
  "headline": "my white inheritance",
  "article_link": "https://www.huffingtonpost.com/entry/my-white-inheritance_us_592307470e4b07617ae4c61g"
}
```

*Analysis: Serious commentary on social issues*

## Dataset Statistics[^2][^3]

| Metric | Value | Comparison |
| :-- | :-- | :-- |
| **Total Records** | 28,619 headlines | Much larger than Twitter datasets |
| **Sarcastic** | 13,635 (47.6%) | The Onion headlines |
| **Non-Sarcastic** | 14,984 (52.4%) | HuffPost headlines |
| **Pre-trained Embeddings Available** | 76.65% | Higher than Twitter (64.47%) |
| **Average Length** | ~12-15 words | Professional headline standard |
| **Language Quality** | High | No misspellings or informal usage |

### Label Distribution

- **Nearly Balanced**: 47.6% sarcastic vs 52.4% non-sarcastic
- **High Quality Labels**: Professional editorial standards reduce noise
- **Self-Contained**: Headlines don't require external context


## Key Dataset Advantages[^8][^2]

### Over Twitter Datasets

1. **Professional Quality**: No spelling mistakes or informal language
2. **Reduced Sparsity**: Higher percentage of words available in pre-trained embeddings
3. **Clean Labels**: The Onion's editorial mission ensures genuine sarcasm
4. **Self-Contained**: Headlines don't require conversational context
5. **Formal Language**: Consistent grammatical structure

### Content Characteristics

- **Domain Coverage**: Current events, politics, lifestyle, entertainment
- **Writing Style**: Professional journalism standards
- **Sarcasm Types**: Satirical commentary, ironic observations, absurd scenarios
- **Language Level**: Adult reading level with sophisticated vocabulary


## Text Analysis and Readability

### Linguistic Features[^7]

| Feature | Sarcastic Headlines | Non-Sarcastic Headlines |
| :-- | :-- | :-- |
| **Tone** | Satirical, ironic, absurd | Straightforward, informative |
| **Content** | Exaggerated scenarios | Factual reporting |
| **Language** | Clever wordplay, unexpected juxtapositions | Direct, clear communication |
| **Structure** | Often setup-punchline format | Standard news headline format |

### Vocabulary Analysis[^3]

- **Word Clouds**: Distinct vocabulary patterns between sarcastic and non-sarcastic categories
- **Complexity**: Sophisticated vocabulary requiring cultural knowledge
- **Formality**: Professional news writing standards maintained


## Preprocessing Requirements

### Text Processing Pipeline

- **Minimal Preprocessing Needed**: Professional quality reduces cleaning requirements
- **Case Normalization**: Optional, as original capitalization may carry meaning
- **Tokenization**: Standard NLP tokenization suitable
- **Punctuation**: Preserve punctuation patterns that signal sarcasm
- **URL Handling**: `article_link` field can be used for additional data extraction


### Feature Engineering Opportunities

- **Length Features**: Headline character/word count
- **Punctuation Patterns**: Exclamation marks, question marks, quotation marks
- **Named Entity Recognition**: Political figures, celebrities, locations
- **Sentiment Analysis**: Contradiction between surface sentiment and true meaning
- **Topic Modeling**: Subject matter classification


## Model Development Applications

### Primary Use Cases[^9][^10]

- **Binary Sarcasm Classification**: Main task with high-quality labels
- **Fake News Detection**: Satirical vs. real news classification
- **Sentiment Analysis**: Understanding complex emotional expressions
- **Natural Language Understanding**: Training contextual language models


### Performance Benchmarks[^9]

- **Transformer Models**: RoBERTa and DistilBERT achieve F1 scores up to 99%
- **Contextual Enhancement**: Adding article descriptions improves performance
- **Training Efficiency**: 35.5% reduction in training time with context summarization
- **Baseline Models**: Traditional ML approaches achieve 75-85% accuracy[^11]


### Model Architectures[^11]

- **Multinomial Naive Bayes**: Traditional baseline approach
- **Logistic Regression**: Linear classification with text features
- **Transformer Models**: BERT, RoBERTa, DistilBERT for state-of-the-art results
- **Ensemble Methods**: Combining multiple approaches for robust performance


## Data Loading and Access[^2]

### Python Implementation

```python
import json

def parse_data(file):
    for line in open(file, 'r'):
        yield json.loads(line)

data = list(parse_data('./Sarcasm_Headlines_Dataset.json'))
```


### Alternative Loading Methods[^2]

```python
def parseJson(fname):
    for line in open(fname, 'r'):
        yield eval(line)

data = list(parseJson('./Sarcasm_Headlines_Dataset.json'))
```


## Research Applications and Impact

### Academic Usage[^12][^3]

- **Sarcasm Detection**: Primary application with consistent high performance
- **Computational Linguistics**: Understanding satirical language patterns
- **Media Analysis**: Distinguishing satirical from legitimate news
- **NLP Benchmarking**: Standard dataset for model evaluation


### Commercial Applications

- **Content Moderation**: Identifying satirical content on platforms
- **Media Monitoring**: Distinguishing real news from satire
- **Sentiment Analysis**: Understanding complex emotional expressions
- **Recommendation Systems**: Content categorization for news aggregators


## Access and Citation

### Official Sources

- **Kaggle**: https://www.kaggle.com/datasets/rmisra/news-headlines-dataset-for-sarcasm-detection[^1]
- **GitHub**: https://github.com/rishabhmisra/News-Headlines-Dataset-For-Sarcasm-Detection[^2]
- **Hugging Face**: Various implementations available[^8]
- **Author's Page**: https://rishabhmisra.github.io/NewsHeadlinesDataset.pdf[^3]


### Citation Information[^4]

```bibtex
@misc{misra2022news,
      title={News Headlines Dataset For Sarcasm Detection}, 
      author={Rishabh Misra},
      year={2022},
      eprint={2212.06035},
      archivePrefix={arXiv},
      primaryClass={cs.CL}
}
```


### Dataset Versions[^11]

- **Version 1**: Original 28K headlines dataset
- **Version 2**: Extended version with additional headlines
- **Combined**: Researchers often merge both versions for larger training sets

The Sarcasm Headlines Dataset represents a significant advancement in sarcasm detection research, providing a clean, professionally curated corpus that overcomes the limitations of noisy social media datasets while maintaining the complexity needed for meaningful natural language understanding research.[^10][^2]
<span style="display:none">[^13][^14][^15][^16][^17]</span>

<div style="text-align: center">⁂</div>

[^1]: https://www.kaggle.com/datasets/rmisra/news-headlines-dataset-for-sarcasm-detection

[^2]: https://github.com/rishabhmisra/News-Headlines-Dataset-For-Sarcasm-Detection

[^3]: https://rishabhmisra.github.io/NewsHeadlinesDataset.pdf

[^4]: https://arxiv.org/abs/2212.06035

[^5]: image.jpg

[^6]: image.jpg

[^7]: image.jpg

[^8]: https://huggingface.co/datasets/raquiba/Sarcasm_News_Headline

[^9]: https://www.nature.com/articles/s41598-024-65217-8

[^10]: https://www.sciencedirect.com/science/article/pii/S2666651023000013

[^11]: https://www.ijraset.com/research-paper/sarcasm-detection-in-news-headlines

[^12]: https://research-archive.org/index.php/rars/preprint/view/1469

[^13]: https://www.kaggle.com/datasets/shariphthapa/sarcasm-json-datasets

[^14]: https://metatext.io/datasets/news-headlines-dataset-for-sarcasm-detection

[^15]: https://www.kaggle.com/datasets/saurabhbagchi/sarcasm-detection-through-nlp

[^16]: https://theonion.com

[^17]: https://ui.adsabs.harvard.edu/abs/2022arXiv221206035M/abstract

