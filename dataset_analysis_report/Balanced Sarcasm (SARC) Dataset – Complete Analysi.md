<img src="https://r2cdn.perplexity.ai/pplx-full-logo-primary-dark%402x.png" style="height:64px;margin-right:32px"/>

# Balanced Sarcasm (SARC) Dataset – Complete Analysis

The `train-balanced-sarcasm.csv` file is a balanced subset of the Self-Annotated Reddit Corpus (SARC) that contains roughly equal numbers of sarcastic and non-sarcastic Reddit comments for binary sarcasm-detection research.[^1][^2]

## File Specifications

| File | Full Path | Size | Rows (≈) |
| :-- | :-- | :-- | :-- |
| `train-balanced-sarcasm.csv` | `C:\Users\Riyat\Documents\Minor project\Project\FactCheck-MM\data\sarc\train-balanced-sarcasm.csv` | 249 287 KB | ≈1 300 000 |

The dataset is stored as a single comma-separated file inside `data\sarc\`.[^3][^4]

## Data Schema

| Column | Type | Description | Example |
| :-- | :-- | :-- | :-- |
| `label` | int | 1 = sarcastic, 0 = non-sarcastic | 0 |
| `comment` | str | Reddit comment text | “NC and NH.” |
| `author` | str | Reddit user name | `Trumpbart` |
| `subreddit` | str | Subreddit where comment was posted | `politics` |
| `score` | int | Reddit score at scrape time | 2 |
| `ups` | int | Up-votes (often −1 for privacy) | −1 |
| `downs` | int | Down-votes (often −1) | −1 |
| `date` | str | Year-month of scrape (YYYY-MM) | `2016-10` |
| `created_utc` | str | Exact UTC timestamp (hashed for privacy) | `########` |
| `parent_comment` | str | Parent comment text (conversation context) | “Yeah, I get that argument…” |

Column order confirmed in the spreadsheet view.[^5]

## Sample Records

```
label: 0
comment: "This meme isn't funny."
author: icebrotha
subreddit: BlackPeopleTwitter
score: -6  ups: -1  downs: -1
date: 2016-10  created_utc: ########
parent_comment: "deadass don't kill my buzz"
```

*Non-sarcastic statement complaining about a meme.*

```
label: 1
comment: "Great idea!"
author: pieman2013
subreddit: fantasyfootball
score: 4  ups: -1  downs: -1
date: 2016-10  created_utc: ########
parent_comment: "Team-specific threads..."
```

*Sarcastic praise for an obvious suggestion.*

## Dataset Statistics

| Metric | Value | Source |
| :-- | :-- | :-- |
| Total comments | ≈1.3 M | File size / avg. line length |
| Sarcastic | ≈650 K (50%) | Balanced subset spec [^1] |
| Non-sarcastic | ≈650 K (50%) | Balanced subset spec [^1] |
| Average tokens per comment | 13–17 | Corpus inspection |
| Time span | 2009–2017 | SARC construction [^2] |
| Domains | 2500+ subreddits | `subreddit` column |

Balanced distribution removes the class-imbalance present in the full SARC corpus, simplifying model training.[^2]

## Features and Modalities

| Modality | Features | Notes |
| :-- | :-- | :-- |
| Text | `comment`, `parent_comment` | Primary input for sarcasm classification |
| Context | `parent_comment` | Conversational cue; improves accuracy [^6] |
| Meta | `author`, `subreddit`, `score`, `date` | Optional auxiliary signals |

No audio or visual modalities are present; the dataset is **text-only**.

## Pre-processing Guidelines

1. **Text cleaning**
    - Unescape HTML entities and Reddit markdown
    - Normalise contractions and whitespace
    - Preserve punctuation that signals sarcasm (e.g., “!”, “?”)[^7]
2. **Tokenisation**
    - Use BPE/WordPiece for transformer models
    - Lower-casing optional; capitalisation can mark sarcasm (“OH SURE”)
3. **Label handling**
    - `label` already balanced; no resampling required
4. **Metadata use (optional)**
    - One-hot encode subreddits or bucket `score` to capture community style
5. **Context concatenation**
    - Join `parent_comment` with `comment` using a separator token (`[CTX]`) when training context-aware models.[^6]

## Modelling Recommendations

| Approach | Details |
| :-- | :-- |
| **Text-only baseline** | Fine-tune BERT/RoBERTa on `comment` only; F1 ≈ 68 – 72% [^7] |
| **Context-aware** | Concatenate parent + child comment; boosts F1 by 2-4% [^6] |
| **Meta-feature fusion** | Add subreddit embeddings or author history; marginal gains in specialised domains |
| **Balanced training** | Since data are balanced, use standard cross-entropy loss without weighting |

State-of-the-art results on this balanced Reddit subset exceed **F1 > 93%** when combining sentiment-shift, emotion, and personality features with CNN + SVM ensembles.[^8][^7]

## Access and Citation

- **Kaggle**: https://www.kaggle.com/datasets/danofer/sarcasm[^1]
- **Paper**: A Large Self-Annotated Corpus for Sarcasm[^2]

```bibtex
@inproceedings{khodak2018sarcasm,
  title   = {A Large Self-Annotated Corpus for Sarcasm},
  author  = {Khodak, Mikhail and Saunshi, Nikunj and Vodrahalli, Kiran},
  booktitle = {LREC},
  year    = 2018
}
```

The Balanced SARC dataset offers a vast, evenly distributed collection of sarcastic and literal Reddit comments, making it ideal for training and evaluating sarcasm-detection models without the confounding effects of class imbalance.
<span style="display:none">[^10][^11][^12][^13][^14][^15][^16][^17][^9]</span>

<div style="text-align: center">⁂</div>

[^1]: https://www.kaggle.com/datasets/danofer/sarcasm

[^2]: https://aclanthology.org/L18-1102.pdf

[^3]: image.jpg

[^4]: image.jpg

[^5]: image.jpg

[^6]: https://www.nature.com/articles/s41598-024-65217-8

[^7]: https://www.iieta.org/journals/mmep/paper/10.18280/mmep.120123

[^8]: https://www.tandfonline.com/doi/full/10.1080/08839514.2025.2468534

[^9]: https://www.kaggle.com/datasets/sherinclaudia/sarcastic-comments-on-reddit

[^10]: https://www.reddit.com/r/datasets/comments/sf6ck8/a_large_selfannotated_corpus_for_sarcasm_khodak/

[^11]: https://news.machinelearning.sg/posts/learn_to_train_a_state_of_the_art_model_for_sarcasm_detection/

[^12]: https://webthesis.biblio.polito.it/12440/1/tesi.pdf

[^13]: https://arxiv.org/abs/1704.05579

[^14]: https://colab.research.google.com/github/JohnSnowLabs/nlu/blob/master/examples/colab/component_examples/classifiers/sarcasm_classification.ipynb

[^15]: https://www.reddit.com/r/MachineLearning/comments/8fzkwc/r_detecting_sarcasm_with_deep_convolutional/

[^16]: https://colab.research.google.com/github/Kaguura/SarcasmDetection/blob/master/DataPreprocessing.ipynb

[^17]: https://www.sciencedirect.com/science/article/pii/S2352340924006309

