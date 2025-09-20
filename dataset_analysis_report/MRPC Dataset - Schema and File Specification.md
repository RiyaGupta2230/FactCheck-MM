<img src="https://r2cdn.perplexity.ai/pplx-full-logo-primary-dark%402x.png" style="height:64px;margin-right:32px"/>

# MRPC Dataset - Schema and File Specification

Based on the attached MRPC files, here is the exact schema and file structure:

## Exact File List and Paths

### MRPC Attached Files

| File | UUID | Size | Characters |
| :-- | :-- | :-- | :-- |
| train.tsv | `aaa38d63-46cb-4878-9ee8-2dc543f512b9` | 945 KB | 944,891 |
| test.tsv | `2dc9c65f-e5c2-4882-a588-023c4677ec45` | 447 KB | 446,948 |
| dev.tsv | `04367937-1dbd-45ba-a1f0-2226c461b95a` | 106 KB | 105,984 |

### Full File Paths

- `C:\Users\Riyat\Documents\Minor project\Project\FactCheck-MM\data\MRPC\train.tsv`
- `C:\Users\Riyat\Documents\Minor project\Project\FactCheck-MM\data\MRPC\test.tsv`
- `C:\Users\Riyat\Documents\Minor project\Project\FactCheck-MM\data\MRPC\dev.tsv`


## Exact Schema and Field Types

### Column Schema[^1]

```
Quality	#1 ID	#2 ID	#1 String	#2 String
```

| Column Position | Field Name | Data Type | Description | Sample Value |
| :-- | :-- | :-- | :-- | :-- |
| 1 | **Quality** | Integer (0/1) | Paraphrase indicator | 1, 0 |
| 2 | \#1 ID | Integer | Unique ID for first sentence | 702876, 2108705 |
| 3 | \#2 ID | Integer | Unique ID for second sentence | 702977, 2108831 |
| 4 | \#1 String | Text | First sentence in pair | "Amrozi accused his brother..." |
| 5 | \#2 String | Text | Second sentence in pair | "Referring to him as only..." |

## Actual Sample Records

### Sample Record 1 (Paraphrase)[^1]

```
Quality: 1
#1 ID: 702876
#2 ID: 702977
#1 String: "Amrozi accused his brother , whom he called " the witness " , of deliberately distorting his evidence ."
#2 String: "Referring to him as only " the witness " , Amrozi accused his brother of deliberately distorting his evidence ."
Classification: Paraphrase (Quality=1)
```


### Sample Record 2 (Non-Paraphrase)[^1]

```
Quality: 0
#1 ID: 2108705
#2 ID: 2108831
#1 String: "Yucaipa owned Dominick 's before selling the chain to Safeway in 1998 for $ 2.5 billion ."
#2 String: "Yucaipa bought Dominick 's in 1995 for $ 693 million and sold it to Safeway for $ 1.8 billion in 1998 ."
Classification: Non-Paraphrase (Quality=0)
```


## Data Format Specifications

- **File Format**: Tab-Separated Values (.tsv)
- **Encoding**: UTF-8 with BOM (﻿)
- **Line Endings**: Windows CRLF (\r\n)
- **Header**: Present in first row
- **Separator**: Tab character (\t)


## Field Type Analysis

| Field | Python Type | SQL Type | Range/Constraints |
| :-- | :-- | :-- | :-- |
| Quality | int | TINYINT | 0 or 1 (binary) |
| \#1 ID | int | INT | Positive integers |
| \#2 ID | int | INT | Positive integers |
| \#1 String | str | TEXT | Variable length, 20-200+ characters |
| \#2 String | str | TEXT | Variable length, 20-200+ characters |

## Key Schema Observations

1. **Quality as Label**: The Quality field serves as the paraphrase indicator (1 = Paraphrase, 0 = Non-paraphrase)
2. **No explicit 'Label' column**: Unlike some GLUE datasets, MRPC uses Quality directly
3. **ID Structure**: Each sentence has unique IDs, but IDs can appear in multiple pairs
4. **Text Preprocessing**: Sentences have spaces around punctuation for tokenization
5. **Standard TSV**: Clean tab-separated format with proper header row
6. **Character Encoding**: UTF-8 with Byte Order Mark for Windows compatibility

This schema follows the original Microsoft Research Paraphrase Corpus format, making it directly compatible with GLUE benchmark evaluation protocols.[^1]

<div style="text-align: center">⁂</div>

[^1]: train.tsv

