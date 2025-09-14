"""
Advanced feature engineering for milestone1 sarcasm detection
Extracts linguistic and contextual features for maximum accuracy - CORRECTED VERSION
"""
import pandas as pd
import numpy as np
import re
import os
import glob
from textblob import TextBlob
import emoji
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.feature_selection import SelectKBest, f_classif
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def clean_text_advanced(text):
    """Advanced text cleaning and normalization"""
    # Robust cast and null guard
    if text is None or (isinstance(text, float) and pd.isna(text)):
        return ""
    try:
        text = str(text)
    except Exception:
        return ""

    # Convert emojis to text descriptions (guarded)
    try:
        text = emoji.demojize(text, delimiters=(" [", "] "))
    except Exception:
        # Fallback to original text if demojize fails
        text = str(text)

    # Normalize URLs, mentions, hashtags
    text = re.sub(r'http[s]?://\S+', ' [URL] ', text)
    text = re.sub(r'@\w+', ' [MENTION] ', text)
    text = re.sub(r'#(\w+)', r' [HASHTAG] \1 ', text)

    # Normalize repetitive punctuation
    text = re.sub(r'([!?]){3,}', r'\1\1', text)
    text = re.sub(r'\.{3,}', '...', text)

    # Normalize whitespace
    text = re.sub(r'\s+', ' ', text).strip()

    return text


def extract_linguistic_features(df):
    """Extract comprehensive linguistic features for sarcasm detection"""
    
    logger.info("🔧 Extracting linguistic features...")
    
    # Clean text first
    df['text_clean'] = df['text'].apply(clean_text_advanced)
    
    # 1. Punctuation features
    df['exclamation_count'] = df['text_clean'].str.count('!')
    df['question_count'] = df['text_clean'].str.count(r'\?')
    df['caps_ratio'] = df['text_clean'].str.count(r'[A-Z]') / (df['text_clean'].str.len() + 1e-6)
    df['ellipsis_count'] = df['text_clean'].str.count(r'\.\.\.')
    
    # 2. Sentiment features
    def get_sentiment_features(text):
        try:
            blob = TextBlob(text)
            return blob.sentiment.polarity, blob.sentiment.subjectivity
        except:
            return 0.0, 0.0
    
    sentiment_data = df['text_clean'].apply(get_sentiment_features)
    df['sentiment_polarity'] = [x[0] for x in sentiment_data]
    df['sentiment_subjectivity'] = [x[1] for x in sentiment_data]
    
    # 3. Sarcasm indicator features
    sarcasm_indicators = [
        'obviously', 'sure', 'yeah right', 'totally', 'absolutely',
        'great job', 'wonderful', 'perfect', 'brilliant', 'fantastic',
        'oh wow', 'amazing', 'incredible', 'unbelievable', 'right',
        'of course', 'naturally', 'clearly', 'definitely'
    ]
    
    df['sarcasm_indicator_count'] = df['text_clean'].str.lower().apply(
        lambda x: sum([x.count(word) for word in sarcasm_indicators])
    )
    
    # 4. Contrast and irony features
    contrast_words = ['but', 'however', 'although', 'though', 'while', 'whereas', 'yet']
    df['contrast_word_count'] = df['text_clean'].str.lower().apply(
        lambda x: sum([x.count(word) for word in contrast_words])
    )
    
    # 5. Text complexity features
    df['word_count'] = df['text_clean'].str.split().str.len()
    df['avg_word_length'] = df['text_clean'].apply(
        lambda x: np.mean([len(word) for word in x.split()]) if x.split() else 0
    )
    df['unique_word_ratio'] = df['text_clean'].apply(
        lambda x: len(set(x.split())) / max(len(x.split()), 1)
    )
    
    # 6. Special characters and patterns
    df['emoji_count'] = df['text'].apply(
        lambda x: len([c for c in str(x) if c in emoji.EMOJI_DATA])
    )
    df['mention_count'] = df['text_clean'].str.count(r'\[MENTION\]')
    df['hashtag_count'] = df['text_clean'].str.count(r'\[HASHTAG\]')
    df['url_count'] = df['text_clean'].str.count(r'\[URL\]')
    
    logger.info("✅ Linguistic features extracted successfully")
    
    return df

def extract_tfidf_features(df, max_features=100):
    """Extract TF-IDF features for important terms - CORRECTED VERSION"""
    
    logger.info("🔍 Extracting TF-IDF features...")
    
    # Get dataset size for parameter adjustment
    total_docs = len(df)
    
    # Dynamically adjust min_df and max_df based on dataset size
    if total_docs < 10:
        # Very small dataset - use minimal constraints
        min_df = 1
        max_df = 1.0
        max_features = min(20, max_features)
    elif total_docs < 100:
        # Small dataset - relaxed constraints
        min_df = 1
        max_df = 0.95
        max_features = min(50, max_features)
    else:
        # Larger dataset - standard constraints
        min_df = 2
        max_df = 0.8
        
    # Ensure min_df doesn't conflict with max_df
    max_docs_for_term = int(total_docs * max_df)
    if min_df >= max_docs_for_term:
        min_df = max(1, max_docs_for_term - 1)
        logger.warning(f"⚠️ Adjusted min_df to {min_df} for dataset size {total_docs}")
    
    # Create TF-IDF vectorizer with safe parameters
    tfidf = TfidfVectorizer(
        max_features=max_features,
        ngram_range=(1, 2),
        stop_words='english',
        min_df=min_df,
        max_df=max_df
    )
    
    try:
        # Fit and transform
        tfidf_features = tfidf.fit_transform(df['text_clean'])
        logger.info(f"✅ TF-IDF created {tfidf_features.shape[1]} features from {total_docs} documents")
        
        # Feature selection if labels available
        if 'label' in df.columns and tfidf_features.shape[1] > 0:
            # Select top features (at most half of available features)
            k = min(50, tfidf_features.shape[1] // 2, max_features // 2)
            if k > 0:
                selector = SelectKBest(f_classif, k=k)
                selected_features = selector.fit_transform(tfidf_features, df['label'])
                
                # Add to dataframe
                feature_names = [f"tfidf_{i}" for i in range(selected_features.shape[1])]
                tfidf_df = pd.DataFrame(selected_features.toarray(), columns=feature_names, index=df.index)
                df = pd.concat([df, tfidf_df], axis=1)
                
                logger.info(f"✅ Added {len(feature_names)} selected TF-IDF features")
            else:
                logger.warning("⚠️ No TF-IDF features selected due to small dataset")
        
    except ValueError as e:
        logger.warning(f"⚠️ TF-IDF extraction failed: {e}. Skipping TF-IDF features.")
        # Continue without TF-IDF features - linguistic features are still valuable
    
    logger.info("✅ TF-IDF features extraction completed")
    return df

def prepare_features_for_training(df):
    """Prepare final feature set for training"""
    
    # Extract all features
    df = extract_linguistic_features(df)
    df = extract_tfidf_features(df)
    
    # Feature columns (linguistic features)
    feature_columns = [
        'exclamation_count', 'question_count', 'caps_ratio', 'ellipsis_count',
        'sentiment_polarity', 'sentiment_subjectivity', 'sarcasm_indicator_count',
        'contrast_word_count', 'word_count', 'avg_word_length', 'unique_word_ratio',
        'emoji_count', 'mention_count', 'hashtag_count', 'url_count'
    ]
    
    # Add TF-IDF features if they exist
    tfidf_columns = [col for col in df.columns if col.startswith('tfidf_')]
    feature_columns.extend(tfidf_columns)
    
    # Fill NaN values
    for col in feature_columns:
        if col in df.columns:
            df[col] = df[col].fillna(0)
    
    logger.info(f"📊 Final feature set: {len(feature_columns)} features")
    
    return df, feature_columns

if __name__ == "__main__":
    import sys
    import argparse
    
    # Command line argument parsing
    parser = argparse.ArgumentParser(description='Extract features for milestone1 sarcasm detection')
    parser.add_argument('--input_dir', required=True, help='Input directory with chunk files')
    parser.add_argument('--output_dir', required=True, help='Output directory for processed files')
    
    args = parser.parse_args()
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Process all chunk files in input directory
    input_files = glob.glob(os.path.join(args.input_dir, "*.csv"))
    
    if not input_files:
        logger.error(f"❌ No CSV files found in {args.input_dir}")
        sys.exit(1)
    
    logger.info(f"🚀 Starting feature engineering on {len(input_files)} files")
    
    for input_file in input_files:
        logger.info(f"📄 Processing: {os.path.basename(input_file)}")
        
        # Load chunk
        df = pd.read_csv(input_file)
        logger.info(f"📊 Loaded {len(df)} samples")
        
        # Extract features
        df_enhanced, feature_columns = prepare_features_for_training(df)
        
        # Save enhanced chunk
        output_filename = f"enhanced_{os.path.basename(input_file)}"
        output_path = os.path.join(args.output_dir, output_filename)
        df_enhanced.to_csv(output_path, index=False)
        
        logger.info(f"💾 Saved enhanced data: {output_path}")
        logger.info(f"✨ Added {len(feature_columns)} features")
    
    logger.info("🎉 Feature engineering completed successfully!")
    print(f"✅ Processed {len(input_files)} chunk files")
    print(f"📁 Enhanced files saved to: {args.output_dir}")
