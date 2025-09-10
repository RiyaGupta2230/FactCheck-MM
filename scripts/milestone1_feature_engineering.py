"""
Advanced feature engineering for sarcasm detection
Extracts linguistic and contextual features for maximum accuracy
"""
import pandas as pd
import numpy as np
import re
from textblob import TextBlob
import emoji
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.feature_selection import SelectKBest, f_classif
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def clean_text_advanced(text):
    """Advanced text cleaning and normalization"""
    if not text or pd.isna(text):
        return ""
    
    # Convert emojis to text descriptions
    text = emoji.demojize(text, delimiters=(" [", "] "))
    
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
        blob = TextBlob(text)
        return blob.sentiment.polarity, blob.sentiment.subjectivity
    
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
    """Extract TF-IDF features for important terms"""
    
    logger.info("🔍 Extracting TF-IDF features...")
    
    # Create TF-IDF vectorizer
    tfidf = TfidfVectorizer(
        max_features=max_features,
        ngram_range=(1, 2),
        stop_words='english',
        min_df=2,
        max_df=0.8
    )
    
    # Fit and transform
    tfidf_features = tfidf.fit_transform(df['text_clean'])
    
    # Select most discriminative features
    if 'label' in df.columns:
        selector = SelectKBest(f_classif, k=min(50, max_features))
        selected_features = selector.fit_transform(tfidf_features, df['label'])
        
        # Add to dataframe
        feature_names = [f"tfidf_{i}" for i in range(selected_features.shape[1])]
        tfidf_df = pd.DataFrame(selected_features.toarray(), columns=feature_names, index=df.index)
        df = pd.concat([df, tfidf_df], axis=1)
    
    logger.info("✅ TF-IDF features extracted successfully")
    
    return df

def prepare_features_for_training(df):
    """Prepare final feature set for training"""
    
    # Extract all features
    df = extract_linguistic_features(df)
    df = extract_tfidf_features(df)
    
    # Feature columns
    feature_columns = [
        'exclamation_count', 'question_count', 'caps_ratio', 'ellipsis_count',
        'sentiment_polarity', 'sentiment_subjectivity', 'sarcasm_indicator_count',
        'contrast_word_count', 'word_count', 'avg_word_length', 'unique_word_ratio',
        'emoji_count', 'mention_count', 'hashtag_count', 'url_count'
    ]
    
    # Add TF-IDF features
    tfidf_columns = [col for col in df.columns if col.startswith('tfidf_')]
    feature_columns.extend(tfidf_columns)
    
    # Fill NaN values
    for col in feature_columns:
        if col in df.columns:
            df[col] = df[col].fillna(0)
    
    logger.info(f"📊 Final feature set: {len(feature_columns)} features")
    
    return df, feature_columns

if __name__ == "__main__":
    # Example usage
    sample_data = pd.DataFrame({
        'text': ['Oh sure, what a brilliant idea!', 'This is genuinely helpful.'],
        'label': [1, 0]
    })
    
    enhanced_data, features = prepare_features_for_training(sample_data)
    print("Feature columns:", features[:10])  # Show first 10 features
