# NLP News Headline Analysis Script
# Haber Başlıklarından Duygu Analizi ve Topic Modeling

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
warnings.filterwarnings('ignore')

# NLP Libraries
import nltk
nltk.download('all') 
import spacy
from textblob import TextBlob
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer

# Preprocessing
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.decomposition import LatentDirichletAllocation, NMF
from sklearn.model_selection import train_test_split

# Download required NLTK data
import ssl
try:
    _create_unverified_https_context = ssl._create_unverified_context
except AttributeError:
    pass
else:
    ssl._create_default_https_context = _create_unverified_https_context

nltk.download('punkt', quiet=True)
nltk.download('stopwords', quiet=True)
nltk.download('wordnet', quiet=True)
nltk.download('averaged_perceptron_tagger', quiet=True)
nltk.download('omw-1.4', quiet=True)

from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
from nltk.tag import pos_tag
import string
import re

class NewsHeadlineAnalyzer:
    def __init__(self):
        self.lemmatizer = WordNetLemmatizer()
        self.stop_words = set(stopwords.words('english'))
        self.vader_analyzer = SentimentIntensityAnalyzer()
        
        # Try to load spacy model
        try:
            self.nlp = spacy.load('en_core_web_sm')
        except:
            print("SpaCy model not found. Please install: python -m spacy download en_core_web_sm")
            self.nlp = None
    
    def load_data(self, file_path):
        """Load the news dataset"""
        print("📦 Loading dataset...")
        try:
            df = pd.read_json(file_path, lines=True)
            print(f"Dataset loaded successfully! Shape: {df.shape}")
            print(f"Columns: {list(df.columns)}")
            return df
        except Exception as e:
            print(f"Error loading dataset: {e}")
            return None
    
    def explore_data(self, df):
        """Explore the dataset"""
        print("\n🔍 Dataset Overview:")
        print(f"Total records: {len(df)}")
        print(f"Columns: {list(df.columns)}")
        
        if 'category' in df.columns:
            print(f"\nCategories distribution:")
            print(df['category'].value_counts().head(10))
        
        if 'headline' in df.columns:
            print(f"\nSample headlines:")
            for i, headline in enumerate(df['headline'].head(5)):
                print(f"{i+1}. {headline}")
        
        return df
    
    def preprocess_text(self, text):
        """
        🔧 NLP Preprocessing Steps
        1. Tokenization
        2. Lowercasing
        3. Stopword removal
        4. Lemmatization
        """
        if pd.isna(text):
            return ""
        
        # Convert to string and lowercase
        text = str(text).lower()
        
        # Remove punctuation and special characters
        text = re.sub(r'[^a-zA-Z\s]', '', text)
        
        # Tokenization
        tokens = word_tokenize(text)
        
        # Remove stopwords and lemmatize
        processed_tokens = []
        for token in tokens:
            if token not in self.stop_words and len(token) > 2:
                lemmatized = self.lemmatizer.lemmatize(token)
                processed_tokens.append(lemmatized)
        
        return ' '.join(processed_tokens)
    
    def demonstrate_preprocessing_steps(self, sample_text):
        """Demonstrate each preprocessing step"""
        print("\n🔧 Preprocessing Steps Demonstration:")
        print(f"Original text: {sample_text}")
        
        # Step 1: Lowercasing
        step1 = sample_text.lower()
        print(f"1. Lowercasing: {step1}")
        
        # Step 2: Remove punctuation
        step2 = re.sub(r'[^a-zA-Z\s]', '', step1)
        print(f"2. Remove punctuation: {step2}")
        
        # Step 3: Tokenization
        step3 = word_tokenize(step2)
        print(f"3. Tokenization: {step3}")
        
        # Step 4: Remove stopwords
        step4 = [token for token in step3 if token not in self.stop_words and len(token) > 2]
        print(f"4. Remove stopwords: {step4}")
        
        # Step 5: Lemmatization
        step5 = [self.lemmatizer.lemmatize(token) for token in step4]
        print(f"5. Lemmatization: {step5}")
        
        # Step 6: POS Tagging (optional)
        if step4:
            pos_tags = pos_tag(step4)
            print(f"6. POS Tagging: {pos_tags}")
        
        return ' '.join(step5)
    
    def apply_preprocessing(self, df, text_column='headline'):
        """Apply preprocessing to the entire dataset"""
        print(f"\n🔄 Applying preprocessing to {text_column} column...")
        
        # Demonstrate on first headline
        if len(df) > 0:
            sample_headline = df[text_column].iloc[0]
            processed_sample = self.demonstrate_preprocessing_steps(sample_headline)
        
        # Apply to all data
        df['processed_text'] = df[text_column].apply(self.preprocess_text)
        
        # Remove empty processed texts
        df = df[df['processed_text'].str.len() > 0]
        
        print(f"Dataset shape after preprocessing: {df.shape}")
        return df
    
    def vectorize_text(self, df):
        """
        📊 Vectorization Methods
        1. CountVectorizer (Bag of Words)
        2. TF-IDF
        """
        print("\n📊 Text Vectorization:")
        
        texts = df['processed_text'].tolist()
        
        # CountVectorizer (Bag of Words)
        print("\n1. CountVectorizer (Bag of Words):")
        count_vectorizer = CountVectorizer(max_features=1000, min_df=2, max_df=0.8)
        count_matrix = count_vectorizer.fit_transform(texts)
        
        print(f"CountVectorizer shape: {count_matrix.shape}")
        print(f"Feature names sample: {count_vectorizer.get_feature_names_out()[:10]}")
        
        # Show sample document representation
        sample_doc = count_matrix[0].toarray()[0]
        feature_names = count_vectorizer.get_feature_names_out()
        non_zero_indices = np.nonzero(sample_doc)[0][:10]
        
        print("\nSample document (BoW representation):")
        for idx in non_zero_indices:
            print(f"  {feature_names[idx]}: {sample_doc[idx]}")
        
        # TF-IDF
        print("\n2. TF-IDF Vectorizer:")
        tfidf_vectorizer = TfidfVectorizer(max_features=1000, min_df=2, max_df=0.8)
        tfidf_matrix = tfidf_vectorizer.fit_transform(texts)
        
        print(f"TF-IDF shape: {tfidf_matrix.shape}")
        print(f"Feature names sample: {tfidf_vectorizer.get_feature_names_out()[:10]}")
        
        # Show sample document representation
        sample_doc_tfidf = tfidf_matrix[0].toarray()[0]
        feature_names_tfidf = tfidf_vectorizer.get_feature_names_out()
        non_zero_indices_tfidf = np.nonzero(sample_doc_tfidf)[0][:10]
        
        print("\nSample document (TF-IDF representation):")
        for idx in non_zero_indices_tfidf:
            print(f"  {feature_names_tfidf[idx]}: {sample_doc_tfidf[idx]:.4f}")
        
        # Compare advantages and disadvantages
        print("\n📈 Vectorization Methods Comparison:")
        print("CountVectorizer (BoW) Advantages:")
        print("  - Simple and intuitive")
        print("  - Good for document classification")
        print("  - Preserves exact word counts")
        
        print("\nCountVectorizer Disadvantages:")
        print("  - Ignores word importance")
        print("  - Common words dominate")
        print("  - No semantic meaning consideration")
        
        print("\nTF-IDF Advantages:")
        print("  - Considers word importance")
        print("  - Reduces impact of common words")
        print("  - Better for information retrieval")
        
        print("\nTF-IDF Disadvantages:")
        print("  - More complex computation")
        print("  - May underweight important common terms")
        print("  - Still no semantic understanding")
        
        return count_matrix, tfidf_matrix, count_vectorizer, tfidf_vectorizer
    
    def sentiment_analysis(self, df, text_column='headline'):
        """
        😊 Sentiment Analysis using TextBlob and VADER
        """
        print("\n😊 Sentiment Analysis:")
        
        # TextBlob Sentiment Analysis
        print("\n1. TextBlob Sentiment Analysis:")
        df['textblob_polarity'] = df[text_column].apply(lambda x: TextBlob(str(x)).sentiment.polarity)
        df['textblob_subjectivity'] = df[text_column].apply(lambda x: TextBlob(str(x)).sentiment.subjectivity)
        
        # Classify sentiment
        def classify_textblob_sentiment(polarity):
            if polarity > 0.1:
                return 'Positive'
            elif polarity < -0.1:
                return 'Negative'
            else:
                return 'Neutral'
        
        df['textblob_sentiment'] = df['textblob_polarity'].apply(classify_textblob_sentiment)
        
        print("TextBlob Sentiment Distribution:")
        print(df['textblob_sentiment'].value_counts())
        
        # VADER Sentiment Analysis
        print("\n2. VADER Sentiment Analysis:")
        
        def get_vader_sentiment(text):
            scores = self.vader_analyzer.polarity_scores(str(text))
            return scores
        
        vader_scores = df[text_column].apply(get_vader_sentiment)
        df['vader_compound'] = vader_scores.apply(lambda x: x['compound'])
        df['vader_pos'] = vader_scores.apply(lambda x: x['pos'])
        df['vader_neu'] = vader_scores.apply(lambda x: x['neu'])
        df['vader_neg'] = vader_scores.apply(lambda x: x['neg'])
        
        # Classify VADER sentiment
        def classify_vader_sentiment(compound):
            if compound >= 0.05:
                return 'Positive'
            elif compound <= -0.05:
                return 'Negative'
            else:
                return 'Neutral'
        
        df['vader_sentiment'] = df['vader_compound'].apply(classify_vader_sentiment)
        
        print("VADER Sentiment Distribution:")
        print(df['vader_sentiment'].value_counts())
        
        # Compare methods
        print("\n📊 Sentiment Analysis Comparison:")
        comparison = pd.crosstab(df['textblob_sentiment'], df['vader_sentiment'])
        print(comparison)
        
        # Sample results
        print("\n📝 Sample Sentiment Analysis Results:")
        sample_df = df[[text_column, 'textblob_sentiment', 'textblob_polarity', 
                       'vader_sentiment', 'vader_compound']].head(10)
        for idx, row in sample_df.iterrows():
            print(f"\nHeadline: {row[text_column]}")
            print(f"TextBlob: {row['textblob_sentiment']} (polarity: {row['textblob_polarity']:.3f})")
            print(f"VADER: {row['vader_sentiment']} (compound: {row['vader_compound']:.3f})")
        
        return df
    
    def topic_modeling(self, tfidf_matrix, vectorizer, n_topics=5):
        """
        📚 Topic Modeling using LDA and NMF
        """
        print(f"\n📚 Topic Modeling with {n_topics} topics:")
        
        # LDA Topic Modeling
        print("\n1. LDA (Latent Dirichlet Allocation):")
        lda_model = LatentDirichletAllocation(
            n_components=n_topics, 
            random_state=42, 
            max_iter=10,
            learning_method='online'
        )
        lda_topics = lda_model.fit_transform(tfidf_matrix)
        
        # Display LDA topics
        feature_names = vectorizer.get_feature_names_out()
        
        print("\nLDA Topics:")
        lda_topic_names = []
        for topic_idx, topic in enumerate(lda_model.components_):
            top_words = [feature_names[i] for i in topic.argsort()[-10:]]
            print(f"Topic {topic_idx + 1}: {', '.join(top_words)}")
            
            # Suggest topic name based on top words
            topic_name = self.suggest_topic_name(top_words)
            lda_topic_names.append(topic_name)
            print(f"  Suggested name: {topic_name}")
        
        # NMF Topic Modeling
        print("\n2. NMF (Non-negative Matrix Factorization):")
        nmf_model = NMF(n_components=n_topics, random_state=42, max_iter=100)
        nmf_topics = nmf_model.fit_transform(tfidf_matrix)
        
        print("\nNMF Topics:")
        nmf_topic_names = []
        for topic_idx, topic in enumerate(nmf_model.components_):
            top_words = [feature_names[i] for i in topic.argsort()[-10:]]
            print(f"Topic {topic_idx + 1}: {', '.join(top_words)}")
            
            # Suggest topic name based on top words
            topic_name = self.suggest_topic_name(top_words)
            nmf_topic_names.append(topic_name)
            print(f"  Suggested name: {topic_name}")
        
        # Compare LDA vs NMF
        print("\n📈 Topic Modeling Methods Comparison:")
        print("LDA Advantages:")
        print("  - Probabilistic model")
        print("  - Documents can belong to multiple topics")
        print("  - Good theoretical foundation")
        
        print("\nLDA Disadvantages:")
        print("  - Requires hyperparameter tuning")
        print("  - Computationally intensive")
        print("  - May produce less coherent topics")
        
        print("\nNMF Advantages:")
        print("  - Faster computation")
        print("  - Often more interpretable topics")
        print("  - Good for sparse data")
        
        print("\nNMF Disadvantages:")
        print("  - Less flexible than LDA")
        print("  - Hard assignment of topics")
        print("  - Sensitive to initialization")
        
        return lda_model, nmf_model, lda_topics, nmf_topics, lda_topic_names, nmf_topic_names
    
    def suggest_topic_name(self, top_words):
        """Suggest a meaningful name for a topic based on top words"""
        # Simple heuristic to suggest topic names
        word_themes = {
            'politics': ['government', 'political', 'election', 'vote', 'president', 'congress', 'policy', 'trump', 'biden'],
            'business': ['business', 'company', 'market', 'stock', 'economic', 'money', 'financial', 'trade', 'economy'],
            'technology': ['tech', 'technology', 'digital', 'data', 'software', 'internet', 'computer', 'app', 'ai'],
            'sports': ['sports', 'game', 'team', 'player', 'match', 'football', 'basketball', 'baseball', 'soccer'],
            'entertainment': ['movie', 'film', 'music', 'celebrity', 'actor', 'show', 'entertainment', 'star'],
            'health': ['health', 'medical', 'doctor', 'hospital', 'disease', 'treatment', 'patient', 'drug'],
            'world': ['world', 'international', 'country', 'global', 'nation', 'foreign', 'war', 'peace'],
            'crime': ['crime', 'police', 'court', 'law', 'legal', 'arrest', 'investigation', 'murder']
        }
        
        # Count matches for each theme
        theme_scores = {}
        for theme, keywords in word_themes.items():
            score = sum(1 for word in top_words if word in keywords)
            theme_scores[theme] = score
        
        # Return theme with highest score, or 'general' if no clear theme
        best_theme = max(theme_scores, key=theme_scores.get)
        if theme_scores[best_theme] > 0:
            return best_theme.capitalize()
        else:
            return "General"
    
    def visualize_results(self, df):
        """Create visualizations for the analysis results"""
        print("\n📊 Creating visualizations...")
        
        plt.figure(figsize=(15, 12))
        
        # Sentiment Distribution
        plt.subplot(2, 3, 1)
        df['textblob_sentiment'].value_counts().plot(kind='bar', color='skyblue')
        plt.title('TextBlob Sentiment Distribution')
        plt.xlabel('Sentiment')
        plt.ylabel('Count')
        plt.xticks(rotation=0)
        
        plt.subplot(2, 3, 2)
        df['vader_sentiment'].value_counts().plot(kind='bar', color='lightcoral')
        plt.title('VADER Sentiment Distribution')
        plt.xlabel('Sentiment')
        plt.ylabel('Count')
        plt.xticks(rotation=0)
        
        # Sentiment Scores Distribution
        plt.subplot(2, 3, 3)
        plt.hist(df['textblob_polarity'], bins=30, alpha=0.7, color='blue', label='TextBlob')
        plt.hist(df['vader_compound'], bins=30, alpha=0.7, color='red', label='VADER')
        plt.title('Sentiment Scores Distribution')
        plt.xlabel('Sentiment Score')
        plt.ylabel('Frequency')
        plt.legend()
        
        # Category distribution (if available)
        if 'category' in df.columns:
            plt.subplot(2, 3, 4)
            top_categories = df['category'].value_counts().head(10)
            top_categories.plot(kind='barh', color='green')
            plt.title('Top 10 Categories')
            plt.xlabel('Count')
        
        # Sentiment by Category (if available)
        if 'category' in df.columns:
            plt.subplot(2, 3, 5)
            sentiment_by_category = pd.crosstab(df['category'], df['textblob_sentiment'])
            sentiment_by_category.head(10).plot(kind='bar', stacked=True)
            plt.title('Sentiment by Category (Top 10)')
            plt.xlabel('Category')
            plt.ylabel('Count')
            plt.xticks(rotation=45)
            plt.legend()
        
        plt.tight_layout()
        plt.show()
    
    def generate_summary_report(self, df, lda_topic_names, nmf_topic_names):
        """Generate a comprehensive summary report"""
        print("\n" + "="*60)
        print("📋 COMPREHENSIVE ANALYSIS REPORT")
        print("="*60)
        
        print(f"\n📊 DATASET OVERVIEW:")
        print(f"  • Total headlines analyzed: {len(df):,}")
        if 'category' in df.columns:
            print(f"  • Number of categories: {df['category'].nunique()}")
            print(f"  • Most common category: {df['category'].mode()[0]}")
        
        print(f"\n🔧 PREPROCESSING RESULTS:")
        avg_length_original = df['headline'].str.len().mean()
        avg_length_processed = df['processed_text'].str.len().mean()
        print(f"  • Average headline length (original): {avg_length_original:.1f} characters")
        print(f"  • Average headline length (processed): {avg_length_processed:.1f} characters")
        print(f"  • Text reduction: {((avg_length_original - avg_length_processed) / avg_length_original * 100):.1f}%")
        
        print(f"\n😊 SENTIMENT ANALYSIS RESULTS:")
        
        # TextBlob results
        textblob_counts = df['textblob_sentiment'].value_counts()
        print(f"  TextBlob Analysis:")
        for sentiment, count in textblob_counts.items():
            percentage = (count / len(df)) * 100
            print(f"    • {sentiment}: {count:,} ({percentage:.1f}%)")
        
        # VADER results
        vader_counts = df['vader_sentiment'].value_counts()
        print(f"  VADER Analysis:")
        for sentiment, count in vader_counts.items():
            percentage = (count / len(df)) * 100
            print(f"    • {sentiment}: {count:,} ({percentage:.1f}%)")
        
        # Agreement between methods
        agreement = (df['textblob_sentiment'] == df['vader_sentiment']).sum()
        agreement_percentage = (agreement / len(df)) * 100
        print(f"  • Agreement between methods: {agreement_percentage:.1f}%")
        
        print(f"\n📚 TOPIC MODELING RESULTS:")
        print(f"  LDA Topics Identified:")
        for i, topic_name in enumerate(lda_topic_names):
            print(f"    • Topic {i+1}: {topic_name}")
        
        print(f"  NMF Topics Identified:")
        for i, topic_name in enumerate(nmf_topic_names):
            print(f"    • Topic {i+1}: {topic_name}")
        
        print(f"\n🎯 KEY INSIGHTS:")
        
        # Most positive/negative headlines
        most_positive = df.loc[df['textblob_polarity'].idxmax()]
        most_negative = df.loc[df['textblob_polarity'].idxmin()]
        
        print(f"  • Most positive headline: '{most_positive['headline']}'")
        print(f"    (TextBlob score: {most_positive['textblob_polarity']:.3f})")
        print(f"  • Most negative headline: '{most_negative['headline']}'")
        print(f"    (TextBlob score: {most_negative['textblob_polarity']:.3f})")
        
        # Sentiment by category if available
        if 'category' in df.columns:
            print(f"\n  • Sentiment by Category:")
            sentiment_by_cat = df.groupby('category')['textblob_polarity'].mean().sort_values(ascending=False)
            print(f"    Most positive category: {sentiment_by_cat.index[0]} ({sentiment_by_cat.iloc[0]:.3f})")
            print(f"    Most negative category: {sentiment_by_cat.index[-1]} ({sentiment_by_cat.iloc[-1]:.3f})")
        
        print(f"\n✅ ANALYSIS COMPLETED SUCCESSFULLY!")
        print("="*60)

def main():
    """Main function to run the complete analysis"""
    analyzer = NewsHeadlineAnalyzer()
    
    print("🧠 NLP News Headline Analysis")
    print("=" * 50)
    
    # Load data
    file_path = "News_Category_Dataset_v3.json"  # Update this path as needed
    df = analyzer.load_data(file_path)
    
    if df is None:
        print("❌ Could not load dataset. Please check the file path.")
        return
    
    # Explore data
    df = analyzer.explore_data(df)
    
    # Sample subset for demonstration (remove this line to process full dataset)
    df_sample = df.sample(n=min(1000, len(df)), random_state=42)
    print(f"\nUsing sample of {len(df_sample)} headlines for demonstration")
    
    # Apply preprocessing
    df_processed = analyzer.apply_preprocessing(df_sample)
    
    # Vectorization
    count_matrix, tfidf_matrix, count_vectorizer, tfidf_vectorizer = analyzer.vectorize_text(df_processed)
    
    # Sentiment Analysis
    df_with_sentiment = analyzer.sentiment_analysis(df_processed)
    
    # Topic Modeling
    lda_model, nmf_model, lda_topics, nmf_topics, lda_topic_names, nmf_topic_names = analyzer.topic_modeling(
        tfidf_matrix, tfidf_vectorizer, n_topics=5
    )
    
    # Visualizations
    analyzer.visualize_results(df_with_sentiment)
    
    # Generate final report
    analyzer.generate_summary_report(df_with_sentiment, lda_topic_names, nmf_topic_names)
    
    return df_with_sentiment, lda_model, nmf_model

if __name__ == "__main__":
    # Run the analysis
    results = main()
    
    # Print completion message
    print("\n🎉 Analysis complete! Check the outputs above for detailed results.")
    print("💾 The processed data is stored in the 'results' variable.")
    print("📊 Visualizations have been displayed.")
    print("📋 Summary report has been generated.")

# Additional utility functions for further analysis
def save_results(df, filename="news_analysis_results.csv"):
    """Save results to CSV file"""
    df.to_csv(filename, index=False)
    print(f"Results saved to {filename}")

def get_topic_distribution(lda_topics, lda_topic_names):
    """Get topic distribution for each document"""
    topic_dist = pd.DataFrame(lda_topics, columns=[f"Topic_{i+1}_{name}" for i, name in enumerate(lda_topic_names)])
    return topic_dist