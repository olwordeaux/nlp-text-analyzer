# Saving the finalized NLP pipeline script into a Python file for the user.
script_content = """
import nltk
from spellchecker import SpellChecker
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from gensim import corpora
from gensim.models import LdaModel
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.cluster import KMeans
from collections import Counter
import re
import logging

# Download necessary NLTK components
nltk.download('punkt')
nltk.download('stopwords')

# Setup logging
logging.basicConfig(filename='nlp_pipeline.log', level=logging.INFO, format='%(asctime)s:%(levelname)s:%(message)s')

# Initialize SpellChecker and stopwords
spell = SpellChecker()
stop_words = set(stopwords.words('english'))

def load_text(file_path):
    \"\"\"
    Loads text from a specified file path.
    Args:
    - file_path (str): Path to the text file.
    Returns:
    - str: Content of the text file.
    \"\"\"
    try:
        with open(file_path, 'r', encoding='utf-8') as file:
            logging.info("File loaded successfully.")
            return file.read()
    except FileNotFoundError as e:
        logging.error(f"File not found: {file_path}")
        raise e

def correct_spelling(text):
    \"\"\"
    Corrects spelling mistakes in the given text.
    Args:
    - text (str): Input text for spell correction.
    Returns:
    - str: Spell-checked text.
    \"\"\"
    try:
        words = word_tokenize(text)
        corrected_words = [spell.correction(word) for word in words]
        logging.info("Spell-checking completed.")
        return ' '.join(corrected_words)
    except Exception as e:
        logging.error("Error during spell-checking.")
        raise e

def clean_text(text):
    \"\"\"
    Cleans the text by removing digits, special characters, emails, and URLs.
    Args:
    - text (str): Input text to be cleaned.
    Returns:
    - str: Cleaned text.
    \"\"\"
    try:
        text = re.sub(r'\\d+', '', text)  # Remove digits
        text = re.sub(r'[^\w\\s]', '', text)  # Remove special characters
        text = re.sub(r'\\S+@\\S+', '', text)  # Remove emails
        text = re.sub(r'http\\S+|www\\S+', '', text)  # Remove URLs
        text = text.lower()  # Convert to lowercase
        logging.info("Text cleaning completed.")
        return text
    except Exception as e:
        logging.error("Error during text cleaning.")
        raise e

def preprocess_text(text):
    \"\"\"
    Tokenizes the text and removes stopwords, returning only alphabetic tokens.
    Args:
    - text (str): Input text for preprocessing.
    Returns:
    - list: List of preprocessed tokens.
    \"\"\"
    try:
        tokens = word_tokenize(text)
        filtered_tokens = [word for word in tokens if word.isalpha() and word not in stop_words]
        logging.info("Text preprocessing completed.")
        return filtered_tokens
    except Exception as e:
        logging.error("Error during preprocessing.")
        raise e

def tfidf_feature_extraction(text):
    \"\"\"
    Extracts top 1000 TF-IDF features from the text.
    Args:
    - text (str): Input text for TF-IDF extraction.
    Returns:
    - tuple: TF-IDF matrix and feature names.
    \"\"\"
    try:
        vectorizer = TfidfVectorizer(stop_words='english', max_features=1000)
        X = vectorizer.fit_transform([text])
        feature_names = vectorizer.get_feature_names_out()
        logging.info("TF-IDF feature extraction completed.")
        return X, feature_names
    except Exception as e:
        logging.error("Error during TF-IDF extraction.")
        raise e

def kmeans_clustering(tfidf_matrix, num_clusters=5):
    \"\"\"
    Applies k-Means clustering on the TF-IDF matrix.
    Args:
    - tfidf_matrix (scipy sparse matrix): TF-IDF matrix.
    - num_clusters (int): Number of clusters to create.
    Returns:
    - KMeans: Trained k-Means model.
    \"\"\"
    try:
        kmeans = KMeans(n_clusters=num_clusters, random_state=42)
        kmeans.fit(tfidf_matrix)
        logging.info("k-Means clustering completed.")
        return kmeans
    except Exception as e:
        logging.error("Error during k-Means clustering.")
        raise e

def lda_topic_modeling(tokens, num_topics=5, num_words=10):
    \"\"\"
    Performs LDA topic modeling on the tokenized text.
    Args:
    - tokens (list): List of tokenized words.
    - num_topics (int): Number of topics to extract.
    - num_words (int): Number of words per topic.
    Returns:
    - list: List of topics with top words.
    \"\"\"
    try:
        dictionary = corpora.Dictionary([tokens])
        corpus = [dictionary.doc2bow(tokens)]
        lda_model = LdaModel(corpus=corpus, id2word=dictionary, num_topics=num_topics, random_state=42, passes=10)
        topics = lda_model.show_topics(num_topics=num_topics, num_words=num_words, formatted=False)
        logging.info("LDA topic modeling completed.")
        return topics
    except Exception as e:
        logging.error("Error during LDA topic modeling.")
        raise e

def extract_keywords(tokens, top_n=10):
    \"\"\"
    Extracts the top N keywords based on frequency.
    Args:
    - tokens (list): List of tokenized words.
    - top_n (int): Number of keywords to extract.
    Returns:
    - list: Top N keywords and their frequencies.
    \"\"\"
    try:
        freq_dist = Counter(tokens)
        logging.info("Keyword extraction completed.")
        return freq_dist.most_common(top_n)
    except Exception as e:
        logging.error("Error during keyword extraction.")
        raise e

def generate_report(original_text, cleaned_text, processed_tokens, keywords, clusters, feature_names, lda_topics, output_file='final_report_with_clustering_and_lda.txt'):
    \"\"\"
    Generates and saves a detailed report summarizing the analysis.
    Args:
    - original_text (str): Original text.
    - cleaned_text (str): Cleaned text.
    - processed_tokens (list): Preprocessed tokens.
    - keywords (list): List of top keywords.
    - clusters (KMeans): Trained k-Means model.
    - feature_names (list): TF-IDF feature names.
    - lda_topics (list): List of LDA topics.
    - output_file (str): Name of the report file to save.
    \"\"\"
    try:
        with open(output_file, 'w', encoding='utf-8') as file:
            file.write("### NLP Analysis Report with Noise Handling, k-Means Clustering, and Gensim LDA ###\\n\\n")
            file.write("**Original Text Preview:**\\n")
            file.write(original_text[:500] + '...\\n\\n')
            file.write("**Cleaned Text Preview:**\\n")
            file.write(cleaned_text[:500] + '...\\n\\n')
            file.write("**Filtered Tokens (First 100):**\\n")
            file.write(' '.join(processed_tokens[:100]) + '\\n\\n')
            file.write("**Top 10 Keywords (Frequency):**\\n")
            for keyword, freq in keywords:
                file.write(f"{keyword}: {freq} occurrences\\n")
            file.write("\\n**k-Means Clusters:**\\n")
            for idx, center in enumerate(clusters.cluster_centers_):
                top_terms = [feature_names[i] for i in center.argsort()[-10:]]
                file.write(f"Cluster {idx+1}: {', '.join(top_terms)}\\n")
            file.write("\\n**LDA Topics:**\\n")
            for idx, topic in enumerate(lda_topics):
                topic_terms = ', '.join([word for word, prob in topic[1]])
                file.write(f"Topic {idx+1}: {topic_terms}\\n")
        logging.info(f"Report generated successfully: {output_file}")
    except Exception as e:
        logging.error("Error during report generation.")
        raise e

# Execute the Pipeline
if __name__ == "__main__":
    file_path = 'sample_text.txt'  # Replace with your file path
    
    try:
        # Load and clean the text
        original_text = load_text(file_path)
        corrected_text = correct_spelling(original_text)
        cleaned_text = clean_text(corrected_text)
        
        # Preprocess the cleaned text
        processed_tokens = preprocess_text(cleaned_text)
        
        # Extract keywords by frequency
        keywords = extract_keywords(processed_tokens)
        
        # Apply TF-IDF feature extraction
        tfidf_matrix, feature_names = tfidf_feature_extraction(cleaned_text)
        
        # Apply k-Means clustering
        kmeans = kmeans_clustering(tfidf_matrix, num_clusters=5)
        
        # Apply LDA topic modeling using Gensim
        lda_topics = lda_topic_modeling(processed_tokens, num_topics=5)
        
        # Generate and save the report
        generate_report(original_text, cleaned_text, processed_tokens, keywords, kmeans, feature_names, lda_topics)
    except Exception as e:
        logging
