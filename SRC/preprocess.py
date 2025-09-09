# Add these imports at the top
import pandas as pd
import re
import nltk
import joblib
import logging
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# Fix NLTK data path - Add this right after imports
import nltk.data
nltk.data.path = [
    r"C:\Users\SOUMEN\AppData\Roaming\nltk_data",
    r"C:\nltk_data",
    nltk.data.path[0]
]

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='[%(asctime)s] %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("preprocess.log", encoding="utf-8"),
        logging.StreamHandler()
    ]
)

logging.info("🚀 Starting preprocessing...")

# Download NLTK resources first
nltk.download('punkt', quiet=True)
nltk.download('punkt_tab', quiet=True)
nltk.download('stopwords', quiet=True)

# Load and sample dataset
try:
    df = pd.read_csv("spotify_millsongdata.csv").sample(10000, random_state=42)
    logging.info("✅ Dataset loaded and sampled: %d rows", len(df))
except Exception as e:
    logging.error("❌ Failed to load dataset: %s", str(e))
    raise e

# Drop link column and preprocess
df = df.drop(columns=['link'], errors='ignore').reset_index(drop=True)

# Text cleaning
stop_words = set(stopwords.words('english'))

# Replace the tokenization part with this modified version:

def preprocess_text(text):
    try:
        # Clean and normalize text
        text = re.sub(r"[^a-zA-Z\s]", "", str(text))
        text = text.lower().strip()
        
        # Tokenize using NLTK
        tokens = word_tokenize(text)
        
        # Remove stopwords
        tokens = [word for word in tokens if word not in stop_words and len(word) > 2]
        
        # Join tokens back
        return " ".join(tokens)
    except Exception as e:
        logging.error(f"Error preprocessing text: {str(e)}")
        return text

logging.info("🧹 Cleaning text...")
df['cleaned_text'] = df['text'].apply(preprocess_text)
logging.info("✅ Text cleaned.")

# Vectorization
logging.info("🔠 Vectorizing using TF-IDF...")
tfidf = TfidfVectorizer(max_features=5000)
tfidf_matrix = tfidf.fit_transform(df['cleaned_text'])
logging.info("✅ TF-IDF matrix shape: %s", tfidf_matrix.shape)

# Cosine similarity
logging.info("📐 Calculating cosine similarity...")
cosine_sim = cosine_similarity(tfidf_matrix, tfidf_matrix)
logging.info("✅ Cosine similarity matrix generated.")

# Save everything
joblib.dump(df, 'df_cleaned.pkl')
joblib.dump(tfidf_matrix, 'tfidf_matrix.pkl')
joblib.dump(cosine_sim, 'cosine_sim.pkl')
logging.info("💾 Data saved to disk.")

logging.info("✅ Preprocessing complete.")