🎵 Music Recommendation System (NLP-Based)

An end-to-end content-based music recommendation system built using NLP techniques on song lyrics.
The system leverages TF-IDF vectorization and cosine similarity to recommend similar songs and is deployed as an interactive web application.

🚀 Live Demo:
👉 https://samacker25-music-recom.streamlit.app/

📌 Project Overview

This project demonstrates how traditional NLP techniques can be used to build an effective recommendation engine.
Large datasets and model artifacts are managed using Hugging Face, while the application is deployed on Streamlit for seamless web access.

Key highlights:

Content-based recommendation (no user history required)

Scalable handling of large datasets

Clean, modular ML pipeline

Production-ready deployment

🧠 How It Works

Text Preprocessing

Cleans song lyrics (lowercasing, punctuation removal, stopword removal)

Feature Engineering

Converts lyrics into numerical vectors using TF-IDF

Similarity Search

Computes pairwise similarity using cosine similarity

Recommendation

Returns the most similar songs based on lyrical content

Deployment

Preprocessed artifacts loaded dynamically from HuggingFace

Served via Streamlit Cloud

🧰 Tech Stack

Core Technologies

Python

Scikit-learn

NLP (TF-IDF, Cosine Similarity)

Pandas, NumPy

Data & Deployment

HuggingFace Datasets

Streamlit Cloud

Tools & Practices

Git & GitHub

Modular code structure

Clean Git history for large-file handling

📁 Project Structure
Music-recommendation-app/
│
├── SRC/
│   ├── main.py           # Streamlit application
│   ├── preprocess.py     # Data preprocessing pipeline
│   ├── recommend.py      # Recommendation logic
│
├── requirements.txt      # Project dependencies
├── .gitignore            # Ignored large files & environments
└── README.md

⚙️ Running Locally
1️⃣ Clone the repository
git clone https://github.com/Samacker25/Music-recommendation-app.git
cd Music-recommendation-app

2️⃣ Create virtual environment & install dependencies
python -m venv .venv
source .venv/bin/activate  # Windows: .venv\Scripts\activate
pip install -r requirements.txt

3️⃣ Run the app
streamlit run SRC/main.py

📦 Dataset & Model Artifacts

Large files are not stored in GitHub.

They are hosted on HuggingFace:

df_cleaned.pkl

cosine_sim.pkl

The app downloads them dynamically at runtime using huggingface_hub.

🎯 Key Learnings

Implemented a complete NLP-based recommendation pipeline

Learned best practices for handling large ML artifacts

Deployed a production-ready ML app using Streamlit Cloud

Gained hands-on experience with real-world MLOps challenges

🔗 Links

🔴 Live App: https://samacker25-music-recom.streamlit.app/

💻 GitHub Repo: https://github.com/Samacker25/Music-recommendation-app

📦 Dataset: https://huggingface.co/datasets/Samacker25/music-recemmondation-data

🙌 Author

Soumen Kundu
Aspiring ML / MLOps Engineer
🔗 GitHub: https://github.com/Samacker25
