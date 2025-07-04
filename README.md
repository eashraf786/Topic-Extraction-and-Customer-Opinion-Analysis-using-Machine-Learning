## 🧩 Topic Extraction and Customer Opinion Analysis using Deep Learning
### 📌 Overview
This project demonstrates a complete pipeline for Topic Modeling and Customer Opinion Analysis of unstructured product reviews from Amazon’s software category. Moving beyond basic sentiment analysis, it uncovers 15 nuanced topics hidden within customer feedback using a combination of unsupervised clustering and deep learning classification.

The workflow integrates Word2Vec, advanced clustering (HDBSCAN), dimensionality reduction (PCA, t-SNE), and a robust Artificial Neural Network (ANN) classifier. This end-to-end framework empowers businesses to understand specific product issues, improvement areas, and customer perspectives directly from raw reviews — enabling data-driven decision-making and product enhancements.

### ⚙️ Key Features
📄 Dataset: 12,800+ raw Amazon software product reviews (from UC San Diego dataset)

🗂️ Feature Extraction: Text preprocessing → Word2Vec embeddings → Dimensionality Reduction (PCA/t-SNE)

🔍 Clustering: Unsupervised discovery of hidden topics using multiple algorithms; HDBSCAN tuned for best cluster separation

🤖 Classification: Deep ANN with 876,651 parameters; trained to categorize new reviews into discovered topics

⚖️ Imbalance Handling: SMOTE to balance minority clusters for robust training

🏷️ Keyword Extraction: YAKE to extract top keywords per review for better interpretability

📊 Evaluation: Multiple clustering metrics (Silhouette, Davies-Bouldin, Calinski–Harabasz) and classifier performance (Accuracy, Precision, F1-Score)

### 📈 Results
✅ Best Clustering: HDBSCAN with t-SNE features — 15 clear clusters, minimal outliers (18%)

✅ Topics Identified: E.g., Puzzle Games, Networking, Language Learning, Family Research, Tax Tools, Navigation, Speech Recognition, Video Editing, Disk Backup, MS Office, Windows, Anti-Virus, Legal Paperwork, Finance

✅ ANN Classifier: 93% accuracy with Tanh activation & Adam optimizer

📌 Real Insights: Generated representative keywords, word clouds, and semantic centroids for each cluster

### 🧰 Tools & Tech Stack
Python, Google Colab, Jupyter Notebook

NLP: NLTK, SpaCy, Gensim, Word2Vec, YAKE

ML/DL: Scikit-learn, Keras, PyTorch

Visualization: Seaborn, Matplotlib

Clustering: KMeans, BIRCH, HDBSCAN, Agglomerative

Dimensionality Reduction: PCA, t-SNE

Data Handling: Pandas, NumPy

Imbalance Handling: SMOTE
