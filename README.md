# Customer Review Sentiment Analysis Using Machine Learning

An NLP-based machine learning project that analyzes e-commerce customer reviews and classifies them into **Positive, Neutral, or Negative** sentiment categories.

The project implements a text classification pipeline using TF-IDF feature extraction and compares Multinomial Naive Bayes with Logistic Regression.

## Overview

Customer reviews contain valuable information about customer satisfaction and product quality. This project explores how classical machine learning techniques can automatically identify sentiment from review text.

The workflow covers data preparation, text preprocessing, feature extraction, model training, evaluation, and sentiment prediction.

## Dataset

- **Source:** [Hugging Face — neonwatty/amazon_reviews_multi](https://huggingface.co/datasets/neonwatty/amazon_reviews_multi)
- **Sample size:** 20,000 reviews
- **Language:** English
- **Task:** Three-class sentiment classification

### Sentiment Labeling

Star ratings are mapped to sentiment labels:

| Star Rating | Sentiment |
|---|---|
| 1–2 stars | Negative |
| 3 stars | Neutral |
| 4–5 stars | Positive |

## Methodology

### 1. Data Preparation

- Load customer reviews from a CSV dataset.
- Inspect missing values and dataset structure.
- Prepare the review text and sentiment labels for model training.
- Split the data into training and testing sets using stratified sampling.

### 2. Text Preprocessing

The text preprocessing pipeline includes:

- Lowercasing and punctuation removal
- Whitespace normalization
- NLTK tokenization
- Stop-word removal while preserving important negation terms
- Porter stemming

These steps help standardize the text before feature extraction.

### 3. Feature Extraction

**TF-IDF (Term Frequency–Inverse Document Frequency)** converts review text into numerical features suitable for machine learning.

The baseline configuration uses unigram features. The vectorizer is fitted on the training data and subsequently used to transform the test data to avoid fitting the vocabulary on the test set.

### 4. Model Training

Two supervised classification algorithms are compared:

**Multinomial Naive Bayes**
- Probabilistic classifier based on Bayes' theorem.
- Commonly used for text classification.
- Provides a computationally efficient baseline.

**Logistic Regression**
- Linear classification algorithm that estimates class probabilities.
- Works effectively with sparse TF-IDF text features.
- Serves as a second baseline for comparison.

### 5. Model Evaluation

Performance is measured using:

- Accuracy
- Weighted precision
- Weighted recall
- Weighted F1-score
- Confusion matrices

## Results

The initial experiment used 16,000 training samples and 4,000 test samples.

| Evaluation Metric | Multinomial Naive Bayes | Logistic Regression |
|---|---:|---:|
| Accuracy | 66.65% | 68.40% |
| Weighted Precision | 63.37% | 65.33% |
| Weighted Recall | 66.65% | 68.40% |
| Weighted F1-score | 60.41% | 65.87% |

**Observation:** Logistic Regression achieved better performance across all reported metrics in the baseline experiment. Further class-wise evaluation is useful for understanding errors, especially when distinguishing neutral reviews from positive or negative reviews.

These results represent the initial configuration and should not be interpreted as guaranteed performance on other datasets.

## Technology Stack

| Category | Technologies |
|---|---|
| Programming Language | Python |
| Data Processing | Pandas, NumPy |
| Natural Language Processing | NLTK |
| Machine Learning | Scikit-learn |
| Visualization | Matplotlib |
| Interactive Development | Jupyter Notebook |

## Repository Structure

```text
sentiment-analysis-nlp/
├── data/
│   └── customer_reviews.csv
├── notebooks/
│   ├── sentiment_analysis.ipynb
│   └── sentiment_analysis.py
├── README.md
├── requirements.txt
└── .gitignore
```

## Installation and Execution

### Prerequisites

- Python installed on your system
- Git
- pip

### 1. Clone the repository

```bash
git clone YOUR_REPOSITORY_URL
cd sentiment-analysis-nlp
```

Replace `YOUR_REPOSITORY_URL` with the URL of your GitHub repository.

### 2. Create a virtual environment

On Windows PowerShell:

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
```

On macOS/Linux:

```bash
python3 -m venv .venv
source .venv/bin/activate
```

### 3. Install dependencies

```bash
python -m pip install -r requirements.txt
```

### 4. Run the Python implementation

```bash
python notebooks/sentiment_analysis.py
```

Alternatively, open `notebooks/sentiment_analysis.ipynb` in Jupyter Notebook and execute the cells sequentially.

Ensure that the dataset is available at `data/customer_reviews.csv` and that the code uses a project-relative path.

## Future Enhancements

- Experiment with unigram and bigram TF-IDF features.
- Tune classification model hyperparameters.
- Evaluate per-class precision, recall, and F1-score.
- Analyze confusion matrices and common misclassifications.
- Build an interactive application for user-entered review predictions.
- Explore additional text classification algorithms.

## Key Learning Outcomes

- Implementing a classical NLP pipeline
- Transforming text into numerical representations using TF-IDF
- Training and comparing supervised machine learning models
- Evaluating multiclass classification performance
- Understanding stratified sampling and data leakage prevention




