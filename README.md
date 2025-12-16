# Twitter-Hate-Speech-Detection
Sentiment analysis project submission for Analytics Vidhya

# Twitter Sentiment Analysis

A machine learning project for classifying tweets using natural language processing techniques and logistic regression models.

## 📋 Project Overview

This project implements a binary classification system to analyze and categorize tweets. The pipeline includes comprehensive data preprocessing, exploratory data analysis, feature engineering, and model training with various text vectorization techniques.

## 🔍 Project Workflow

### 1. Data Preprocessing and EDA (`preprocessing_eda.ipynb`)

**Data Loading:**
- Training and test datasets loaded from CSV files
- Initial data exploration and summary statistics

**Text Preprocessing Steps:**
- **Remove @user mentions:** Eliminates user mentions and isolated @ symbols
- **Remove &amp; codes:** Cleans HTML entity codes (ampersands)
- **Remove non-alphanumeric characters:** Strips special characters and punctuation
- **Remove duplicates:** Eliminates duplicate entries from the dataset
- **Handle missing values:** Identifies and processes null values

**Exploratory Data Analysis:**
- Class distribution visualization (0s and 1s frequency)
- Word cloud generation
- Text statistics and patterns analysis

**Libraries Used:**
- pandas, numpy for data manipulation
- matplotlib, seaborn for visualization
- re for regular expressions
- wordcloud for text visualization
- symspellpy for spell checking
- nltk for NLP tasks (stopwords, tokenization, lemmatization, stemming)

### 2. Model Training (`logistic_regression.ipynb`)

**Text Processing:**
- **Lemmatization:** Reduces words to their base form using WordNetLemmatizer
- **Stemming:** Applies SnowballStemmer for word root extraction
- **Duplicate removal:** Final cleanup of processed text

**Vectorization Techniques:**

The project experiments with multiple vectorization approaches:

1. **CountVectorizer:**
   - Unigrams (1,1)
   - Bigrams (2,2)
   - Unigrams + Bigrams (1,2)
   - Parameters: `min_df=10`, `max_features=500`

2. **TF-IDF Vectorizer:**
   - Unigrams (1,1)
   - Bigrams (2,2)
   - Unigrams + Bigrams (1,2)
   - Parameters: `min_df=10`, `max_features=500`

**Model Configuration:**
- **Algorithm:** Logistic Regression
- **Hyperparameters:**
  - `C=2` (regularization strength)
  - `max_iter=1000`
  - `n_jobs=-1` (parallel processing)
  - `class_weight='balanced'` (handles class imbalance)
  - `random_state=42` (reproducibility)

**Handling Class Imbalance:**
- Undersampling techniques
- SMOTE (Synthetic Minority Over-sampling Technique) for data augmentation
- Class weight balancing in the model

**Model Persistence:**
- Best performing model saved as `logistic_model.pkl`
- Corresponding TF-IDF vectorizer saved as `tfidf_vectorizer_logistic.pkl`

### 3. Feature Importance and Selection (`feature_importance_selection.ipynb`)

**Feature Analysis:**
- Evaluation of feature importance from trained models
- Feature selection based on relevance and impact
- Comparison of different vectorization strategies

**Resampling:**
- Implements various resampling techniques to address class imbalance
- Evaluates model performance across different sampling strategies

**Vectorization Comparison:**
- CountVectorizer vs TF-IDF performance analysis
- Unigram, bigram, and combined n-gram evaluation
- Optimal feature selection for production model

## 🛠️ Technologies and Dependencies

```python
# Core Libraries
pandas
numpy

# Machine Learning
scikit-learn
  - LogisticRegression
  - CountVectorizer
  - TfidfVectorizer
  - train_test_split
  - various metrics

# NLP Libraries
nltk
  - stopwords
  - word_tokenize
  - WordNetLemmatizer
  - SnowballStemmer
symspellpy

# Data Visualization
matplotlib
seaborn
wordcloud

# Data Handling
pickle  # for model serialization
```

## 🚀 Getting Started

### Prerequisites

Install the required packages:

```bash
pip install pandas numpy scikit-learn nltk matplotlib seaborn wordcloud symspellpy
```

Download NLTK data:

```python
import nltk
nltk.download('stopwords')
nltk.download('punkt')
nltk.download('wordnet')
```

### Running the Project

1. **Data Preprocessing and EDA:**
   ```bash
   jupyter notebook preprocessing_eda.ipynb
   ```
   - Loads and explores the raw data
   - Performs text cleaning and preprocessing
   - Generates visualizations and statistics

2. **Model Training:**
   ```bash
   jupyter notebook logistic_regression.ipynb
   ```
   - Applies lemmatization and stemming
   - Tests multiple vectorization strategies
   - Trains logistic regression models
   - Evaluates performance metrics
   - Saves the best model

3. **Feature Analysis:**
   ```bash
   jupyter notebook feature_importance_selection.ipynb
   ```
   - Analyzes feature importance
   - Selects optimal features
   - Compares vectorization techniques

## 📊 Model Evaluation

The project uses comprehensive evaluation metrics including:
- Accuracy
- Precision
- Recall
- F1-Score
- Confusion Matrix
- Classification Report

Models are compared across different configurations to identify the optimal setup.

## 💾 Model Output

The final trained models are saved in the `models/` directory:
- `logistic_model.pkl`: Trained logistic regression model
- `tfidf_vectorizer_logistic.pkl`: Fitted TF-IDF vectorizer

These can be loaded for inference on new data:

```python
import pickle

# Load model and vectorizer
with open('models/logistic_model.pkl', 'rb') as f:
    model = pickle.load(f)

with open('models/tfidf_vectorizer_logistic.pkl', 'rb') as f:
    vectorizer = pickle.load(f)

# Make predictions
def predict_tweet(text):
    text_vectorized = vectorizer.transform([text])
    prediction = model.predict(text_vectorized)
    return prediction[0]
```

## 🔬 Key Findings

- Text preprocessing significantly impacts model performance
- TF-IDF vectorization generally outperforms simple count-based methods
- Handling class imbalance through SMOTE and class weights improves results
- Combined unigrams and bigrams capture more contextual information
- Feature limitation (`max_features=500`) helps prevent overfitting

## 📈 Future Improvements

- Experiment with deep learning models (LSTM, BERT)
- Implement cross-validation for more robust evaluation
- Add hyperparameter tuning (GridSearchCV, RandomizedSearchCV)
- Explore ensemble methods
- Deploy model as a web API
- Add real-time tweet classification functionality

## 📝 Notes

- The dataset files (`train_E6oV3lV.csv` and `test_tweets_anuFYb8.csv`) should be placed in the project root directory
- Preprocessing includes cleaning for Twitter-specific elements (@mentions, URLs, etc.)
- The project uses reproducible random seeds for consistency
- Currently ranked 547 out of 1400+ submissions with F1-score of **0.6964769648**

## 🤝 Contributing

Contributions are welcome! Please feel free to submit pull requests or open issues for bugs and feature requests.

## 📄 License

This project is available for educational and research purposes.

---

**Note:** This project was developed as part of a machine learning classification task focusing on natural language processing and text classification techniques.
