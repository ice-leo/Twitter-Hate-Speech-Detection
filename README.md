# Twitter Sentiment Analysis

A sentiment analysis project submission for classifying tweets using natural language processing techniques, implementing both traditional machine learning (Logistic Regression) and transformer-based approaches (BERT/RoBERTa). Submitted to Analytics Vidhya Hackathon (Currently ranked 7 out of 1400+ submissions with F1-score of **0.8583941606**).

## 📋 Project Overview

This project implements a binary classification system to analyze and categorize tweets. The pipeline includes comprehensive data preprocessing, exploratory data analysis, feature engineering, and model training using two distinct approaches:

1. **Traditional ML Approach**: Logistic Regression with TF-IDF/CountVectorizer
2. **Deep Learning Approach**: BERT-based models (RoBERTa)

---

## 🔬 Approach 1: Logistic Regression

### Workflow

#### 1. Data Preprocessing and EDA (`logistic_regression/preprocessing_eda.ipynb`)

**Data Loading:**
- Training and test datasets loaded from CSV files
- Initial data exploration and summary statistics

**Text Preprocessing Steps:**
- **Remove @user mentions:** Eliminates user mentions and isolated @ symbols
- **Remove & codes:** Cleans HTML entity codes (ampersands)
- **Remove non-alphanumeric characters:** Strips special characters and punctuation
- **Remove duplicates:** Eliminates duplicate entries from the dataset
- **Handle missing values:** Identifies and processes null values
- **Data Augmentation:** Performed Synonym Replacement, Random Insertion, Random Swap, and Random Deletion via [EDA](https://github.com/jasonwei20/eda_nlp) by Wei, Jason and Zou, Kai to augment data

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

#### 2. Model Training (`logistic_regression/logistic_regression.ipynb`)

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
  + `C=2` (regularization strength)
  + `max_iter=1000`
  + `n_jobs=-1` (parallel processing)
  + `class_weight='balanced'` (handles class imbalance)
  + `random_state=42` (reproducibility)

**Handling Class Imbalance:**
- Undersampling techniques
- SMOTE (Synthetic Minority Over-sampling Technique) for data augmentation
- Class weight balancing in the model

**Model Persistence:**
- Best performing model saved as `models/logistic_model.pkl`
- Corresponding TF-IDF vectorizer saved as `models/tfidf_vectorizer_logistic.pkl`

#### 3. Feature Importance and Selection (`logistic_regression/feature_importance_selection.ipynb`)

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

### Logistic Regression Results

- **Performance Metrics:**
  - F1-Score: 0.696 (Current ranking: 547 out of 1400+)
  - Accuracy, Precision, Recall reported in notebooks
  - Confusion Matrix analysis included

### Usage - Logistic Regression

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

---

## 🤖 Approach 2: BERT/RoBERTa

### Workflow

#### 1. Model Implementation (`roberta/`)

**Model Architecture:**
- Uses pre-trained RoBERTa (Robustly Optimized BERT Approach)
- Fine-tuned on Twitter hate speech detection task
- Transformer-based architecture with attention mechanisms

**Key Features:**
- **Contextual Understanding:** Captures bidirectional context of words
- **Transfer Learning:** Leverages pre-trained language representations
- **Attention Mechanism:** Focuses on relevant parts of the text
- **No Manual Feature Engineering:** End-to-end learning from raw text

**Training Configuration:**
- Pre-trained model: RoBERTa-base or RoBERTa-large
- Fine-tuning on Twitter dataset
- Optimization using AdamW optimizer
- Learning rate scheduling
- Batch processing for efficiency

**Preprocessing for BERT:**
- Tokenization using RoBERTa tokenizer
- Special tokens: [CLS], [SEP]
- Padding and truncation to fixed sequence length
- Attention masks for variable-length inputs

### BERT/RoBERTa Results

- **Performance Metrics:**
  - F1-Score: **0.8583941606** (Current ranking: 7 out of 1400+)
  - Significantly outperforms Logistic Regression approach
  - Better handling of context and semantic meaning
  - Improved detection of nuanced hate speech patterns

### Key Advantages of BERT Approach

1. **Contextual Understanding:** Captures word meaning based on surrounding context
2. **No Feature Engineering:** Learns representations automatically
3. **Transfer Learning:** Benefits from pre-training on large text corpora
4. **Better Generalization:** Performs well on diverse language patterns
5. **State-of-the-Art Performance:** Achieves much higher F1-score (0.858 vs 0.696)

### Usage - BERT/RoBERTa

```python
from transformers import RobertaTokenizer, RobertaForSequenceClassification
import torch

# Load model and tokenizer
model = RobertaForSequenceClassification.from_pretrained('models/roberta_model')
tokenizer = RobertaTokenizer.from_pretrained('models/roberta_model')

def predict_tweet_bert(text):
    # Tokenize input
    inputs = tokenizer(text, return_tensors="pt", padding=True, truncation=True, max_length=512)
    
    # Get prediction
    with torch.no_grad():
        outputs = model(**inputs)
        prediction = torch.argmax(outputs.logits, dim=1).item()
    
    return prediction
```

---

## 📊 Comparison: Logistic Regression vs BERT

| Metric | Logistic Regression | BERT/RoBERTa |
|--------|-------------------|--------------|
| **F1-Score** | 0.6964769648 | **0.8583941606** |
| **Ranking** | 547 / 1400+ | **7 / 1400+** |
| **Training Time** | Fast (minutes) | Slower (hours) |
| **Inference Time** | Very Fast | Moderate |
| **Model Size** | Small (~1 MB) | Large (~500 MB) |
| **Interpretability** | High | Low |
| **Context Understanding** | Limited | Excellent |
| **Feature Engineering** | Required | Not Required |
| **Hardware Requirements** | CPU-friendly | GPU recommended |

### When to Use Each Approach

**Use Logistic Regression when:**
- You need fast training and inference
- Model interpretability is important
- Limited computational resources
- Baseline model for comparison
- Simple deployment requirements

**Use BERT/RoBERTa when:**
- Maximum performance is critical
- Context and nuance matter
- Sufficient computational resources available
- State-of-the-art results needed
- Complex language patterns expected

---

## 🛠️ Technologies and Dependencies

### Core Libraries
```
# Data Processing
pandas
numpy

# Traditional ML
scikit-learn
  - LogisticRegression
  - CountVectorizer
  - TfidfVectorizer
  - train_test_split
  - various metrics

# Deep Learning
transformers  # Hugging Face Transformers
torch  # PyTorch
```

### NLP Libraries
```
# Text Processing
nltk
  - stopwords
  - word_tokenize
  - WordNetLemmatizer
  - SnowballStemmer
symspellpy

# Visualization
matplotlib
seaborn
wordcloud

# Model Persistence
pickle
```

---

## 🚀 Getting Started

### Prerequisites

**For Logistic Regression:**
```bash
pip install pandas numpy scikit-learn nltk matplotlib seaborn wordcloud symspellpy
```

**For BERT/RoBERTa:**
```bash
pip install transformers torch pandas numpy
```

Download NLTK data:
```python
import nltk
nltk.download('stopwords')
nltk.download('punkt')
nltk.download('wordnet')
```

### Running the Project

#### Logistic Regression Approach

1. **Data Preprocessing and EDA:**
   ```bash
   jupyter notebook logistic_regression/preprocessing_eda.ipynb
   ```
   - Loads and explores the raw data
   - Performs text cleaning and preprocessing
   - Generates visualizations and statistics

2. **Model Training:**
   ```bash
   jupyter notebook logistic_regression/logistic_regression.ipynb
   ```
   - Applies lemmatization and stemming
   - Tests multiple vectorization strategies
   - Trains logistic regression models
   - Evaluates performance metrics
   - Saves the best model

3. **Feature Analysis:**
   ```bash
   jupyter notebook logistic_regression/feature_importance_selection.ipynb
   ```
   - Analyzes feature importance
   - Selects optimal features
   - Compares vectorization techniques

#### BERT/RoBERTa Approach

1. **Navigate to RoBERTa directory:**
   ```bash
   cd roberta/
   ```

2. **Run training/inference notebooks:**
   - Follow notebooks in the `roberta/` directory
   - Fine-tune pre-trained RoBERTa model
   - Evaluate on test set
   - Generate predictions

---

## 📁 Project Structure

```
.
├── logistic_regression/          # Traditional ML approach
│   ├── preprocessing_eda.ipynb   # Data preprocessing and EDA
│   ├── logistic_regression.ipynb # Model training
│   └── feature_importance_selection.ipynb
│
├── roberta/                       # BERT/RoBERTa approach
│   └── [RoBERTa training notebooks]
│
├── models/                        # Saved models
│   ├── logistic_model.pkl
│   ├── tfidf_vectorizer_logistic.pkl
│   └── roberta_model/            # Fine-tuned RoBERTa
│
├── train_E6oV3lV.csv             # Training dataset
├── test_tweets_anuFYb8.csv       # Test dataset
├── requirements.txt               # Python dependencies
└── README.md                      # This file
```

---

## 📈 Key Findings

### Overall Insights
- Text preprocessing significantly impacts model performance
- Handling class imbalance is crucial for both approaches
- Deep learning models (BERT) significantly outperform traditional ML

### Logistic Regression Findings
- TF-IDF vectorization generally outperforms simple count-based methods
- Combined unigrams and bigrams capture more contextual information
- Feature limitation (`max_features=500`) helps prevent overfitting
- SMOTE and class weights improve results on imbalanced data

### BERT/RoBERTa Findings
- Transformer architecture captures nuanced language patterns
- Pre-training on large corpora provides strong foundation
- Fine-tuning adapts model to hate speech detection task
- Attention mechanisms identify key words and phrases
- Achieves **22% improvement** in F1-score over Logistic Regression

---

## 🎯 Performance Summary

| Model | F1-Score | Rank | Key Strength |
|-------|----------|------|--------------|
| **RoBERTa** | **0.8584** | **7 / 1400+** | Context understanding, nuance detection |
| **Logistic Regression** | 0.6965 | 547 / 1400+ | Fast, interpretable, lightweight |

The RoBERTa model demonstrates that transformer-based architectures are far superior for complex NLP tasks like hate speech detection, where context and semantic understanding are critical.

---

## 🔬 Future Improvements

### For Both Approaches
- Implement k-fold cross-validation for more robust evaluation
- Add hyperparameter tuning (GridSearchCV, RandomizedSearchCV)
- Explore ensemble methods combining both approaches
- Deploy models as REST API
- Add real-time tweet classification functionality
- Create web interface for model demonstration

### Logistic Regression Specific
- Test additional feature engineering techniques
- Experiment with other traditional ML algorithms (SVM, Random Forest)
- Implement stacking/voting ensembles

### BERT/RoBERTa Specific
- Try other transformer models (BERT-large, DistilBERT, ALBERT)
- Implement multi-task learning
- Add data augmentation techniques
- Optimize for inference speed
- Experiment with model distillation for deployment

---

## 📊 Model Evaluation Metrics

Both approaches use comprehensive evaluation including:
- **Accuracy:** Overall correctness
- **Precision:** True positive rate
- **Recall:** Sensitivity
- **F1-Score:** Harmonic mean of precision and recall
- **Confusion Matrix:** Detailed error analysis
- **Classification Report:** Per-class metrics

---

## 📝 Notes

- The dataset files (`train_E6oV3lV.csv` and `test_tweets_anuFYb8.csv`) should be placed in the project root directory
- Preprocessing includes cleaning for Twitter-specific elements (@mentions, URLs, etc.)
- Both approaches use reproducible random seeds for consistency
- BERT models require GPU for efficient training (CPU possible but slow)
- Logistic Regression can run efficiently on CPU

---

## 🤝 Contributing

Contributions are welcome! Please feel free to submit pull requests or open issues for:
- Bug reports
- Feature requests
- Performance improvements
- Documentation enhancements
- New model implementations

---

## 📄 License

This project is available for educational and research purposes.

---

## 🙏 Acknowledgments

- Analytics Vidhya for hosting the hackathon
- Hugging Face for the Transformers library
- EDA implementation by Wei, Jason and Zou, Kai
- Open-source community for various tools and libraries

---

**Note:** This project demonstrates the evolution from traditional machine learning to deep learning approaches in NLP tasks. The significant performance improvement from Logistic Regression (F1: 0.696) to BERT (F1: 0.858) highlights the power of transformer-based architectures for understanding complex linguistic patterns in hate speech detection.
