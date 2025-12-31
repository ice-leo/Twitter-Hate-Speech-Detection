# Twitter Sentiment Analysis

A sentiment analysis project submission for classifying tweets using natural language processing techniques, implementing both traditional machine learning (Logistic Regression) and transformer-based approaches (BERT/RoBERTa). Submitted to Analytics Vidhya Hackathon (Currently ranked 6 out of 1400+ submissions with F1-score of **0.8621700880**).

## 📋 Project Overview

This project implements a binary classification system to analyze and categorize tweets. The pipeline includes comprehensive data preprocessing, exploratory data analysis, feature engineering, and model training using two distinct approaches:

1. **Traditional ML Approach**: Logistic Regression with TF-IDF/CountVectorizer
2. **Deep Learning Approach**: DistilRoBERTa (Distilled RoBERTa)

---

## 🔬 Approach 1: Logistic Regression

### Workflow

#### 1. Data Preprocessing and EDA (`logistic_regression/preprocessing_eda_LR.ipynb`)

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

#### 2. Model Training (`logistic_regression/LR.ipynb`)

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

#### 3. Feature Importance and Selection (`logistic_regression/feature_importance_selection_LR.ipynb`)

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

## 🤖 Approach 2: DistilRoBERTa

### Workflow

#### 1. Model Training (`roberta.ipynb`)

**Model Architecture:**
- Uses pre-trained **DistilRoBERTa-base** model
- Distilled version of RoBERTa - faster and lighter while maintaining performance
- Fine-tuned for binary sequence classification (hate speech detection)
- Transformer-based architecture with self-attention mechanisms

**Data Preparation:**
- **Dataset Split:** 80/20 train-validation split with stratification
  - Training: 25,569 samples
  - Validation: 6,393 samples
- **Stratification:** Maintains class balance across splits
- **Format:** Converted to Hugging Face Dataset format for efficient processing

**Tokenization Configuration:**
```python
tokenizer = AutoTokenizer.from_pretrained("distilroberta-base")

# Tokenization parameters:
- padding: 'max_length'
- truncation: True
- max_length: 128 tokens
```

**Training Configuration:**
```python
TrainingArguments:
  - output_dir: "./results"
  - eval_strategy: "epoch"
  - save_strategy: "epoch"
  - learning_rate: 2e-5
  - per_device_train_batch_size: 16
  - per_device_eval_batch_size: 16
  - num_train_epochs: 3
  - weight_decay: 0.01
  - load_best_model_at_end: True
  - metric_for_best_model: "f1"
```

**Training Process:**
- **Optimizer:** AdamW (default in Transformers Trainer)
- **Epochs:** 3
- **Total Training Steps:** 4,797
- **Training Time:** ~51 minutes 38 seconds
- **Training Loss:** Decreased from 0.0857 (Epoch 1) to 0.0337 (Epoch 3)
- **Evaluation:** Performed at the end of each epoch
- **Best Model Selection:** Based on F1-score

**Training Progress:**

| Epoch | Training Loss | Validation Loss | Accuracy | Precision | Recall | F1-Score |
|-------|--------------|-----------------|----------|-----------|---------|----------|
| 1 | 0.0857 | 0.0823 | 0.9779 | 0.9137 | 0.7567 | **0.8278** |
| 2 | 0.0458 | 0.0851 | 0.9801 | 0.8867 | 0.8214 | **0.8528** |
| 3 | 0.0337 | 0.0900 | 0.9814 | 0.9042 | 0.8214 | **0.8608** |

**Evaluation Metrics:**
```python
def compute_metrics(eval_pred):
    - Accuracy
    - Precision (binary)
    - Recall (binary)
    - F1-Score (binary)
```

**Model Persistence:**
- Saved as `model_roberta/` directory
- Includes model weights, configuration, and tokenizer files

#### 2. Model Inference (`apply_roberta.ipynb`)

**Inference Setup:**
- **Device:** CUDA if available, otherwise CPU
- **Model Loading:** From saved `model.safetensors` and `config.json`
- **Batch Processing:** Batch size of 2 for efficient inference
- **Max Sequence Length:** 48 tokens (optimized for inference speed)

**Inference Pipeline:**
```python
# 1. Load test data
test_df = pd.read_csv('test_tweets_anuFYb8.csv')

# 2. Tokenize with inference settings
tokenizer = AutoTokenizer.from_pretrained("distilroberta-base")
encodings = tokenizer(tweets, padding=True, truncation=True, max_length=48)

# 3. Create DataLoader
dataset = TweetDataset(encodings)
loader = DataLoader(dataset, batch_size=2)

# 4. Run inference
model.eval()
with torch.no_grad():
    for batch in loader:
        outputs = model(**batch)
        predictions = torch.argmax(outputs.logits, dim=1)

# 5. Save predictions
output_df.to_csv('test_predictions.csv')
```

**Inference Optimization:**
- **Gradient Computation:** Disabled with `torch.no_grad()` for speed
- **Model Mode:** Set to `eval()` to disable dropout and batch normalization updates
- **Memory Management:** `torch.cuda.empty_cache()` for GPU memory cleanup
- **Shorter Sequences:** 48 tokens vs 128 during training for faster processing

### DistilRoBERTa Results

**Final Performance (Epoch 3 - Best Model):**
- **Accuracy:** 98.14%
- **Precision:** 90.42%
- **Recall:** 82.14%
- **F1-Score:** **0.8608** (Validation set)
- **Test Set F1-Score:** **0.8584** (Competition submission)
- **Ranking:** **7 out of 1400+** submissions

**Performance Progression:**
- Steady improvement across epochs
- F1-Score improved by ~3.3% from Epoch 1 to Epoch 3
- Training loss reduced by 60% (0.0857 → 0.0337)
- Model converged well without overfitting

### Key Advantages of DistilRoBERTa Approach

1. **Contextual Understanding:** Captures bidirectional context and semantic meaning
2. **No Feature Engineering:** Learns representations automatically from raw text
3. **Transfer Learning:** Benefits from pre-training on large text corpora
4. **Efficiency:** DistilRoBERTa is 40% smaller and 60% faster than RoBERTa-base
5. **Superior Performance:** Achieves **24% relative improvement** in F1-score over Logistic Regression (0.8608 vs 0.6965)
6. **Better Minority Class Detection:** Higher recall for hate speech detection
7. **Robust Generalization:** Strong performance on unseen test data

### Usage - DistilRoBERTa

**For Training:**
```python
from transformers import AutoTokenizer, AutoModelForSequenceClassification, Trainer, TrainingArguments
from datasets import Dataset

# Load tokenizer and model
tokenizer = AutoTokenizer.from_pretrained("distilroberta-base")
model = AutoModelForSequenceClassification.from_pretrained("distilroberta-base", num_labels=2)

# Tokenize data
train_dataset = Dataset.from_pandas(train_df)
train_dataset = train_dataset.map(
    lambda x: tokenizer(x['tweet'], padding='max_length', truncation=True, max_length=128),
    batched=True
)

# Configure training
training_args = TrainingArguments(
    output_dir="./results",
    num_train_epochs=3,
    learning_rate=2e-5,
    per_device_train_batch_size=16,
    eval_strategy="epoch"
)

# Train
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=val_dataset
)
trainer.train()
```

**For Inference:**
```python
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification, AutoConfig

# Load model and tokenizer
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
config = AutoConfig.from_pretrained('config.json')
model = AutoModelForSequenceClassification.from_pretrained('model.safetensors', config=config).to(DEVICE)
tokenizer = AutoTokenizer.from_pretrained("distilroberta-base")

def predict_tweet(text):
    # Tokenize input
    inputs = tokenizer(text, return_tensors="pt", padding=True, truncation=True, max_length=48)
    inputs = {k: v.to(DEVICE) for k, v in inputs.items()}
    
    # Get prediction
    model.eval()
    with torch.no_grad():
        outputs = model(**inputs)
        prediction = torch.argmax(outputs.logits, dim=1).item()
    
    return prediction

# Example usage
tweet = "@user this is a sample tweet"
label = predict_tweet(tweet)
print(f"Predicted label: {label}")  # 0: non-hate, 1: hate speech
```

---

## 📊 Comparison: Logistic Regression vs DistilRoBERTa

| Metric | Logistic Regression | DistilRoBERTa |
|--------|-------------------|--------------|
| **F1-Score (Validation)** | 0.74 | **0.8608** |
| **F1-Score (Test/Competition)** | 0.6965 | **0.8622** |
   | **Accuracy** | 91% | **98.14%** |
| **Precision** |  75% | **90.42%** |
| **Recall** | 73% | **82.14%** |
| **Ranking** | 547 / 1400+ | **7 / 1400+** |
| **Training Time** | Fast (minutes) | Moderate (~52 minutes) |
| **Inference Time** | Very Fast | Fast (with GPU) |
| **Model Size** | Small (~1 MB) | Medium (~82 MB) |
| **Parameters** | Thousands | ~82 Million |
| **Interpretability** | High | Low |
| **Context Understanding** | Limited | Excellent |
| **Feature Engineering** | Required | Not Required |
| **Hardware Requirements** | CPU-friendly | GPU recommended |
| **Max Sequence Length** | N/A (uses TF-IDF) | 128 tokens (train), 48 (inference) |
| **Preprocessing Complexity** | High (lemmatization, stemming) | Low (tokenization only) |

### When to Use Each Approach

**Use Logistic Regression when:**
- You need fast training and inference
- Model interpretability is critical for business requirements
- Limited computational resources (CPU only)
- Baseline model for comparison needed
- Simple deployment requirements (lightweight)
- Feature importance analysis is needed
- Budget constraints limit GPU access

**Use DistilRoBERTa when:**
- Maximum performance is the priority
- Context and semantic understanding matter
- GPU resources are available
- State-of-the-art results are required
- Complex language patterns expected (sarcasm, nuanced hate speech)
- You need production-ready performance
- 24% performance improvement justifies computational cost

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

# Deep Learning (DistilRoBERTa)
transformers  # Hugging Face Transformers library
torch  # PyTorch
datasets  # Hugging Face Datasets library
tqdm  # Progress bars
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

**For DistilRoBERTa:**
```bash
pip install transformers torch datasets pandas numpy tqdm
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
   jupyter notebook logistic_regression/preprocessing_eda_LR.ipynb
   ```
   - Loads and explores the raw data
   - Performs text cleaning and preprocessing
   - Generates visualizations and statistics

2. **Model Training:**
   ```bash
   jupyter notebook logistic_regression/LR.ipynb
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

#### DistilRoBERTa Approach

1. **Training the model:**
   ```bash
   jupyter notebook roberta.ipynb
   ```
   - Loads and splits the training data (80/20 split)
   - Converts to Hugging Face Dataset format
   - Tokenizes with DistilRoBERTa tokenizer
   - Fine-tunes pre-trained DistilRoBERTa model (3 epochs)
   - Evaluates on validation set
   - Saves the best model based on F1-score

2. **Running inference on test data:**
   ```bash
   jupyter notebook apply_roberta.ipynb
   ```
   - Loads test dataset
   - Loads trained model and configuration
   - Tokenizes test tweets
   - Generates predictions in batches
   - Saves predictions to CSV file

---

## 📁 Project Structure

```
.
├── logistic_regression/          # Traditional ML approach
│   ├── preprocessing_eda_LR.ipynb   # Data preprocessing and EDA
│   ├── LR.ipynb # Model training
│   └── feature_importance_selection_LR.ipynb
│
├── roberta.ipynb                  # DistilRoBERTa training notebook
├── apply_roberta.ipynb            # DistilRoBERTa inference notebook
│
├── models/                        # Saved models
│   ├── logistic_model.pkl
│   ├── tfidf_vectorizer_logistic.pkl
│   └── model_roberta/            # Fine-tuned DistilRoBERTa
│       ├── config.json
│       ├── model.safetensors
│       └── [other model files]
│
├── train_E6oV3lV.csv             # Training dataset
├── test_tweets_anuFYb8.csv       # Test dataset
├── test_predictions.csv          # Model predictions output
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

### DistilRoBERTa Findings
- Transformer architecture captures nuanced language patterns and context
- Pre-training on large corpora provides strong foundation for transfer learning
- Fine-tuning adapts model effectively to hate speech detection task
- Self-attention mechanisms identify key words and phrases automatically
- Achieves **24% relative improvement** in F1-score over Logistic Regression (0.8608 vs 0.6965)
- DistilRoBERTa provides excellent balance between performance and efficiency
- Model shows strong convergence with steady improvement across epochs
- Validation F1-score (0.8608) closely matches test F1-score (0.8584), indicating good generalization

---

## 🎯 Performance Summary

| Model | F1-Score (Val) | F1-Score (Test) | Rank | Key Strength |
|-------|----------------|-----------------|------|--------------|
| **DistilRoBERTa** | **0.8608** | **0.8622** | **7 / 1400+** | Context understanding, semantic analysis, efficiency |
| **Logistic Regression** | 0.74 | 0.6965 | 547 / 1400+ | Fast, interpretable, lightweight |

The DistilRoBERTa model demonstrates that efficient transformer-based architectures can achieve superior performance for complex NLP tasks like hate speech detection, where context and semantic understanding are critical. The distilled version provides an excellent balance between the performance of large transformers and the efficiency needed for practical deployment.

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

### DistilRoBERTa Specific
- Experiment with full RoBERTa-base/large for potentially higher performance
- Try other efficient transformers (ALBERT, MobileBERT, TinyBERT)
- Implement curriculum learning strategies
- Add data augmentation techniques (back-translation, synonym replacement)
- Optimize inference speed with ONNX runtime or TensorRT
- Experiment with different max_length configurations
- Try ensemble methods combining multiple transformer checkpoints
- Implement model distillation for even faster inference
- Explore domain-adaptive pre-training on Twitter data

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

**Note:** This project demonstrates the evolution from traditional machine learning to modern deep learning approaches in NLP tasks. The significant performance improvement from Logistic Regression (F1: 0.6965) to DistilRoBERTa (F1: 0.8608) highlights the power of transformer-based architectures for understanding complex linguistic patterns in hate speech detection. The use of DistilRoBERTa specifically shows that efficient, distilled models can achieve near state-of-the-art performance while maintaining practical deployment characteristics.
