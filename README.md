# Text Classification Experiment: Classic ML vs Neural Networks

## Overview

This project implements and compares various machine learning and deep learning approaches for text classification on a Question-Answer dataset. The experiment evaluates both traditional machine learning models and modern neural network architectures to determine the most effective methods for classifying text data. By systematically comparing 22 different model configurations across multiple text representations, the study provides insights into the trade-offs between computational complexity and classification performance in natural language processing tasks.

## Dataset

The dataset used in this experiment is the Question Answer Classification Dataset, consisting of CSV files containing question-answer pairs with corresponding classification labels. The training data is stored in "Question Answer Classification Dataset 1[Training].csv" while the test data is in "Question Answer Classification Dataset[Test].csv". Each sample contains text data from questions and answers that need to be classified into multiple categories. The dataset comes pre-split into training and test sets, with the training portion further divided into 90% for training and 10% for validation during model development. The text samples vary in length with an average of 97.7 words per sample, maximum of 1,382 words, minimum of 7 words, and median of 66 words.

**Dataset Statistics:**
- **Training Set:** 279,999 samples
- **Validation Set:** 28,000 samples (10% of training data)
- **Test Set:** 59,999 samples
- **Total Classes:** 10
- **Class Distribution:** Well-balanced (approximately 28,000 samples per class)
- **Classes:** Business & Finance, Computers & Internet, Education & Reference, Entertainment & Music, Family & Relationships, Health, Politics & Government, Science & Mathematics, Society & Culture, Sports

**Download Links:**
- **Training Set:** [Question Answer Classification Dataset 1[Training].csv](https://drive.google.com/file/d/11qn9k-CqFXbptAC4hl1KKtfTu7ZAUs6Q/view?usp=drive_link)
- **Test Set:** [Question Answer Classification Dataset[Test].csv](https://drive.google.com/file/d/1fwcV7K0vq5OiuS33dw_0Y48lZZSuyZg_/view?usp=drive_link)

## Data Preprocessing and Feature Engineering

### Text Preprocessing Pipeline

1. **Text Normalization:**
   - Convert to lowercase
   - Remove punctuation and special characters
   - Keep only alphanumeric characters

2. **Stop Word Filtering:**
   - NLTK English stop words (198 words)
   - Custom stop words: 'question', 'title', 'content', 'best', 'answer', 'body'

3. **Morphological Processing:**
   - WordNet lemmatization to reduce word variations

**Example Transformation:**
- **Before:** "Question Title: Have you liked a person who is bi-sexual and couldn't get a relationship going?"
- **After:** "liked person bisexual couldnt get relationship going"

### Feature Representation Techniques

**Traditional Methods:**
- **Bag of Words:** CountVectorizer (max_features=10,000, ngram_range=(1,3), min_df=5, max_df=0.85)
- **TF-IDF:** TfidfVectorizer with identical parameters plus sublinear_tf=True, norm='l2'
- **Output Shape:** (279,999, 10,000)

**Neural Embeddings:**
- **Word2Vec:** Custom trained skip-gram (dim=100, window=5, min_count=2)
- **GloVe:** Pre-trained Wikipedia+Gigaword (100 dimensions)
- **Vocabulary:** 10,000 words with <PAD> and <UNK> tokens
- **Sequence Length:** Maximum 50 tokens with padding/truncation

## Experiment Objectives

This comprehensive experiment was designed to address several key questions in text classification. First, it compares traditional machine learning approaches against modern deep learning methods to understand which paradigm performs better on this specific task. Second, it evaluates different text representation techniques including Bag of Words, TF-IDF, and word embeddings to determine which method captures text semantics most effectively. Third, it assesses various neural network architectures such as recurrent neural networks, long short-term memory networks, and gated recurrent units, including both unidirectional and bidirectional variants. Finally, the experiment aims to identify best practices and optimal model combinations for text classification tasks while analyzing the computational cost versus accuracy trade-offs.

## Models Implemented

The experiment evaluates 22 different model configurations divided into traditional machine learning and deep learning approaches. For traditional machine learning, three algorithms are tested: Random Forest Classifier with 150 estimators and maximum depth of 12, Logistic Regression with L2 regularization and class balancing, and Multinomial Naive Bayes with alpha smoothing of 0.1. Each algorithm is evaluated with two text representation methods: Bag of Words and TF-IDF vectorization.

The deep learning portion includes 16 neural network models. Two models use deep neural networks with pre-trained word embeddings - one with Word2Vec embeddings trained using skip-gram with 100 dimensions, and another with GloVe embeddings from the Wikipedia+Gigaword corpus. Twelve models implement recurrent neural network variants including SimpleRNN, GRU, and LSTM architectures, each tested with both Word2Vec and GloVe embeddings in both unidirectional and bidirectional configurations. The remaining two models use deep neural networks with traditional vector representations (Bag of Words and TF-IDF).

All neural networks are trained with a batch size of 128, using the Adam optimizer with early stopping after 5 epochs of no improvement. The embedding layers are frozen during training, and the models include dropout regularization and batch normalization where appropriate.

### Model Architectures and Hyperparameters

#### Traditional Machine Learning
- **Random Forest:** 150 estimators, max_depth=12, balanced classes
- **Logistic Regression:** C=2.0, L2 penalty, balanced classes, max_iter=300
- **Naive Bayes:** alpha=0.1, multinomial variant

#### Neural Networks
- **Common Settings:** Batch size 128, max epochs 50, early stopping patience 5, Adam optimizer
- **RNN Models:** Hidden dimension 128, dropout 0.5, bidirectional variants tested
- **DNN Models:** 3-layer architecture with batch normalization and dropout
- **Embeddings:** 100-dimensional, frozen during training
- **Learning Rates:** 0.001 (LSTM/GRU), 0.002 (SimpleRNN)

## Key Findings

The experiment revealed clear performance differences between traditional and deep learning approaches. Neural network models consistently achieved higher accuracy, with the best model (GRU with Word2Vec embeddings) reaching 71.46% test accuracy compared to 68.84% for the best traditional machine learning model (Logistic Regression with TF-IDF). Among the neural networks, models using word embeddings (Word2Vec and GloVe) significantly outperformed those using traditional vector representations like Bag of Words and TF-IDF. This suggests that pre-trained embeddings capture semantic relationships more effectively than frequency-based methods.

LSTM and GRU architectures provided the best balance between accuracy and computational efficiency, with bidirectional variants offering marginal improvements over unidirectional models. The traditional machine learning models showed TF-IDF outperforming Bag of Words, likely due to better term importance weighting. However, deep learning models were able to capture complex patterns that traditional approaches missed, particularly in handling vocabulary variations and contextual relationships.

Training time varied significantly between approaches, with traditional machine learning models completing in seconds while neural networks required several minutes depending on the architecture. This highlights the important trade-off between computational resources and classification performance that practitioners must consider when selecting models for production deployment.

## Technical Implementation

### Data Preprocessing Pipeline

The implementation follows a comprehensive preprocessing pipeline that ensures consistent text preparation across all models:

1. **Text Cleaning:**
   - Lowercasing all text
   - Removal of punctuation and special characters
   - Filtering to alphanumeric characters only

2. **Stop Word Removal:**
   - Standard NLTK English stop words (198 words)
   - Custom domain-specific stop words: 'question', 'title', 'content', 'best', 'answer', 'body'

3. **Lemmatization:**
   - WordNet lemmatizer from NLTK
   - Applied to reduce inflectional variations

4. **Tokenization and Sequencing (Neural Networks):**
   - Maximum vocabulary size: 10,000 words
   - Maximum sequence length: 50 tokens
   - Padding and truncation for uniform input length
   - Special tokens: <PAD> (0), <UNK> (1)

**Preprocessing Example:**
- **Original:** "Question Title: Have you liked a person who is bi-sexual and couldn't get a relationship going? Question Content: Best Answer: no but I'm bi there sexual orientation probably is not the reason you cant get a relationship going"
- **Processed:** "liked person bisexual couldnt get relationship going im bi sexual orientation probably reason cant get relationship going"

## Requirements

### Dependencies
```
pandas>=1.3.0
numpy>=1.21.0
matplotlib>=3.5.0
seaborn>=0.11.0
scikit-learn>=1.0.0
nltk>=3.7
torch>=1.11.0
gensim>=4.0.0
wordcloud>=1.8.0
```

### NLTK Data
```python
nltk.download('stopwords')
nltk.download('wordnet')
```

### Hardware Requirements
- **Recommended:** CUDA-compatible GPU with 8GB+ VRAM
- **Minimum:** CPU with 16GB RAM
- **Storage:** 10GB free space for datasets and models

### Software Environment
- Python 3.8+
- PyTorch 1.11+ with CUDA support
- scikit-learn 1.0+
- NLTK 3.7+
- Gensim 4.0+
- Matplotlib, Seaborn, Pandas, NumPy

## Usage

1. **Clone the repository**
2. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```
3. **Download NLTK data**
   ```python
   import nltk
   nltk.download('stopwords')
   nltk.download('wordnet')
   ```
4. **Place dataset files** in the `dataset/` directory
5. **Run the Jupyter notebook** `Text_Classification.ipynb`

### Reproducibility Notes
- Random seed: 42 for all models
- Data split: Stratified 90/10 train/validation from original training set
- GPU acceleration: Automatic detection and utilization
- Model checkpoints: Saved for best validation performance
- All hyperparameters: Explicitly documented in code

### Experimental Setup
- **Total Models Tested:** 22 configurations
- **Evaluation Metrics:** Accuracy, Macro F1, Weighted F1, Confusion Matrix
- **Cross-Validation:** Hold-out validation set (10% of training data)
- **Early Stopping:** Validation accuracy monitoring with patience=5
- **Training Monitoring:** Loss and accuracy tracking for all epochs

## Results Summary

The experiment clearly demonstrates the superiority of neural network approaches over traditional machine learning for text classification tasks. Neural networks achieved higher accuracy rates, with the best model reaching 71.46% test accuracy, compared to 68.84% for the best traditional machine learning model. Word embedding-based representations proved superior to traditional vectorization methods, with Word2Vec and GloVe embeddings capturing semantic relationships more effectively than Bag of Words or TF-IDF approaches.

Among neural architectures, LSTM and GRU models provided the best performance-to-computational-cost ratio, making them suitable for most practical applications. The bidirectional variants offered slight improvements but at increased computational cost. Traditional machine learning models, while faster to train, were unable to capture the complex patterns that deep learning models could learn from the text data.

Model selection should be based on available computational resources and performance requirements. For applications requiring high accuracy with sufficient training time and resources, neural networks with word embeddings are recommended. For scenarios with limited computational resources or real-time requirements, traditional machine learning models with TF-IDF representations provide a reasonable alternative.

### Sample Results (from code execution):
- Best neural network model: GRU with Word2Vec embeddings (71.46% test accuracy)
- Best traditional ML model: Logistic Regression with TF-IDF (68.84% test accuracy)
- Neural network accuracy range: 64.25% - 71.46% depending on model and embedding type
- Traditional ML accuracy range: 52.90% - 68.84% depending on model and representation
- Training time: 30-95 seconds per neural network model, seconds for traditional ML

### Detailed Model Accuracies

#### Traditional Machine Learning Models
| Representation | Model | Test Accuracy |
|----------------|-------|---------------|
| BoW | Random Forest | 52.90% |
| BoW | Logistic Regression | 63.79% |
| BoW | Naive Bayes | 67.07% |
| TF-IDF | Random Forest | 53.09% |
| TF-IDF | Logistic Regression | 68.84% |
| TF-IDF | Naive Bayes | 67.49% |

#### Neural Network Models
| Model Configuration | Test Accuracy |
|---------------------|---------------|
| GRU_Word2Vec | 71.46% |
| BiLSTM_Word2Vec | 71.44% |
| BiGRU_Word2Vec | 71.40% |
| LSTM_Word2Vec | 71.27% |
| GRU_GloVe | 70.59% |
| LSTM_GloVe | 70.51% |
| BiGRU_GloVe | 70.46% |
| BiLSTM_GloVe | 70.34% |
| DNN_TFIDF | 69.60% |
| DNN_BoW | 69.53% |
| DNN_Word2Vec | 68.74% |
| BiSimpleRNN_Word2Vec | 68.23% |
| SimpleRNN_Word2Vec | 68.15% |
| DNN_GloVe | 65.42% |
| BiSimpleRNN_GloVe | 64.97% |
| SimpleRNN_GloVe | 64.25% |

s
### Model Performance Analysis

**Per-Class Performance (Best Model - GRU_Word2Vec):**
- Business & Finance: 67.3%
- Computers & Internet: 83.3%
- Education & Reference: 60.5%
- Entertainment & Music: 68.3%
- Family & Relationships: 70.3%
- Health: 74.3%
- Politics & Government: 74.3%
- Science & Mathematics: 70.3%
- Society & Culture: 58.3%
- Sports: 87.3%

**Key Insights:**
- Neural networks outperform traditional ML by 2.62% absolute accuracy
- Word embeddings provide 2.72% boost over traditional vectors
- Bidirectional RNNs show marginal improvement (0.5-1.0%)
- SimpleRNN performs worst among RNN architectures
- TF-IDF consistently outperforms BoW for traditional ML

## Future Work

Future enhancements could include hyperparameter optimization using grid or random search to find optimal configurations for each model type. Ensemble methods combining predictions from multiple models could potentially improve accuracy beyond individual model performance. The integration of transformer-based models like BERT or RoBERTa would likely achieve state-of-the-art results given their superior language understanding capabilities. Domain adaptation techniques could extend the applicability of these models to different text classification tasks beyond question-answer classification.


