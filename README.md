# Text Classification Experiment: Classic ML vs Neural Networks

## Overview

This project implements and compares various machine learning and deep learning approaches for text classification on a Question-Answer dataset. The experiment evaluates both traditional machine learning models and modern neural network architectures to determine the most effective methods for classifying text data. By systematically comparing 22 different model configurations across multiple text representations, the study provides insights into the trade-offs between computational complexity and classification performance in natural language processing tasks.

## Dataset

The dataset used in this experiment is the Question Answer Classification Dataset, consisting of CSV files containing question-answer pairs with corresponding classification labels. The training data is stored in "Question Answer Classification Dataset 1[Training].csv" while the test data is in "Question Answer Classification Dataset[Test].csv". Each sample contains text data from questions and answers that need to be classified into multiple categories. The dataset comes pre-split into training and test sets, with the training portion further divided into 90% for training and 10% for validation during model development. The text samples vary in length with an average of 15-20 words per sample, and after preprocessing, the vocabulary size is approximately 10,000 unique words.

**Download Links:**
- **Training Set:** [Question Answer Classification Dataset 1[Training].csv](https://drive.google.com/file/d/11qn9k-CqFXbptAC4hl1KKtfTu7ZAUs6Q/view?usp=drive_link)
- **Test Set:** [Question Answer Classification Dataset[Test].csv](https://drive.google.com/file/d/1fwcV7K0vq5OiuS33dw_0Y48lZZSuyZg_/view?usp=drive_link)

## Experiment Objectives

This comprehensive experiment was designed to address several key questions in text classification. First, it compares traditional machine learning approaches against modern deep learning methods to understand which paradigm performs better on this specific task. Second, it evaluates different text representation techniques including Bag of Words, TF-IDF, and word embeddings to determine which method captures text semantics most effectively. Third, it assesses various neural network architectures such as recurrent neural networks, long short-term memory networks, and gated recurrent units, including both unidirectional and bidirectional variants. Finally, the experiment aims to identify best practices and optimal model combinations for text classification tasks while analyzing the computational cost versus accuracy trade-offs.

## Models Implemented

The experiment evaluates 22 different model configurations divided into traditional machine learning and deep learning approaches. For traditional machine learning, three algorithms are tested: Random Forest Classifier with 150 estimators and maximum depth of 12, Logistic Regression with L2 regularization and class balancing, and Multinomial Naive Bayes with alpha smoothing of 0.1. Each algorithm is evaluated with two text representation methods: Bag of Words and TF-IDF vectorization.

The deep learning portion includes 16 neural network models. Two models use deep neural networks with pre-trained word embeddings - one with Word2Vec embeddings trained using skip-gram with 100 dimensions, and another with GloVe embeddings from the Wikipedia+Gigaword corpus. Twelve models implement recurrent neural network variants including SimpleRNN, GRU, and LSTM architectures, each tested with both Word2Vec and GloVe embeddings in both unidirectional and bidirectional configurations. The remaining two models use deep neural networks with traditional vector representations (Bag of Words and TF-IDF).

All neural networks are trained with a batch size of 128, using the Adam optimizer with early stopping after 5 epochs of no improvement. The embedding layers are frozen during training, and the models include dropout regularization and batch normalization where appropriate.

## Key Findings

The experiment revealed clear performance differences between traditional and deep learning approaches. Neural network models consistently achieved higher accuracy, typically in the 80-85% range compared to 70-80% for traditional machine learning models. Among the neural networks, models using word embeddings (Word2Vec and GloVe) significantly outperformed those using traditional vector representations like Bag of Words and TF-IDF. This suggests that pre-trained embeddings capture semantic relationships more effectively than frequency-based methods.

LSTM and GRU architectures provided the best balance between accuracy and computational efficiency, with bidirectional variants offering marginal improvements over unidirectional models. The traditional machine learning models showed TF-IDF outperforming Bag of Words, likely due to better term importance weighting. However, deep learning models were able to capture complex patterns that traditional approaches missed, particularly in handling vocabulary variations and contextual relationships.

Training time varied significantly between approaches, with traditional machine learning models completing in seconds while neural networks required several minutes to hours depending on the architecture. This highlights the important trade-off between computational resources and classification performance that practitioners must consider when selecting models for production deployment.

## Technical Implementation

The implementation follows a comprehensive preprocessing pipeline that ensures consistent text preparation across all models. Raw text undergoes lowercasing, punctuation removal, and alphanumeric filtering. Stop words are removed using both NLTK's standard list and custom domain-specific terms like "question", "title", "content", "best", "answer", and "body". Words are then lemmatized using NLTK's WordNetLemmatizer to reduce inflectional variations.

For embedding-based models, text is tokenized and converted to sequences with a maximum length of 50 tokens, using special padding and unknown tokens. Traditional vectorization methods use scikit-learn's CountVectorizer and TfidfVectorizer with optimized parameters including n-gram ranges up to trigrams, minimum document frequency thresholds, and maximum feature limits.

Neural network architectures incorporate multiple regularization techniques including dropout layers, batch normalization, and early stopping to prevent overfitting. The recurrent neural networks use a combination of final hidden states and global average pooling for more robust representations. All models are evaluated using comprehensive metrics including accuracy, macro and weighted F1-scores, confusion matrices, and training history analysis.

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

## Results Summary

The experiment clearly demonstrates the superiority of neural network approaches over traditional machine learning for text classification tasks. Neural networks achieved higher accuracy rates, typically in the 80-85% range, compared to the 70-80% range of traditional methods. Word embedding-based representations proved superior to traditional vectorization methods, with Word2Vec and GloVe embeddings capturing semantic relationships more effectively than Bag of Words or TF-IDF approaches.

Among neural architectures, LSTM and GRU models provided the best performance-to-computational-cost ratio, making them suitable for most practical applications. The bidirectional variants offered slight improvements but at increased computational cost. Traditional machine learning models, while faster to train, were unable to capture the complex patterns that deep learning models could learn from the text data.

Model selection should be based on available computational resources and performance requirements. For applications requiring high accuracy with sufficient training time and resources, neural networks with word embeddings are recommended. For scenarios with limited computational resources or real-time requirements, traditional machine learning models with TF-IDF representations provide a reasonable alternative.

### Sample Results (from code execution):
- Best model: Typically LSTM or BiLSTM with Word2Vec embeddings
- Accuracy range: 75-85% depending on model and representation
- Training time: 2-15 minutes per neural network model

## Future Work

Future enhancements could include hyperparameter optimization using grid or random search to find optimal configurations for each model type. Ensemble methods combining predictions from multiple models could potentially improve accuracy beyond individual model performance. The integration of transformer-based models like BERT or RoBERTa would likely achieve state-of-the-art results given their superior language understanding capabilities. Domain adaptation techniques could extend the applicability of these models to different text classification tasks beyond question-answer classification.


