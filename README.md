# SMS Spam Detection using Machine Learning 📱🛡️

A comprehensive machine learning project that classifies SMS messages as spam or ham (legitimate) using various classification algorithms and natural language processing techniques.

## 📊 Project Overview

This project implements and compares multiple machine learning models to detect spam SMS messages with high accuracy. The system uses TF-IDF vectorization and advanced text preprocessing to achieve optimal results.

## 🎯 Key Results

| Model | Accuracy | Precision | Recall | F1-Score |
|-------|----------|-----------|--------|----------|
| **Random Forest** | **97.4%** | **97.5%** | **97.4%** | **97.4%** |
| SVM | 97.0% | 97.1% | 97.0% | 97.0% |
| Logistic Regression | 96.8% | 96.9% | 96.8% | 96.8% |
| Naive Bayes | 96.6% | 96.7% | 96.6% | 96.6% |

## 📁 Dataset

- **Source**: SMS Spam Collection Dataset
- **Total Messages**: 5,572
- **Classes**: 
  - Ham (Legitimate): 4,825 messages (86.6%)
  - Spam: 747 messages (13.4%)

## 🛠️ Technologies Used

- **Python 3.x**
- **Libraries**:
  - `pandas` - Data manipulation
  - `numpy` - Numerical operations
  - `scikit-learn` - Machine learning models
  - `nltk` - Natural language processing
  - `matplotlib` & `seaborn` - Data visualization

## 🔧 Features

### Text Preprocessing
- Lowercasing
- Tokenization
- Stopword removal
- Stemming (Porter Stemmer)
- Special character removal

### Feature Extraction
- TF-IDF Vectorization
- Maximum 5000 features
- Unigram and bigram support

### Model Implementation
- Logistic Regression
- Multinomial Naive Bayes
- Random Forest Classifier
- Support Vector Machine (SVM)

### Evaluation Metrics
- Accuracy Score
- Confusion Matrix
- Classification Report (Precision, Recall, F1-Score)
- Model Comparison Visualizations

## 📈 Visualizations

The project includes comprehensive visualizations:
- Class distribution analysis
- Confusion matrices for all models
- Model performance comparison charts
- Word frequency distributions
- Message length analysis

## 🚀 Getting Started

### Prerequisites
```bash
pip install pandas numpy scikit-learn nltk matplotlib seaborn
```

### Installation

1. Clone the repository
```bash
git clone https://github.com/AmiraAbdEl-Rahman/SMS_Spam_Detection.git
cd SMS_Spam_Detection
```

2. Install required packages
```bash
pip install -r requirements.txt
```

3. Download NLTK data
```python
import nltk
nltk.download('punkt')
nltk.download('stopwords')
```

### Usage
```python
# Run the main script
python spam_detection.py

# Or use Jupyter Notebook
jupyter notebook SMS_Spam_Detection.ipynb
```

## 📝 Project Structure
```
SMS_Spam_Detection/
│
├── data/
│   └── SMSSpamCollection.txt
│
├── notebooks/
│   └── SMS_Spam_Detection.ipynb
│
├── images/
│   └── (visualizations)
│
├── requirements.txt
├── README.md
└── spam_detection.py
```

## 🔍 Key Insights

1. **Random Forest** achieved the best performance with 97.4% accuracy
2. All models showed excellent performance (>96% accuracy)
3. Spam messages typically contain promotional words and special characters
4. Length analysis shows spam messages tend to be shorter on average
5. The system has very low false positive rates, protecting legitimate messages

## 📊 Sample Results
```
Random Forest Classifier Performance:
- Accuracy: 97.4%
- Precision: 97.5%
- Recall: 97.4%
- F1-Score: 97.4%

Confusion Matrix:
              Predicted
              Ham    Spam
Actual Ham    950    15
       Spam   14     136
```

## 🔮 Future Improvements

- [ ] Implement deep learning models (LSTM, BERT)
- [ ] Add real-time prediction API
- [ ] Create web interface for user interaction
- [ ] Expand dataset with multilingual messages
- [ ] Implement ensemble methods
- [ ] Add explainability features (LIME, SHAP)

## 🤝 Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

1. Fork the project
2. Create your feature branch (`git checkout -b feature/AmazingFeature`)
3. Commit your changes (`git commit -m 'Add some AmazingFeature'`)
4. Push to the branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request

---

**#MachineLearning #NLP #Python #DataScience #SpamDetection**