# 📧 Email Spam Classification – NLP & Machine Learning

This project is a **binary classification system** that uses **Natural Language Processing (NLP)** and **Machine Learning (ML)** to determine whether an email or message is **spam** or **ham** (not spam). It uses real-world SMS data and explores various ML models, including Naive Bayes, Logistic Regression, and Support Vector Machine (SVM).

---

## 🧠 Project Overview

Spam detection is a vital application of text classification. In this project, we:
- Clean and preprocess text data (tokenization, stopword removal, stemming).
- Convert the text into numerical vectors using **TF-IDF**.
- Train multiple classification algorithms.
- Evaluate performance using **precision**, **recall**, **F1-score**, and **confusion matrix**.

This is an excellent beginner-friendly NLP project for learning how to work with unstructured text data.

---

## 🗂️ Dataset

This project uses the **Spam Data** dataset by [Satyam Patel on Kaggle](https://www.kaggle.com/datasets/satyampatell/spamdata), which contains labeled SMS/email messages.

📁 **Source**: [Kaggle - Spam Data](https://www.kaggle.com/datasets/satyampatell/spamdata)

📌 **Columns:**
- `Category`: Indicates whether the message is `ham` (not spam) or `spam`
- `Message`: The content of the email/SMS

> In the project, we map `ham → 0` and `spam → 1` for model training.

> ⚠️ Please make sure to download the dataset manually from Kaggle and rename it as `spam.csv` (or update the filename in the script accordingly).

---

## ⚙️ Features

- Cleaned and preprocessed SMS/email text
- NLP pipeline using NLTK
- Text vectorization with **TF-IDF**
- Binary classification using:
  - 📌 Multinomial Naive Bayes
  - 📌 Logistic Regression
  - 📌 Support Vector Machine (SVM)
- Model performance evaluated using:
  - Accuracy
  - Precision
  - Recall
  - F1-score
  - Confusion Matrix

---

## 🔧 Technologies & Libraries

| Category         | Tools & Libraries               |
|------------------|----------------------------------|
| Language         | Python 3.x                       |
| Data Handling    | `pandas`, `numpy`                |
| NLP              | `nltk`, `re`, `string`           |
| ML Models        | `scikit-learn`                   |
| Vectorization    | `TfidfVectorizer`                |
| Evaluation       | `classification_report`, `confusion_matrix` |

---

## 🛠️ Installation

1. **Clone the Repository**
   ```bash
   git clone https://github.com/yourusername/email-spam-classification.git
   cd email-spam-classification
   ```

2. **Install Dependencies**
   ```bash
   pip install -r requirements.txt
   ```

3. **Download NLTK Resources**
   ```python
   import nltk
   nltk.download('stopwords')
   ```

---

## 📁 Project Structure

```
email-spam-classification/
│
├── spam.csv                   # Dataset file (downloaded from Kaggle)
├── spam_classifier.py         # Main script with preprocessing, training, and evaluation
├── requirements.txt           # Python libraries required
├── README.md                  # Project documentation
└── .gitignore                 # Ignored files/folders
```

---

## 📜 How It Works

### 🔹 Step 1: Data Cleaning
- Convert to lowercase
- Remove numbers, punctuation, and extra whitespace
- Tokenize text
- Remove stopwords
- Apply stemming

### 🔹 Step 2: Vectorization
- Apply `TfidfVectorizer` to convert cleaned text into numerical format

### 🔹 Step 3: Model Training
- Split data using `train_test_split`
- Train models using Naive Bayes, Logistic Regression, and SVM

### 🔹 Step 4: Model Evaluation
- Predict on test data
- Print classification report and confusion matrix

---

## 🧪 Sample Output (Multinomial Naive Bayes)

```
--- Multinomial Naive Bayes ---
Confusion Matrix:
[[949   5]
 [ 21 140]]

Classification Report:
              precision    recall  f1-score   support

           0       0.98      0.99      0.98       954
           1       0.97      0.87      0.92       161

    accuracy                           0.97      1115
   macro avg       0.97      0.93      0.95      1115
weighted avg       0.97      0.97      0.97      1115
```

---

## ✅ Models Used

| Model                   | Suitable For Text? | Performance Summary |
|------------------------|-------------------|----------------------|
| Multinomial Naive Bayes| ✅ Yes            | Fast and accurate    |
| Logistic Regression     | ✅ Yes            | Good baseline        |
| Support Vector Machine  | ✅ Yes            | High precision       |

---

## 📈 Future Enhancements

- Use deep learning models like LSTM, BERT
- Build a web app using Flask or Django
- Host model as an API with FastAPI
- Integrate with an email client for real-time classification
- Visualize word frequencies and message length distributions

---

## 💼 Use Cases

- Email filtering systems (Gmail, Outlook)
- SMS spam detectors
- Content moderation tools
- Chatbot pre-filtering
- Customer service automation

---

## 📚 References

- [Kaggle Dataset](https://www.kaggle.com/datasets/satyampatell/spamdata)
- [Scikit-learn Documentation](https://scikit-learn.org/)
- [NLTK Documentation](https://www.nltk.org/)

---

## 🧑‍💻 Author

**Ali Hassan Atif**  
📧 thealihassanatif@gmail.com  
🔗 [LinkedIn](https://www.linkedin.com/in/alihassanatif)  
🌍 Based in Sahiwal, Punjab, Pakistan  

---

## 📝 License

This project is licensed under the [MIT License](LICENSE).

---

## 🌟 Star this repo

If you found this project useful or interesting, please consider giving it a ⭐ to support the work!
