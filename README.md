# 🤖 Fake News Detection with BERT

[![Method](https://img.shields.io/badge/Method-Text%20Mining-blue)](https://en.wikipedia.org/wiki/Text_mining)
[![Model](https://img.shields.io/badge/Model-BERT-orange)](https://github.com/google-research/bert)
[![Library](https://img.shields.io/badge/Library-TensorFlow%20%26%20Keras-brightgreen)](https://www.tensorflow.org/)

This project aims to build a text classification model capable of automatically detecting fake news (hoax) using Natural Language Processing (NLP) and deep learning approaches.

---

## 📄 Background

[cite_start]In the digital era, the massive spread of information brings the significant challenge of fake news, which can manipulate public opinion and disrupt social stability[cite: 14, 15]. Therefore, an intelligent system is needed to validate the authenticity of news. [cite_start]This project implements the **BERT** (Bidirectional Encoder Representations from Transformers) model to address this issue[cite: 18].

## 🛠️ Research Methodology

The research process was conducted through several systematic stages, as illustrated in the following workflow diagram:

![Experiment Stages](https://i.imgur.com/your-diagram-image.png)
*Image: Workflow diagram of the experiment stages*

1.  [cite_start]**Data Collection**: Utilized the "Fake News Detection" dataset from Kaggle, containing approximately 21,417 real news articles and 23,481 fake news articles[cite: 49].
2.  [cite_start]**Exploratory Data Analysis (EDA)**: Analyzed the initial distribution and characteristics of the data[cite: 27].
3.  **Data Preprocessing**:
    * [cite_start]Combined the `title` and `text` columns[cite: 58, 70].
    * [cite_start]Converted text to lowercase[cite: 64].
    * [cite_start]Performed tokenization using NLTK and `BertTokenizer`[cite: 65, 72].
    * [cite_start]Removed stopwords and non-alphanumeric characters[cite: 66].
4.  **BERT Modeling**:
    * [cite_start]Used the pre-trained `bert-base-uncased` model from Hugging Face[cite: 77].
    * [cite_start]Added `Dense`, `Dropout`, and `BatchNormalization` layers for classification[cite: 80].
    * [cite_start]The model was trained for 20 epochs with the Adam optimizer and `binary_crossentropy` loss function[cite: 84, 85].
5.  [cite_start]**Model Evaluation**: Measured the model's performance using various metrics, including Accuracy, Precision, Recall, F1-score, ROC AUC, and others[cite: 90].

---

## 📊 Results and Performance

The trained model demonstrated excellent performance in the classification task.

### Key Evaluation Metrics:
| Metric | Value |
| :--- | :--- |
| **Accuracy Score** | [cite_start]`0.9480` [cite: 124] |
| **ROC AUC Score** | [cite_start]`0.9897` [cite: 124] |
| **F1-Score (Macro Avg)**| [cite_start]`0.95` [cite: 122] |
| **Cohen's Kappa Score** | [cite_start]`0.8959` [cite: 124] |
| **Log Loss** | [cite_start]`0.1356` [cite: 124] |

### Confusion Matrix:
- [cite_start]**True Positive (Real News)**: 4111 [cite: 117]
- [cite_start]**True Negative (Fake News)**: 4402 [cite: 113]
- [cite_start]**False Positive**: 294 [cite: 114]
- [cite_start]**False Negative**: 173 [cite: 116]

The evaluation results indicate that the model is not only accurate but also balanced in predicting both classes (real and fake news).

---

## 💡 Conclusion

[cite_start]The **BERT** model proved to be highly effective for the task of fake news detection, achieving an accuracy of **94.8%**[cite: 127]. This high performance is supported by BERT's ability to understand text context deeply. [cite_start]For future development, the model's performance could be enhanced through hyperparameter fine-tuning and the exploration of other BERT variants like RoBERTa or DistilBERT[cite: 133].

---

## 👥 Team Contributors (Group 4 - LA05)

| NIM | Name |
| :--- | :--- |
| `2602231993` | [cite_start]Christopher Parulian Marpaung [cite: 7] |
| `2602128176` | [cite_start]Patrick Adrian Nelwan [cite: 8] |
| `2602109832` | [cite_start]Andika Rizky Putrahutama [cite: 9] |
| `2602146172` | [cite_start]M.Kenny Ryanta [cite: 10] |
