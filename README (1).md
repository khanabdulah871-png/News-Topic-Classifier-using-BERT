# 📰 News Topic Classifier Using BERT

## Objective
Fine-tune **TinyBERT** (`huawei-noah/TinyBERT_General_4L_312D`) transformer model to automatically classify news headlines into 4 topic categories: **World**, **Sports**, **Business**, and **Sci/Tech** using the AG News Dataset.

---

## Dataset
- **Name:** HuyAugie/Smaller_AG_News_Dataset
- **Source:** Hugging Face Datasets
- **Classes:** World | Sports | Business | Sci/Tech
- **Splits:** Train / Validation (10% of train) / Test

---

## Methodology / Approach

### 1. Environment Setup
Installed required libraries: `transformers`, `datasets`, `torch`, `scikit-learn`, `gradio`, `matplotlib`, `seaborn`

### 2. Data Preprocessing
- Loaded dataset using Hugging Face `datasets` library
- Checked for null values and class distribution
- Tokenized text using `BertTokenizer` with:
  - `max_length = 128`
  - `padding = max_length`
  - `truncation = True`
- Created Train / Validation / Test splits using `DataLoader`

### 3. Model Development
- Loaded `huawei-noah/TinyBERT_General_4L_312D` pretrained model
- Added classification head with `BertForSequenceClassification` (4 output classes)
- Used `AdamW` optimizer with `lr = 2e-5` and `weight_decay = 0.01`
- Applied linear warmup scheduler (10% warmup steps)
- Gradient clipping (`max_norm = 1.0`) to prevent exploding gradients
- Trained for **3 epochs**

### 4. Training Loop
- Each epoch: forward pass → loss backward → gradient clip → optimizer step
- Validation check after every epoch to monitor overfitting
- Tracked: Train Loss, Validation Loss, Validation Accuracy

### 5. Evaluation
- Evaluated on test set using:
  - **Accuracy**
  - **Weighted F1-Score**
  - **Classification Report** (per-class precision, recall, F1)
  - **Confusion Matrix** (visualized using seaborn heatmap)

### 6. Deployment
- Model and tokenizer saved using `save_pretrained()`
- Deployed using **Streamlit** for live headline classification

---

## Key Results

| Metric | Score |
|---|---|
| Accuracy | ~92% |
| Weighted F1-Score | ~92% |

### Per-Class Performance
| Class | Precision | Recall | F1-Score |
|---|---|---|---|
| World | High | High | High |
| Sports | Very High | Very High | Very High |
| Business | High | High | High |
| Sci/Tech | High | High | High |

---

## Project Structure

```
News Topic Classifier Using BERT/
│
├── bert_ag_news/              ← Fine-tuned model files
│   ├── config.json
│   ├── model.safetensors
│   ├── tokenizer.json
│   └── tokenizer_config.json
│
├── app.py                     ← Streamlit deployment app
├── requirements.txt           ← Project dependencies
└── README.md                  ← This file
```

---

## How to Run Locally

### 1. Clone the repository
```bash
git clone https://github.com/your-username/news-topic-classifier-using-bert
cd news-topic-classifier-using-bert
```

### 2. Create virtual environment
```bash
python -m venv venv
venv\Scripts\activate       # Windows
source venv/bin/activate    # Mac/Linux
```

### 3. Install dependencies
```bash
pip install -r requirements.txt
```

### 4. Run the app
```bash
streamlit run app.py
```

---

## Dependencies

```txt
streamlit
transformers
torch
scikit-learn
numpy
seaborn
matplotlib
datasets
```

---

## Skills Gained
- NLP using Transformer models (BERT)
- Transfer learning and fine-tuning
- Evaluation metrics for text classification
- Lightweight model deployment with Streamlit

---

## Notes
> ⚠️ **Note:** The Streamlit app could not be deployed on a local device due to insufficient storage space. The fine-tuned BERT model (`model.safetensors`) is approximately 400MB in size, which exceeded the available local storage. The model training and evaluation were successfully completed on Kaggle (GPU: NVIDIA Tesla T4).
