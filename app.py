import streamlit as st
import torch
from transformers import BertTokenizer, BertForSequenceClassification
from pathlib import Path

# ---- Page config ----
st.set_page_config(
    page_title="News Topic Classifier",
    page_icon="📰",
    layout="centered"
)

# ---- Model path ----
MODEL_PATH = Path(__file__).parent / "bert_ag_news"

# ---- Model load karo ----
@st.cache_resource
def load_model():
    tokenizer = BertTokenizer.from_pretrained(str(MODEL_PATH))
    model     = BertForSequenceClassification.from_pretrained(str(MODEL_PATH))
    model.eval()
    return tokenizer, model

tokenizer, model = load_model()

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)

label_names = ["🌍 World", "⚽ Sports", "💼 Business", "🔬 Sci/Tech"]

# ---- Predict function ----
def predict(headline):
    inputs = tokenizer(
        headline,
        return_tensors="pt",
        truncation=True,
        padding=True,
        max_length=128
    )
    inputs = {k: v.to(device) for k, v in inputs.items()}

    with torch.no_grad():
        outputs = model(**inputs)

    probs     = torch.softmax(outputs.logits, dim=1).squeeze()
    label_idx = probs.argmax().item()
    return label_idx, probs

# ---- UI ----
st.title("📰 News Topic Classifier")
st.markdown("**BERT** model fine-tuned on AG News Dataset")
st.divider()

# Input box
headline = st.text_input(
    "Enter a news headline:",
    placeholder="e.g. NASA launches new rocket to explore Mars..."
)

# Example buttons
st.markdown("**Try an example:**")
col1, col2 = st.columns(2)
col3, col4 = st.columns(2)

if col1.button("🌍 World News"):
    headline = "UN calls emergency meeting over border conflict"
if col2.button("⚽ Sports News"):
    headline = "Real Madrid defeats Barcelona 3-1 in El Clasico"
if col3.button("💼 Business News"):
    headline = "Federal Reserve raises interest rates again"
if col4.button("🔬 Sci/Tech News"):
    headline = "NASA successfully launches new Mars rover mission"

# Predict karo
if headline:
    with st.spinner("Classifying..."):
        label_idx, probs = predict(headline)

    st.divider()

    # Result
    st.success(f"**Predicted Category: {label_names[label_idx]}**")
    st.markdown(f"**Confidence: {probs[label_idx].item()*100:.2f}%**")

    st.divider()

    # Progress bars
    st.markdown("**All Category Scores:**")
    for i, name in enumerate(label_names):
        score = probs[i].item()
        st.markdown(f"{name}")
        st.progress(score, text=f"{score*100:.2f}%")