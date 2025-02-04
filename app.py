import streamlit as st
import os
os.system("pip install datasets")
import torch
from transformers import (
    BertTokenizer,
    BertForQuestionAnswering,
    BartForConditionalGeneration,
    BartTokenizer,
    pipeline,
)
import pandas as pd
from datasets import load_dataset

def load_squad():
    """Load the SQuAD dataset from a local CSV file."""
    return pd.read_csv("main/squad_sample.csv")

@st.cache_resource
def load_cnn_dailymail():
    """Load the CNN/DailyMail dataset from a local CSV file."""
    return pd.read_csv("main/cnn_dailymail_sample.csv")

def load_summarization_model():
    """Load BART model and tokenizer for summarization"""
    model_name = "facebook/bart-large-cnn"
    tokenizer = BartTokenizer.from_pretrained(model_name)
    model = BartForConditionalGeneration.from_pretrained(model_name).to("cuda" if torch.cuda.is_available() else "cpu")
    return tokenizer, model


def question_answering(squad_df):
    st.subheader("Question Answering")
    model_name = "bert-large-uncased-whole-word-masking-finetuned-squad"
    tokenizer = BertTokenizer.from_pretrained(model_name)
    model = BertForQuestionAnswering.from_pretrained(model_name).to("cuda" if torch.cuda.is_available() else "cpu")
    
    unique_contexts = squad_df["context"].unique()
    context = st.selectbox("Select a context:", unique_contexts[:10])
    question = st.text_input("Enter your question:")
    
    if st.button("Get Answer"):
        inputs = tokenizer(question, context, return_tensors="pt", truncation=True).to(model.device)
        outputs = model(**inputs)
        start_index = torch.argmax(outputs.start_logits)
        end_index = torch.argmax(outputs.end_logits) + 1
        answer = tokenizer.convert_tokens_to_string(
            tokenizer.convert_ids_to_tokens(inputs["input_ids"][0][start_index:end_index])
        )
        st.write(f"**Answer:** {answer}")


def text_summarization(cnn_df):
    st.subheader("Text Summarization")
    tokenizer, model = load_summarization_model()
    
    option = st.radio("Choose input type:", ("Select an article", "Manually enter text"))
    
    if option == "Select an article":
        article = st.selectbox("Select an article:", cnn_df["article"].head(10))
    else:
        article = st.text_area("Enter the paragraph to summarize:")

    if article and st.button("Summarize"):
        inputs = tokenizer(article, return_tensors="pt", max_length=2024, truncation=True).to(model.device)
        summary_ids = model.generate(
            inputs["input_ids"], max_length=100, min_length=25, length_penalty=2.0, num_beams=4, early_stopping=True
        )
        summary = tokenizer.decode(summary_ids[0], skip_special_tokens=True)
        st.write(f"**Summary:** {summary}")

def document_comprehension(squad_df):
    st.subheader("Document Comprehension")
    model_name = "bert-large-uncased-whole-word-masking-finetuned-squad"
    tokenizer = BertTokenizer.from_pretrained(model_name)
    model = BertForQuestionAnswering.from_pretrained(model_name).to("cuda" if torch.cuda.is_available() else "cpu")
    nlp = pipeline("question-answering", model=model, tokenizer=tokenizer)
    
    unique_contexts = squad_df["context"].unique()
    context = st.selectbox("Select a context:", unique_contexts[:10])
    question = st.text_input("Enter your question:")
    
    if st.button("Get Answer"):
        response = nlp(question=question, context=context)
        st.write(f"**Answer:** {response['answer']}")

def main():
    st.title("Transformer-based NLP Toolkit")
    squad_df = load_squad()
    cnn_df = load_cnn_dailymail()
    
    task = st.sidebar.selectbox("Select a Task", ["Question Answering", "Text Summarization", "Document Comprehension"])
    
    if task == "Question Answering":
        question_answering(squad_df)
    elif task == "Text Summarization":
        text_summarization(cnn_df)
    elif task == "Document Comprehension":
        document_comprehension(squad_df)

if __name__ == "__main__":
    main()
