import streamlit as st
from transformers import GPT2LMHeadModel, GPT2Tokenizer
import torch

@st.cache_resource
def load_model():
    tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
    model = GPT2LMHeadModel.from_pretrained("gpt2")
    return tokenizer, model

tokenizer, model = load_model()

st.title("📝 Story Generator with GPT-2")
prompt = st.text_input("Enter a story prompt:", "Once upon a time...")

if st.button("Generate"):
    inputs = tokenizer.encode(prompt, return_tensors="pt")
    outputs = model.generate(inputs, max_length=150, do_sample=True, top_k=50)
    story = tokenizer.decode(outputs[0], skip_special_tokens=True)
    st.markdown("**Generated Story:**")
    st.write(story)
