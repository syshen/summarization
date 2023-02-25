import requests
import openai
import torch
import streamlit as st
from readability import Document
from bs4 import BeautifulSoup
from transformers import AutoTokenizer

tokenizer = AutoTokenizer.from_pretrained("gpt2")

def count_tokens(text):
    input_ids = torch.tensor(tokenizer.encode(text)).unsqueeze(0)
    num_tokens = input_ids.shape[1]
    return num_tokens

def break_up_file_to_chunks(text, chunk_size=2000, overlap=100):
    tokenizer = AutoTokenizer.from_pretrained("gpt2")

    tokens = tokenizer.encode(text)
    num_tokens = len(tokens)
    
    chunks = []
    for i in range(0, num_tokens, chunk_size - overlap):
        chunk = tokens[i:i + chunk_size]
        chunks.append(chunk)
    
    return chunks

def summarize(prompt_request, max_tokens=500):
  response = openai.Completion.create(
            model="text-davinci-003",
            prompt=prompt_request,
            temperature=.5,
            max_tokens=max_tokens,
            top_p=1,
            frequency_penalty=0,
            presence_penalty=0
  )
  return response.choices[0].text

st.write("""
This website is designed to summarize articles from a given URL using OpenAI. To begin with it, the first step is to obtain an OpenAI API key.
""")
openai.api_key = st.text_input("OpenAI API Key")

if openai.api_key:
  article_url = st.text_input("Provide the URL for the article that requires a summary:")
  if article_url:
    with st.spinner("Loading article ..."):
      response = requests.get(article_url)
      doc = Document(response.text)
      st.title(doc.title())

    with st.spinner("Analyzing article..."):
      soup = BeautifulSoup(doc.summary(), 'html.parser')
      text_string = soup.get_text()

      tokens_count = count_tokens(text_string)
      st.write(f"Number of tokens: {tokens_count}")

    with st.expander("Original Text"):
      st.write(text_string)

    with st.spinner("Summarizing..."):
      with st.expander("Summary"):
        chunks = break_up_file_to_chunks(text_string)
        summaries = []

        for i, chunk in enumerate(chunks):
          text = tokenizer.decode(chunks[i])
          # st.write(text)
          summary = summarize(f"Summarize: {text}")
          summaries.append(summary)    

        if len(summaries) > 0:
          final_summary = summarize(f"Consolidate the summaries with paragraphs: {str(summaries)}", max_tokens=2000)
          st.subheader("Summary")
          final_summary

    clicked = st.button("Translate to Mandarin")
    if clicked:
      with st.spinner("Translating..."):
        translation = summarize(f"Translate to Traditional Chinese: {final_summary}", max_tokens=2000)
        st.write(translation)