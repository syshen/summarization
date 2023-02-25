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

def text_to_chunks(text, chunk_size=2000, overlap=100):
  punctuation = '.!?'

  sentences = []
  start = 0
  for i, char in enumerate(text):
    if char in punctuation:
      sentences.append(text[start:i+1])
      start = i+1
  if i < len(text):
    sentences.append(text[start:])

  chunks = []
  chunk = None
  current_chunk_size = 0
  for i, sentence in enumerate(sentences):
    tokens = tokenizer.encode(sentence)
    num_tokens = len(tokens)
    if (current_chunk_size + num_tokens) >= chunk_size:
      current_chunk_size = num_tokens
      chunks.append(chunk)
      chunk = tokens
    else:
      if chunk is None:
        chunk = tokens
      else:
        chunk.extend(tokens)
      current_chunk_size += num_tokens

  return chunks

def callOpenAI(prompt_request, max_tokens=500):
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

@st.cache_data
def summarize_text(full_text):
  chunks = text_to_chunks(full_text)
  summaries = []

  for i, chunk in enumerate(chunks):
    text = tokenizer.decode(chunks[i])
    summary = callOpenAI(f"Summarize: {text}", max_tokens=2000)
    summaries.append(summary)

  if len(summaries) > 0:
    intermedia_summary = " ".join(summaries)
    final_summary = callOpenAI(f"Consolidate the summaries: {intermedia_summary}", max_tokens=2000)
  return final_summary

@st.cache_data
def translate_text(full_text):
  chunks = text_to_chunks(full_text, overlap=0)
  translated = []

  for i, chunk in enumerate(chunks):
    text = tokenizer.decode(chunks[i])
    summary = callOpenAI(f"Translate to Traditional Chinese: {text}", max_tokens=2000)
    translated.append(summary)    

  return ''.join(translated)

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
          final_summary = summarize_text(text_string)
          final_summary

    bt1_clicked = st.button("Translate Origianl Text to Mandarin")
    bt2_clicked = st.button("Translate Summary to Mandarin")

    if bt1_clicked:
      with st.spinner("Translating..."):
        translation = translate_text(text_string)
      with st.expander("Original Text Translation"): 
        translation

    if bt2_clicked:
      with st.spinner("Translating..."):
        translation = translate_text(final_summary)

      with st.expander("Summary Translation"): 
        translation