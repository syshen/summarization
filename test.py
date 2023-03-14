import requests
import openai
import torch
from readability import Document
from bs4 import BeautifulSoup
from transformers import AutoTokenizer

tokenizer = AutoTokenizer.from_pretrained("gpt2")

def count_tokens(text):
  input_ids = torch.tensor(tokenizer.encode(text)).unsqueeze(0)
  num_tokens = input_ids.shape[1]
  return num_tokens

def text_to_chunks(text, chunk_size=2000):
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
    if (current_chunk_size + num_tokens) > chunk_size:
      current_chunk_size = num_tokens
      chunks.append(chunk)
      chunk = tokens
    else:
      if chunk is None:
        chunk = tokens
      else:
        chunk.extend(tokens)
      current_chunk_size += num_tokens
  chunks.append(chunk)

  return chunks

def callOpenAI(prompt_request, max_tokens=500):
  try:
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
  except:
    print("error")

def summarize_text(full_text, debug=False):
  chunks = text_to_chunks(full_text)
  summaries = []

  for i, chunk in enumerate(chunks):
    text = tokenizer.decode(chunks[i])
    summary = callOpenAI(f"Summarize: {text}", max_tokens=2000)
    summaries.append(summary)

  if len(summaries) > 0:
    final = callOpenAI(f"Consolidate the summaries: {str(summaries)}", max_tokens=2000)
  return final

def translate_text(translating):
  chunks = text_to_chunks(translating, chunk_size=500)
  translated = []

  for i, chunk in enumerate(chunks):
    text = tokenizer.decode(chunks[i])
    print(f"Translate: {text}")
    result = callOpenAI(f"Translate to Traditional Chinese: {text}", max_tokens=3500)
    print("----")
    print(f"Result: {result}")
    translated.append(result)    

  return ''.join(translated)

def load_article(url):
  response = requests.get(url)
  doc = Document(response.text)
  soup = BeautifulSoup(doc.summary(), 'html.parser')
  text_string = soup.get_text()

  tokens_count = count_tokens(text_string)
  return doc.title(), text_string, tokens_count

openai.api_key = "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"

if openai.api_key:
  article_url = "https://medium.com/@pravse/the-maze-is-in-the-mouse-980c57cfd61a"
  if article_url:
    final_summary = None
    text_string = None

    title, text_string, tokens_count = load_article(article_url)
    print(f"Number of tokens: {tokens_count}")

    # final_summary = summarize_text(text_string)
    # print("Summary: ")
    # print(final_summary)
    translation = translate_text(text_string)
    # print("Translation: ")
    # print(translation)

    #translation = translate_text(text_string)
