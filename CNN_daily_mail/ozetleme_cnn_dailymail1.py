# CNN/DailyMail Abstractive Summarization Pipeline
# Tam açıklamalı, uçtan uca Python scripti
# Gereksinimler: datasets, transformers, evaluate, rouge-score, matplotlib

import os
import random
import re
import numpy as np
import torch
import matplotlib.pyplot as plt
from datasets import load_dataset
from transformers import (
    AutoTokenizer, AutoModelForSeq2SeqLM, DataCollatorForSeq2Seq,
    Seq2SeqTrainer, Seq2SeqTrainingArguments, pipeline
)
import evaluate

# Model ve tokenizer'ı yükle
model_checkpoint = 'facebook/bart-base'
tokenizer = AutoTokenizer.from_pretrained(model_checkpoint)
model = AutoModelForSeq2SeqLM.from_pretrained(model_checkpoint)
summarizer = pipeline("summarization", model=model, tokenizer=tokenizer)

# CNN/DailyMail veri setini yükle (küçük bir test alt kümesi)
dataset = load_dataset("cnn_dailymail", "3.0.0")
small_test_dataset = dataset['test'].shuffle(seed=42).select(range(500))

# Rastgele bir haber seç
idx = random.randint(0, len(small_test_dataset)-1)
article = small_test_dataset[idx]['article']
original_summary = small_test_dataset[idx]['highlights']

print(f"\n--- Rastgele Seçilen Haber (index: {idx}) ---")
print("Makale:", article[:500], "...\n")  # İlk 500 karakteri göster
print("Orijinal Özet:", original_summary)

# Pipeline ile özetleme
pipeline_summary = summarizer(article, max_length=64, min_length=10, do_sample=False)[0]['summary_text']
print("Pipeline Özeti:", pipeline_summary)

# Doğrudan model.generate ile özetleme
inputs = tokenizer([article], max_length=512, truncation=True, return_tensors="pt")
summary_ids = model.generate(**inputs, max_length=64, min_length=10, num_beams=4)
generate_summary = tokenizer.decode(summary_ids[0], skip_special_tokens=True)
print("Model.generate Özeti:", generate_summary)

import torch
print(torch.cuda.is_available())

output_lines = []
output_lines.append(f"--- Rastgele Seçilen Haber (index: {idx}) ---\n")
output_lines.append("Makale: " + article[:500] + "...\n")
output_lines.append("Orijinal Özet: " + original_summary + "\n")
output_lines.append("Pipeline Özeti: " + pipeline_summary + "\n")
output_lines.append("Model.generate Özeti: " + generate_summary + "\n")
output_lines.append("-" * 80 + "\n")

with open("ornek_ozetler.txt", "a", encoding="utf-8") as f:
    f.writelines([line + "\n" for line in output_lines])

print("\nÇıktılar 'ornek_ozetler.txt' dosyasına EKLENDİ.") 