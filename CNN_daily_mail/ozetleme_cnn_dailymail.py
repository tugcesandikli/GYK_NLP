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
    Seq2SeqTrainer, Seq2SeqTrainingArguments
)
import evaluate

# 1. Veri Seti Yükleme
print("Veri seti yükleniyor...")
dataset = load_dataset("cnn_dailymail", "3.0.0")
print(dataset)
print("Örnek veri:", dataset['train'][0])

# 2. Veri Ön İşleme Fonksiyonları
def clean_text(text):
    text = text.lower()
    text = re.sub(r'\s+', ' ', text)
    text = re.sub(r'[^a-zA-Z0-9.,!?\'\" ]', '', text)
    text = text.strip()
    return text

max_input_length = 512
max_target_length = 64

# 3. Model ve Tokenizer Seçimi
model_checkpoint = 'facebook/bart-base'  # Alternatif: 't5-small'
tokenizer = AutoTokenizer.from_pretrained(model_checkpoint)

# 4. Preprocessing
print("Veri ön işleniyor...")
def preprocess_function(examples):
    inputs = [clean_text(doc) for doc in examples['article']]
    targets = [clean_text(summary) for summary in examples['highlights']]
    model_inputs = tokenizer(inputs, max_length=max_input_length, truncation=True, padding="max_length")
    labels = tokenizer(targets, max_length=max_target_length, truncation=True, padding="max_length")['input_ids']
    model_inputs['labels'] = labels
    return model_inputs

# Hızlı prototip için küçük bir alt küme
small_train_dataset = dataset['train'].shuffle(seed=42).select(range(2000))
small_val_dataset = dataset['validation'].shuffle(seed=42).select(range(500))
small_test_dataset = dataset['test'].shuffle(seed=42).select(range(500))

tokenized_train = small_train_dataset.map(preprocess_function, batched=True)
tokenized_val = small_val_dataset.map(preprocess_function, batched=True)
tokenized_test = small_test_dataset.map(preprocess_function, batched=True)

# 5. Model Kurulumu ve Eğitim Ayarları
print("Model ve eğitim ayarları hazırlanıyor...")
model = AutoModelForSeq2SeqLM.from_pretrained(model_checkpoint)
data_collator = DataCollatorForSeq2Seq(tokenizer, model=model)

training_args = Seq2SeqTrainingArguments(
    output_dir="./results",
    eval_strategy="epoch",
    learning_rate=2e-5,
    per_device_train_batch_size=4,
    per_device_eval_batch_size=4,
    weight_decay=0.01,
    save_total_limit=2,
    num_train_epochs=3,
    predict_with_generate=True,
    fp16=torch.cuda.is_available(),
    logging_dir="./logs",
    logging_steps=50,
    report_to="none"
)

# 6. Eğitim
print("Eğitim başlıyor...")
rouge = evaluate.load('rouge')
def compute_metrics(eval_pred):
    predictions, labels = eval_pred
    decoded_preds = tokenizer.batch_decode(predictions, skip_special_tokens=True)
    labels = np.where(labels != -100, labels, tokenizer.pad_token_id)
    decoded_labels = tokenizer.batch_decode(labels, skip_special_tokens=True)
    result = rouge.compute(predictions=decoded_preds, references=decoded_labels, use_stemmer=True)
    result = {k: round(v * 100, 2) for k, v in result.items()}
    return result

trainer = Seq2SeqTrainer(
    model,
    training_args,
    train_dataset=tokenized_train,
    eval_dataset=tokenized_val,
    data_collator=data_collator,
    tokenizer=tokenizer,
    compute_metrics=compute_metrics
)

train_result = trainer.train()
metrics = train_result.metrics
trainer.save_model()

# 7. Eğitim ve Doğrulama Kayıplarını Çiz
train_loss = trainer.state.log_history
train_losses = [x['loss'] for x in train_loss if 'loss' in x]
eval_losses = [x['eval_loss'] for x in train_loss if 'eval_loss' in x]
plt.plot(train_losses, label='Train Loss')
plt.plot(eval_losses, label='Eval Loss')
plt.xlabel('Step')
plt.ylabel('Loss')
plt.legend()
plt.title('Eğitim ve Doğrulama Kayıpları')
plt.savefig('egitim_kayiplari.png')
plt.close()
print("Eğitim ve doğrulama kayıpları 'egitim_kayiplari.png' olarak kaydedildi.")

# 8. Değerlendirme: ROUGE Skorları ve Örnek Karşılaştırmalar
print("Test setinde değerlendirme yapılıyor...")
test_results = trainer.evaluate(tokenized_test, max_length=max_target_length, num_beams=4)
print('Test ROUGE-L:', test_results['eval_rougeL'])

print("\n5 örnek karşılaştırmalı çıktı:")
for i in range(5):
    sample = small_test_dataset[i]
    input_ids = tokenizer.encode(clean_text(sample['article']), return_tensors='pt', max_length=max_input_length, truncation=True)
    summary_ids = model.generate(input_ids, max_length=max_target_length, num_beams=4, early_stopping=True)
    generated_summary = tokenizer.decode(summary_ids[0], skip_special_tokens=True)
    print(f'--- Haber {i+1} ---')
    print('Orijinal Özet:', sample['highlights'])
    print('Model Özeti   :', generated_summary)
    print()

# 9. Kısa Geliştirme Süreci Raporu (console'a yazılır, ayrıca gelistirme_raporu.md dosyasında da var)
print("\n--- Geliştirme Süreci Raporu ---")
print("Model: facebook/bart-base (alternatif: t5-small)")
print("Epoch: 3, Learning rate: 2e-5, Max input: 512, Max summary: 64, Batch size: 4")
print("Bellek kısıtı nedeniyle küçük bir alt küme ile prototip alındı. Daha iyi sonuç için batch size ve epoch artırılabilir, model büyütülebilir.")
print("Kod, GPU varsa fp16 ile çalışır. Tüm hiperparametreler kodda açıkça belirtilmiştir.") 

# 10. Pipeline ile ve Doğrudan Model ile Özetleme Örnekleri
from transformers import pipeline, AutoModelForSeq2SeqLM, AutoTokenizer

print("\n--- Pipeline ile Özetleme Örneği ---")
summarizer = pipeline("summarization", model="facebook/bart-base")
example_text = """
The quick brown fox jumps over the lazy dog. This is a famous pangram used for testing typing and fonts. It contains every letter of the English alphabet at least once.
"""
summary = summarizer(example_text, max_length=32, min_length=5, do_sample=False)
print("Pipeline özeti:", summary[0]['summary_text'])

print("\n--- Doğrudan Model.generate ile Özetleme Örneği ---")
tokenizer = AutoTokenizer.from_pretrained("facebook/bart-base")
model = AutoModelForSeq2SeqLM.from_pretrained("facebook/bart-base")
inputs = tokenizer([example_text], max_length=128, truncation=True, return_tensors="pt")
summary_ids = model.generate(**inputs, max_length=32, min_length=5, num_beams=4)
summary_text = tokenizer.decode(summary_ids[0], skip_special_tokens=True)
print("Model.generate özeti:", summary_text) 

import random

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
print(torch.cuda.get_device_name(0)) 