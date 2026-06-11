# 🤗 Hugging Face AI Workstation

A comprehensive Jupyter Notebook showcasing core Natural Language Processing (NLP) workflows using the Hugging Face Transformers ecosystem. This project demonstrates tokenization, embeddings, text generation, sentiment analysis, summarization, hardware profiling, and performance benchmarking in a single notebook.

---

## 📌 Overview

This notebook acts as an AI workstation for experimenting with pre-trained transformer models while monitoring system resources and execution performance. It automatically detects available hardware (CPU/GPU) and runs multiple NLP tasks using industry-standard Hugging Face models.

---

## ✨ Features

### 🔹 Hardware Profiling

* Detects CUDA-enabled GPUs automatically.
* Switches execution between CPU and GPU dynamically.
* Displays device information before execution.

### 🔹 Tokenization Analysis

* Uses **DistilGPT-2** tokenizer.
* Displays generated tokens and token counts.
* Helps understand text preprocessing mechanics.

### 🔹 Text Embeddings

* Generates dense vector representations using **BERT Base Uncased**.
* Extracts hidden-state embeddings for input text.
* Useful for semantic similarity and downstream NLP tasks.

### 🔹 Text Generation

* Uses **DistilGPT-2** for autoregressive text generation.
* Produces context-aware continuations from prompts.

### 🔹 Sentiment Analysis

* Performs sentiment classification on user text.
* Returns labels and confidence scores.

### 🔹 Text Summarization

* Uses **BART Large CNN** for abstractive summarization.
* Converts long passages into concise summaries.

### 🔹 Performance Benchmarking

* Measures notebook execution time.
* Calculates generation throughput (tokens/sec).

### 🔹 Resource Monitoring

* Reports RAM utilization using `psutil`.
* Displays GPU memory usage through `torch.cuda`.

---

## 🛠 Technologies Used

* Python
* Jupyter Notebook
* Hugging Face Transformers
* PyTorch
* Accelerate
* SentencePiece
* Psutil

---

## 📦 Installation

Install the required dependencies:

```bash
pip install transformers==4.46.3
pip install torch accelerate sentencepiece psutil
```

---

## 🚀 Running the Notebook

1. Clone this repository:

```bash
git clone <repository-url>
```

2. Navigate to the project directory:

```bash
cd <repository-folder>
```

3. Launch Jupyter Notebook:

```bash
jupyter notebook
```

4. Open the notebook file and execute all cells sequentially.

---

## 💻 Example Code

```python
import torch
from transformers import pipeline

device = 0 if torch.cuda.is_available() else -1

print(f"Running on {'GPU' if device == 0 else 'CPU'}")

generator = pipeline(
    "text-generation",
    model="distilgpt2",
    device=device
)

result = generator(
    "Machine Learning is",
    max_new_tokens=50
)

print(result[0]["generated_text"])
```

---

## 📊 Sample Output

```text
============================================================
HUGGING FACE AI WORKSTATION
============================================================

DEVICE INFORMATION
Device: CUDA
GPU: NVIDIA GPU

TOKENIZATION
Token Count: 8

EMBEDDINGS
Embedding Shape: torch.Size([1, 10, 768])

TEXT GENERATION
Machine Learning is transforming industries...

SENTIMENT ANALYSIS
POSITIVE (0.999)

BENCHMARK
Execution Time: 1.12 seconds
Tokens Per Second: 89.28

MEMORY
RAM Usage: 1422 MB
GPU Memory: 2.14 GB

Execution Complete
============================================================
```

---

## 📂 Project Structure

```text
├── HuggingFace_AI_Workstation.ipynb
├── README.md
```

---

## 🎯 Learning Outcomes

This notebook helps users understand:

* Transformer architectures
* Tokenization workflows
* Embedding generation
* Text generation pipelines
* Sentiment classification
* Summarization models
* GPU acceleration in NLP
* Performance benchmarking

---

## 🔒 License

This project is released under the MIT License. Feel free to use, modify, and distribute it for educational and research purposes.

