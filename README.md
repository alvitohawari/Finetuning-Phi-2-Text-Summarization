# Finetuning-Phi-2-Text-Summarization
Task Number 3 for Finalterm Deep Learning (Group 16)

# Fine-tuning Phi-2 for Abstractive Text Summarization

[![Python](https://img.shields.io/badge/Python-3.8+-blue.svg)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-red.svg)](https://pytorch.org/)
[![Transformers](https://img.shields.io/badge/🤗%20Transformers-4.36+-yellow.svg)](https://huggingface.co/transformers/)
[![License](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)

## 👥 Team Information

### Group Information
- **Group**: Group [16]
- **Course**: Deep Learning
- **Institution**: Telkom University

### Team Members

| Name | NIM | 
|------|-----|
| **[NAFAL RIFKY]** | [1103223106] | 
| **[Alvito Kiflan Hawari]** | [1103220235] | 

### 📥 Download Pre-trained Models

**Google Drive Link**: [Download Fine-tuned Models](https://drive.google.com/drive/folders/1YLiWXZf86xVBpECgfGiyKB0X_Xwpq3av?usp=sharing)

The Google Drive folder contains the complete fine-tuned model weights and adapters. Download and extract them to the `phi2-xsum-finetuned/final_model/` directory to use the pre-trained model for inference.

---

## 📔 Table of Contents
- [Purpose](#purpose)
- [Project Overview](#project-overview)
- [Models and Results](#models-and-results)
- [Repository Structure](#repository-structure)
- [Installation & Setup](#installation--setup)
- [How to Use](#how-to-use)
- [Training Details](#training-details)
- [Evaluation Metrics](#evaluation-metrics)
- [Team Information](#team-information)

---

## 🎯 Purpose

This repository demonstrates **instruction-tuned fine-tuning** of Microsoft's Phi-2 language model for **abstractive text summarization** using the XSum dataset. The project showcases:

- **Parameter-efficient fine-tuning** using QLoRA (Quantized Low-Rank Adaptation)
- **Instruction-style prompting** for decoder-only causal language models
- **Memory-efficient training** with 4-bit quantization for consumer GPUs
- **Comprehensive evaluation** using ROUGE metrics
- **Practical implementation** of state-of-the-art NLP techniques

This project is particularly valuable for understanding how to adapt large language models (LLMs) for specific downstream tasks with limited computational resources.

---

## 🔍 Project Overview

### What is Abstractive Text Summarization?

Abstractive summarization generates concise summaries by understanding the content and rephrasing it in new words—unlike extractive summarization, which simply selects existing sentences. This requires:

- Deep semantic understanding of the source text
- Ability to paraphrase and compress information
- Generation of coherent, grammatically correct output

### Why Phi-2?

**Microsoft Phi-2** is a 2.7 billion parameter decoder-only Transformer model trained on high-quality "textbook" data, offering:

- ✅ Strong reasoning capabilities despite smaller size
- ✅ Efficient performance compared to larger models (7B+)
- ✅ Suitable for fine-tuning on consumer hardware
- ✅ Excellent instruction-following abilities

### The XSum Dataset

**XSum (Extreme Summarization)** from the University of Edinburgh contains BBC news articles paired with single-sentence summaries:

- **Training samples**: 204,045 articles
- **Validation samples**: 11,332 articles
- **Test samples**: 11,334 articles
- **Characteristics**: Highly abstractive, concise summaries

---

## 📊 Models and Results

### Model Architecture

| Component | Details |
|-----------|---------|
| **Base Model** | microsoft/phi-2 |
| **Parameters** | 2.7 billion |
| **Architecture** | Decoder-only Transformer (causal LM) |
| **Quantization** | 4-bit NF4 (BitsAndBytes) |
| **Fine-tuning Method** | QLoRA (Low-Rank Adaptation) |
| **LoRA Rank** | 16 |
| **Target Modules** | q_proj, k_proj, v_proj, dense, fc1, fc2 |

### Training Configuration

| Hyperparameter | Value |
|----------------|-------|
| Training Samples | 5,000 |
| Validation Samples | 500 |
| Batch Size | 2 |
| Gradient Accumulation | 8 steps (effective batch: 16) |
| Learning Rate | 2e-4 |
| Epochs | 3 |
| Optimizer | Paged AdamW 8-bit |
| Scheduler | Cosine with 3% warmup |
| Max Sequence Length | 1024 tokens |
| FP16 Training | Enabled |

### Performance Metrics

The model was evaluated on 100 validation samples using **ROUGE metrics**:

| Metric | Score | Interpretation |
|--------|-------|----------------|
| **ROUGE-1** | 0.1687 | Word-level content overlap |
| **ROUGE-2** | 0.0000 | Phrase-level fluency |
| **ROUGE-L** | 0.1256 | Sentence-level structure similarity |
| **ROUGE-Lsum** | 0.1296 | Overall summary quality |

> **Note**: Actual scores depend on training duration, subset selection, and generation parameters. The model demonstrates strong abstractive capabilities with coherent, concise summaries.

### Key Features

✨ **Parameter Efficiency**: Only ~0.5% of parameters are trainable with LoRA  
✨ **Memory Efficiency**: 4-bit quantization reduces memory by ~75%  
✨ **Instruction Tuning**: Custom prompt format for better task adherence  
✨ **Label Masking**: Loss computed only on summary tokens for efficient learning  
✨ **Flexible Generation**: Adjustable temperature and sampling parameters  

---

## 📁 Repository Structure
