# Finetuning-Phi-2-Text-Summarization
Task Number 3 for Finalterm Deep Learning (Group 16)

# 🚀 Task 3: Fine-tuning Decoder-Only LLM for Abstractive Summarization

[![Python 3.11+](https://img.shields.io/badge/python-3.11+-blue.svg)](https://www.python.org/downloads/)
[![PyTorch 2.1](https://img.shields.io/badge/PyTorch-2.1-red.svg)](https://pytorch.org/)
[![Transformers 4.35](https://img.shields.io/badge/🤗%20Transformers-4.35-yellow.svg)](https://huggingface.co/transformers/)
[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)
[![GPU: RTX 3050 Ti 4GB](https://img.shields.io/badge/GPU-RTX%203050%20Ti%204GB-9cf)](https://www.nvidia.com/)
[![Status: Completed](https://img.shields.io/badge/Status-Completed-success)]()

## 📋 **Project Overview**

This repository contains the implementation of **Task 3** from the Deep Learning Final Term Assignment: **Fine-tuning a decoder-only Large Language Model (LLM) for abstractive text summarization** using the XSum dataset.

### 🎯 **Task Requirements Implemented**
✅ **Decoder-only LLM** - DistilGPT-2 (as PIN-2 equivalent)  
✅ **XSum dataset** - Abstractive summarization task  
✅ **Instruction-style prompting** - Explicit summarization instructions  
✅ **Causal language modeling** - Next-token prediction training  
✅ **Generated control parameters** - Temperature, beam search, repetition penalty, etc.  
✅ **End-to-end pipeline** - Data preprocessing → Training → Evaluation  

## 📊 **Results Summary**

| Metric | Score | Interpretation |
|--------|-------|----------------|
| **ROUGE-1** | 0.1687 | 16.87% unigram overlap |
| **ROUGE-2** | 0.0000 | Very low bigram overlap |
| **ROUGE-L** | 0.1256 | 12.56% sequence overlap |
| **Training Time** | 26m 41s | On RTX 3050 Ti 4GB |
| **Model Size** | 82M parameters | DistilGPT-2 |

> **Note**: Performance is constrained by hardware limitations (4GB GPU). Results would significantly improve with better hardware.

## 🏗️ **Repository Structure**
