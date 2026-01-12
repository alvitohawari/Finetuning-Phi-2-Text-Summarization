# Finetuning-Phi-2-Text-Summarization
Task Number 3 for Finalterm Deep Learning (Group 16)

from transformers import AutoTokenizer, AutoModelForCausalLM

model = AutoModelForCausalLM.from_pretrained("./models/gpt2-xsum-finetuned")
tokenizer = AutoTokenizer.from_pretrained("./models/gpt2-xsum-finetuned")

# Generate summary dengan kontrol parameter
summary = generate_summary_with_control(article, model, tokenizer, temperature=0.7)
