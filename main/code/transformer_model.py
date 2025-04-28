import os
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'
import tensorflow as tf
from transformers import GPT2LMHeadModel, GPT2Tokenizer

# Load pre-trained model and tokenizer
model_name = 'gpt2'
model = GPT2LMHeadModel.from_pretrained(model_name)
tokenizer = GPT2Tokenizer.from_pretrained(model_name)

# Add a special padding token
tokenizer.add_special_tokens({'pad_token': '[PAD]'})

# 🧠 Take input from user
input_text = input("Enter your prompt: ")

# Tokenize user input
inputs = tokenizer(input_text, return_tensors='pt', padding=True)

# Generate text with adjusted parameters
output = model.generate(
    inputs['input_ids'],
    attention_mask=inputs['attention_mask'],
    pad_token_id=tokenizer.pad_token_id,
    max_length=500,
    num_return_sequences=1,  # Changed to 1 for simpler output
    do_sample=True,
    temperature=0.7,
    top_k=50,
    top_p=0.95
)

# Decode and print the output
generated_text = tokenizer.decode(output[0], skip_special_tokens=True)
print("\nGenerated Response:\n")
print(generated_text)
