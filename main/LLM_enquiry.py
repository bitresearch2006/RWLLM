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

# Tokenize input text

input_text = "Write a c function to calculate the factorial of a number."
inputs = tokenizer(input_text, return_tensors='pt', padding=True)

# Generate text with adjusted parameters
output = model.generate(
    inputs['input_ids'],
    attention_mask=inputs['attention_mask'],
    pad_token_id=tokenizer.pad_token_id,
    max_length=500,
    num_return_sequences=5,
    do_sample=True,
    temperature=0.7,
    top_k=50,
    top_p=0.95
)

generated_text = tokenizer.decode(output[0], skip_special_tokens=True)

print(generated_text)
