import os
from transformers import GPT2LMHeadModel, GPT2Tokenizer

# Disable oneDNN optimizations warning for TensorFlow backend (optional)
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'

local_dir = './local_gpt2_model'
log_file = "chat_log.txt"

# Load or download GPT-2 model and tokenizer
if not os.path.exists(local_dir):
    print("Local model not found. Downloading GPT-2...")
    model = GPT2LMHeadModel.from_pretrained("gpt2")
    tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
    tokenizer.add_special_tokens({'pad_token': '[PAD]'})
    model.resize_token_embeddings(len(tokenizer))
    os.makedirs(local_dir, exist_ok=True)
    model.save_pretrained(local_dir)
    tokenizer.save_pretrained(local_dir)
else:
    model = GPT2LMHeadModel.from_pretrained(local_dir)
    tokenizer = GPT2Tokenizer.from_pretrained(local_dir)
    tokenizer.add_special_tokens({'pad_token': '[PAD]'})
    model.resize_token_embeddings(len(tokenizer))

chat_history = []

def rwllm(user_input):
    global chat_history

    # Append user input to history
    chat_history.append(f"User: {user_input}")

    # Use last 6 entries (last 3 turns) as prompt for the model
    prompt = "\n".join(chat_history[-6:]) + "\nAI:"

    # Tokenize the prompt
    inputs = tokenizer(prompt, return_tensors='pt', truncation=True, max_length=1024)

    # Generate output from GPT-2
    output = model.generate(
        input_ids=inputs['input_ids'],
        attention_mask=inputs['attention_mask'],
        pad_token_id=tokenizer.pad_token_id,
        max_new_tokens=100,
        do_sample=True,
        temperature=0.7,
        top_k=50,
        top_p=0.95,
        eos_token_id=tokenizer.eos_token_id
    )

    # Decode the generated tokens to string
    full_output = tokenizer.decode(output[0], skip_special_tokens=True, clean_up_tokenization_spaces=True)

    # Extract response text after the last "AI:" and before next "User:"
    response = full_output.split("AI:")[-1].strip().split("User:")[0].strip()

    # If response is empty or invalid, give fallback response
    if not response or all(c in "-\n " for c in response):
        response = "I'm sorry, I didn’t quite understand that."

    # Append model response to chat history
    chat_history.append(f"AI: {response}")

    # Log chat to file
    with open(log_file, "a", encoding="utf-8") as log:
        log.write(f"User: {user_input}\nGPT-2: {response}\n\n")

    return response
