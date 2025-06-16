import os
import logging
from transformers import GPT2LMHeadModel, GPT2Tokenizer

# Disable oneDNN warning (optional)
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'

local_dir = './local_gpt2_model'
log_file = "chat_log.txt"
diagnostic_log = "rwllm_log.log"

# Load GPT-2 model and tokenizer
if not os.path.exists(local_dir):
    print("Local model not found. Downloading GPT-2...")
    model = GPT2LMHeadModel.from_pretrained("gpt2")
    tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
    tokenizer.add_special_tokens({'pad_token': '[PAD]'})
    model.resize_token_embeddings(len(tokenizer))
    os.makedirs(local_dir, exist_ok=True)
    model.save_pretrained(local_dir)
    tokenizer.save_pretrained(local_dir)

model = GPT2LMHeadModel.from_pretrained(local_dir)
tokenizer = GPT2Tokenizer.from_pretrained(local_dir)
tokenizer.add_special_tokens({'pad_token': '[PAD]'})
model.resize_token_embeddings(len(tokenizer))

chat_history = []

def rwllm(user_input, diagnostics=False):
    global chat_history

    if diagnostics:
        logging.basicConfig(filename=diagnostic_log, level=logging.INFO, format='%(asctime)s - %(message)s')

    def log_event(message, error_message=None):
        if diagnostics:
            if error_message:
                logging.error(f"{message} - Error: {error_message}")
            else:
                logging.info(message)

    try:
        log_event("Starting rwllm()")
        chat_history.append(f"User: {user_input}")
        prompt = "\n".join(chat_history[-6:]) + "\nAI:"
        log_event(f"Prompt Prepared:\n{prompt}")

        inputs = tokenizer(prompt, return_tensors='pt', truncation=True, max_length=1024)
        log_event(f"Tokenized Prompt - Token Count: {len(inputs['input_ids'][0])}")

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
        log_event("Text generation completed")

        full_output = tokenizer.decode(output[0], skip_special_tokens=True, clean_up_tokenization_spaces=True)
        response = full_output.split("AI:")[-1].strip().split("User:")[0].strip()
        log_event(f"Generated Response: {response}")

        if not response or all(c in "-\n " for c in response):
            response = "I'm sorry, I didn’t quite understand that."
            log_event("Fallback response used")

        chat_history.append(f"AI: {response}")

        with open(log_file, "a", encoding="utf-8") as log:
            log.write(f"User: {user_input}\nGPT-2: {response}\n\n")
            log_event("Chat logged to file")

        return response

    except Exception as e:
        log_event("Error in rwllm()", error_message=str(e))
        print(f"Error in rwllm: {e}")
        return "Sorry, an error occurred while generating a response."
