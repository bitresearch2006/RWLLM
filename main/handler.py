import os
import json
import logging
import torch
from transformers import GPT2LMHeadModel, GPT2Tokenizer

# --- Configuration ---
LOCAL_MODEL_DIR = './local_gpt2_model'
LOG_FILE = "chat_log.txt"
DIAGNOSTIC_LOG = "rwllm_log.log"
SESSION_LOGS_DIR = "session_logs"

# In-memory session histories
chat_histories = {}

# Logging setup
logging.basicConfig(filename=DIAGNOSTIC_LOG, level=logging.INFO, format='%(asctime)s - %(message)s')
chat_logger = logging.getLogger(__name__)

# Detect device
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# --- Load or Download GPT-2 Model ---
def load_model():
    if not os.path.exists(LOCAL_MODEL_DIR):
        print("Downloading GPT-2 model...")
        model = GPT2LMHeadModel.from_pretrained("gpt2")
        tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
        tokenizer.add_special_tokens({'pad_token': '[PAD]'})
        model.resize_token_embeddings(len(tokenizer))
        os.makedirs(LOCAL_MODEL_DIR, exist_ok=True)
        model.save_pretrained(LOCAL_MODEL_DIR)
        tokenizer.save_pretrained(LOCAL_MODEL_DIR)

    model = GPT2LMHeadModel.from_pretrained(LOCAL_MODEL_DIR)
    tokenizer = GPT2Tokenizer.from_pretrained(LOCAL_MODEL_DIR)
    return model.to(DEVICE), tokenizer

# Load once at cold start
model, tokenizer = load_model()
tokenizer.add_special_tokens({'pad_token': '[PAD]'})
model.resize_token_embeddings(len(tokenizer))

# --- Session Management ---
def load_session(session_id):
    os.makedirs(SESSION_LOGS_DIR, exist_ok=True)
    path = os.path.join(SESSION_LOGS_DIR, f"session_{session_id}.txt")
    if not os.path.exists(path):
        return False
    with open(path, "r", encoding="utf-8") as f:
        lines = [l.strip() for l in f if l.strip()]
    chat_histories[session_id] = [l for l in lines if l.startswith(("User:", "AI:"))]
    return True

def save_session_log(session_id, user_input, ai_response):
    os.makedirs(SESSION_LOGS_DIR, exist_ok=True)
    with open(os.path.join(SESSION_LOGS_DIR, f"session_{session_id}.txt"), "a", encoding="utf-8") as s:
        s.write(f"User: {user_input}\nAI: {ai_response}\n\n")

def get_last_messages(session_id, n=6):
    """Retrieve last n messages (User / AI) from memory."""
    return chat_histories.get(session_id, [])[-n:]

# --- Core LLM Function ---
def handle(
    user_input,
    session_id,
    session_start=False,
    session_end=False,
    diagnostics=False,
    context_turns=6,
    max_new_tokens=100,
    temperature=0.7,
    top_k=50,
    top_p=0.95
):
    # Resume or create session
    if session_start:
        if not load_session(session_id):
            chat_histories[session_id] = []
            if diagnostics:
                chat_logger.info(f"New session created: {session_id}")
        else:
            if diagnostics:
                chat_logger.info(f"Resumed previous session: {session_id}")

    if session_id not in chat_histories:
        chat_histories[session_id] = []

    history = chat_histories[session_id]
    history.append(f"User: {user_input}")

    prompt = "\n".join(history[-context_turns:]) + "\nAI:"
    inputs = tokenizer(
        prompt,
        return_tensors='pt',
        truncation=True,
        max_length=1024 - max_new_tokens
    ).to(DEVICE)

    output = model.generate(
        **inputs,
        pad_token_id=tokenizer.pad_token_id,
        max_new_tokens=max_new_tokens,
        do_sample=True,
        temperature=temperature,
        top_k=top_k,
        top_p=top_p,
        no_repeat_ngram_size=2,
        repetition_penalty=1.2,
        eos_token_id=tokenizer.eos_token_id
    )

    generated_text = tokenizer.decode(
        output[0][inputs['input_ids'].shape[-1]:],
        skip_special_tokens=True
    )
    response = generated_text.strip() or "I'm sorry, I didn’t quite understand that."

    history.append(f"AI: {response}")

    # Save logs
    with open(LOG_FILE, "a", encoding="utf-8") as g:
        g.write(f"Session {session_id} | User: {user_input}\n")
        g.write(f"Session {session_id} | AI: {response}\n\n")
    save_session_log(session_id, user_input, response)

    if session_end and diagnostics:
        chat_logger.info(f"Session ended: {session_id}")

    return response

