import os
import logging
from transformers import GPT2LMHeadModel, GPT2Tokenizer

# Disable oneDNN warning (optional)
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'

# Paths
local_dir = './local_gpt2_model'
log_file = "chat_log.txt"
diagnostic_log = "rwllm_log.log"
session_logs_dir = "session_logs"

# Load GPT-2 model & tokenizer
if not os.path.exists(local_dir):
    print("Downloading GPT-2 model...")
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

# In-memory sessions
chat_histories = {}

# Logging config
logging.basicConfig(filename=diagnostic_log, level=logging.INFO, format='%(asctime)s - %(message)s')
chat_logger = logging.getLogger(__name__)

def load_session(session_id):
    """Load previous session history from disk into memory."""
    os.makedirs(session_logs_dir, exist_ok=True)
    path = os.path.join(session_logs_dir, f"session_{session_id}.txt")
    if not os.path.exists(path):
        return False
    lines = [l.strip() for l in open(path, "r", encoding="utf-8") if l.strip()]
    chat_histories[session_id] = [l for l in lines if l.startswith(("User:", "AI:"))]
    return True

def get_last_messages(session_id, n=6):
    """Retrieve last n messages (User / AI) from memory."""
    return chat_histories.get(session_id, [])[-n:]

def rwllm(
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
    global chat_histories

    # Parameter validation
    try:
        assert 0 < temperature <= 1, "Temperature must be between 0 and 1"
        assert 0 < top_p <= 1, "Top-p must be between 0 and 1"
        assert top_k >= 0, "Top-k must be non-negative"
        assert max_new_tokens > 0, "max_new_tokens must be positive"
    except AssertionError as err:
        return f"Parameter Error: {err}"

    # Optional diagnostic logging
    if diagnostics:
        chat_logger.info(f"rwllm called (session={session_id})")

    # Start or resume session
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
    inputs = tokenizer(prompt, return_tensors='pt', truncation=True, max_length=1024)

    output = model.output = model.generate(
    input_ids=inputs['input_ids'],
    attention_mask=inputs['attention_mask'],
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

    full_output = tokenizer.decode(output[0], skip_special_tokens=True, clean_up_tokenization_spaces=True)
    response = full_output.split("AI:")[-1].split("User:")[0].strip()

    if not response or all(c in "-\n " for c in response):
        response = "I'm sorry, I didn’t quite understand that."

    history.append(f"AI: {response}")

    # Global chat log
    with open(log_file, "a", encoding="utf-8") as g:
        g.write(f"Session {session_id} | User: {user_input}\n")
        g.write(f"Session {session_id} | AI: {response}\n\n")

    # Per-session log
    os.makedirs(session_logs_dir, exist_ok=True)
    with open(os.path.join(session_logs_dir, f"session_{session_id}.txt"), "a", encoding="utf-8") as s:
        s.write(f"User: {user_input}\nAI: {response}\n\n")

    if session_end and diagnostics:
        chat_logger.info(f"Session end flagged: {session_id}")

    return response
