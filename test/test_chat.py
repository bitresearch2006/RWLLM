import uuid
from llm import rwllm, get_last_messages

print("🔹 Welcome to GPT-2 Chat!")
print("🔹 Type 'bye', 'stop', or 'exit' to end. Type '/history' to view last messages.\n")

# Begin with a hidden auto-generated session ID
session_id = str(uuid.uuid4())

# Let user customize generation parameters
try:
    temperature = float(input("Enter temperature (0.0–1.0, default 0.7): ") or 0.7)
    top_k = int(input("Enter top_k (e.g., 40,50; default 50): ") or 50)
    top_p = float(input("Enter top_p (0.0–1.0, default 0.95): ") or 0.95)
    max_new_tokens = int(input("Enter max_new_tokens (default 100): ") or 100)
    context_turns = int(input("Enter context_turns (default 6): ") or 6)
except ValueError:
    print("⚠️ Invalid input; using defaults.")
    temperature, top_k, top_p, max_new_tokens, context_turns = 0.7, 50, 0.95, 100, 6

# Optionally resume previous session if ID known (uncomment)
# session_id = "YOUR_PREVIOUS_SESSION_ID"
response = rwllm(
    "Hello!",
    session_id=session_id,
    session_start=True,
    temperature=temperature,
    top_k=top_k,
    top_p=top_p,
    max_new_tokens=max_new_tokens,
    context_turns=context_turns
)
print("\nGPT-2:", response, "\n")

while True:
    user_input = input("You: ").strip()
    if user_input.lower() in ['bye', 'stop', 'exit']:
        response = rwllm(
            user_input,
            session_id=session_id,
            session_end=True,
            temperature=temperature,
            top_k=top_k,
            top_p=top_p,
            max_new_tokens=max_new_tokens,
            context_turns=context_turns
        )
        print("\nGPT-2:", response, "\n🔚 Goodbye!\n")
        break

    if user_input == "/history":
        history = get_last_messages(session_id, n=context_turns*2)
        print("\n📝 Recent history:")
        for msg in history:
            print(msg)
        print()
        continue

    response = rwllm(
        user_input,
        session_id=session_id,
        temperature=temperature,
        top_k=top_k,
        top_p=top_p,
        max_new_tokens=max_new_tokens,
        context_turns=context_turns
    )
    print("\nGPT-2:", response, "\n")
