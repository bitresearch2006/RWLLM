from llm import rwllm  # not get_response anymore

print("Welcome to GPT-2 Chat! Type 'bye', 'stop', or 'exit' to end.\n")

while True:
    user_input = input("You: ")
    if user_input.lower() in ['bye', 'stop', 'exit']:
        print("Goodbye!")
        break

    response = rwllm(user_input)
    print("\nGPT-2:", response, "\n")
