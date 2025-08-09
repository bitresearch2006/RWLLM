#!/bin/bash

read -p "Enter your message for GPT-2: " USER_INPUT

curl -X POST "http://localhost:8080/function/llm" \
  -H "Content-Type: application/json" \
  -d "{
    \"user_input\": \"$USER_INPUT\",
    \"session_id\": \"test1\",
    \"session_start\": true,
    \"session_end\": false,
    \"diagnostics\": true,
    \"context_turns\": 6,
    \"max_new_tokens\": 100,
    \"temperature\": 0.7,
    \"top_k\": 50,
    \"top_p\": 0.95
  }"
