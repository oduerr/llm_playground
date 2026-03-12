import json
from openai import OpenAI
from transformers import AutoTokenizer

# Setting up the model and the tokenizer
model_name_for_API = "gemma3:12b"
model_name_for_API = "qwen2.5:3b-instruct"
model_name_for_API = "qwen2.5:7b-instruct"
tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen2.5-3B-Instruct") #Same tokenizer can be used for 3b and 7b models
client = OpenAI(
    base_url="http://localhost:11434/v1",
    api_key="unused"
)
TASK = "What is 3141 + 4242?"
TASK = "Tim has 5652 apples and Jane has 10272727 apples, how many apples do they have together?"

def print_prompt(messages):
    if tokenizer is None:
        prompt = ""
    else:
        prompt = tokenizer.apply_chat_template(
            messages,
            tools=tools,
            tokenize=False,
            add_generation_prompt=True
        )
    print("=== PROMPT GOING INTO MODEL ===")
    print(prompt)
    print("================================\n")


tools = [
    {
        "type": "function",
        "function": {
            "name": "add",
            "description": "Add two numbers",
            "parameters": {
                "type": "object",
                "properties": {
                    "a": {"type": "number"},
                    "b": {"type": "number"}
                },
                "required": ["a", "b"]
            }
        }
    }
]

messages = [{"role": "user", "content": TASK}]

# ── ReAct Loop ───────────────────────────────────────────────────────────────
MAX_STEPS = 5

for step in range(MAX_STEPS):
    print(f"\n{'='*10} STEP {step+1} {'='*10}")
    print_prompt(messages)

    response = client.chat.completions.create(
        model=model_name_for_API,
        messages=messages,
        tools=tools
    )

    assistant_message = response.choices[0].message

    # ── ACT: model wants to call a tool ─────────────────────────────────────
    if assistant_message.tool_calls:
        tool_call = assistant_message.tool_calls[0]
        args = json.loads(tool_call.function.arguments)
        print(f"[Requesting tool call]    {tool_call.function.name}({args})")

        # Execute
        result = args["a"] + args["b"]
        print(f"[Result of tool call]    {args['a']} + {args['b']} = {result}")

        # Feed back into context
        messages.append(assistant_message)
        messages.append({
            "role": "tool",
            "tool_call_id": tool_call.id,
            "content": str(result)
        })

    # ── FINISH: model produced a final answer ────────────────────────────────
    else:
        print(f"[ANSWER] {assistant_message.content}")
        break

else:
    print("Max steps reached without final answer")