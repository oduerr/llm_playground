########## Testing sampling from local LLAMA2 Model ##########

import os
import time
from torch import cuda, bfloat16
import transformers

# Fetch auth_token from environment variable
auth_token = os.environ.get('HUGGING_FACE_AUTH_TOKEN')
if auth_token is None:
    raise EnvironmentError("Environment variable HUGGING_FACE_AUTH_TOKEN not set")

#model_id = 'meta-llama/Llama-2-13b-chat-hf'
#model_id = 'meta-llama/Llama-2-70b-chat-hf'
model_id = 'meta-llama/Llama-2-7b-chat-hf'

device = f'cuda:{cuda.current_device()}' if cuda.is_available() else 'cpu'

quant_config = transformers.BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type='nf4',
    bnb_4bit_use_double_quant=True,
    bnb_4bit_compute_dtype=bfloat16
)

model_config = transformers.AutoConfig.from_pretrained(
    model_id,
    use_auth_token=auth_token
)

model = transformers.AutoModelForCausalLM.from_pretrained(  
    model_id,
    trust_remote_code=True,
    config=model_config,
    quantization_config=quant_config,
    use_auth_token=auth_token
)

print("Number of attention heads:", model.config.num_attention_heads)
print("Hidden size:", model.config.hidden_size)
print("Number of hidden layers:", model.config.num_hidden_layers)
print("Output vocabulary size:", model.config.vocab_size)

model.eval()
print(f"Model loaded on {device}")

tokenizer = transformers.AutoTokenizer.from_pretrained(
    model_id,
    use_auth_token=auth_token
)

prompt = 'I have butter, bread and jam. What can I make with them? Give me a recipe.'
input_text = prompt


########## Generate text (one by one) ##########
input_ids = tokenizer.encode(input_text, return_tensors='pt').to(device)
import torch
max_length = 512 # Set a maximum length for the generated text
# Loop to generate tokens
start_time = time.time()
gen_tokens = 0
generated_text = input_text
for _ in range(max_length - len(input_ids[0])):
    # Generate the next token
    output = model.generate(input_ids, max_length=input_ids.shape[1] + 1, num_return_sequences=1)
    
    # Get the last token ID
    gen_tokens += 1
    next_token = output[0][-1].item()
    next_token_tensor = torch.tensor([[next_token]], dtype=torch.long).to(input_ids.device)
    # Decode the entire generated text
    new_generated_text = tokenizer.decode(input_ids[0])
    # Print the newly added characters
    print(new_generated_text[len(generated_text):], end='', flush=True)
    # Update the generated text
    generated_text = new_generated_text
    
    input_ids = torch.cat([input_ids, next_token_tensor], dim=1)
    
    # Optional: Stop generating if the end-of-sentence token is reached
    if next_token == tokenizer.eos_token_id:
        break

print(generated_text)
print("\n\nTokens per second (When sampling one-by-one): {}".format(gen_tokens/(time.time() - start_time)))


# Now we use the model in a pipeline
from transformers import pipeline
pipline = pipeline('text-generation', model=model, tokenizer=tokenizer, framework='pt')
#pipline = pipeline('text-generation', model=model, tokenizer=tokenizer, framework='tf')
for i in range(10):
    print("Starting the pipeline with {}\n".format(prompt))
    #Starts timer
    start_time = time.time()
    result = pipline(prompt)[0]
    print(f"label: {result['generated_text']}")
    print("Tokens per second (Then using a pipeline): {}".format(len(result['generated_text'])/(time.time() - start_time)))
