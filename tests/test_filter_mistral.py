import json
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
data = json.load(open("/raid/phogpt_team/chitb/Subnet47_condense/data/validation.json"))
sample = data[0]


#model_id = "namespace-Pt/ultragist-llama2-7b-chat"
model_id = "Condense-AI/Mistral-7B-Instruct-v0.2"
tokenizer = AutoTokenizer.from_pretrained(
  model_id, 
  trust_remote_code=True,
)
model = AutoModelForCausalLM.from_pretrained(
  model_id, 
  trust_remote_code=True, 
  torch_dtype=torch.float16, 
  attn_implementation="sdpa", 
  # load the entire model on the default gpu
  device_map={"": "cuda"}, 
  # you can manually set the compression ratio, otherwise the model will automatically choose the most suitable compression ratio from [2,4,8,16,32]
  # ultragist_ratio=[8],
).eval()

total_samples = sample['positive_chunks'] + sample['negative_chunks']
for idx in range(len(total_samples)):
    PROMPT = f"""
    You are a precise and objective fact-checker. Your task is to determine whether the following quoted text appears in the provided context or is a direct paraphrase of it. 

    Instructions:
    - Consider the context to include information that might have been rephrased but retains the original meaning.
    - Return 'yes' if the quoted text appears or is a clear paraphrase of the context.
    - Return 'no' if the quoted text does not appear or if it is not a valid paraphrase.
    - Your response should contain exactly one word: either 'yes' or 'no'. No additional text or explanations are required.
    Context:
    ```
    {sample['context']}
    ```
    Quote:
    ```
    {total_samples[idx]}
    ```
    Only return single yes or no response.
    [/INST] """
    with torch.no_grad(): 
        messages = [{"role": "user", "content": PROMPT}]
        inputs = tokenizer.apply_chat_template(messages, tokenize=True, add_generation_prompt=True, return_tensors="pt", return_dict=True).to("cuda")

        # reset memory before new compression task
        #model.memory.reset()

        # directly call generate to progressively compress the context while generating next tokens
        outputs = model.generate(**inputs, do_sample=False, max_new_tokens=256)[:, inputs["input_ids"].shape[1]:]
        print("*"*20)
        # print(f"Input size:       {inputs['input_ids'].shape[1]}") 
        print(f"Prediction {idx}       {tokenizer.decode(outputs[0], skip_special_tokens=True)}")
        print("*"*20)

        # # extract the compressed memory (including the generated tokens)
        # compressed_memory = model.memory.get_memory()
        # ultragist_size, raw_size, sink_size = model.memory.get_memory_size()
        # print(f"UltraGist size:   {ultragist_size}")
        # print(f"Raw size:         {raw_size}")
        # print(f"Sink size:        {sink_size}")
        # print(f"Memory:           {compressed_memory[0][0].shape}")
        # print("*"*20)