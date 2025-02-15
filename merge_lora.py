from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel
import sys
base_model = AutoModelForCausalLM.from_pretrained(sys.argv[1],
                                                  trust_remote_code=True)
tokenizer = AutoTokenizer.from_pretrained(sys.argv[1])
peft_model_id = sys.argv[2] 
model = PeftModel.from_pretrained(base_model, peft_model_id)
model.merge_and_unload()
print("Model loaded, ", model)
model.save_pretrained(sys.argv[3])
tokenizer.save_pretrained(sys.argv[3])
