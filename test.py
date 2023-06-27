from transformers import AutoTokenizer,AutoModelForSeq2SeqLM

# 1. Load the fine-tuned model and tokenizer
model_name = './results'  # or specify the path to where you saved the fine-tuned model
tokenizer = AutoTokenizer.from_pretrained('t5-base')  # assuming you fine-tuned T5-base
model = AutoModelForSeq2SeqLM.from_pretrained(model_name)

# 2. Encode the input text
input_text = "Classify,Who gave the UN the land in NY to build their HQ"
input_ids = tokenizer.encode(input_text, return_tensors='pt')

# 3. Generate predictions
output = model.generate(input_ids, max_length=10, num_beams=5, early_stopping=True)

# 4. Decode the output
decoded_output = tokenizer.decode(output[0], skip_special_tokens=True)

print(decoded_output)
