import torch
from transformers import GPT2Tokenizer, GPT2LMHeadModel

tokenizer = GPT2Tokenizer.from_pretrained('gpt2')
model = GPT2LMHeadModel.from_pretrained('gpt2')


def surprise(string, next_word):
	inputs = tokenizer(string, return_tensors="pt")
	outputs = model(**inputs, labels=inputs["input_ids"])

	logits = outputs.logits
	probs = torch.nn.Softmax(dim=2)(logits)

	next_token = tokenizer.encode(next_word)

	return logits[0][-1][next_token].item()