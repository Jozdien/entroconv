import torch
from transformers import GPT2Tokenizer, GPT2LMHeadModel

tokenizer = GPT2Tokenizer.from_pretrained('gpt2')
model = GPT2LMHeadModel.from_pretrained('gpt2')


def get_entropy(string, next_word):
	inputs = tokenizer(string, return_tensors="pt")
	outputs = model(**inputs, labels=inputs["input_ids"])

	logits = outputs.logits

	next_token = tokenizer.encode(next_word)

	return torch.mean(logits[0][-1][next_token]).item()


def get_entropy_full(string, next_word):
	inputs = tokenizer(string, return_tensors="pt")
	outputs = model(**inputs, labels=inputs["input_ids"])

	logits = outputs.logits

	next_token = tokenizer.encode(next_word)

	return logits[0][-1][next_token]


def most_prob_cont(string):
	inputs = tokenizer(string, return_tensors="pt")
	outputs = model(**inputs, labels=inputs["input_ids"])

	logits = outputs.logits

	return tokenizer.decode(torch.argmax(logits[0][-1])), torch.max(logits[0][-1])


def get_prob(string, next_word):
	inputs = tokenizer(string, return_tensors="pt")
	outputs = model(**inputs, labels=inputs["input_ids"])

	logits = outputs.logits
	probs = torch.nn.Softmax(dim=2)(logits)

	next_token = tokenizer.encode(next_word)

	return torch.mean(probs[0][-1][next_token]).item()