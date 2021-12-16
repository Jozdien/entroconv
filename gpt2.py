import torch
from transformers import GPT2Tokenizer, GPT2LMHeadModel

tokenizer = GPT2Tokenizer.from_pretrained('gpt2')
model = GPT2LMHeadModel.from_pretrained('gpt2')


# Returns raw predictions for next token relative to string
def get_logits(string):
	inputs = tokenizer(string, return_tensors="pt")
	outputs = model(**inputs, labels=inputs["input_ids"])

	logits = outputs.logits

	return logits


# Returns probabilities for next token relative to string
def get_probs(string):
	inputs = tokenizer(string, return_tensors="pt")
	outputs = model(**inputs, labels=inputs["input_ids"])

	probs = torch.nn.Softmax(dim=2)(outputs.logits)

	return probs


# Returns the most likely next token relative to string
def most_prob_cont(string):
	logits = get_logits(string)

	return tokenizer.decode(torch.argmax(logits[0][-1])), torch.max(logits[0][-1])


# Returns the entropy value (mean of logit values of comprising token(s)) for next_word relative to string
def word_entropy(string, next_word):
	logits = get_logits(string)

	next_token = tokenizer.encode(next_word)

	return torch.mean(logits[0][-1][next_token]).item()


# Returns a vector of entropy values (logit values) for all comprising token(s) of next_word relative to string
def word_entropy_full(string, next_word):
	logits = get_logits(string)

	next_token = tokenizer.encode(next_word)

	return logits[0][-1][next_token]


# Returns the probability of next_word being continuation to string
def word_prob(string, next_word):
	probs = get_probs(string)

	next_token = tokenizer.encode(next_word)

	return torch.mean(probs[0][-1][next_token]).item()