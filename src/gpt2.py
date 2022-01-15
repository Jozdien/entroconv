import torch
import math
from transformers import GPT2Tokenizer, GPT2LMHeadModel

tokenizer = GPT2Tokenizer.from_pretrained('gpt2')
model = GPT2LMHeadModel.from_pretrained('gpt2')


# Returns tokenized version of string for debugging
def tokenize(string):
	return [tokenizer.decode(i) for i in tokenizer.encode(string)]


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
	probs = get_probs(string)

	return tokenizer.decode(torch.argmax(probs[0][-1])), torch.log(torch.max(probs[0][-1]))


# Returns the entropy value (mean of logit values of comprising token(s)) for a sequence of next tokens relative to string
def entropy_of_next(string, next_str):
	if string == "":
		tokens = tokenizer.encode(next_str)
		if len(tokens) == 1:
			raise ValueError("Prefix string argument to function should not be empty; or barring that, next_str argument should be of multiple tokens")
		else:
			first_token = tokenizer.decode(tokens[0])
			string = first_token
			next_str = next_str[len(first_token):]
	elif len(string) > 200:
		string = string[string[-200:].find("."):]

	probs = get_probs(string)

	next_tokens = tokenizer.encode(next_str)

	running_sum = []
	for token in next_tokens:
		running_sum.append(torch.log(probs[0][-1][token]))

		string += tokenizer.decode(token)
		if len(string) > 200:
			string = string[string[-200:].find("."):]

		probs = get_probs(string)

	return (sum(running_sum) / len(running_sum)).item()


# Returns a vector of entropy values (logit values) for all comprising token(s) of a sequence of next tokens relative to string
def entropy_of_next_full(string, next_str):
	probs = get_probs(string)

	next_tokens = tokenizer.encode(next_str)

	running_sum = []
	for token in next_tokens:
		running_sum.append(torch.log(probs[0][-1][token]))

		string += tokenizer.decode(token)
		probs = get_probs(string)

	return running_sum


# Returns the probability of next_word being continuation to string
def word_prob(string, next_word):
	probs = get_probs(string)

	next_token = tokenizer.encode(next_word)

	return torch.mean(probs[0][-1][next_token]).item()