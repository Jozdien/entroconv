from sentence_transformers import util
import nltk
# import gpt2
import gpt3

classify_file = "../samples/test_classify.txt"
heuristic_file = "../samples/test_heuristic.txt"
cannelize_file = "../samples/test_cannelize.txt"

prompt_heuristic = ""

def classify():
	# Shows some promise, could just be overfitting, need more examples
	with open(classify_file) as f:
		raw_text = f.read()

	examples = []

	while True:
		examples.append([raw_text[:raw_text.find("\n\n")], raw_text[raw_text.find("\n\n")+2:raw_text.find("\n\n\n")]])

		raw_text = raw_text[raw_text.find("\n\n\n")+3:]
		if raw_text.find("\n\n") == -1:
			query = raw_text
			break

	label, logprob = gpt3.classification(examples=examples, query=query, labels=["Contentful", "Structuralist"])

	print(f"Query: \n{query}")
	print(f"Label: {label}\nLogprob: {logprob}")

def heuristic():
	# Might work, GPT-3 shows fair predictive performance, but Joe Rogan isn't acting as a good heuristic in 1-1 conversations
	# Try multi-person conversation transcripts?  JRE #1258 if you can find it
	with open(heuristic_file) as f:
		prompt_heuristic = f.read()

	# cont = gpt2.greedy_seq(prompt_heuristic, seq_len=5)[len(prompt_heuristic):]

	# print("Using GPT-2")
	# print("-------------------")
	
	# print("Ground Truth")
	# print(cont)
	# print("Ground truth after newline")
	# print(gpt2.greedy_seq(prompt_heuristic+"\n", seq_len=5)[len(prompt_heuristic)+1:])

	# print("Pred: newline")
	# print(gpt2.entropy_of_next(prompt_heuristic, "\n"))
	# print("Pred: Jack")
	# print(gpt2.entropy_of_next(prompt_heuristic+"\n\n\n", "Jack"))

	cont, logprobs = gpt3.completion(prompt=prompt_heuristic, num_logprobs=5, response_length=10)

	print("Using GPT-3")
	print("-------------------")
	# print(cont)

	if "\n" not in logprobs[0].keys():
		upper_bound = min(logprobs[0], key=logprobs[0].get)
		print(f"<{upper_bound} likelihood")
	elif "Joe" not in logprobs[2].keys():
		upper_bound = (min(logprobs[2], key=logprobs[2].get) + logprobs[0]["\n"]) / 2
		print(f"<{upper_bound} likelihood")
	else:
		newline_logprob = logprobs[0]["\n"]  # Necessary because \n within f"{}" causes syntax errors
		joe_logprob = logprobs[2]["Joe"]
		print(f"Logprob of \\n: {newline_logprob}\nLogprob of Joe: {joe_logprob}\nAverage logprob: {(newline_logprob + joe_logprob) / 2}")


def cos_sim(sentence_1, sentence_2):
	embedding_1, embedding_2 = gpt3.embedding(input=sentence_1), gpt3.embedding(input=sentence_2)

	return util.pytorch_cos_sim(embedding_1, embedding_2).item()


def sim_of_conts(prompt, seq_no=2):
	cont_seqs = []

	for i in range(seq_no):
		cont, _ = gpt3.completion(prompt=prompt, temperature=1, stop_seq=".")
		cont += "."
		cont_seqs.append(cont)

	sims = []
	for i in range(seq_no):
		for j in range(i+1, seq_no):
			if i != j:
				sim = cos_sim(cont_seqs[i], cont_seqs[j])
				sims.append(sim)

	print(cont_seqs)
	print(sims)
	return sum(sims)/len(sims)


def cannelize():
	# Ran one single test with poor results - might very well work, but testing super expensive unless switch to GPT-2, which might not work
	# If running more tests, increase seq_no when calling sim_of_conts, that's where I see the biggest increase coming from for now
	with open(cannelize_file) as f:
		raw_text = f.read()

	try:
		nltk.data.find('tokenizers/punkt')
	except:
		nltk.download('punkt')

	sentences = nltk.tokenize.sent_tokenize(raw_text)

	print(sim_of_conts(prompt=raw_text[:-len(sentences[-1])-1], seq_no=5) - sim_of_conts(prompt=raw_text, seq_no=5))
	print(sentences[-1])
	print("The above sentence makes the posterior sequences this much more uncertain.")


if __name__ == "__main__":

	# classify()
	# heuristic()
	cannelize()