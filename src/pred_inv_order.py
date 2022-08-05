'''
Observations with SpanBERT: Unlikely to be powerful enough to test the Schmidhuber hypothesis.

NOTE
To run this code, you'll need to download and extract the model from the following link: https://github.com/facebookresearch/SpanBERT/issues/38 into models/spanbert_large_with_head/
Rename the .bin file as pytorch_model_old.bin and run configure_model.py, which changes the layer names to match.
'''

# returns all the results for masks in text, denoted by [MASK] in-text, as a list of lists
# each sub-list corresponds to one mask, and contains dicts of possible completions
# each dict contains keys 'score', 'token', 'token_str', and 'sequence'
def span_mask_pipeline(text, model='../models/spanbert_large_with_head/', tokenizer='bert-large-cased'):
    from transformers import pipeline

    unmasker = pipeline('fill-mask', model=model, tokenizer=tokenizer)
    all_mask_completions = unmasker(text)
    return all_mask_completions

# formatting function for printing completions
def print_mask_completions(text, completions):
    print("\nInput: {}".format(text))
    if isinstance(completions[0], dict):
        for completion in completions:
            print("\nScore: {}\nSequence: {}\n".format(completion['score'], completion['sequence']))
    else:
        for mask_completions in completions:
            print("\n")
            for completion in mask_completions:
                print("Score: {}\nCompletion: {}".format(completion['score'], completion['token_str']))
        top_str = top_completion(text, completions)
        print("\nComposite string from top completions for all masks: \n{}".format(top_str))

# returns composite string from top completions for all masks and a list of each of their probabilities
def top_completion(text, completions=None, model='../models/spanbert_large_with_head/', tokenizer='bert-large-cased'):
    if not completions:
        completions = span_mask_pipeline(text=text, model=model, tokenizer=tokenizer)
    if isinstance(completions[0], dict):
        words = [mask_completions['token_str'] for mask_completions in completions]
        probs = [mask_completions['score'] for mask_completions in completions]
    else:
        words = [mask_completions[0]['token_str'] for mask_completions in completions]  # top completions for each mask
        probs = [mask_completions[0]['score'] for mask_completions in completions]  # probabilities for top completions
    s = iter(text.split("[MASK]"))  # splitting input text by [MASK]
    top_str = next(s) + "".join(str(y)+x for x,y in zip(s, words)) # joining text by top completions
    return top_str, probs

# prints top results for the masks in text, denoted by [MASK] in-text
# use the pipeline function instead of this
def span_mask(text, model='../models/spanbert_large_with_head/', tokenizer='bert-large-cased'):
    from transformers import BertForMaskedLM, BertTokenizer
    import torch

    bert = BertForMaskedLM.from_pretrained('../models/spanbert_large_with_head/')
    tokenizer = BertTokenizer.from_pretrained('bert-large-cased')
    inputs = tokenizer(text, return_tensors='pt')
    with torch.no_grad():
        logits = bert(**inputs).logits
    mask_token_index = (inputs.input_ids == tokenizer.mask_token_id)[0].nonzero(as_tuple=True)[0]
    predicted_token_id = logits[0, mask_token_index].argmax(axis=-1)
    print(tokenizer.decode(predicted_token_id))

# returns completions for one mask in text, denoted by [MASK] in-text, as a list of dicts
# each dict contains keys 'score', 'token', 'token_str', and 'sequence'
def mono_mask(text, model='bert-base-uncased'):
    from transformers import pipeline

    unmasker = pipeline('fill-mask', model=model)
    completions = unmasker(text)
    return completions

# function that splits text into sentences
def sent_split(text):
    import nltk
    try:
        nltk.data.find('tokenizers/punkt')
    except:
        nltk.download('punkt')
    return nltk.tokenize.sent_tokenize(text)

# function to mask all but the last n sentences of a given list of sentences
def mask_except_n_sentences(sentences, n, mask="[MASK]"):
    for (index, sent) in enumerate(sentences):
        if index < (len(sentences)-n):
            sent = mask_except_n(sent, 0)
            sentences[index] = sent
    return ' '.join(sentences)

# function to mask all but the last n words of a given piece of text
def mask_except_n(text, n, mask="[MASK]"):
    words = text.split(' ')
    words[:len(words)-n] = [mask for _ in range(len(words)-n)]
    return ' '.join(words)

# function to mask first n words of a given piece of text
def mask_first_n(text, n, mask="[MASK]"):
    words = text.split(' ')
    words[:n] = [mask for _ in range(n)]
    return ' '.join(words)

if __name__ == "__main__":
    model = '../models/spanbert_large_with_head/'
    tokenizer = 'bert-large-cased'

    text = "[MASK] [MASK] to me that you are just as flawed as I am unless you can tell me what you plan to do about it. Afterward you will still have plenty of flaws left, but that's not the point; the important thing is to do better, to keep moving ahead, to take one more step forward. Tsuyoku naritai!"
    sentences = sent_split(text)
    masked_sentences = mask_except_n_sentences(sentences, n=2)
    completions = span_mask_pipeline(text, model=model, tokenizer=tokenizer)
    print_mask_completions(masked_sentences, completions)