import os
import openai

openai.api_key = os.getenv('OPENAI_KEY')


def classification(examples, query, labels, model="davinci", search_model="davinci",
         temperature=0.7, num_logprobs=1):

  response = openai.Classification.create(
      search_model=search_model,
      model=model,
      examples=examples,
      query=query,
      labels=labels,
      temperature=temperature,
      logprobs=num_logprobs,
  )

  label = response.label
  logprob = response.completion.choices[0]['logprobs']['token_logprobs'][0]

  return label, logprob


def completion(prompt, engine='text-davinci-001', model=None, response_length=64,
         temperature=0.7, top_p=1, frequency_penalty=0, presence_penalty=0,
         start_text='', restart_text='', stop_seq=None, num_logprobs=1):
    
  response = openai.Completion.create(
      engine=engine,
      prompt=prompt + start_text,
      max_tokens=response_length,
      temperature=temperature,
      top_p=top_p,
      frequency_penalty=frequency_penalty,
      presence_penalty=presence_penalty,
      stop=stop_seq,
      logprobs=num_logprobs,
  )

  answer = response.choices[0]['text']
  token_logprobs = zip(response.choices[0]['logprobs']['tokens'], response.choices[0]['logprobs']['token_logprobs'])
  top_logprobs = response.choices[0]['logprobs']['top_logprobs']
  
  return answer, token_logprobs, top_logprobs


def embedding(input, engine="text-similarity-davinci-001"):
  response = openai.Embedding.create(input=input, engine=engine)

  return response.data[0]["embedding"]