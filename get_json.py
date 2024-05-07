from openai import OpenAI
import os
from dateutil.parser import parse
import pandas as pd
from tqdm import tqdm
import pickle
# Set your OpenAI API key
api_key = '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'
client = OpenAI()

OpenAI.api_key = os.getenv('OPENAI_API_KEY')


def get_openai_response(prompt, max_tokens=100):
    response = client.chat.completions.create(
        model="gpt-3.5-turbo-1106",  # Adjust the model as necessary
        messages=[
            {"role": "system", "content": "You are are an expert data analyst. You are part of a research team studying the role of hierarchies in financial earnings calls."},
            {"role": "user", "content": prompt},
        ]
        # max_tokens=500  # Adjust max tokens as necessary
    )
    return response.choices[0].message.content.strip()

with open('prompt.txt','r') as file:
    prompt_start = file.read()

# with open('past', 'wb') as file:
#     pickle.dump(past, file)

with open('past', 'rb') as file:
    old_folders = pickle.load(file)


past = old_folders

folders = os.listdir("MAEC_DATASET")
for i in tqdm(range(len(folders))):
    folder = folders[i]
    if folder in old_folders:
        print("Already done")
        continue
    year, ticker = folder.split('_')
    parsed = parse(year)
    true_date = parsed.strftime('%Y-%m-%d')
    path = f"MAEC_DATASET/{folders[i]}/text.txt"
    with open(path, 'r') as file:
        call = file.read()
    prompt = prompt_start + call
    response= get_openai_response(prompt)
    output = folder + "*\n" + response + ";" + "\n"
    with open("responses.txt", "a+") as file:
        file.write(output)
    past.append(folder)
    if i % 5 == 0:
        with open('past', 'wb') as file:
            pickle.dump(past, file)
    


