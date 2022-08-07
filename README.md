# Just a Dumb Language Model

Nothing to see here folks. Just a pretty dumb language model. Like seriously, it's pretty dumb. I trained a GPT-like model on Wikitext103 for about 15 hours on a single GPU. Because I was cheap, only a single Nvidia T4 GPU was used. The biggest transformer I could fit into GPU memory was 8 layers deep (about 90M parameters), but other than that it's got a very GPT-like architecture. Do not mistake it for GPT-3 though. It's got the IQ of an AI carrot.

**Update:** You can now use Dumb Language Model online via Huggingface Spaces. Just click [here](https://huggingface.co/spaces/rkingery/dumb-language-model) to waste your time for a few minutes.

## Installation
If you want to run the DLM locally, do the following to get a local copy of the repo setup. You will need to have [Git LFS](https://git-lfs.github.com/) installed to fetch the models. I recommend doing all of this in a virtual environment (e.g. conda).
```
git clone https://github.com/rkingery/dumb-language-model.git
cd dumb-language-model
bash setup.sh
```

## API Instructions
To use the DLM via API you'll first want to run flask inside the repo root directory (it will run at `http://127.0.0.1:5000` by default):
```
FLASK_ENV=development FLASK_APP=app.py flask run
```


Once flask is running, pass to it a json containing the 3 fields.
- `text`: String of text you want to seed the model with. (required)
- `max_len`: Max number of words you want the model to generate. The longer this is the slower it will run. (default=50)
- `temperature`: How much randomness you want the generated output to have. A number between 0 and 1, with 1 being "best" output and 0 completely random. (default=0.5)

Here is an example of how to use the DLM API in python.
```
import requests, json

data = {
    'text': 'This is a story about',
    'max_len': None,
    'temperature': None
}

resp = requests.post("http://127.0.0.1:5000/predict", json=data)
output = json.loads(resp.content)

print(output)
```

## Frontend Instructions
The frontend for this app is streamlit based. To get the UI, do the following. By default it will run in your browser at `http://127.0.0.1:8501`.
```
streamlit run frontend.py
```
