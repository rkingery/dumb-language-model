# Just a Dumb Language Model

Nothing to see here folks. Just a pretty dumb language model.

## Installation
If you want to run the DLM locally, do the following to get a local copy of the repo setup. I recommend doing this in a virtual environment.
```
git clone https://github.com/rkingery/dumb-language-model.git
cd dumb-language-model
pip install -r requirements.txt
```

## API Instructions
To use the DLM via API you can run flask and pass to it a json containing the 3 fields.

- `text`: String of text you want to seed the model with. (required)
- `max_len`: Max number of words you want the model to generate. The longer this is the slower it will run. (default=512)
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
Coming Soon...
