#!/bin/bash
git lfs install
git clone https://huggingface.co/rkingery/dumb-language-model
mv dumb-language-model/* models
rm -rf dumb-language-model
pip install -r requirements.txt
