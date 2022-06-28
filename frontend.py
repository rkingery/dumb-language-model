import streamlit as st
import torch
import torchtext

from utils import generate_text, get_model, get_vocab, MAX_LEN, TEMPERATURE

st.title('Just a Dumb Language Model')

text = st.text_input('Enter some text (this will be used to seed the language model)', value='Tell me a story about')
max_len = st.number_input('Enter a max number of words to generate', min_value=0, max_value=512, value=20)
temperature = st.number_input('Enter a temperature (the higher it is, the more random the output will be)', 
                              min_value=0., max_value=1., value=0.5)

if st.button('Click to run'):
    max_len = max_len if max_len is not None else MAX_LEN
    temperature = temperature if temperature is not None else TEMPERATURE
    vocab = get_vocab()
    model = get_model()
    generated = generate_text(text, model, vocab, max_len=max_len, temperature=temperature)
    
    st.markdown('### Generated Text')
    st.markdown(f'{generated}')