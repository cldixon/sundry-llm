import streamlit as st 
from typing import List, Tuple
from transformers import AutoTokenizer

def show_decoding(tokenizer, encoding:List[int]) -> List[Tuple[int, str]]:
    """Show encoded/decoded pairs for example sentence."""
    return [(_enc, tokenizer.decode(_enc)) for _enc in encoding]

st.title("Tokenizer Testing")

selected_checkpoint = st.selectbox(
    label="select tokenizer checkpoint", 
    options=["bert-base-uncased", "bert-base-cased", "bert-large-uncased", "bert-large-cased"],
    index=0
)


tokenizer = AutoTokenizer.from_pretrained(selected_checkpoint)
user_input = st.text_input(
    label="enter text for tokenizer here...",
    value="gobbledygook!"
)

encoded = tokenizer(user_input)["input_ids"]

decoded = show_decoding(tokenizer, encoded)
st.markdown("- " + "\n- ".join([
    f"`{_id}: {word}`" for _id, word in decoded
]))

encoded_wo_spec_tokens = tokenizer.encode(user_input, add_special_tokens=False)
num_tokens = len(encoded_wo_spec_tokens)
st.metric(label="tokens", value=num_tokens)