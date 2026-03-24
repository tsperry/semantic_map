import streamlit as st
import pandas as pd

from embeddings import embed_texts
from matcher import get_top_matches

st.title("Field Harmonization Tool")

# Load data
target_df = pd.read_csv("data/source_30.csv")
source_df = pd.read_csv("data/target_1k.csv")

# Precompute embeddings 
target_texts = (
    target_df["field_id"] + " | " + target_df["field_desc"]
).to_list()

target_embs = embed_texts(target_texts)

# select source field
idx = st.number_input("Select source field index", 0, len(source_df)-1, 0)

row = source_df.iloc[idx]

st.subheader("Source Field")
st.write(row["field_id"])
st.write(row["field_desc"])

source_text = row["field_id"] + " | " + row["field_desc"]
source_emb = embed_texts([source_text])[0]

# get matches
matches = get_top_matches(row, source_emb, target_df, target_embs)

st.subheader("suggested Matches")

for m in matches:
    st.write(f"**{m['target_field']}**")
    st.write(m["description"])
    
    st.write(f"Final Score: {m['score']:.3f}")
    st.write(f"- Desc: {m['desc_sim']:.3f}")
    st.write(f"- Field ID match: {m['field_match']}")
    st.write(f"- Response: {m['response_sim']:.3f}")
    
    st.divider()

