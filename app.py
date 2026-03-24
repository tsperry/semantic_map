import streamlit as st
import pandas as pd

from embeddings import embed_texts
from matcher import get_top_matches

st.title("Field Harmonization Tool")

# Load data
target_df = pd.read_csv("data/target.csv")
source_df = pd.read_csv("data/soruce.csv")

# Precompute embeddings 
target_texts = (
    target_df["field_id"] + " | " + target_df["field_desc"]
).to_list()

target_embs = embed_texts(target_texts)

# select source field
idx = st.number_input("Select source field index", 0, len(source_df)-1, 0)

row = source_df.iloc[idx]

st.subheader("Source Field")
st.write(row["field_name"])
st.write(row["description"])

source_text = row["field_name"] + " | " + row["description"]
source_emb = embed_texts([source_text])[0]

# get matches
matches = get_top_matches(source_emb, target_embs, target_df)

st.subheader("suggested Matches")

for m in matches:
    st.write(f"**{m['target_field']}**")
    st.write(m["description"])
    st.write(f"Score: {m['score']:.3f}")
    st.divider()

