import streamlit as st
import pandas as pd

from embeddings import embed_texts
from matcher import get_top_matches, parse_responses

st.set_page_config(layout="wide")
st.title("Field Harmonization Tool")

# Load data
target_df = pd.read_csv("data/target_1k.csv")
source_df = pd.read_csv("data/source_30.csv")

# Precompute embeddings 
target_texts = (
    target_df["field_id"] + " | " + target_df["field_desc"]
).to_list()

target_embs = embed_texts(target_texts)


left, right = st.columns([1,3])


with left: 
    # select source field
    st.subheader("Select a Source Field")
    selection = st.dataframe(
        source_df[['field_id', 'field_desc']],
        use_container_width=True,
        on_select="rerun",
        selection_mode="single-row",
    )

selected_rows = selection.selection.rows
idx = selected_rows[0] if selected_rows else 0

row = source_df.iloc[idx]

with right:
    st.subheader("Source Field")
    st.dataframe(source_df.iloc[[idx]][['field_id', 'field_desc']], 
                 use_container_width=True, hide_index=True)

    st.write(parse_responses(source_df.at[idx,'values']))

    source_text = row["field_id"] + " | " + row["field_desc"]
    source_emb = embed_texts([source_text])[0]

    # get matches
    matches = get_top_matches(row, source_emb, target_df, target_embs)

    st.subheader("Suggested Matches")

    # Build a tidy dataframe from the matches list
    matches_df = pd.DataFrame([{
        "Field ID":      m["target_field"],
        "Description":   m["description"],
        "Score":         round(m["score"], 3),
        "Desc Sim":      round(m["desc_sim"], 3),
        "Field Match":   m["field_match"],
        "Response Sim":  round(m["response_sim"], 3),
    } for m in matches])


    st.caption("Top Matches")
    match_selection = st.dataframe(matches_df[['Field ID', 'Description', 'Score']], 
                 use_container_width=True, 
                 hide_index=True, 
                 on_select="rerun",
                 selection_mode="single-row",
                 )
    selected_match_rows = match_selection.selection.rows
    match_idx = selected_match_rows[0] if selected_match_rows else 0
    selected_match = matches[match_idx]

    st.subheader("Value Comparison")
    val_left, val_right = st.columns(2)

    with val_left:
        st.caption(f"Source: {row['field_id']}")
        source_vals = parse_responses(row["values"])
        st.text("\n".join(source_vals) if source_vals else "—")

    with val_right:
        st.caption(f"Target: {selected_match['target_field']}")
        target_row = target_df[target_df["field_id"] == selected_match["target_field"]].iloc[0]
        target_vals = parse_responses(target_row["values"])
        st.text("\n".join(target_vals) if target_vals else "—")