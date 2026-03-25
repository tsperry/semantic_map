import os
import streamlit as st
import pandas as pd

from embeddings import embed_texts
from matcher import get_top_matches, parse_responses

st.set_page_config(layout="wide")
st.title("Field Harmonization Tool")

@st.cache_data
def load_data():
    return pd.read_csv("data/target.csv"), pd.read_csv("data/source.csv")

@st.cache_data
def get_target_embeddings(df):
    texts = (df["field_id"] + " | " + df["field_desc"]).to_list()
    return embed_texts(texts)

@st.cache_data
def get_source_embedding(text):
    return embed_texts([text])[0]

MAPPING_FILE = "data/mapping.csv"

def load_mappings():
    if os.path.exists(MAPPING_FILE):
        return pd.read_csv(MAPPING_FILE)
    return pd.DataFrame(columns=["source_field_id", "decision", "target_field_id"])

def save_decision(source_id, decision, target_id=None):
    mappings = load_mappings()
    # Remove existing decision if it exists
    mappings = mappings[mappings["source_field_id"] != source_id]
    
    new_row = pd.DataFrame([{
        "source_field_id": source_id,
        "decision": decision,
        "target_field_id": target_id
    }])
    mappings = pd.concat([mappings, new_row], ignore_index=True)
    mappings.to_csv(MAPPING_FILE, index=False)

# Load data
target_df, source_df = load_data()

# Load mappings and merge into source_df
mappings_df = load_mappings()
source_df = source_df.merge(
    mappings_df[["source_field_id", "decision"]], 
    left_on="field_id", 
    right_on="source_field_id", 
    how="left"
)
source_df["decision"] = source_df["decision"].fillna("Unmapped")

# Precompute target embeddings
target_embs = get_target_embeddings(target_df)


left, right = st.columns([1,2])


with left: 
    # select source field
    st.subheader("Select a Source Field")
    selection = st.dataframe(
        source_df[['field_id', 'field_desc', 'decision']],
        use_container_width=True,
        on_select="rerun",
        selection_mode="single-row",
    )

selected_rows = selection.selection.rows
if selected_rows:
    idx = selected_rows[0]
else:
    # Auto-advance to the first unmapped record
    unmapped = source_df[source_df['decision'] == 'Unmapped']
    idx = unmapped.index[0] if not unmapped.empty else 0

row = source_df.iloc[idx]

with right:

    st.subheader("Decision")
    st.write(f"Current Status: **{row['decision']}**")

    act_col1, act_col2, act_col3 = st.columns(3)

    st.subheader("Source Field")
    st.dataframe(source_df.iloc[[idx]][['field_id', 'field_desc']], 
                 use_container_width=True, hide_index=True)

    #st.write(parse_responses(source_df.at[idx,'values']))

    source_text = row["field_id"] + " | " + row["field_desc"]
    source_emb = get_source_embedding(source_text)

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

    with act_col1:
        if st.button("Map to Selected Match", use_container_width=True, type="primary"):
            save_decision(row["field_id"], "Map", selected_match["target_field"])
            st.rerun()
            
    with act_col2:
        if st.button("Add as New", use_container_width=True):
            save_decision(row["field_id"], "New")
            st.rerun()
            
    with act_col3:
        if st.button("Skip / Drop", use_container_width=True):
            save_decision(row["field_id"], "Skip")
            st.rerun()