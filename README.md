# Field Harmonization Tool

A Streamlit-based utility for mapping source data dictionary fields to a target data dictionary using semantic embeddings.

## Current Functionality
- **Interactive Browser:** View and select source data dictionary fields.
- **Semantic Matching:** Uses pre-computed embeddings to rank top candidate matches from a target dataset.
- **Detailed Comparison:** Side-by-side view of source and target field details (description and categorical values) for manual validation.

## Tech Stack
- **Python:** Data manipulation and analysis.
- **Streamlit:** Interactive web interface.
- **Pandas:** Tabular data processing.
- **Sentence-Transformers:** Semantic vector embedding generation.

### setup
```
pip install sentence-transformers streamlit pandas scikit-learn
```
### run it
```
streamlit run app.py
```

## TODO
- [ ] **value mapping** - add value mapping for categorical fields  
- [ ] **edit field_id** - allow user to edit field_id for new fields
- [ ] **edit field_desc** - allow user to edit field_desc for new fields
- [ ] **edit values** - allow user to edit values for new fields

- [ ] **Data Upload:** Add st.file_uploader for dynamic CSV source/target selection.
- [ ] **UI Polish:** Add pagination/filtering for large source datasets.

### visualization
- [ ] look into st.column_config for dataframe selection




