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
- [ ] **Data Upload:** Add st.file_uploader for dynamic CSV source/target selection.
- [ ] **Persistence:** Save user mapping decisions (Map, Skip, Manual) to a mapping.csv file.
- [ ] **State Management:** Optimize caching/session state to prevent redundant embedding recomputation.
- [ ] **Export:** Add functionality to download the final mapping file.
- [ ] **UI Polish:** Add pagination/filtering for large source datasets.
