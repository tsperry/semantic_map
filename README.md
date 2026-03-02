Project: Human-in-the-loop Semantic Data Dictionary Mapper (v1)
Goal
Compare a source data dictionary to a target data dictionary and produce a mapping.csv where each source field is marked as:
	•	mapped (to a target field)
	•	new (no match)
	•	skipped (decide later)
	•	manual (user-entered target)
Inputs
Two CSVs with required columns:
	•	field_name, descriptionOptional:
	•	type, categories (pipe-delimited, e.g. Yes|No|missing)
Core matching logic
	1	Exact ID override
	•	If source.field_name == target.field_name, force that target as top candidate (preselected).
	2	Semantic candidates (otherwise)
	•	Build a per-field text for embeddings, e.g.:
	•	question: {description}\ncategories: {categories_or_none}\ntype: {type_or_none}
	•	Use SentenceTransformers to embed source + target fields and rank targets by cosine similarity.
	3	Rule-based adjustments (predictable behavior)
	•	Apply a type mismatch penalty (optional but recommended).
	•	Apply a category overlap bonus (Jaccard overlap on category tokens).
	4	Return top k candidates (e.g. 5) with scores.
UI (Gradio v1)
	•	Upload source CSV + target CSV
	•	For each source field:
	•	Show source details
	•	Show top 5 suggested target fields (radio list with scores)
	•	Actions: Map, New field, Skip, Manual
	•	At end (or anytime): download mapping.csv
Performance / state
	•	Cache embeddings so you don’t recompute per click:
	•	Precompute target embeddings once per upload.
	•	Precompute source embeddings once per upload.
	•	Store progress + decisions in gr.State.
Outputs
	•	mapping.csv columns (suggested):
	•	source_field, source_description
	•	decision
	•	target_field (blank if new/skip)
	•	score
	•	notes
Milestones
	1	Data loader + normalization (CSV parsing, categories split/clean, fill missing with none)
	2	Embedding + scoring (ST model + cosine; add penalty/bonus; top-k retrieval)
	3	CLI prototype (prove the loop + output mapping)
	4	Gradio UI (review flow + state + download)
	5	Polish (basic validation, README with run instructions, example CSVs)
Tech stack
	•	Python: pandas, numpy, sentence-transformers, (optional) rapidfuzz, gradio
