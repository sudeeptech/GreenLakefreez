import streamlit as st
import json
import numpy as np
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity

# ----------------------------------
# PAGE CONFIG
# ----------------------------------
st.set_page_config(page_title="GreenLake QA", page_icon="💬")
st.title("💬 GreenLake Local QA System (No API)")

# ----------------------------------
# RELOAD BUTTON (for updated QA data)
# ----------------------------------
if st.button("🔄 Reload QA Data"):
    st.cache_resource.clear()
    st.experimental_rerun()

# ----------------------------------
# LOAD QA DATA & LOCAL MODEL
# ----------------------------------
@st.cache_resource(show_spinner=True)
def load_data():
    # Load QA JSON file
    try:
        with open("qa_data.json", "r", encoding="utf-8") as f:
            data = json.load(f)
    except FileNotFoundError:
        st.error("❌ qa_data.json file not found!")
        st.stop()
    except Exception as e:
        st.error(f"Error loading QA data: {e}")
        st.stop()

    # Extract questions and answers
    questions = [item["question"] for item in data]
    answers = [item["answer"] for item in data]

    # Load local frozen embedding model
    model = SentenceTransformer("local_model")

    # Convert all questions to embeddings
    embeddings = model.encode(questions, convert_to_numpy=True)

    return model, questions, answers, embeddings


# Load once (cached)
model, questions, answers, embeddings = load_data()
st.success("✅ Knowledge Base Loaded")

# ----------------------------------
# USER INPUT
# ----------------------------------
query = st.text_input("Ask your question")

if query:
    # Convert user question to vector
    query_vector = model.encode([query], convert_to_numpy=True)

    # Compute cosine similarity
    scores = cosine_similarity(query_vector, embeddings)[0]
    best_match_index = np.argmax(scores)

    # Threshold check
    if scores[best_match_index] > 0.5:
        st.write("### ✅ Answer")
        st.write(answers[best_match_index])
    else:
        st.write("I don't know")
