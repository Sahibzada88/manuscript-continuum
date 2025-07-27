import streamlit as st
from langchain_groq import ChatGroq
from rag_handler import create_vector_store, get_context
from rl_training import RLTrainer
from prompts import HISTORICAL_PROMPT
import os
from dotenv import load_dotenv

# Initialize
load_dotenv()
rl_trainer = RLTrainer()
st.set_page_config(page_title="📜 Manuscript Continuum", layout="wide")

# Sidebar configuration
st.sidebar.title("Configuration")
groq_api_key = st.sidebar.text_input("GROQ API Key", type="password", 
                                     value=os.getenv("GROQ_API_KEY", ""))
os.environ["GROQ_API_KEY"] = groq_api_key

# Initialize session state
if "story" not in st.session_state:
    st.session_state.story = ""
    st.session_state.century = "18th"
    st.session_state.vector_store = None

# UI Elements
st.title("📜 Manuscript Continuum")
st.caption("Collaborative storytelling across centuries")
st.divider()

# Century selection
era = st.selectbox("Select Historical Era", 
                  ["14th", "18th", "19th"], 
                  key="era_selector")

# Create vector store when era changes
if st.session_state.century != era:
    with st.spinner(f"Loading {era} century literature..."):
        st.session_state.century = era
        st.session_state.vector_store = create_vector_store(era)

# Story display
st.subheader("Current Story")
story_placeholder = st.empty()
story_placeholder.markdown(f"`{st.session_state.story}`")

# User input
user_input = st.text_area("Add your contribution:", height=150)
generate_btn = st.button("Continue Story")

# Generation logic
if generate_btn and groq_api_key:
    # Get historical context
    context = get_context(st.session_state.vector_store, 
                         st.session_state.story + user_input)
    
    # Build enhanced prompt
    optimized_prompt = rl_trainer.optimize_prompt(
        HISTORICAL_PROMPT.format(
            century=era,
            context=context,
            story=st.session_state.story,
            new_input=user_input
        )
    )
    
    # Generate continuation
    with st.spinner("Writing in historical style..."):
        continuation = rl_trainer.generate_with_feedback(optimized_prompt)
        st.session_state.story += f"\n\n**User:** {user_input}\n**AI:** {continuation}"
        story_placeholder.markdown(st.session_state.story)
        
        # Store for feedback
        st.session_state.last_continuation = continuation
        st.session_state.last_user_input = user_input

# Feedback system
if "last_continuation" in st.session_state:
    st.divider()
    st.subheader("Improve the AI")
    st.write("Rate the last continuation to help the AI learn:")
    
    col1, col2 = st.columns(2)
    if col1.button("👍 Good Continuation"):
        rl_trainer.save_feedback(
            era,
            st.session_state.last_user_input,
            st.session_state.last_continuation,
            1  # Positive
        )
        st.success("Thanks! This will improve future writing.")
        del st.session_state.last_continuation
        
    if col2.button("👎 Needs Improvement"):
        rl_trainer.save_feedback(
            era,
            st.session_state.last_user_input,
            st.session_state.last_continuation,
            0  # Negative
        )
        st.info("Thanks! The AI will learn from this.")
        del st.session_state.last_continuation

# Instructions
if not st.session_state.story:
    st.divider()
    st.subheader("How to Use:")
    st.markdown("""
    1. Select a historical era (14th, 18th, or 19th century)
    2. Add your story continuation in the text box
    3. Click "Continue Story" to generate AI-written text
    4. Rate the AI's writing to help it improve
    5. Continue building the story collaboratively!
    """)