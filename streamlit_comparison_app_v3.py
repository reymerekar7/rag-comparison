import streamlit as st
import base64
import os
import tempfile
import gc
import uuid

from RAG import CohereRAGPipeline, OpenAIRAGPipeline

import pandas as pd

# Initialize session state variables
if "id" not in st.session_state:
    st.session_state.id = uuid.uuid4()
    st.session_state.file_cache = {}
    st.session_state.index = None
    st.session_state.query_engine = None
    st.session_state.selected_model = None  # Track the currently selected model
    st.session_state.api_key = ""  # Track the API key

session_id = st.session_state.id

def reset_chat():
    st.session_state.messages = []
    st.session_state.context = None
    gc.collect()

def display_pdf(file):
    # Opening file from file path
    st.markdown("### PDF Preview")
    base64_pdf = base64.b64encode(file.read()).decode("utf-8")
    pdf_display = f"""<iframe src="data:application/pdf;base64,{base64_pdf}" width="400" height="100%" type="application/pdf" style="height:100vh; width:100%"> </iframe>"""
    st.markdown(pdf_display, unsafe_allow_html=True)

def clear_on_model_switch(selected_option):
    """
    Clears session state variables and API key when the model is switched.
    """
    if st.session_state.selected_model is None or st.session_state.selected_model != selected_option:
        st.session_state.index = None
        st.session_state.query_engine = None
        st.session_state.api_key = ""
        st.session_state.selected_model = selected_option

# Define a dictionary that maps model names to their corresponding pipeline classes
MODEL_PIPELINES = {
    "Cohere ⌘ R+": CohereRAGPipeline,
    "OpenAI GPT-4": OpenAIRAGPipeline,
    # Add more models here as needed
}

# Sidebar for model selection and API key input
with st.sidebar:
    option = st.selectbox("Choose a model to get started", list(MODEL_PIPELINES.keys()))

    # Clear session state and API key on model switch
    clear_on_model_switch(option)

    api_key_label = option.split()[0]  # Dynamically create the API label
    st.header(f"Set your {api_key_label} API Key")
    API_KEY = st.text_input("password", type="password", value=st.session_state.api_key, label_visibility="collapsed")

    if API_KEY != st.session_state.api_key:
        st.session_state.api_key = API_KEY

    st.link_button(f"get one @ {api_key_label} 🔗", f"https://{api_key_label.lower()}.com/api/")
    uploaded_file = st.file_uploader("Choose your `.pdf` file", type="pdf")

    if uploaded_file and API_KEY:

        st.markdown(f"**Uploaded file:** `{uploaded_file.name}`")

        with tempfile.TemporaryDirectory() as temp_dir:
            file_path = os.path.join(temp_dir, uploaded_file.name)
            with open(file_path, "wb") as f:
                f.write(uploaded_file.getvalue())

            file_key = f"{session_id}-{uploaded_file.name}"
            index_key = f"{session_id}-index"

            st.write("Indexing your document...")

            if st.session_state.index is None:
                if os.path.exists(temp_dir):
                    # Dynamically create and initialize the selected model pipeline
                    PipelineClass = MODEL_PIPELINES[option]
                    rag_pipeline = PipelineClass(API_KEY)
                    llm, *pipeline_args = rag_pipeline.initialize_models()

                    st.write("Generating embeddings...")

                    index = rag_pipeline.load_data_and_create_index(temp_dir, pipeline_args[0])
                    query_engine = rag_pipeline.create_query_engine(llm, *pipeline_args[1:])

                    st.write('Embeddings generated!')

                    # Cache the index and query engine
                    st.session_state.index = index
                    st.session_state.query_engine = query_engine
                    st.session_state.file_cache[file_key] = query_engine
                    st.session_state.file_cache[index_key] = index
                else:
                    st.error('Could not find the file you uploaded, please check again')
                    st.stop()
            else:
                query_engine = st.session_state.file_cache[file_key]
                index = st.session_state.file_cache[index_key]

            st.success("Ready to Chat!")
            display_pdf(uploaded_file)

tab1, tab2 = st.tabs(["Chat", "Context"])

with tab1:
    col1, col2 = st.columns([6, 1])

    with col1:
        st.header("Chat with Your Docs!")

    with col2:
        st.button("Clear ↺", on_click=reset_chat)

    chat_container = st.container(height=500)

    # Initialize chat history
    if "messages" not in st.session_state:
        reset_chat()

    # Display chat messages from history on app rerun
    for message in st.session_state.messages:
        with chat_container.chat_message(message["role"]):
            st.markdown(message["content"])

    # Accept user input
    if prompt := st.chat_input("What's up?"):
        # Add user message to chat history
        st.session_state.messages.append({"role": "user", "content": prompt})
        # Display user message in chat message container
        with chat_container.chat_message("user"):
            st.markdown(prompt)

        # Display assistant response in chat message container
        with chat_container.chat_message("assistant"):
            message_placeholder = st.empty()
            full_response = ""
            
            # Simulate stream of response with milliseconds delay
            query_engine = st.session_state.query_engine
            streaming_response = query_engine.query(prompt)
            
            for chunk in streaming_response.response_gen:
                full_response += chunk
                message_placeholder.markdown(full_response + "▌")

            message_placeholder.markdown(full_response)

        # Add assistant response to chat history
        st.session_state.messages.append({"role": "assistant", "content": full_response})

with tab2:
    if prompt:
        # Logic to retrieve top k pieces of context
        index = st.session_state.file_cache[index_key]

        retriever = index.as_retriever(similarity_top_k=3)  # Assuming top_k = 3
        retrieved_context = retriever.retrieve(prompt)

        retrieved_texts = [doc.text for doc in retrieved_context]
        retrieved_scores = [doc.score for doc in retrieved_context]

        df = pd.DataFrame({
            'Retrieved Text': retrieved_texts,
            'Similarity Score': retrieved_scores
        })

        # Display the retrieved context table
        st.markdown("### Retrieved Context and Similarity Scores")
        st.markdown(prompt)
        st.table(df)