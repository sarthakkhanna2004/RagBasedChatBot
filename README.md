# RAG-Based Chatbot using Large Language Models

This project implements a **Retrieval-Augmented Generation (RAG)** based chatbot that combines
document retrieval with Large Language Models (LLMs) to generate accurate, context-aware responses.
The system reduces hallucinations by grounding responses in relevant retrieved data.

---

## Problem Statement
Traditional LLM-based chatbots generate responses solely based on their training data,
which can lead to hallucinations and lack of domain-specific knowledge.
This project addresses the issue by implementing a **RAG pipeline** that retrieves
relevant information before generating answers.

---

## Solution Overview
The chatbot follows a **Retrieval-Augmented Generation (RAG)** architecture:
1. User query is converted into embeddings
2. Relevant documents are retrieved using vector similarity search
3. Retrieved context is passed to an LLM
4. The LLM generates a grounded and context-aware response

---

## Key Features
- Context-aware responses using document retrieval
- Reduced hallucinations compared to vanilla LLMs
- Interactive chatbot interface
- End-to-end deployed application
- Modular and scalable RAG pipeline

---

## Tech Stack
- Python
- Large Language Models (LLMs)
- Embeddings & Vector Similarity Search
- Retrieval-Augmented Generation (RAG)
- Streamlit
- GitHub & Streamlit Cloud (Deployment)

---

## Application Workflow
1. Document ingestion and preprocessing
2. Embedding generation for documents
3. Vector-based retrieval of relevant context
4. LLM-based response generation
5. User interaction via Streamlit UI

---

## Live Demo
🔗 **Streamlit App:**  
https://ragbasedchatbot-lucpm7xqn6cdtzkfe4wayd.streamlit.app/

---

