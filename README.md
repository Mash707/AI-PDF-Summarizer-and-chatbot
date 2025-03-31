# Basic RAG Agent

## Overview

This project provides a Streamlit based UI to analyze and extract meaningful insights from documents. It leverages **Cohere's AI models** for **document embedding, retrieval-augmented generation (RAG), and summarization**. The tool is particularly useful for understanding PDF research papers by extracting key sections, summarizing content, and answering user queries.

## Features

- **Load and process PDF files**: Uses LangChain’s `PyPDFLoader` to extract and split text from documents.
- **Generate document embeddings**: Utilizes Cohere’s `embed-english-v3.0` model to generate vector representations.
- **Query-based document retrieval**: Implements **cosine similarity** and Cohere’s `rerank-v3.5` model to refine search results.
- **Summarization**: Generates concise summaries of the document using Cohere’s `command-r-plus-08-2024` model.

## File Structure

```
├── main.py  # Core logic for document processing and AI-powered query handling
├── ui.py   # Streamlit based interface for user interaction
├── .env     # Stores API keys (not included in the repository)
```

## Dependencies

Ensure you have the following Python packages installed:

```bash
pip install langchain_community numpy cohere rich python-dotenv argparse streamlit
```

## Setup

1. **Get a Cohere API Key**  
   Sign up at [Cohere](https://cohere.com/) and obtain an API key.

2. **Create a `.env` file**  
   Inside the project directory, create a `.env` file and add:

   ```
   COHERE_API_KEY=your_api_key_here
   ```

3. **Run the UI**  

   To ask a question about a document:

   ```bash
   streamlit run ui.py"
   ```

## How It Works

1. **Load the document**: The `Agent` class extracts and chunks text.
2. **Embed the text**: Cohere’s embedding model generates vector representations.
3. **Process queries**:
   - Uses **cosine similarity** to retrieve the most relevant document chunks.
   - Refines search results with **Cohere’s RAG reranking**.
   - Generates a final response using **Cohere’s chat model**.
4. **Summarization**: Uses Cohere’s chat model to produce a concise summary.

## Future Enhancements

- **Support for additional file formats** (e.g., DOCX, TXT)
- **Integration with other AI models** (e.g., OpenAI, Hugging Face)
- **Add tests** 



