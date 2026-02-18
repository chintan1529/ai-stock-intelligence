# AI Financial Intelligence Platform

AI-powered stock analysis system for Indian equities using Retrieval-Augmented Generation (RAG), sentiment analysis, and large language models.

## Live Demo

[Launch the App](https://cc894189-ai-stock-intelligence.hf.space)

---

## Overview

This project is a real-world AI financial assistant that combines live stock data, news sentiment, semantic search, and large language models to generate investor-oriented insights.

Users can:

* Ask questions about any Indian stock
* Compare two stocks
* Analyze a portfolio

The system uses a RAG pipeline to ensure answers are grounded in real financial data.

---

## Key Features

### Stock Q&A

Ask natural language questions like:

* “What does TCS do?”
* “Is this stock overvalued?”
* “What are the risks of this company?”

The AI responds using real fundamentals and news context.

---

### Stock Comparison

Compare two companies and receive:

* Sector analysis
* Valuation comparison
* Investor suitability insights

---

### Portfolio Analyzer

Input multiple tickers and get:

* Diversification analysis
* Sector exposure insights
* Risk observations

---

## AI Architecture

### Core Components

* Retrieval-Augmented Generation (RAG)
* FAISS vector search
* Financial sentiment analysis (FinBERT)
* LLM reasoning via Groq API
* Real-time stock data via yfinance

### Pipeline

1. Fetch stock fundamentals and news
2. Compute sentiment and price trend
3. Convert data into semantic chunks
4. Store embeddings in FAISS index
5. Retrieve relevant context for user query
6. Generate analysis using LLM

---

## Tech Stack

**AI & NLP**

* Sentence Transformers (MiniLM)
* FinBERT sentiment model
* Groq-hosted LLM

**Backend**

* Python
* FAISS
* yfinance

**Frontend**

* Streamlit

**Deployment**

* Hugging Face Spaces

---

## Project Structure

```
ai-stock-intelligence/
│
├── app.py
├── requirements.txt
└── README.md
```

---

## Example Use Cases

* Quick stock research assistant
* Retail investor analysis tool
* Portfolio diversification insights
* Financial Q&A chatbot

---

## Author

AI Engineering student focused on building real-world AI systems in finance and applied machine learning.
