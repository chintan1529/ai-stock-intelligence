AI Financial Intelligence Platform

An AI-powered financial analysis system for Indian equities using Retrieval-Augmented Generation (RAG), sentiment analysis, and large language models.

[Open the AI Stock Assistant](https://cc894189-ai-stock-intelligence.hf.space)


Open the AI Stock Assistant

Features
1. Stock Q&A

Ask natural language questions about any Indian stock

AI provides contextual financial analysis

2. Stock Comparison

Compare two stocks

Get investor-oriented insights

3. Portfolio Analyzer

Input multiple tickers

Receive diversification and risk analysis

AI Architecture

Core Components:

Retrieval-Augmented Generation (RAG)

FAISS vector search

Financial sentiment analysis (FinBERT)

LLM reasoning via Groq API

Real-time financial data via yfinance

Pipeline:

Fetch stock fundamentals and news

Compute sentiment and price trend

Build vector knowledge base

Retrieve relevant context

Generate financial analysis using LLM

Tech Stack

Python

Streamlit

Sentence Transformers

FAISS

FinBERT

Groq LLM API

yfinance

Example Use Cases

Quick stock research

Investment comparison

Portfolio diversification insights

Financial Q&A assistant

Project Structure
ai-stock-intelligence/
│
├── app.py
├── requirements.txt
└── README.md
Author

AI Engineering Student
Focused on building real-world AI systems and financial intelligence tools.
