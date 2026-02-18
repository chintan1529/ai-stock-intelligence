import streamlit as st
import os
import numpy as np
import yfinance as yf
import faiss
from sentence_transformers import SentenceTransformer
from transformers import pipeline
from openai import OpenAI

# ==============================
# PAGE CONFIG
# ==============================
st.set_page_config(page_title="AI Stock Intelligence", layout="wide")
st.title("AI Financial Intelligence Platform")

# ==============================
# API SETUP
# ==============================
GROQ_API_KEY = st.sidebar.text_input("Enter Groq API Key", type="password")

if not GROQ_API_KEY:
    st.warning("Please enter your Groq API key in the sidebar.")
    st.stop()

client = OpenAI(
    api_key=GROQ_API_KEY,
    base_url="https://api.groq.com/openai/v1"
)

# ==============================
# LOAD MODELS (cached)
# ==============================
@st.cache_resource
def load_models():
    embed_model = SentenceTransformer("all-MiniLM-L6-v2")
    sentiment_model = pipeline(
        "sentiment-analysis",
        model="ProsusAI/finbert"
    )
    return embed_model, sentiment_model

embed_model, sentiment_model = load_models()

# ==============================
# UTILITY FUNCTIONS
# ==============================
def chunk_text(text, chunk_size=300, overlap=50):
    chunks = []
    start = 0
    while start < len(text):
        end = start + chunk_size
        chunk = text[start:end]
        chunks.append(chunk)
        start = end - overlap
    return chunks

def analyze_news_sentiment(news_items):
    results = []
    score_map = {"positive": 1, "neutral": 0, "negative": -1}
    total_score = 0

    for item in news_items[:5]:
        title = item.get("title", "")
        if title.strip() == "":
            continue

        result = sentiment_model(title)[0]
        label = result["label"].lower()
        score = score_map.get(label, 0)

        total_score += score
        results.append((title, label))

    if total_score > 0:
        overall = "Positive"
    elif total_score < 0:
        overall = "Negative"
    else:
        overall = "Neutral"

    return results, overall

def get_financial_insights(info):
    market_cap = info.get("marketCap", 0)
    pe = info.get("trailingPE", None)

    if market_cap < 50000000000:
        size = "Small-cap"
    elif market_cap < 200000000000:
        size = "Mid-cap"
    else:
        size = "Large-cap"

    if pe is None:
        pe_comment = "PE ratio not available."
    elif pe < 15:
        pe_comment = "Valuation appears low."
    elif pe < 30:
        pe_comment = "Valuation appears moderate."
    else:
        pe_comment = "Valuation appears relatively high."

    return (
        f"Market Cap Classification: {size}\n"
        f"PE Interpretation: {pe_comment}"
    )

def get_price_trend_insights(ticker_symbol):
    stock = yf.Ticker(ticker_symbol)
    hist = stock.history(period="1y")

    if hist.empty:
        return "Price trend data not available."

    start_price = hist["Close"].iloc[0]
    end_price = hist["Close"].iloc[-1]

    yearly_return = ((end_price - start_price) / start_price) * 100
    volatility = hist["Close"].pct_change().std() * (252 ** 0.5) * 100

    if yearly_return > 15:
        trend = "strong upward momentum"
    elif yearly_return > 0:
        trend = "moderate positive trend"
    elif yearly_return > -15:
        trend = "slightly negative trend"
    else:
        trend = "strong downward trend"

    return (
        f"1-Year Return: {yearly_return:.2f}%\n"
        f"Annualized Volatility: {volatility:.2f}%\n"
        f"Trend Interpretation: The stock shows a {trend}."
    )

def build_stock_knowledge_base(ticker_symbol):
    stock = yf.Ticker(ticker_symbol)
    info = stock.info

    company_text = (
        "Company: " + str(info.get("longName", "N/A")) + "\n"
        "Sector: " + str(info.get("sector", "N/A")) + "\n"
        "Industry: " + str(info.get("industry", "N/A")) + "\n"
        "Market Cap: " + str(info.get("marketCap", "N/A")) + "\n"
        "PE Ratio: " + str(info.get("trailingPE", "N/A")) + "\n"
        "Revenue: " + str(info.get("totalRevenue", "N/A")) + "\n\n"
        "Business Summary:\n" + str(info.get("longBusinessSummary", "N/A"))
    )

    insights_text = get_financial_insights(info)
    price_text = get_price_trend_insights(ticker_symbol)

    news_items = stock.news or []
    sentiments, overall_sentiment = analyze_news_sentiment(news_items)
    sentiment_text = "Overall News Sentiment: " + overall_sentiment

    news_texts = []
    for title, label in sentiments:
        news_texts.append(f"News: {title}\nSentiment: {label}")

    documents = [company_text, insights_text, price_text, sentiment_text] + news_texts

    all_chunks = []
    for doc in documents:
        if doc.strip():
            all_chunks.extend(chunk_text(doc))

    embeddings = embed_model.encode(all_chunks).astype("float32")
    dimension = embeddings.shape[1]

    index = faiss.IndexFlatL2(dimension)
    index.add(embeddings)

    return index, all_chunks

def search_chunks(query, index, chunks, top_k=3):
    query_vec = embed_model.encode([query]).astype("float32")
    distances, indices = index.search(query_vec, top_k)
    return [chunks[i] for i in indices[0]]

def ask_llm(question, context_chunks):
    context = "\n\n".join(context_chunks)

    prompt = f"""
You are a professional equity research analyst.

Use only the provided context to answer the question.
Do not give ratings or target prices.

Context:
{context}

Question:
{question}

Answer:
"""

    response = client.chat.completions.create(
        model="llama-3.1-8b-instant",
        messages=[
            {"role": "system", "content": "You are a financial analyst."},
            {"role": "user", "content": prompt}
        ],
        temperature=0.3,
    )

    return response.choices[0].message.content

# ==============================
# UI MODES
# ==============================
mode = st.sidebar.selectbox(
    "Choose Mode",
    ["Stock Q&A", "Stock Comparison", "Portfolio Analysis"]
)

# -------- STOCK Q&A --------
if mode == "Stock Q&A":
    ticker = st.text_input("Enter Indian stock ticker (e.g., TCS.NS)")

    if ticker:
        index, chunks = build_stock_knowledge_base(ticker)
        question = st.text_input("Ask a question about the stock")

        if question:
            context = search_chunks(question, index, chunks)
            answer = ask_llm(question, context)

            st.subheader("AI Analysis")
            st.write(answer)

# -------- COMPARISON --------
elif mode == "Stock Comparison":
    t1 = st.text_input("First stock ticker")
    t2 = st.text_input("Second stock ticker")

    if t1 and t2:
        info1 = yf.Ticker(t1).info
        info2 = yf.Ticker(t2).info

        prompt = f"""
Compare the following companies:

Company 1:
Name: {info1.get('longName')}
Sector: {info1.get('sector')}
PE: {info1.get('trailingPE')}

Company 2:
Name: {info2.get('longName')}
Sector: {info2.get('sector')}
PE: {info2.get('trailingPE')}

Provide a comparison for investors.
"""

        response = client.chat.completions.create(
            model="llama-3.1-8b-instant",
            messages=[{"role": "user", "content": prompt}]
        )

        st.write(response.choices[0].message.content)

# -------- PORTFOLIO --------
elif mode == "Portfolio Analysis":
    portfolio = st.text_input("Enter tickers separated by comma")

    if portfolio:
        tickers = portfolio.split(",")
        summary = ""

        for t in tickers:
            info = yf.Ticker(t.strip()).info
            summary += (
                f"Company: {info.get('longName')}\n"
                f"Sector: {info.get('sector')}\n"
                f"PE: {info.get('trailingPE')}\n\n"
            )

        prompt = f"""
Analyze this portfolio:

{summary}

Provide diversification insights and risks.
"""

        response = client.chat.completions.create(
            model="llama-3.1-8b-instant",
            messages=[{"role": "user", "content": prompt}]
        )

        st.write(response.choices[0].message.content)
