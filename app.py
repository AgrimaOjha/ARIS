
import os
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["TOKENIZERS_PARALLELISM"] = "false"

import streamlit as st
import pandas as pd
import numpy as np
import joblib
import plotly.graph_objects as go
from dotenv import load_dotenv

from langchain_groq import ChatGroq
from langgraph.graph import StateGraph, END
from typing import TypedDict, List

load_dotenv()

# =========================
# PAGE CONFIG
# =========================
st.set_page_config(
    page_title="ARIS – Agentic Retention Intelligence System",
    layout="wide"
)

# =========================
# LOAD MODEL
# =========================
model = joblib.load("churn_model.pkl")
model_columns = joblib.load("model_columns.pkl")

# =========================
# LLM
# =========================
llm = ChatGroq(model_name="llama-3.1-8b-instant", temperature=0.3)

# =========================
# SAFE RAG
# =========================
retention_kb = [
    {"keywords": ["fiber"], "text": "Fiber issue → Provide service credit + technical fix"},
    {"keywords": ["contract", "month"], "text": "Offer long-term discounted contract"},
    {"keywords": ["electronic", "check"], "text": "Encourage autopay with incentives"},
    {"keywords": ["charges"], "text": "Suggest lower pricing bundle"}
]

# =========================
# HELPERS
# =========================
def clean_feature(f):
    return f.replace("_", " ").replace("InternetService", "Internet").replace("Contract", "Contract Type")

def format_chat_history(history):
    formatted = ""
    for chat in history[-3:]:
        formatted += f"User: {chat['user']}\n"
        formatted += f"Assistant: {chat['assistant']}\n\n"
    return formatted

# =========================
# AGENT STATE
# =========================
class AgentState(TypedDict):
    risk: str
    drivers: List[str]
    context: str
    chat_history: List[dict]
    query: str
    recommendation: str

# =========================
# RETRIEVE
# =========================
def retrieve_node(state: AgentState):
    text = " ".join(state["drivers"]).lower()
    matched = []

    for doc in retention_kb:
        if any(k in text for k in doc["keywords"]):
            matched.append(f"[SOURCE] {doc['text']}")

    context = "\n".join(matched) if matched else "General retention strategy"
    return {**state, "context": context}

# =========================
# PLANNER (🔥 FINAL UPGRADE)
# =========================
def planner_node(state: AgentState):

    chat_context = format_chat_history(state["chat_history"])

    prompt = f"""
You are ARIS (Agentic Retention Intelligence System).

STRICT DOMAIN:
- You ONLY handle topics related to:
  • Customer churn
  • Telecom industry
  • Retention strategies
  • Risk analysis

ALLOWED:
- You ARE allowed to:
  • Explain concepts like "what is churn"
  • Define telecom-related terms
  • Answer basic understanding questions

RESTRICTION:
- If the query is COMPLETELY unrelated (e.g., coding, math, Fibonacci),
  respond with:

"❌ This query is outside the scope of ARIS. This system is restricted to telecom churn and retention analysis only."

CONTEXT:
- Risk Level: {state['risk']}
- Drivers: {state['drivers']}
- Knowledge: {state['context']}

Previous Conversation:
{chat_context}

User Query:
{state['query']}

INSTRUCTIONS:
- Allow concept explanations within domain
- Stay business-focused
- Use retrieved knowledge
- Avoid hallucinations

OUTPUT FORMAT:

1. Executive Summary
2. Problem Understanding
3. Root Causes
4. Intervention Plan
5. Recommendations
6. Business Impact
7. Sources
8. Disclaimer
"""

    response = llm.invoke(prompt)
    return {**state, "recommendation": response.content}

# =========================
# GRAPH
# =========================
builder = StateGraph(AgentState)
builder.add_node("retrieve", retrieve_node)
builder.add_node("planner", planner_node)

builder.set_entry_point("retrieve")
builder.add_edge("retrieve", "planner")
builder.add_edge("planner", END)

agent = builder.compile()

# =========================
# SESSION STATE
# =========================
if "analysis_done" not in st.session_state:
    st.session_state.analysis_done = False

if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

if "agent_result" not in st.session_state:
    st.session_state.agent_result = None

# =========================
# UI
# =========================
st.title("📡 ARIS – Agentic Retention Intelligence System")
st.markdown("---")

col1, col2 = st.columns(2)

with col1:
    tenure = st.slider("Tenure", 0, 72, 24)
    monthly = st.slider("Monthly Charges", 10.0, 120.0, 65.0)
    total = st.number_input("Total Charges", 0.0, 10000.0, 2000.0)

with col2:
    contract = st.selectbox("Contract", ["Month-to-month", "One year", "Two year"])
    internet = st.selectbox("Internet", ["DSL", "Fiber optic", "No"])
    paperless = st.selectbox("Paperless Billing", ["Yes", "No"])

# =========================
# PREDICTION
# =========================
if st.button("🚀 Run Analysis"):

    data = {
        "tenure": tenure,
        "MonthlyCharges": monthly,
        "TotalCharges": total,
        "Charge_per_Tenure": monthly / (tenure + 1),
        "Contract_" + contract: 1,
        "InternetService_" + internet: 1,
        "PaperlessBilling_Yes": 1 if paperless == "Yes" else 0
    }

    df = pd.DataFrame([data]).reindex(columns=model_columns, fill_value=0)

    prob = model.predict_proba(df)[0][1]

    st.session_state.prob = prob
    st.session_state.analysis_done = True

# =========================
# RESULTS
# =========================
if st.session_state.analysis_done:

    prob = st.session_state.prob
    risk_pct = prob * 100
    risk = "High Risk" if prob > 0.7 else "Moderate Risk" if prob > 0.4 else "Low Risk"

    st.subheader("🔎 Risk Summary")
    st.write(f"Churn Probability: {risk_pct:.2f}%")
    st.write(f"Risk Level: {risk}")

    # Gauge
    fig = go.Figure(go.Indicator(
        mode="gauge+number",
        value=risk_pct,
        gauge={'axis': {'range': [0, 100]}}
    ))
    st.plotly_chart(fig, use_container_width=True)

    # Drivers
    importances = model.feature_importances_
    top_idx = np.argsort(importances)[-3:][::-1]
    drivers = [clean_feature(model_columns[i]) for i in top_idx]

    st.subheader("⚠️ Key Drivers")
    for d in drivers:
        st.write("•", d)

    # =========================
    # AGENT
    # =========================
    query = st.text_input("💬 Ask AI", value="Generate full retention strategy")

    if st.button("🧠 Generate Strategy"):

        state = {
            "risk": risk,
            "drivers": drivers,
            "context": "",
            "chat_history": st.session_state.chat_history,
            "query": query,
            "recommendation": ""
        }

        result = agent.invoke(state)

        st.session_state.agent_result = result["recommendation"]

        st.session_state.chat_history.append({
            "user": query,
            "assistant": result["recommendation"][:300]
        })

    if st.session_state.agent_result:
        st.markdown("### 🤖 AI Report")
        st.write(st.session_state.agent_result)

    # =========================
    # FOLLOW-UP
    # =========================
    follow = st.text_input("Ask follow-up")

    if follow:

        state = {
            "risk": risk,
            "drivers": drivers,
            "context": "",
            "chat_history": st.session_state.chat_history,
            "query": follow,
            "recommendation": ""
        }

        result = agent.invoke(state)

        st.session_state.chat_history.append({
            "user": follow,
            "assistant": result["recommendation"][:300]
        })

        st.info(result["recommendation"])

