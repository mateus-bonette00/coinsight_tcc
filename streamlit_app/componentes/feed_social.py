import os
import requests
import streamlit as st
from dotenv import load_dotenv

load_dotenv()
BEARER_TOKEN = os.getenv("TWITTER_BEARER_TOKEN")

TWEETS_SIMULADOS = [
    {
        "autor": "elonmusk",
        "texto": "Dogecoin é o futuro da economia interplanetária 🚀🌕",
        "data": "2025-07-27 14:25"
    },
    {
        "autor": "nayibbukele",
        "texto": "Bitcoin é agora a moeda oficial para transações internacionais em El Salvador 🇸🇻",
        "data": "2025-07-27 12:00"
    },
    {
        "autor": "CryptoWhale",
        "texto": "Rumores de que grandes fundos estão acumulando Ethereum silenciosamente 👀",
        "data": "2025-07-27 09:45"
    },
    {
        "autor": "WatcherGuru",
        "texto": "China está considerando legalizar pagamentos com stablecoins em setores estratégicos 🇨🇳",
        "data": "2025-07-26 22:15"
    },
]


def mostrar_feed_social():
    st.markdown("### 💬 Feed Social")

    query = "from:elonmusk OR from:nayibbukele OR from:CryptoWhale OR from:WatcherGuru"
    url = (
        "https://api.twitter.com/2/tweets/search/recent"
        f"?query={query}&max_results=5&tweet.fields=created_at,text,author_id"
    )

    headers = {
        "Authorization": f"Bearer {BEARER_TOKEN}"
    }

    try:
        response = requests.get(url, headers=headers)
        response.raise_for_status()
        data = response.json()

        tweets = data.get("data", [])
        if not tweets:
            raise ValueError("Nenhum tweet encontrado.")

        for tweet in tweets:
            st.markdown(f"""
                <div style='background-color:#262730; border-radius:10px; padding:14px; margin-bottom:10px; color:#fff; font-size:0.9rem'>
                    <b>@{tweet['author_id']}:</b> {tweet['text']}<br>
                    <span style='color:#888;'>🕒 {tweet['created_at']}</span>
                </div>
            """, unsafe_allow_html=True)

    except Exception as e:
        st.warning("⚠️ Erro ao buscar tweets ao vivo. Mostrando feed simulado.")

        for tweet in TWEETS_SIMULADOS:
            st.markdown(f"""
                <div style='background-color:#262730; border-radius:10px; padding:14px; margin-bottom:10px; color:#fff; font-size:0.9rem'>
                    <b>@{tweet['autor']}:</b> {tweet['texto']}<br>
                    <span style='color:#888;'>🕒 {tweet['data']}</span>
                </div>
            """, unsafe_allow_html=True)

