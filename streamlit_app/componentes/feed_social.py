import os
import requests
import streamlit as st
from datetime import datetime, timezone, timedelta
from dotenv import load_dotenv

load_dotenv()
BEARER_TOKEN = os.getenv("TWITTER_BEARER_TOKEN", "")

POSTS_CURADOS = [
    {
        "autor": "elonmusk",
        "nome": "Elon Musk",
        "avatar": "🧑‍💼",
        "texto": "Dogecoin is people's crypto. The fees are low and the transaction speed is high. ♥ Doge",
        "data": "há 2h",
        "likes": "142.3K",
        "reposts": "18.7K",
        "fonte": "X (Twitter)"
    },
    {
        "autor": "nayibbukele",
        "nome": "Nayib Bukele",
        "avatar": "🇸🇻",
        "texto": "El Salvador's Bitcoin treasury is up. We will keep buying. 🇸🇻 #Bitcoin",
        "data": "há 5h",
        "likes": "89.1K",
        "reposts": "12.4K",
        "fonte": "X (Twitter)"
    },
    {
        "autor": "CryptoWhale",
        "nome": "Crypto Whale 🐋",
        "avatar": "🐋",
        "texto": "Large wallets have been quietly accumulating ETH over the past 72 hours. Something is coming. 👀 #Ethereum",
        "data": "há 7h",
        "likes": "31.2K",
        "reposts": "5.8K",
        "fonte": "X (Twitter)"
    },
    {
        "autor": "WatcherGuru",
        "nome": "Watcher Guru",
        "avatar": "👁️",
        "texto": "BREAKING: BlackRock's Bitcoin ETF surpasses $20 billion in assets under management.",
        "data": "há 9h",
        "likes": "44.9K",
        "reposts": "9.2K",
        "fonte": "X (Twitter)"
    },
    {
        "autor": "DocumentingBTC",
        "nome": "Documenting Bitcoin",
        "avatar": "📄",
        "texto": "Banks have been declaring Bitcoin as a Ponzi scheme for over a decade. Bitcoin is still here. Banks are still afraid. 🟠",
        "data": "há 11h",
        "likes": "22.6K",
        "reposts": "4.1K",
        "fonte": "X (Twitter)"
    },
]


def _tentar_api_twitter():
    if not BEARER_TOKEN:
        return None
    query = "bitcoin OR ethereum OR crypto lang:en -is:retweet"
    url = (
        "https://api.twitter.com/2/tweets/search/recent"
        f"?query={query}&max_results=5"
        "&tweet.fields=created_at,text,author_id,public_metrics"
        "&expansions=author_id"
        "&user.fields=name,username"
    )
    try:
        r = requests.get(url, headers={"Authorization": f"Bearer {BEARER_TOKEN}"}, timeout=8)
        if r.status_code == 200:
            return r.json()
    except Exception:
        pass
    return None


def mostrar_feed_social():
    st.markdown("""
        <div style='display:flex;align-items:center;gap:10px;margin-bottom:12px;'>
            <span style='font-size:22px;'>💬</span>
            <h4 style='margin:0;color:#e2e8f0;'>Feed Social — Cripto</h4>
        </div>
    """, unsafe_allow_html=True)

    dados_api = _tentar_api_twitter()

    if dados_api and dados_api.get("data"):
        tweets = dados_api["data"]
        users = {u["id"]: u for u in dados_api.get("includes", {}).get("users", [])}
        for tw in tweets:
            user = users.get(tw.get("author_id"), {})
            nome = user.get("name", "Usuário")
            username = user.get("username", "user")
            metrics = tw.get("public_metrics", {})
            likes = metrics.get("like_count", 0)
            reposts = metrics.get("retweet_count", 0)
            created = tw.get("created_at", "")
            st.markdown(f"""
                <div style='background:#141923;border:1px solid #1f2633;border-radius:14px;
                            padding:16px;margin-bottom:10px;'>
                    <div style='display:flex;justify-content:space-between;align-items:center;'>
                        <div>
                            <b style='color:#e2e8f0;'>{nome}</b>
                            <span style='color:#64748b;font-size:13px;'> @{username}</span>
                        </div>
                        <span style='font-size:11px;color:#64748b;'>{created[:10] if created else ''}</span>
                    </div>
                    <p style='color:#cbd5e1;margin:10px 0 8px 0;font-size:14px;'>{tw['text']}</p>
                    <div style='display:flex;gap:16px;'>
                        <span style='color:#64748b;font-size:12px;'>❤️ {likes:,}</span>
                        <span style='color:#64748b;font-size:12px;'>🔁 {reposts:,}</span>
                        <span style='color:#0996ca;font-size:12px;'>X (Twitter)</span>
                    </div>
                </div>
            """, unsafe_allow_html=True)
    else:
        for post in POSTS_CURADOS:
            st.markdown(f"""
                <div style='background:#141923;border:1px solid #1f2633;border-radius:14px;
                            padding:16px;margin-bottom:10px;'>
                    <div style='display:flex;justify-content:space-between;align-items:center;'>
                        <div style='display:flex;align-items:center;gap:8px;'>
                            <span style='font-size:22px;'>{post['avatar']}</span>
                            <div>
                                <b style='color:#e2e8f0;'>{post['nome']}</b>
                                <span style='color:#64748b;font-size:13px;'> @{post['autor']}</span>
                            </div>
                        </div>
                        <span style='font-size:11px;color:#64748b;'>{post['data']}</span>
                    </div>
                    <p style='color:#cbd5e1;margin:10px 0 8px 0;font-size:14px;'>{post['texto']}</p>
                    <div style='display:flex;gap:16px;'>
                        <span style='color:#64748b;font-size:12px;'>❤️ {post['likes']}</span>
                        <span style='color:#64748b;font-size:12px;'>🔁 {post['reposts']}</span>
                        <span style='color:#0996ca;font-size:12px;'>{post['fonte']}</span>
                    </div>
                </div>
            """, unsafe_allow_html=True)
