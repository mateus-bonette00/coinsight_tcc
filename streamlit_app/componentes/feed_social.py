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
        "avatar": "E",
        "texto": "Dogecoin is people's crypto. The fees are low and the transaction speed is high. #Doge",
        "data": "há 2h",
        "likes": "142.3K",
        "reposts": "18.7K",
        "fonte": "X (Twitter)"
    },
    {
        "autor": "nayibbukele",
        "nome": "Nayib Bukele",
        "avatar": "N",
        "texto": "El Salvador's Bitcoin treasury is up. We will keep buying. #Bitcoin",
        "data": "há 5h",
        "likes": "89.1K",
        "reposts": "12.4K",
        "fonte": "X (Twitter)"
    },
    {
        "autor": "CryptoWhale",
        "nome": "Crypto Whale",
        "avatar": "C",
        "texto": "Large wallets have been quietly accumulating ETH over the past 72 hours. Something is coming. #Ethereum",
        "data": "há 7h",
        "likes": "31.2K",
        "reposts": "5.8K",
        "fonte": "X (Twitter)"
    },
    {
        "autor": "WatcherGuru",
        "nome": "Watcher Guru",
        "avatar": "W",
        "texto": "BREAKING: BlackRock's Bitcoin ETF surpasses $20 billion in assets under management.",
        "data": "há 9h",
        "likes": "44.9K",
        "reposts": "9.2K",
        "fonte": "X (Twitter)"
    },
    {
        "autor": "DocumentingBTC",
        "nome": "Documenting Bitcoin",
        "avatar": "D",
        "texto": "Banks have been declaring Bitcoin as a Ponzi scheme for over a decade. Bitcoin is still here. Banks are still afraid.",
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


ICON_HEART   = '<svg xmlns="http://www.w3.org/2000/svg" width="13" height="13" fill="#64748b" viewBox="0 0 16 16"><path d="M8 2.748l-.717-.737C5.6.281 2.514.878 1.4 3.053c-.523 1.023-.641 2.5.314 4.385.92 1.815 2.834 3.989 6.286 6.357 3.452-2.368 5.365-4.542 6.286-6.357.955-1.886.838-3.362.314-4.385C13.486.878 10.4.28 8.717 2.01L8 2.748z"/></svg>'
ICON_REPOST  = '<svg xmlns="http://www.w3.org/2000/svg" width="13" height="13" fill="#64748b" viewBox="0 0 16 16"><path d="M11 5.466V4H5a4 4 0 0 0-3.584 5.777.5.5 0 1 1-.896.446A5 5 0 0 1 5 3h6V1.534a.25.25 0 0 1 .41-.192l2.36 1.966c.12.1.12.284 0 .384l-2.36 1.966a.25.25 0 0 1-.41-.192zm3.81 6.069a.5.5 0 0 1 .38.47V13a5 5 0 0 1-5 5H4v1.466a.25.25 0 0 1-.41.192l-2.36-1.966a.25.25 0 0 1 0-.384l2.36-1.966a.25.25 0 0 1 .41.192V13h6a4 4 0 0 0 3.585-5.777.5.5 0 0 1 .896-.447 5.049 5.049 0 0 1 .328 1.76z"/></svg>'
ICON_CHAT    = '<svg xmlns="http://www.w3.org/2000/svg" width="16" height="16" fill="#9db1cb" viewBox="0 0 16 16"><path d="M2.678 11.894a1 1 0 0 1 .287.801 10.97 10.97 0 0 1-.398 2c1.395-.323 2.247-.697 2.634-.893a1 1 0 0 1 .71-.074A8.06 8.06 0 0 0 8 14c3.996 0 7-2.807 7-6 0-3.192-3.004-6-7-6S1 4.808 1 8c0 1.468.617 2.83 1.678 3.894zm-.493 3.905a21.682 21.682 0 0 1-.713.129c-.2.032-.352-.176-.273-.362a9.68 9.68 0 0 0 .244-.637l.003-.01c.248-.72.45-1.548.524-2.319C.743 11.37 0 9.76 0 8c0-3.866 3.582-7 8-7s8 3.134 8 7-3.582 7-8 7a9.06 9.06 0 0 1-2.347-.306c-.52.263-1.639.742-3.468 1.105z"/></svg>'

def mostrar_feed_social():
    st.markdown(f"""
        <div style='display:flex;align-items:center;gap:10px;margin-bottom:16px;'>
            {ICON_CHAT}
            <h4 style='margin:0;color:#e2e8f0;font-weight:600;'>Feed Social — Cripto</h4>
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
                    <div style='display:flex;gap:16px;align-items:center;'>
                        <span style='color:#64748b;font-size:12px;display:flex;gap:4px;align-items:center;'>{ICON_HEART} {likes:,}</span>
                        <span style='color:#64748b;font-size:12px;display:flex;gap:4px;align-items:center;'>{ICON_REPOST} {reposts:,}</span>
                        <span style='color:#0996ca;font-size:12px;'>X (Twitter)</span>
                    </div>
                </div>
            """, unsafe_allow_html=True)
    else:
        for post in POSTS_CURADOS:
            inicial = post['avatar']
            st.markdown(f"""
                <div style='background:#141923;border:1px solid #1f2633;border-radius:14px;
                            padding:16px;margin-bottom:10px;'>
                    <div style='display:flex;justify-content:space-between;align-items:center;'>
                        <div style='display:flex;align-items:center;gap:10px;'>
                            <div style='width:36px;height:36px;border-radius:50%;background:#1e3a5f;
                                        display:flex;align-items:center;justify-content:center;
                                        color:#06b6d4;font-weight:700;font-size:14px;flex-shrink:0;'>
                                {inicial}
                            </div>
                            <div>
                                <b style='color:#e2e8f0;'>{post['nome']}</b>
                                <span style='color:#64748b;font-size:13px;'> @{post['autor']}</span>
                            </div>
                        </div>
                        <span style='font-size:11px;color:#64748b;'>{post['data']}</span>
                    </div>
                    <p style='color:#cbd5e1;margin:10px 0 8px 0;font-size:14px;line-height:1.5;'>{post['texto']}</p>
                    <div style='display:flex;gap:16px;align-items:center;'>
                        <span style='color:#64748b;font-size:12px;display:flex;gap:4px;align-items:center;'>{ICON_HEART} {post['likes']}</span>
                        <span style='color:#64748b;font-size:12px;display:flex;gap:4px;align-items:center;'>{ICON_REPOST} {post['reposts']}</span>
                        <span style='color:#0996ca;font-size:12px;'>{post['fonte']}</span>
                    </div>
                </div>
            """, unsafe_allow_html=True)
