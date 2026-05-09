# componentes/noticias.py
import re
import time
import unicodedata
import xml.etree.ElementTree as ET
from datetime import datetime, timezone, timedelta
from urllib.parse import urlparse

import requests
import streamlit as st

RSS_FEEDS = [
    ("CoinDesk",       "https://www.coindesk.com/arc/outboundfeeds/rss/",  "CD"),
    ("CoinTelegraph",  "https://cointelegraph.com/rss",                     "CT"),
    ("Decrypt",        "https://decrypt.co/feed",                           "DC"),
    ("The Block",      "https://www.theblock.co/rss.xml",                   "TB"),
]

HEADERS = {"User-Agent": "Mozilla/5.0 (CoinSight TCC/1.0)"}

COUNTRY_HINTS = {
    "US": ["united states", "fed", "fomc", "sec", "white house", "congress", "washington"],
    "CN": ["china", "chinese", "pboC", "beijing", "yuan"],
    "BR": ["brasil", "brazil", "bcb", "copom", "lula"],
    "RU": ["russia", "russian", "kremlin", "moscow", "putin"],
    "EU": ["europe", "european", "ecb", "frankfurt"],
}

ICON_GLOBE = '<svg xmlns="http://www.w3.org/2000/svg" width="16" height="16" fill="#9db1cb" viewBox="0 0 16 16"><path d="M0 8a8 8 0 1 1 16 0A8 8 0 0 1 0 8zm7.5-6.923c-.67.204-1.335.82-1.887 1.855A7.97 7.97 0 0 0 5.145 4H7.5V1.077zM4.09 4a9.267 9.267 0 0 1 .64-1.539 6.7 6.7 0 0 1 .597-.933A7.025 7.025 0 0 0 2.255 4H4.09zm-.582 3.5c.03-.877.138-1.718.312-2.5H1.674a6.958 6.958 0 0 0-.656 2.5h2.49zM4.847 5a12.5 12.5 0 0 0-.338 2.5H7.5V5H4.847zM8.5 5v2.5h2.99a12.495 12.495 0 0 0-.337-2.5H8.5zM4.51 8.5a12.5 12.5 0 0 0 .337 2.5H7.5V8.5H4.51zm3.99 0V11h2.653c.187-.765.306-1.608.338-2.5H8.5zM5.145 12c.138.386.295.744.468 1.068.552 1.035 1.218 1.65 1.887 1.855V12H5.145zm.182 2.472a6.696 6.696 0 0 1-.597-.933A9.268 9.268 0 0 1 4.09 12H2.255a7.024 7.024 0 0 0 3.072 2.472zM3.82 11a13.652 13.652 0 0 1-.312-2.5h-2.49c.062.89.291 1.733.656 2.5H3.82zm6.853 3.472A7.024 7.024 0 0 0 13.745 12H11.91a9.27 9.27 0 0 1-.64 1.539 6.688 6.688 0 0 1-.597.933zM8.5 12v2.923c.67-.204 1.335-.82 1.887-1.855.173-.324.33-.682.468-1.068H8.5zm3.68-1h2.146c.365-.767.594-1.61.656-2.5h-2.49a13.65 13.65 0 0 1-.312 2.5zm2.802-3.5a6.959 6.959 0 0 0-.656-2.5H12.18c.174.782.282 1.623.312 2.5h2.49zM11.27 2.461c.247.464.462.98.64 1.539h1.835a7.024 7.024 0 0 0-3.072-2.472c.218.284.418.598.597.933zM10.855 4a7.966 7.966 0 0 0-.468-1.068C9.835 1.897 9.17 1.282 8.5 1.077V4h2.355z"/></svg>'

def _flag(cc: str | None) -> str:
    if not cc or len(cc) != 2:
        return ICON_GLOBE
    base = 127397
    return chr(ord(cc[0].upper()) + base) + chr(ord(cc[1].upper()) + base)

def _infer_country(text: str) -> str | None:
    tl = text.lower()
    for cc, hints in COUNTRY_HINTS.items():
        for h in hints:
            if h in tl:
                return cc
    return None

def _parse_date(s: str) -> datetime | None:
    if not s:
        return None
    formatos = [
        "%a, %d %b %Y %H:%M:%S %z",
        "%a, %d %b %Y %H:%M:%S GMT",
        "%Y-%m-%dT%H:%M:%SZ",
        "%Y-%m-%dT%H:%M:%S%z",
    ]
    for fmt in formatos:
        try:
            dt = datetime.strptime(s.strip(), fmt)
            if dt.tzinfo is None:
                dt = dt.replace(tzinfo=timezone.utc)
            return dt
        except:
            continue
    return None

def _fmt_time(dt: datetime | None) -> str:
    if not dt:
        return ""
    delta = datetime.now(timezone.utc) - dt
    horas = int(delta.total_seconds() / 3600)
    if horas < 1:
        return "há poucos minutos"
    if horas < 24:
        return f"há {horas}h"
    dias = horas // 24
    return f"há {dias}d"

def _strip_tags(s: str) -> str:
    return re.sub(r"<[^>]+>", "", s or "").strip()

def _normalize(s: str) -> str:
    return unicodedata.normalize("NFKC", " ".join((s or "").split()))

@st.cache_data(ttl=600, show_spinner=False)
def _buscar_rss(url: str) -> list:
    try:
        r = requests.get(url, headers=HEADERS, timeout=12)
        r.raise_for_status()
        root = ET.fromstring(r.content)
        items = root.findall(".//item")
        result = []
        for it in items:
            title = _normalize(_strip_tags(it.findtext("title", "")))
            desc  = _normalize(_strip_tags(it.findtext("description", "") or it.findtext("summary", "")))
            link  = (it.findtext("link", "") or "").strip()
            pub   = _parse_date(it.findtext("pubDate", "") or it.findtext("published", ""))
            if title and link:
                result.append({
                    "title": title,
                    "description": desc[:250] if desc else "",
                    "url": link,
                    "published": pub,
                })
        return result
    except Exception:
        return []

KW_CRIPTO = re.compile(r"\b(bitcoin|ethereum|crypto|btc|eth|blockchain|defi|nft|altcoin|solana|cardano|ada|sol|binance|coinbase|ETF|CBDC)\b", re.I)
KW_GEO    = re.compile(r"\b(fed|fomc|sec|regulation|trump|russia|china|war|sanction|inflation|interest rate|central bank)\b", re.I)

def _score(item: dict, escopo: str) -> float:
    now = datetime.now(timezone.utc)
    pub = item.get("published") or now
    horas = max(1, (now - pub).total_seconds() / 3600.0)
    recency = 1.0 / (1.0 + horas / 12.0)
    text = f"{item.get('title','')} {item.get('description','')}"
    kw = 0.0
    if KW_CRIPTO.search(text): kw += 1.0
    if KW_GEO.search(text):    kw += 0.5
    if escopo == "geopolitica" and KW_GEO.search(text):   kw += 0.8
    if escopo == "cripto"      and KW_CRIPTO.search(text): kw += 0.8
    return 0.6 * recency + 0.4 * (kw / 2.0)

def _dedup(items: list) -> list:
    seen = set()
    out = []
    for it in items:
        key = (it.get("url", ""), (it.get("title") or "").lower().strip()[:60])
        if key in seen:
            continue
        seen.add(key)
        out.append(it)
    return out

def _card(a: dict, fonte: str, emoji_fonte: str):
    title = a.get("title") or ""
    desc  = a.get("description") or ""
    url   = a.get("url") or "#"
    tempo = _fmt_time(a.get("published"))
    cc    = _infer_country(f"{title} {desc}")
    flag  = _flag(cc)
    st.markdown(f"""
        <div style='background:#141a24;border:1px solid #1e2a3a;border-radius:12px;
                    padding:16px;margin-bottom:12px;'>
          <div style="display:flex;justify-content:space-between;align-items:flex-start;gap:8px;">
            <div style="display:flex;gap:10px;align-items:flex-start;flex:1;">
              <span style="font-size:18px;flex-shrink:0;margin-top:2px;">{flag}</span>
              <b style="color:#e2e8f0;font-size:.95rem;line-height:1.4;">{title}</b>
            </div>
            <span style="font-size:11px;color:#64748b;white-space:nowrap;">{tempo}</span>
          </div>
          {'<div style="color:#94a3b8;margin-top:8px;font-size:.84rem;line-height:1.5;">' + desc[:200] + ('…' if len(desc) > 200 else '') + '</div>' if desc else ''}
          <div style="display:flex;justify-content:space-between;align-items:center;margin-top:12px;">
            <span style="color:#64748b;font-size:.78rem;">{emoji_fonte} {fonte}</span>
            <a href="{url}" target="_blank"
               style="color:#06b6d4;text-decoration:none;font-size:.82rem;font-weight:600;">
               Ler notícia →
            </a>
          </div>
        </div>
    """, unsafe_allow_html=True)


def mostrar_noticias_geopoliticas(
    max_articles: int = 6,
    titulo: str = "📰 Notícias",
    escopo: str = "cripto",
    layout_cols: int = 1,
    **kwargs,
):
    st.markdown(f"### {titulo}")

    with st.spinner("Buscando notícias..."):
        todos = []
        for nome, url, emoji in RSS_FEEDS:
            arts = _buscar_rss(url)
            for a in arts:
                a["_fonte"] = nome
                a["_emoji"] = emoji
            todos += arts

    if not todos:
        st.warning("Não foi possível carregar notícias agora. Verifique sua conexão.")
        return

    cutoff = datetime.now(timezone.utc) - timedelta(days=3)
    recentes = [a for a in todos if a.get("published") and a["published"] >= cutoff]
    if not recentes:
        recentes = todos

    recentes = _dedup(recentes)
    recentes.sort(key=lambda x: _score(x, escopo), reverse=True)

    if escopo in ("cripto", "ambos"):
        filtrados = [a for a in recentes if KW_CRIPTO.search(f"{a['title']} {a['description']}")]
    else:
        filtrados = [a for a in recentes if KW_GEO.search(f"{a['title']} {a['description']}")]

    if not filtrados:
        filtrados = recentes

    filtrados = filtrados[:max_articles]

    if layout_cols == 1:
        for a in filtrados:
            _card(a, a.get("_fonte", ""), a.get("_emoji", "📰"))
    else:
        row = []
        for i, a in enumerate(filtrados):
            row.append(a)
            if len(row) == layout_cols or i == len(filtrados) - 1:
                cols = st.columns(layout_cols)
                for j, art in enumerate(row):
                    with cols[j]:
                        _card(art, art.get("_fonte", ""), art.get("_emoji", "📰"))
                row = []
