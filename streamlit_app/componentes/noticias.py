# componentes/noticias.py
import os, re, math, unicodedata, requests, streamlit as st
from datetime import datetime, timezone, timedelta
from urllib.parse import urlparse
from dotenv import load_dotenv

load_dotenv()
GNEWS_API_KEY = os.getenv("GNEWS_API_KEY")  # necessário para GNews

# ===== vocabulário =====
BASE_CRYPTO = "(bitcoin OR ethereum OR crypto OR btc OR eth)"
ECON_TERMS  = "(juros OR inflação OR CPI OR PCE OR FOMC OR Fed OR \"Federal Reserve\" OR Copom)"
POLI_TERMS  = "(guerra OR conflito OR sanções OR embargo OR OTAN OR Kremlin)"
INOV_TERMS  = "(ETF OR \"spot ETF\" OR CBDC OR \"moeda digital\" OR blockchain)"
GEOPOL_TERMS = "(" + " OR ".join([
    "Trump","Biden","Harris","Xi Jinping","Putin","Zelensky","Netanyahu","Lula","Bolsonaro",
    "UE","OTAN","Kremlin","Casa Branca","Congresso","Parlamento",
    "eleições","debate","campanha","sanções","embargo",
    "guerra","conflito","ataques","mísseis","cessar-fogo","mobilização","escalada",
    "Ucrânia","Rússia","Gaza","Israel","Irã","Iêmen","Síria","Taiwan","\"Mar do Sul da China\"",
    "FOMC","Fed","Banco Central","Copom","taxa de juros","inflação"
]) + ")"

COUNTRY_HINTS = {
    "US": ["United States","EUA","Fed","FOMC","SEC","Casa Branca"],
    "CN": ["China","PBoC","People's Bank of China","yuan digital"],
    "BR": ["Brasil","BCB","Banco Central do Brasil","Copom"],
    "RU": ["Rússia","Russia","Kremlin","Moscou"],
}

WHITELIST = {
    # globais
    "reuters.com","apnews.com","bbc.com","theguardian.com","aljazeera.com","dw.com",
    "wsj.com","bloomberg.com","ft.com","economist.com","nytimes.com","washingtonpost.com",
    # br/pt
    "valor.globo.com","estadao.com.br","folha.uol.com.br","g1.globo.com","oglobo.globo.com",
    # cripto
    "coindesk.com","cointelegraph.com","theblock.co","decrypt.co"
}
BLACKLIST = {"marketrealist.com","fxstreet.com","ambcrypto.com"}

# ===== utils =====
def _normalize(s:str)->str: return unicodedata.normalize("NFKC", " ".join((s or "").split()))
def _host(u:str)->str:
    try: return (urlparse(u).hostname or "").lower()
    except: return ""
def _now(): return datetime.now(timezone.utc)

def _flag(cc: str|None)->str:
    if not cc or len(cc)!=2: return "🌐"
    base=127397; return chr(ord(cc[0].upper())+base)+chr(ord(cc[1].upper())+base)

def _infer_country(title:str, desc:str, source:str, url:str)->str|None:
    text = " ".join([title or "", desc or "", source or "", url or ""]).lower()
    host = _host(url)
    def has(h:str)->bool:
        h=h.lower(); return (h in text) or (host.endswith(h) if host else False)
    for cc,hints in COUNTRY_HINTS.items():
        for h in hints:
            if has(h): return cc
    if re.search(r"\b(USA|U\.S\.|US)\b", text, re.I): return "US"
    if re.search(r"\b(China|Chinese)\b", text, re.I): return "CN"
    if re.search(r"\b(Brasil|Brazil)\b", text, re.I): return "BR"
    if re.search(r"\b(Russia|Russian|Kremlin)\b", text, re.I): return "RU"
    return None

def _parse_time(s:str)->datetime|None:
    if not s: return None
    try:
        if "T" in s:  # GNews ISO
            return datetime.strptime(s, "%Y-%m-%dT%H:%M:%SZ").replace(tzinfo=timezone.utc)
        s = s[:14]   # GDELT YYYYMMDDHHMMSS
        return datetime.strptime(s, "%Y%m%d%H%M%S").replace(tzinfo=timezone.utc)
    except:
        return None

def _fmt_time(dt:datetime|None)->str:
    if not dt: return ""
    return dt.astimezone(timezone.utc).strftime("%d/%m/%Y %H:%M")

def _q_join(*parts): return _normalize(" AND ".join([p for p in parts if p]))

# ===== fontes =====
def _search_gnews(query:str, lang:str, n:int):
    if not GNEWS_API_KEY: return []
    r = requests.get("https://gnews.io/api/v4/search",
        params={"q":query,"lang":lang,"max":n,"sortby":"publishedAt","token":GNEWS_API_KEY},
        timeout=15)
    r.raise_for_status()
    out=[]
    for a in (r.json() or {}).get("articles",[]) or []:
        out.append({
            "title": _normalize(a.get("title")),
            "description": _normalize(a.get("description")),
            "url": a.get("url") or "",
            "source": (a.get("source") or {}).get("name",""),
            "published": _parse_time(a.get("publishedAt") or "")
        })
    return out

def _search_gdelt(query:str, n:int):
    # restrito a PT/EN; traduz quando possível (googtrans)
    r = requests.get("https://api.gdeltproject.org/api/v2/doc/doc",
        params={
            "query": query,
            "mode": "ArtList",
            "maxrecords": n,
            "format": "JSON",
            "timespan": "3d",
            "sort": "DateDesc",
            "sourcelang": "Portuguese,English",
            "trans": "googtrans"
        },
        timeout=20)
    r.raise_for_status()
    out=[]
    for a in (r.json() or {}).get("articles",[]) or []:
        out.append({
            "title": _normalize(a.get("title")),
            "description": _normalize(a.get("seendesc") or a.get("snippet") or ""),
            "url": a.get("url") or "",
            "source": a.get("domain") or a.get("sourceCountry") or "",
            "published": _parse_time(a.get("seendate") or a.get("date") or "")
        })
    return out

# ===== rerank =====
KW_GEOPOL = re.compile(r"\b(guerra|conflit|sanç|embarg|míssil|cessar[- ]fogo|escalad|ataque|Ucrân|Gaza|Israel|Irã|Taiwan|OTAN|Kremlin)\b", re.I)
KW_LIDERES = re.compile(r"\b(Trump|Biden|Harris|Xi|Putin|Zelensk|Netanyahu|Lula|Bolsonaro)\b", re.I)
KW_MACRO   = re.compile(r"\b(Fed|FOMC|Copom|taxa de juros|inflaç|PCE|CPI)\b", re.I)
KW_CRYPTO  = re.compile(r"\b(crypto|bitcoin|ethereum|btc|eth|ETF|CBDC)\b", re.I)

def _score(item, escopo:str):
    now = _now()
    pub = item["published"] or now
    hours = max(1, (now - pub).total_seconds() / 3600.0)
    recency = 1.0 / (1.0 + hours/12.0)
    host = _host(item["url"])

    source = 0.0
    if host in WHITELIST:  source += 1.0
    if host in BLACKLIST:  source -= 0.8

    text = f"{item['title']} {item['description']}"
    kw = 0.0
    if KW_GEOPOL.search(text): kw += 0.9
    if KW_LIDERES.search(text): kw += 0.6
    if KW_MACRO.search(text):   kw += 0.5
    if KW_CRYPTO.search(text):  kw += 0.8

    if escopo == "geopolitica": kw += (0.6 if (KW_GEOPOL.search(text) or KW_LIDERES.search(text)) else -0.3)
    if escopo == "cripto":      kw += (0.6 if KW_CRYPTO.search(text) else -0.3)

    return 0.55*recency + 0.30*source + 0.15*kw

def _dedup(items):
    seen=set(); out=[]
    for it in items:
        key = (_host(it["url"]), (it["title"] or "").lower().strip())
        if key in seen: continue
        seen.add(key); out.append(it)
    return out

# ===== interface principal =====
def mostrar_noticias_geopoliticas(
    max_articles:int=12,
    termos_extra:list[str]|None=None,
    lang:str|None=None,           # ignorado para GDELT; GNews usa PT/EN
    categoria:str|None=None,
    paises:list[str]|None=None,
    moedas:list[str]|None=None,
    busca_texto:str|None=None,
    titulo:str="🌐 Notícias",
    layout_cols:int=2,
    escopo:str="ambos",                    # "geopolitica" | "cripto" | "ambos"
    provedores:tuple[str,...]=("gdelt","gnews")
):
    st.markdown(f"### {titulo}", unsafe_allow_html=True)

    langs = ["pt","en"]  # PT/EN somente
    cat = (categoria or "Todos").casefold()
    qs=[]

    wants_geo = escopo in ("geopolitica","ambos")
    wants_cry = escopo in ("cripto","ambos")

    if wants_cry:
        if cat in ("todos","econômico","economico"): qs.append(_q_join(BASE_CRYPTO, ECON_TERMS))
        if cat in ("todos","político","politico"):   qs.append(_q_join(BASE_CRYPTO, POLI_TERMS))
        if cat in ("todos","inovação","inovacao"):   qs.append(_q_join(BASE_CRYPTO, INOV_TERMS))
    if wants_geo:
        if cat in ("todos","político","politico"):   qs.append(GEOPOL_TERMS)
        if cat in ("todos","econômico","economico"): qs.append("(FOMC OR Fed OR taxa de juros OR inflação OR Copom)")

    if busca_texto: qs.append("(" + _normalize(busca_texto) + ")")
    if termos_extra:
        extra = " OR ".join([_normalize(t) for t in termos_extra if t])[:120]
        if extra: qs.append("(" + extra + ")")
    if not qs: qs = [GEOPOL_TERMS] if wants_geo else [_q_join(BASE_CRYPTO, ECON_TERMS)]

    raw=[]
    for q in qs:
        if "gnews" in provedores and GNEWS_API_KEY:
            for lg in langs:
                try: raw += _search_gnews(q, lg, min(10, max_articles))
                except: pass
        if "gdelt" in provedores:
            try: raw += _search_gdelt(q, min(20, max_articles*2))
            except: pass

    # filtra: precisa ter data + últimas 72h + remover idiomas inapropriados (GNews já vem PT/EN; GDELT já filtrado)
    cutoff = _now() - timedelta(hours=72)
    raw = [r for r in raw if (r["published"] is not None and r["published"] >= cutoff)]

    if not raw:
        st.info("Nenhuma notícia relevante encontrada (PT/EN, últimas 72h).")
        return

    items = _dedup(raw)
    items.sort(key=lambda x: _score(x, escopo), reverse=True)
    items = items[:max_articles]

    def _card(a):
        title = a.get("title") or ""
        desc  = a.get("description") or ""
        url   = a.get("url") or "#"
        pub   = _fmt_time(a.get("published"))
        src   = a.get("source") or _host(url)
        cc    = _infer_country(title, desc, src, url)
        flag  = _flag(cc)
        st.markdown(f"""
            <div style='background:#141a24;border:1px solid #222a3a;border-radius:12px;padding:14px;margin-bottom:12px;color:#e6f3f1;'>
              <div style="display:flex;justify-content:space-between;align-items:center;">
                <div style="display:flex;gap:10px;align-items:center;">
                  <span style="font-size:20px">{flag}</span>
                  <b style="font-size:1rem">{title}</b>
                </div>
                <span style="font-size:12px;color:#9db1cb;">{pub}</span>
              </div>
              <div style="color:#cbd5e1;margin-top:6px;">{desc}</div>
              <div style="display:flex;justify-content:space-between;align-items:center;margin-top:10px;">
                <i style="color:#9db1cb;">{src}</i>
                <a href="{url}" target="_blank" style="color:#06b6d4;text-decoration:none;">🔗 Abrir notícia</a>
              </div>
            </div>
        """, unsafe_allow_html=True)

    row=[]
    for i, art in enumerate(items):
        row.append(art)
        if len(row)==layout_cols or i==len(items)-1:
            cols = st.columns(layout_cols)
            for j, a in enumerate(row):
                with cols[j]: _card(a)
            row=[]
