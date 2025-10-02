# app_mobile_tabs_en_best.py — STA (K140) Chest Counter — Mobile tabs + pies + single best player per page (EN)
import pandas as pd
import plotly.express as px
import streamlit as st
from pathlib import Path
from datetime import datetime, timezone
import pytz, re

# ------------- CONFIG -------------
TITLE = "STA (K140) – Chest Counter"
WEEK_NOTE = "Counting window: **Sunday → Saturday** (Europe time)."

st.set_page_config(page_title=TITLE, layout="wide")
BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR / "web_export"
LOGO = BASE_DIR / "logo.png"
TZ = pytz.timezone("Europe/Rome")  # Europe time

SECTIONS = [
    ("Crypts",   "Chest Type"),
    ("Citadels", "Citadel Type"),
    ("Heroics",  "Heroic Type"),
    ("Olympus",  "Olympus Type"),
    ("Epics",    "Epic Type"),
    ("Ancient",  "Ancient Type"),
    ("Other",    "Other Type"),
]

# ------------- HELPERS -------------
def load_csv(name: str) -> pd.DataFrame:
    p = DATA_DIR / f"{name}.csv"
    if not p.exists():
        return pd.DataFrame()
    return pd.read_csv(p, encoding="utf-8")

def normalize_list_df(df: pd.DataFrame):
    if df.empty: 
        return pd.DataFrame(columns=["Player","Type","Quantity"])
    player_col = next((c for c in df.columns if c.lower()=="player"), None)
    qty_col    = next((c for c in df.columns if c.lower()=="quantity"), None)
    type_col   = next((c for c in df.columns if "type" in c.lower()), None)
    if not (player_col and qty_col and type_col):
        cols = [c for c in ["Player","Type","Quantity"] if c in df.columns]
        if len(cols)==3:
            df = df[cols]
        else:
            return pd.DataFrame(columns=["Player","Type","Quantity"])
    else:
        df = df.rename(columns={player_col:"Player", qty_col:"Quantity", type_col:"Type"})
    df["Quantity"] = pd.to_numeric(df["Quantity"], errors="coerce").fillna(0).astype(int)
    df["Player"] = df["Player"].astype(str)
    df["Type"] = df["Type"].astype(str)
    return df

def compute_last_update():
    latest = None
    for p in DATA_DIR.glob("*.csv"):
        ts = datetime.fromtimestamp(p.stat().st_mtime, tz=timezone.utc).astimezone(TZ)
        if (latest is None) or (ts > latest):
            latest = ts
    return latest

def human_delta(a: datetime, b: datetime) -> str:
    if b < a: a,b = b,a
    delta = b - a
    m = int(delta.total_seconds()//60)
    h,m = divmod(m,60); d,h = divmod(h,24)
    parts=[]
    if d: parts.append(f"{d} day{'s' if d!=1 else ''}")
    if h: parts.append(f"{h} hour{'s' if h!=1 else ''}")
    if m and not d: parts.append(f"{m} minute{'s' if m!=1 else ''}")
    return " ".join(parts) if parts else "just now"

# ---- CRYPTS helpers ----
LEVEL_RE = re.compile(r"level\s*(\d+)", re.IGNORECASE)

def to_common_if_plain_crypt(s: str) -> str:
    low = s.lower()
    if any(k in low for k in ["epic","rare","legend","tartarus"]):
        return s
    if "crypt" in low and LEVEL_RE.search(low):
        num = LEVEL_RE.search(low).group(1)
        return f"Common Crypt Level {num}"
    return s

def crypt_subtype(s: str) -> str:
    low = to_common_if_plain_crypt(s).lower()
    if "tartarus" in low: return "Tartarus"
    if "epic" in low:     return "Epic"
    if "rare" in low:     return "Rare"
    if "legend" in low:   return "Legendary"
    if "common crypt level" in low: return "Common"
    return "Other"

def crypt_level(s: str) -> str:
    m = LEVEL_RE.search(s)
    return f"Level {m.group(1)}" if m else "Level ?"

def pie(df: pd.DataFrame, names_col: str, values_col: str, title: str):
    if df.empty or df[values_col].sum() == 0:
        st.info("No data available.")
        return
    fig = px.pie(df, names=names_col, values=values_col, title=title)
    fig.update_layout(legend=dict(orientation='h', yanchor='bottom', y=-0.2, xanchor='center', x=0.5))
    st.plotly_chart(fig, use_container_width=True)

def kpis(df: pd.DataFrame):
    c1, c2, c3 = st.columns(3)
    total = int(df["Quantity"].sum()) if not df.empty else 0
    players = df["Player"].nunique() if not df.empty else 0
    types = df["Type"].nunique() if not df.empty else 0
    c1.metric("Total captured", f"{total:,}".replace(",", "."))
    c2.metric("Players", players)
    c3.metric("Types", types)

def best_player_by_total(df: pd.DataFrame):
    if df.empty:
        return None, 0
    agg = df.groupby("Player", as_index=False)["Quantity"].sum().sort_values("Quantity", ascending=False)
    row = agg.iloc[0]
    return str(row["Player"]), int(row["Quantity"])

# ------------- HEADER -------------
c1, c2 = st.columns([1,6])
with c1:
    if LOGO.exists():
        st.image(str(LOGO), width=56)
with c2:
    st.markdown(f"### **{TITLE}**")  # compact for mobile
    st.caption(WEEK_NOTE)

box1, box2 = st.columns([1,2])
with box1:
    now = datetime.now(TZ)
    st.info(f"**Server Time**\n{now.strftime('%d/%m/%Y %H:%M')}")
with box2:
    lu = compute_last_update()
    if lu:
        st.info(f"**Last Update**\n{lu.strftime('%d/%m/%Y %H:%M')} (*{human_delta(lu, now)} ago*)")
    else:
        st.info("**Last Update**\nNo exports found")

# ------------- GLOBAL FILTERS (Expander) -------------
dfs_all = {name: normalize_list_df(load_csv(name)) for name,_ in SECTIONS}
all_players = sorted(pd.unique(pd.concat([d["Player"] for d in dfs_all.values() if not d.empty], ignore_index=True))) if any(not d.empty for d in dfs_all.values()) else []

with st.expander("Filters", expanded=False):
    players_sel = st.multiselect("Players", all_players, default=all_players)
def apply_player_filter(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty: return df
    if players_sel:
        return df[df["Player"].isin(players_sel)]
    return df

st.divider()

# ------------- TABS -------------
tab_titles = ["🏁 Rules / Scores","🧊 Crypts","🏰 Citadels","⚔️ Heroics","⚡ Olympus","🌟 Epics","🏺 Ancient","📦 Other","📘 Rules Reference"]
tabs = st.tabs(tab_titles)

# --- Rules / Scores ---
with tabs[0]:
    st.subheader("Rules / Scores")
    dn = load_csv("Norme")
    if dn.empty:
        st.warning("Export **Norme.csv** from Excel to view this tab.")
    else:
        ren={}
        for c in dn.columns:
            cl=c.lower()
            if cl=="player": ren[c]="Player"
            elif cl=="score": ren[c]="Score"
            elif cl=="result": ren[c]="Result"
            elif cl=="level": ren[c]="Level"
        dn = dn.rename(columns=ren)
        st.dataframe(dn.reset_index(drop=True), width='stretch', height=420)
        if "Score" in dn.columns:
            dn["Score"] = pd.to_numeric(dn["Score"], errors="coerce").fillna(0).astype(int)
            top = dn.sort_values("Score", ascending=False).head(1)
            if not top.empty:
                st.success(f"🏆 Top scorer: {top['Player'].iloc[0]} — {int(top['Score'].iloc[0])} pts")

    st.markdown("### Total chests captured (overall)")
    tot = load_csv("Foglio1_TotaleD")
    total_value = None
    if not tot.empty:
        s = pd.to_numeric(tot.select_dtypes(include='number').stack(), errors='coerce')
        if not s.empty: total_value = int(s.iloc[0])
    if total_value is None:
        total_value = 0
        for n,_ in SECTIONS:
            d = dfs_all[n]
            total_value += int(d["Quantity"].sum()) if not d.empty else 0
    st.metric("Total chests captured", f"{total_value:,}".replace(",", "."))

# --- Crypts ---
with tabs[1]:
    st.subheader("Crypts")
    df = apply_player_filter(dfs_all["Crypts"])
    if df.empty:
        st.warning("Export **Crypts.csv** from Excel to view this tab.")
    else:
        df["Type"] = df["Type"].apply(to_common_if_plain_crypt)
        kpis(df)
        # Best player by total crypts (not per type)
        best_name, best_qty = best_player_by_total(df)
        if best_name:
            st.success(f"🏆 Best player (total crypts): {best_name} — {best_qty}")
        # Pivot still useful on desktop; keep but compact
        st.markdown("**Totals by player and crypt type** (rotate phone to landscape for a better view)")
        pvt = df.pivot_table(index="Player", columns="Type", values="Quantity", aggfunc="sum", fill_value=0)
        st.dataframe(pvt.sort_index(), width='stretch', height=380)
        # Pies
        d1 = df.copy()
        d1["Level"] = d1["Type"].apply(crypt_level)
        agg = d1.groupby("Level", as_index=False)["Quantity"].sum().sort_values("Quantity", ascending=False)
        pie(agg, "Level", "Quantity", "Crypts by level")
        d2 = df.copy()
        d2["Subtype"] = d2["Type"].apply(crypt_subtype)
        agg2 = d2.groupby("Subtype", as_index=False)["Quantity"].sum().sort_values("Quantity", ascending=False)
        pie(agg2, "Subtype", "Quantity", "Crypts by subtype (Epic / Tartarus / Rare / Common)")

# --- Generic sections with single best player + pie by type ---
def render_generic_tab(name: str, type_label: str):
    st.subheader(name)
    df = apply_player_filter(dfs_all[name])
    if df.empty:
        st.warning(f"Export **{name}.csv** from Excel to view this tab.")
        return
    kpis(df)
    # Best player by total quantity
    best_name, best_qty = best_player_by_total(df)
    if best_name:
        st.success(f"🏆 Best player (total): {best_name} — {best_qty}")
    st.markdown("**Detail (list)**")
    st.dataframe(df.sort_values(['Player','Type']).reset_index(drop=True), width='stretch', height=380)
    st.markdown("**Distribution by type**")
    agg = df.groupby('Type', as_index=False)['Quantity'].sum().sort_values('Quantity', ascending=False)
    pie(agg, 'Type', 'Quantity', f"{name} by type")

with tabs[2]: render_generic_tab("Citadels", "Citadel Type")
with tabs[3]: render_generic_tab("Heroics", "Heroic Type")
with tabs[4]: render_generic_tab("Olympus", "Olympus Type")
with tabs[5]: render_generic_tab("Epics", "Epic Type")
with tabs[6]: render_generic_tab("Ancient", "Ancient Type")
with tabs[7]: render_generic_tab("Other", "Other Type")

# --- Rules Reference ---
with tabs[8]:
    st.subheader("Rules Reference — Troop Grades / Requirements / Scoring")
    st.caption("Tables are scrollable; rotate your phone to landscape for a wider view.")
    st.markdown("**Troop Grades (G Levels)**")
    lg = load_csv("LivelliG"); st.dataframe(lg, width='stretch', height=320)
    st.markdown("**Requirements**")
    rq = load_csv("Requisiti"); st.dataframe(rq, width='stretch', height=320)
    st.markdown("**Scoring Table**")
    pt = load_csv("Punteggi"); st.dataframe(pt, width='stretch', height=320)

# --- MOBILE CSS tweaks ---
st.markdown('''
<style>
.block-container {padding-top: .5rem; padding-left: .5rem; padding-right: .5rem;}
.stTabs [data-baseweb="tab-list"] { flex-wrap: nowrap; overflow-x: auto; gap: .25rem; }
.stTabs [data-baseweb="tab"] { padding: 6px 10px; }
[data-testid="stMetricValue"] { font-size: 1.2rem; }
[data-testid="stMetricLabel"] { font-size: .9rem; }
</style>
''', unsafe_allow_html=True)
