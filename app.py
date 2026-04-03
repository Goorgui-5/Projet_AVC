import streamlit as st
import pandas as pd
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
import warnings
warnings.filterwarnings('ignore')

st.set_page_config(
    page_title="AVC Risk Predictor — Sénégal",
    page_icon="🇸🇳",
    layout="centered"
)

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&family=Syne:wght@700;800&display=swap');

*, *::before, *::after { box-sizing: border-box; }

html, body, .stApp {
    background-color: #F9FAFB;
    font-family: 'Inter', sans-serif;
    color: #111827;
}

/* ── BANDE DRAPEAU ── */
.flag-bar {
    width: 100%;
    height: 4px;
    background: linear-gradient(to right, #00853F 33.3%, #FDEF42 33.3%, #FDEF42 66.6%, #E31B23 66.6%);
    position: fixed;
    top: 0; left: 0;
    z-index: 9999;
}

/* ── HEADER ── */
.header {
    padding: 40px 0 28px;
    border-bottom: 1px solid #E5E7EB;
    margin-bottom: 32px;
}
.header-badge {
    display: inline-flex;
    align-items: center;
    gap: 8px;
    background: #F3F4F6;
    border: 1px solid #E5E7EB;
    border-radius: 100px;
    padding: 5px 14px;
    font-size: 11px;
    font-weight: 600;
    color: #6B7280;
    letter-spacing: 0.08em;
    text-transform: uppercase;
    margin-bottom: 16px;
}
# .flag-mini {
#     width: 18px; height: 12px;
#     border-radius: 2px;
#     background: linear-gradient(to right, #00853F 33%, #FDEF42 33%, #FDEF42 66%, #E31B23 66%);
#     display: inline-block;
#     flex-shrink: 0;
# }
.header h1 {
    font-family: 'Syne', sans-serif;
    font-size: 40px;
    font-weight: 700;
    color: #111827;
    line-height: 1.15;
    margin-bottom: 10px;
}
.header h1 span { color: #2563EB; }
.header p {
    color: #6B7280;
    font-size: 14px;
    line-height: 1.7;
    max-width: 480px;
}

/* ── SECTION LABEL ── */
.section-label {
    font-size: 11px;
    font-weight: 600;
    letter-spacing: 0.1em;
    text-transform: uppercase;
    color: #9CA3AF;
    margin-bottom: 14px;
    margin-top: 8px;
}

/* ── INPUTS ── */
label[data-testid="stWidgetLabel"] p {
    color: #374151 !important;
    font-size: 13px !important;
    font-weight: 500 !important;
}
.stSelectbox [data-baseweb="select"] > div {
    background-color: #FFFFFF !important;
    border: 1px solid #D1D5DB !important;
    border-radius: 10px !important;
    color: #111827 !important;
}
.stSelectbox [data-baseweb="select"] > div:focus-within {
    border-color: #2563EB !important;
    box-shadow: 0 0 0 3px rgba(37,99,235,0.1) !important;
}
.stNumberInput input {
    background-color: #FFFFFF !important;
    border: 1px solid #D1D5DB !important;
    border-radius: 10px !important;
    color: #111827 !important;
    font-size: 14px !important;
}
.stNumberInput input:focus {
    border-color: #2563EB !important;
    box-shadow: 0 0 0 3px rgba(37,99,235,0.1) !important;
}
.stRadio [data-baseweb="radio"] {
    background: #FFFFFF !important;
    border: 1px solid #D1D5DB !important;
    border-radius: 10px !important;
    padding: 10px 16px !important;
}
.stRadio label { color: #374151 !important; font-size: 14px !important; }

/* ── BOUTON ── */
.stButton > button {
    background: #2563EB !important;
    color: white !important;
    font-family: 'Inter', sans-serif !important;
    font-size: 15px !important;
    font-weight: 600 !important;
    border-radius: 10px !important;
    padding: 13px 0 !important;
    border: none !important;
    width: 100% !important;
    letter-spacing: 0.02em !important;
    transition: background 0.2s !important;
}
.stButton > button:hover { background: #1D4ED8 !important; }

/* ── RÉSULTATS ── */
.result-danger {
    background: #FFF5F5;
    border: 1px solid #FECACA;
    border-left: 4px solid #EF4444;
    border-radius: 12px;
    padding: 24px 28px;
    margin: 20px 0 10px;
}
.result-warning {
    background: #FFFBEB;
    border: 1px solid #FDE68A;
    border-left: 4px solid #F59E0B;
    border-radius: 12px;
    padding: 24px 28px;
    margin: 20px 0 10px;
}
.result-safe {
    background: #F0FDF4;
    border: 1px solid #BBF7D0;
    border-left: 4px solid #22C55E;
    border-radius: 12px;
    padding: 24px 28px;
    margin: 20px 0 10px;
}
.res-label {
    font-size: 11px;
    font-weight: 600;
    letter-spacing: 0.1em;
    text-transform: uppercase;
    margin-bottom: 6px;
}
.res-pct {
    font-family: sans-serif;
    font-size: 48px;
    font-weight: 600;
    line-height: 1;
    margin-bottom: 4px;
}
.res-sub { font-size: 13px; margin-top: 4px; }
.c-red    { color: #DC2626; }
.c-yellow { color: #B45309; }
.c-green  { color: #15803D; }

/* ── PROGRESS ── */
.prog-wrap {
    background: #E5E7EB;
    border-radius: 100px;
    height: 6px;
    margin: 10px 0 20px;
    overflow: hidden;
}
.prog-fill { height: 100%; border-radius: 100px; }

/* ── TAGS ── */
.tag {
    display: inline-block;
    border-radius: 7px;
    padding: 6px 12px;
    font-size: 12px;
    font-weight: 500;
    margin: 3px 3px 3px 0;
}
.tag-on  { background: #FEF2F2; border: 1px solid #FECACA; color: #DC2626; }
.tag-off { background: #F9FAFB; border: 1px solid #E5E7EB; color: #9CA3AF; }

/* ── ALERTE URGENCE ── */
.urgence-box {
    background: #FFF5F5;
    border: 1px solid #FECACA;
    border-radius: 12px;
    padding: 18px 22px;
    margin-top: 16px;
}
.urgence-title {
    font-size: 12px;
    font-weight: 600;
    letter-spacing: 0.08em;
    text-transform: uppercase;
    color: #DC2626;
    margin-bottom: 10px;
}
.urgence-box p {
    font-size: 13px;
    color: #6B7280;
    line-height: 1.9;
}
.urgence-num {
    font-size: 14px;
    font-weight: 700;
    color: #DC2626;
    margin-top: 10px;
}

/* ── FOOTER ── */
.footer {
    border-top: 1px solid #E5E7EB;
    padding: 20px 0;
    margin-top: 40px;
    display: flex;
    justify-content: space-between;
}
.footer p { font-size: 12px; color: #9CA3AF; line-height: 1.7; }

/* ── CLEANUP ── */
#MainMenu, footer, header { visibility: hidden; }
.block-container { padding: 0 32px 40px !important; max-width: 760px !important; }
section[data-testid="stSidebar"] { display: none; }
hr { border-color: #E5E7EB !important; }
</style>

<div class="flag-bar"></div>
""", unsafe_allow_html=True)


# ══════════════════════════════════════════════
# MODÈLE
# ══════════════════════════════════════════════
@st.cache_resource
def train_model():
    df = pd.read_csv("healthcare-dataset-stroke-data.csv")
    df = df.drop("id", axis=1).dropna()
    df = df[df['gender'] != 'Other']
    df_ml = df.copy()
    le, encoders = LabelEncoder(), {}
    for col in ['gender', 'ever_married', 'work_type', 'Residence_type', 'smoking_status']:
        df_ml[col] = le.fit_transform(df_ml[col].astype(str))
        encoders[col] = dict(zip(le.classes_, le.transform(le.classes_)))
    features = ['age', 'hypertension', 'heart_disease', 'avg_glucose_level',
                'bmi', 'gender', 'ever_married', 'smoking_status']
    X, y = df_ml[features], df_ml['stroke']
    Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
    m = LogisticRegression(max_iter=1000, class_weight='balanced', random_state=42)
    m.fit(Xtr, ytr)
    return m, encoders

model, encoders = train_model()


# ══════════════════════════════════════════════
# HEADER
# ══════════════════════════════════════════════
st.markdown("""
<div class="header">
    <div class="header-badge">
         <span class="flag-mini"></span>
         🇸🇳 Sénégal &nbsp;·&nbsp; Avril 2026 &nbsp;·&nbsp; Mois de sensibilisation aux AVC (Accident Vasculaire Cérébral).
    </div>
    <h1>Évaluez votre risque <span>d'AVC</span></h1>
    <p>
        L’AVC touche 1 personne sur 4 dans le monde. Grâce à notre modèle de Machine Learning, anticipez les risques avant qu’il ne soit trop tard.<br>
        Renseignez vos informations et obtenez une estimation rapide et personnalisée de votre risque d’AVC.
</div>
""", unsafe_allow_html=True)


# ══════════════════════════════════════════════
# FORMULAIRE
# ══════════════════════════════════════════════
st.markdown('<div class="section-label">Profil patient</div>', unsafe_allow_html=True)
col1, col2 = st.columns(2, gap="medium")

with col1:
    age    = st.slider("Âge", 1, 100, 45)
    gender = st.selectbox("Genre", ["Male", "Female"],
                          format_func=lambda x: "Homme" if x == "Male" else "Femme")
    ever_married = st.selectbox("Statut marital", ["Yes", "No"],
                                format_func=lambda x: "Marié(e)" if x == "Yes" else "Célibataire")

with col2:
    bmi               = st.number_input("IMC (Indice de Masse Corporelle)", 10.0, 60.0, 25.0, 0.1)
    avg_glucose_level = st.number_input("Glycémie moyenne (mg/dL)", 50.0, 300.0, 100.0, 0.5)
    smoking_status    = st.selectbox("Tabagisme", ["never smoked", "formerly smoked", "smokes", "Unknown"],
                                     format_func=lambda x: {
                                         "never smoked"   : "Jamais fumé",
                                         "formerly smoked": "Ex-fumeur",
                                         "smokes"         : "Fumeur actif",
                                         "Unknown"        : "Non renseigné"
                                     }[x])

st.markdown('<div class="section-label" style="margin-top:20px">Antécédents médicaux</div>', unsafe_allow_html=True)
col3, col4 = st.columns(2, gap="medium")

with col3:
    hypertension  = st.radio("Hypertension artérielle", ["Non", "Oui"], horizontal=True)
with col4:
    heart_disease = st.radio("Maladie cardiaque", ["Non", "Oui"], horizontal=True)

st.markdown("<br>", unsafe_allow_html=True)
predict_btn = st.button("Lancer l'analyse prédictive")


# ══════════════════════════════════════════════
# RÉSULTAT
# ══════════════════════════════════════════════
if predict_btn:
    inp = {
        'age'               : age,
        'hypertension'      : 1 if hypertension == "Oui" else 0,
        'heart_disease'     : 1 if heart_disease == "Oui" else 0,
        'avg_glucose_level' : avg_glucose_level,
        'bmi'               : bmi,
        'gender'            : encoders['gender'].get(gender, 0),
        'ever_married'      : encoders['ever_married'].get(ever_married, 0),
        'smoking_status'    : encoders['smoking_status'].get(smoking_status, 0),
    }
    proba = model.predict_proba(pd.DataFrame([inp]))[0][1] * 100

    if proba >= 50:
        st.markdown(f"""
        <div class="result-danger">
            <div class="res-label c-red">Niveau de risque — Élevé</div>
            <div class="res-pct c-red">{proba:.1f}%</div>
            <div class="res-sub c-red">Consultez un médecin rapidement.</div>
        </div>
        <div class="prog-wrap"><div class="prog-fill" style="width:{min(proba,100):.0f}%;background:#EF4444"></div></div>
        """, unsafe_allow_html=True)

    elif proba >= 25:
        st.markdown(f"""
        <div class="result-warning">
            <div class="res-label c-yellow">Niveau de risque — Modéré</div>
            <div class="res-pct c-yellow">{proba:.1f}%</div>
            <div class="res-sub c-yellow">Surveillance médicale conseillée.</div>
        </div>
        <div class="prog-wrap"><div class="prog-fill" style="width:{min(proba,100):.0f}%;background:#F59E0B"></div></div>
        """, unsafe_allow_html=True)

    else:
        st.markdown(f"""
        <div class="result-safe">
            <div class="res-label c-green">Niveau de risque — Faible</div>
            <div class="res-pct c-green">{proba:.1f}%</div>
            <div class="res-sub c-green">Continuez à prendre soin de vous.</div>
        </div>
        <div class="prog-wrap"><div class="prog-fill" style="width:{min(proba,100):.0f}%;background:#22C55E"></div></div>
        """, unsafe_allow_html=True)

    # ── Facteurs ──
    st.markdown('<div class="section-label" style="margin-top:8px">Facteurs identifiés</div>', unsafe_allow_html=True)
    facteurs = [
        (age >= 50,                                       f"Âge ≥ 50 ans ({age} ans)"),
        (hypertension == "Oui",                           "Hypertension artérielle"),
        (heart_disease == "Oui",                          "Maladie cardiaque"),
        (smoking_status in ["smokes", "formerly smoked"], "Tabagisme actif ou passé"),
        (avg_glucose_level > 140,                         f"Glycémie élevée ({avg_glucose_level:.0f} mg/dL)"),
        (bmi > 30,                                        f"Surpoids / Obésité (IMC {bmi:.1f})"),
    ]
    tags = "".join(
        f'<span class="tag {"tag-on" if c else "tag-off"}">{l}</span>'
        for c, l in facteurs
    )
    actifs = sum(1 for c, _ in facteurs if c)
    sous_titre = "Aucun facteur de risque majeur détecté." if actifs == 0 else f"{actifs} facteur(s) de risque identifié(s)."
    st.markdown(f'{tags}<p style="font-size:12px;color:#9CA3AF;margin-top:10px">{sous_titre}</p>', unsafe_allow_html=True)

    # ── Signes d'alerte ──
    st.markdown("""
    <div class="urgence-box">
        <div class="urgence-title">Signes d'alerte AVC</div>
        <p>
            Visage asymétrique &nbsp;·&nbsp; Faiblesse soudaine d'un bras<br>
            Difficulté à parler &nbsp;·&nbsp; Maux de tête brutaux<br>
            Trouble visuel soudain
        </p>
        <div class="urgence-num">Numéro vert Sénégal : 800 00 50 50</div>
    </div>
    """, unsafe_allow_html=True)


# ══════════════════════════════════════════════
# FOOTER
# ══════════════════════════════════════════════
st.markdown("""
<div class="footer">
    <p>Projet Data Science · Logistic Regression · Accuracy 74%<br>
    Outil éducatif — ne remplace pas un avis médical.</p>
    <p style="text-align:right">Orange Digital Center / Sonatel Académie <br> Data Engineer Junior | Data Analyste Junior | ML Engineer Junior <br> Guédiawaye, Dakar, Sénégal</p>
</div>
""", unsafe_allow_html=True)
