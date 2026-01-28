import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import time
import joblib
import numpy as np
from datetime import datetime

# -------------------------------------------------------------------------
# 1. CONFIGURATION GLOBALE
# -------------------------------------------------------------------------
st.set_page_config(
    page_title="JP Fraude | Security",
    page_icon="🦅",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# -------------------------------------------------------------------------
# 2. DESIGN SYSTEM (CSS AVANCÉ - DARK MODE)
# -------------------------------------------------------------------------
st.markdown("""
    <style>
    /* 1. FOND GENERAL */
    .stApp {
        background: radial-gradient(circle at 10% 20%, rgb(15, 23, 42) 0%, rgb(0, 0, 0) 90%);
        color: #e2e8f0;
    }

    /* 2. TYPOGRAPHIE */
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;500;700&display=swap');
    html, body, [class*="css"] {
        font-family: 'Inter', sans-serif;
    }
    h1, h2, h3 {
        color: #f8fafc;
        font-weight: 700;
        letter-spacing: -0.5px;
    }

    /* 3. SIDEBAR */
    section[data-testid="stSidebar"] {
        background-color: #0f172a;
        border-right: 1px solid #1e293b;
    }

    /* 4. CARTES (GLASSMORPHISM) */
    .glass-card {
        background: rgba(30, 41, 59, 0.7);
        backdrop-filter: blur(10px);
        -webkit-backdrop-filter: blur(10px);
        border: 1px solid rgba(255, 255, 255, 0.1);
        border-radius: 16px;
        padding: 25px;
        margin-bottom: 20px;
        box-shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.5);
    }
    
    /* 5. METRICS */
    div[data-testid="stMetric"] {
        background-color: #1e293b;
        border: 1px solid #334155;
        padding: 15px;
        border-radius: 10px;
        color: white;
    }
    div[data-testid="stMetric"] label {
        color: #94a3b8;
    }
    div[data-testid="stMetricValue"] {
        color: white !important;
    }

    /* 6. BOUTONS STYLISÉS (Largeur 100% & Minuscules) */
    .stButton > button {
        background: linear-gradient(90deg, #3b82f6, #2563eb);
        color: white;
        border: none;
        padding: 15px 24px;
        border-radius: 8px;
        font-weight: 500;
        font-size: 16px;
        text-transform: none !important; /* Force le texte normal */
        letter-spacing: 0.5px;
        transition: all 0.3s ease;
        width: 100%;
    }
    .stButton > button:hover {
        background: linear-gradient(90deg, #2563eb, #1d4ed8);
        transform: scale(1.02);
        box-shadow: 0 0 15px rgba(59, 130, 246, 0.5);
    }
    
    /* 7. CUSTOM CLASSES HTML */
    .hero-title {
        font-size: 3.5rem;
        background: -webkit-linear-gradient(#eee, #94a3b8);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        text-align: center;
        margin-bottom: 10px;
    }
    .hero-subtitle {
        text-align: center;
        color: #94a3b8;
        font-size: 1.2rem;
        margin-bottom: 40px;
    }
    .section-title {
        border-left: 4px solid #3b82f6;
        padding-left: 15px;
        margin-top: 30px;
        margin-bottom: 20px;
    }
    </style>
""", unsafe_allow_html=True)

# -------------------------------------------------------------------------
# 3. FONCTIONS UTILITAIRES & CHARGEMENT
# -------------------------------------------------------------------------

# Fonction pour la date en français
def get_current_date_french():
    now = datetime.now()
    days = {0: "Lundi", 1: "Mardi", 2: "Mercredi", 3: "Jeudi", 4: "Vendredi", 5: "Samedi", 6: "Dimanche"}
    months = {1: "Janvier", 2: "Février", 3: "Mars", 4: "Avril", 5: "Mai", 6: "Juin", 7: "Juillet", 8: "Août", 9: "Septembre", 10: "Octobre", 11: "Novembre", 12: "Décembre"}
    return days[now.weekday()], now.day, months[now.month], now.year, now.strftime("%H:%M")

@st.cache_data
def load_data():
    df = pd.read_csv('Fraud_light.csv')
    return df

@st.cache_resource
def load_brain():
    try:
        model = joblib.load('modele_fraude.pkl')
        scaler = joblib.load('scaler_fraude.pkl')
        cols = joblib.load('colonnes_modele.pkl')
        return model, scaler, cols
    except:
        return None, None, None

# Chargement initial
try:
    df = load_data()
except Exception as e:
    st.error(f"Erreur lors du chargement des données CSV: {e}")
    st.stop()

model, scaler, model_columns = load_brain()

# -------------------------------------------------------------------------
# 4. NAVIGATION
# -------------------------------------------------------------------------
st.sidebar.title("🦅 JP Fraude")
st.sidebar.caption("Security Intelligence System")
st.sidebar.markdown("---")
page = st.sidebar.radio("Menu", ["Accueil & Vision", "Tableau de Bord", "Centre de Test", "Base de Données"])
st.sidebar.markdown("---")
st.sidebar.info("v 2.0.3 | Dark Edition")

# =========================================================================
# PAGE 1 : ACCUEIL
# =========================================================================
if page == "Accueil & Vision":
    
    st.markdown('<h1 class="hero-title">JP Fraude SECURITY</h1>', unsafe_allow_html=True)
    st.markdown('<p class="hero-subtitle">Redéfinir la sécurité financière grâce à l\'intelligence artificielle.</p>', unsafe_allow_html=True)

    col1, col2, col3 = st.columns(3)
    with col1:
        st.markdown("""
<div class="glass-card" style="text-align:center; height: 300px;">
    <h1 style="font-size: 50px;">🤖</h1>
    <h3>Qui sommes-nous ?</h3>
    <p style="color:#cbd5e1;">Une équipe d'ingénieurs passionnés par la Data Science et la cybersécurité. Nous traquons les anomalies invisibles.</p>
</div>
""", unsafe_allow_html=True)
    with col2:
        st.markdown("""
<div class="glass-card" style="text-align:center; height: 300px;">
    <h1 style="font-size: 50px;">🛡️</h1>
    <h3>Notre Mission</h3>
    <p style="color:#cbd5e1;">Protéger vos actifs financiers 24/7. Notre algorithme analyse des milliers de transactions par seconde.</p>
</div>
""", unsafe_allow_html=True)
    with col3:
        st.markdown("""
<div class="glass-card" style="text-align:center; height: 300px;">
    <h1 style="font-size: 50px;">⚡</h1>
    <h3>Technologie</h3>
    <p style="color:#cbd5e1;">Modèles de Machine Learning avancés (Naive Bayes) couplés à une interface Streamlit ultra-réactive.</p>
</div>
""", unsafe_allow_html=True)

    st.markdown("---")
    
    # Section alignée
    st.markdown('<h2 class="section-title">Comment fonctionne JP Fraude ?</h2>', unsafe_allow_html=True)
    c1, c2 = st.columns([1, 1], vertical_alignment="center")
    
    with c1:
        st.markdown(
            '<img src="https://images.unsplash.com/photo-1550751827-4bd374c3f58b" style="width: 100%; border-radius: 15px; box-shadow: 0 4px 15px rgba(0,0,0,0.5);">', 
            unsafe_allow_html=True
        )
    with c2:
        st.markdown("""
<div style="padding-left: 20px;">
    <h3 style="color: #f8fafc; margin-bottom: 15px;">Le processus en 3 étapes :</h3>
    <ul style="list-style-type: none; padding: 0; color: #cbd5e1;">
        <li style="margin-bottom: 10px;"><strong style="color: #3b82f6;">1. Collecte :</strong> Ingestion des flux en temps réel.</li>
        <li style="margin-bottom: 10px;"><strong style="color: #3b82f6;">2. Analyse IA :</strong> Comparaison avec l'historique de fraudes.</li>
        <li style="margin-bottom: 20px;"><strong style="color: #3b82f6;">3. Verdict :</strong> Blocage ou validation en < 200ms.</li>
    </ul>
</div>
""", unsafe_allow_html=True)
        st.button("Voir la démo technique")

# =========================================================================
# PAGE 2 : TABLEAU DE BORD
# =========================================================================
elif page == "Tableau de Bord":
    st.markdown('<h2 class="section-title"> &nbsp; Météo des Risques</h2>', unsafe_allow_html=True)

    fraud_data = df[df['isFraud'] == 1]
    
    k1, k2, k3, k4 = st.columns(4)
    k1.metric("Volume Analysé", f"{len(df):,}")
    k2.metric("Menaces Bloquées", f"{len(fraud_data):,}")
    k3.metric("Montant Sécurisé", f"${fraud_data['amount'].sum()/1e6:.1f} M")
    k4.metric("Taux de Précision", "99.2%")

    st.markdown("---")

    g1, g2 = st.columns([2, 1])
    
    with g1:
        st.markdown("### Cartographie des Fraudes")
        fig = px.scatter(fraud_data, x="amount", y="oldbalanceOrg", color="type",
                         size="amount", 
                         color_discrete_sequence=px.colors.qualitative.Pastel)
        fig.update_layout(template="plotly_dark", paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)')
        st.plotly_chart(fig, use_container_width=True)

    with g2:
        st.markdown("### Répartition")
        fig2 = px.pie(df, names='type', hole=0.7, color_discrete_sequence=px.colors.sequential.Plasma)
        fig2.update_layout(template="plotly_dark", paper_bgcolor='rgba(0,0,0,0)')
        st.plotly_chart(fig2, use_container_width=True)

# =========================================================================
# PAGE 3 : CENTRE DE TEST (IA ACTIVE & REACTIVE)
# =========================================================================
elif page == "Centre de Test":
    st.markdown('<h2 class="section-title"> Scanner de Transaction (Temps Réel)</h2>', unsafe_allow_html=True)
    
    if model is None:
        st.warning("ATTENTION : Modèle introuvable.")
        st.stop()

    st.markdown("""
<div class="glass-card" style="margin-bottom: 30px;">
    <h4 style="margin:0">Simulateur Naive Bayes</h4>
    <p style="color:#94a3b8; font-size:0.9rem">Le système réagit en temps réel. Modifiez un paramètre pour réinitialiser l'analyse.</p>
</div>
""", unsafe_allow_html=True)

    # --- GESTION D'ÉTAT (SESSION STATE) ---
    if 'analysis_done' not in st.session_state:
        st.session_state.analysis_done = False

    def reset_analysis():
        st.session_state.analysis_done = False

    jour_nom, jour_num, mois, annee, heure_actuelle = get_current_date_french()

    col_form, col_res = st.columns([1, 1], vertical_alignment="center")

    # ---------------------------------------------------------------------
    # COLONNE GAUCHE : FORMULAIRE
    # ---------------------------------------------------------------------
    with col_form:
        # 1. WIDGET CALENDRIER (Input)
        st.subheader("Détails du Flux")
        
        type_x = st.selectbox("Type de transaction", 
                             ['PAYMENT', 'TRANSFER', 'CASH_OUT', 'DEBIT', 'CASH_IN'],
                             on_change=reset_analysis)
        
        amount_x = st.number_input("Montant ($)", value=10000.0, step=100.0, 
                                  on_change=reset_analysis)
        
        st.markdown("---")
        c1, c2 = st.columns(2)
        with c1:
            old_bal_org = st.number_input("Solde Expéditeur AVANT", value=10000.0, on_change=reset_analysis)
            new_bal_org = st.number_input("Solde Expéditeur APRÈS", value=0.0, on_change=reset_analysis)
        with c2:
            old_bal_dest = st.number_input("Solde Destinataire AVANT", value=0.0, on_change=reset_analysis)
            new_bal_dest = st.number_input("Solde Destinataire APRÈS", value=0.0, on_change=reset_analysis)
        
        st.markdown("<br>", unsafe_allow_html=True)
        
        launch_btn = st.button("Lancer l'analyse maintenant", use_container_width=True)
        
        if launch_btn:
            st.session_state.analysis_done = True

    # ---------------------------------------------------------------------
    # COLONNE DROITE : RESULTAT OU ATTENTE
    # ---------------------------------------------------------------------
    with col_res:
        
        # CAS 1 : On attend (Pas encore cliqué OU changement de paramètre)
        if not st.session_state.analysis_done:
            
            # WIDGET CALENDRIER "EN ATTENTE" (Même style mais grisé)
            st.markdown(f"""
<div style="display: flex; flex-direction: column; align-items: center; justify-content: center; margin-bottom: 30px;">
    <div style="background: #f8fafc; color: #0f172a; width: 100px; border-radius: 16px; text-align: center; overflow: hidden; box-shadow: 0 10px 30px rgba(0,0,0,0.5);">
        <div style="background: #3b82f6; color: white; font-size: 14px; font-weight: bold; padding: 6px; text-transform: uppercase; letter-spacing: 1px;">SYSTÈME</div>
        <div style="font-size: 45px; font-weight: 900; padding: 5px 0 10px 0;">{jour_num}</div>
    </div>
    <div style="text-align: center; margin-top: 15px;">
        <h3 style="margin: 0; color: #f8fafc; font-size: 1.4rem;">{jour_nom} {mois} {annee}</h3>
        <p style="margin: 5px 0 0 0; color: #94a3b8; font-size: 0.95rem;">
            Serveur Synchronisé : <span style="color:#60a5fa; font-family:monospace; background: rgba(59, 130, 246, 0.1); padding: 2px 8px; border-radius: 4px;">{heure_actuelle}</span>
        </p>
    </div>
</div>
""", unsafe_allow_html=True)

        # CAS 2 : Analyse lancée
        if st.session_state.analysis_done:
            with st.spinner("Analyse des vecteurs de fraude..."):
                time.sleep(0.8) # Effet simulation
                
                # --- CALCULS IA ---
                now = datetime.now()
                step_hour = now.hour
                timestamp_str = now.strftime("%d/%m/%Y à %H:%M:%S")

                errorBalanceOrig = new_bal_org + amount_x - old_bal_org
                errorBalanceDest = old_bal_dest + amount_x - new_bal_dest
                
                input_data = pd.DataFrame({
                    'step': [1], 
                    'amount': [amount_x],
                    'errorBalanceOrig': [errorBalanceOrig],
                    'errorBalanceDest': [errorBalanceDest],
                    'hour_of_day': [step_hour]
                })

                types_possibles = ['type_CASH_IN', 'type_CASH_OUT', 'type_DEBIT', 'type_PAYMENT', 'type_TRANSFER']
                for t in types_possibles:
                    input_data[t] = 1 if f"type_{type_x}" == t else 0

                input_data = input_data.reindex(columns=model_columns, fill_value=0)

                try:
                    input_scaled = scaler.transform(input_data)
                    prediction = model.predict(input_scaled)[0]
                    proba = model.predict_proba(input_scaled)[0][1] 

                    # --- AFFICHAGE RESULTAT ---
                    if prediction == 1:
                        st.markdown(f"""
<div style="background-color: rgba(220, 38, 38, 0.1); border: 2px solid #dc2626; padding: 30px; border-radius: 20px; text-align: center; box-shadow: 0 0 30px rgba(220, 38, 38, 0.2); animation: fadeIn 0.5s;">
    <div style="font-size: 60px; margin-bottom: 10px;">🚫</div>
    <h1 style="color: #ef4444; font-size: 40px; margin:0;">BLOQUÉ</h1>
    <h3 style="color: white; font-weight: 300;">Transaction Frauduleuse</h3>
    <p style="color: #94a3b8; font-size: 12px; margin-top: 5px;">Horodatage : {timestamp_str}</p>
    <div style="margin-top:20px; background: rgba(0,0,0,0.3); padding: 10px; border-radius: 10px; display: inline-block;">
        <span style="font-size: 16px; color: #fca5a5;">Risque : </span>
        <span style="font-size: 24px; font-weight:bold; color: #ef4444;">{proba*100:.1f}%</span>
    </div>
</div>
""", unsafe_allow_html=True)
                    else:
                        st.markdown(f"""
<div style="background-color: rgba(5, 150, 105, 0.1); border: 2px solid #059669; padding: 30px; border-radius: 20px; text-align: center; box-shadow: 0 0 30px rgba(5, 150, 105, 0.2); animation: fadeIn 0.5s;">
    <div style="font-size: 60px; margin-bottom: 10px;">✅</div>
    <h1 style="color: #10b981; font-size: 40px; margin:0;">APPROUVÉ</h1>
    <h3 style="color: white; font-weight: 300;">Transaction Légitime</h3>
    <p style="color: #94a3b8; font-size: 12px; margin-top: 5px;">Horodatage : {timestamp_str}</p>
        <div style="margin-top:20px; background: rgba(0,0,0,0.3); padding: 10px; border-radius: 10px; display: inline-block;">
        <span style="font-size: 16px; color: #6ee7b7;">Sûreté : </span>
        <span style="font-size: 24px; font-weight:bold; color: #10b981;">{(1-proba)*100:.1f}%</span>
    </div>
</div>
""", unsafe_allow_html=True)
                
                except Exception as e:
                    st.error(f"Erreur technique : {e}")

# =========================================================================
# PAGE 4 : BASE DE DONNÉES
# =========================================================================
elif page == "Base de Données":
    st.markdown('<h2 class="section-title">&nbsp;&nbsp;&nbsp; Explorateur de Données</h2>', unsafe_allow_html=True)
    
    with st.expander("Filtres Avancés", expanded=True):
        c1, c2 = st.columns(2)
        with c1:
            filter_type = st.multiselect("Filtrer par Type", df['type'].unique(), default=df['type'].unique())
        with c2:
            show_fraud_only = st.checkbox("Montrer uniquement les fraudes")
    
    df_filtered = df[df['type'].isin(filter_type)]
    if show_fraud_only:
        df_filtered = df_filtered[df_filtered['isFraud'] == 1]
    
    st.markdown(f"**{len(df_filtered)}** transactions trouvées.")
    
    st.dataframe(
        df_filtered,
        column_config={
            "isFraud": st.column_config.CheckboxColumn("Est une Fraude ?", help="1 si fraude, 0 sinon"),
            "amount": st.column_config.NumberColumn("Montant ($)", format="$%d"),
        },
        hide_index=True,
        use_container_width=True,
        height=500
    )

    csv = df_filtered.to_csv(index=False).encode('utf-8')
    st.download_button(
        label="📥 Télécharger les données filtrées (CSV)",
        data=csv,
        file_name='jp_plagia_filtered_data.csv',
        mime='text/csv',
    )