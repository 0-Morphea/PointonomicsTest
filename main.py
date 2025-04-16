import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go

# ------------------------
# Fonctions Utilitaires
# ------------------------
def compute_lorenz_curve(values):
    """Calcule la courbe de Lorenz à partir d'un vecteur de valeurs."""
    sorted_vals = np.sort(values)
    cum_vals = np.cumsum(sorted_vals)
    cum_vals_norm = cum_vals / cum_vals[-1]
    # Ajout du point d'origine
    lorenz = np.insert(cum_vals_norm, 0, 0)
    return lorenz

def gini_coefficient(values):
    """Calcule le coefficient de Gini pour mesurer l'inégalité."""
    sorted_vals = np.sort(values)
    n = len(values)
    index = np.arange(1, n+1)
    return (2 * np.sum(index * sorted_vals)) / (n * np.sum(sorted_vals)) - (n + 1) / n

def classify_user(total_points):
    """Attribue un classement en fonction du total de points."""
    if total_points < 500:
        return "Bronze"
    elif total_points < 1000:
        return "Argent"
    else:
        return "Or"

# ------------------------
# Simulation sans Quêtes
# ------------------------
def simulate_points(num_users, weeks, scenario, bonus_multiplier, prob_success, penalty, use_bonus_rules, use_incitation_mode):
    """
    Simule l'accumulation de points sur plusieurs semaines en appliquant divers scénarios.
    """
    users_data = []
    
    for user in range(num_users):
        weekly_points = []
        total_points = 0
        # Simulation d'une durée d'engagement aléatoire si le mode incitations non dilutives est activé
        engagement_duration = np.random.randint(1, weeks+1) if use_incitation_mode else weeks
        
        for week in range(weeks):
            base_points = np.random.randint(50, 150)
            # Application du scénario choisi
            if scenario == "Optimiste":
                week_points = base_points * bonus_multiplier
            elif scenario == "Pessimiste":
                week_points = max(base_points - penalty, 0)
            else:  # Scénario Mixte
                if np.random.rand() < prob_success:
                    week_points = base_points * bonus_multiplier
                else:
                    week_points = max(base_points - penalty, 0)
            
            # Bonus pour séries (exemple : trois semaines consécutives avec plus de 100 points)
            if use_bonus_rules and week >= 2:
                if (weekly_points[-1] > 100) and (weekly_points[-2] > 100) and (week_points > 100):
                    week_points *= 1.2  # bonus de 20%
            
            # Pondération de l'engagement si activé
            if use_incitation_mode:
                week_points *= np.sqrt(engagement_duration / float(weeks))
            
            total_points += week_points
            weekly_points.append(week_points)
        
        classement = classify_user(total_points)
        users_data.append({
            "Utilisateur": f"User_{user+1}",
            "Total Points": total_points,
            "Classement": classement,
            "Weekly Points": weekly_points
        })
    
    return pd.DataFrame(users_data)

# ------------------------
# Simulation avec Quêtes
# ------------------------
def simulate_with_quests(num_users, weeks, quests_df):
    """
    Simule l'accumulation de points à partir d'un tableau de quêtes.
    
    Pour chaque utilisateur et chaque semaine, la réussite de chaque quête est simulée
    selon la probabilité définie et ajoute le score correspondant.
    """
    results = []
    
    for user in range(num_users):
        total_points = 0
        for week in range(weeks):
            week_points = 0
            for idx, quest in quests_df.iterrows():
                # Si la quête est réussie (selon sa probabilité définie)
                if np.random.rand() < quest["Probabilité de réussite"]:
                    week_points += quest["Score attribué"]
            total_points += week_points
        results.append({"Utilisateur": f"User_{user+1}", "Total Points": total_points})
    
    return pd.DataFrame(results)

# ------------------------
# Interface Streamlit
# ------------------------
st.title("Prototype de Simulation Points Autonomics")

st.markdown("""
Ce prototype vous permet de simuler différentes approches d'attribution de points.  
Vous pouvez configurer des scénarios dynamiques, activer des bonus, simuler des incitations non dilutives,  
et gérer dynamiquement un tableau de quêtes dans le menu principal.
""")

# Sidebar – Configuration Générale de la Simulation
st.sidebar.header("Configuration de la Simulation")
num_users = st.sidebar.number_input("Nombre d'utilisateurs simulés", min_value=1, max_value=1000, value=100, step=1)
weeks = st.sidebar.slider("Durée de la simulation (en semaines)", min_value=1, max_value=52, value=12)
scenario = st.sidebar.selectbox("Choix du scénario", options=["Optimiste", "Pessimiste", "Mixte"])
bonus_multiplier = st.sidebar.slider("Multiplicateur Bonus", min_value=1.0, max_value=5.0, value=1.5, step=0.1)
prob_success = st.sidebar.slider("Probabilité de succès (Mixte)", min_value=0.0, max_value=1.0, value=0.7, step=0.05)
penalty = st.sidebar.number_input("Pénalité d'inactivité", min_value=0, max_value=100, value=20, step=1)
use_bonus_rules = st.sidebar.checkbox("Activer Bonus pour séries", value=True)
use_incitation_mode = st.sidebar.checkbox("Activer incitations non dilutives", value=False)
compare_token_points = st.sidebar.checkbox("Analyse comparative Token vs Points", value=False)

st.markdown("---")
st.header("Gestion des Quêtes")

st.markdown("""
Dans cette section, vous pouvez gérer les quêtes en attribuant un nom, un score et une probabilité de réussite à chaque quête.  
Vous pourrez ensuite intégrer ces quêtes dans la simulation.
""")
# Tableau des quêtes éditable dans le menu principal
default_quests = pd.DataFrame({
    "Nom de la Quête": ["Connexion quotidienne", "Publication d'article", "Participation à un vote"],
    "Score attribué": [50, 150, 75],
    "Probabilité de réussite": [0.9, 0.7, 0.8]
})
quests_df = st.experimental_data_editor(default_quests, num_rows="dynamic", key="quests_editor")
st.write("Tableau des quêtes actuellement définies:")
st.dataframe(quests_df)

st.markdown("---")
st.header("Lancement de la Simulation Globale")
if st.sidebar.button("Lancer la Simulation Globale"):
    st.write("Exécution de la simulation...")
    
    # Simulation sans quêtes
    df_sim = simulate_points(num_users, weeks, scenario, bonus_multiplier,
                             prob_success, penalty, use_bonus_rules, use_incitation_mode)
    
    st.subheader("Résultats de la Simulation Globale")
    st.dataframe(df_sim[["Utilisateur", "Total Points", "Classement"]])
    
    # Histogramme de distribution des Total Points
    st.subheader("Distribution des Points Totaux")
    fig_hist = px.histogram(df_sim, x="Total Points", nbins=20, title="Histogramme des Points Totaux")
    st.plotly_chart(fig_hist)
    
    # Calcul de la courbe de Lorenz et du coefficient de Gini
    total_points = df_sim["Total Points"].values
    lorenz = compute_lorenz_curve(total_points)
    gini = gini_coefficient(total_points)
    
    st.subheader("Indicateur de Fairness")
    st.write(f"Coefficient de Gini : {gini:.2f}")
    n = len(lorenz)
    x_vals = np.linspace(0.0, 1.0, n)
    fig_lorenz = go.Figure()
    fig_lorenz.add_trace(go.Scatter(x=x_vals, y=lorenz, mode='lines+markers', name='Courbe de Lorenz'))
    fig_lorenz.add_trace(go.Scatter(x=[0,1], y=[0,1], mode='lines', name='Égalité parfaite', line=dict(dash='dash')))
    fig_lorenz.update_layout(title="Courbe de Lorenz", xaxis_title="Part cumulée de la population", yaxis_title="Part cumulée des points")
    st.plotly_chart(fig_lorenz)
    
    # Visualisation du système de classement (répartition par niveau)
    st.subheader("Répartition par Classement")
    classement_counts = df_sim["Classement"].value_counts().reset_index()
    classement_counts.columns = ["Classement", "Nombre d'utilisateurs"]
    fig_class = px.pie(classement_counts, names="Classement", values="Nombre d'utilisateurs", title="Distribution des Classements")
    st.plotly_chart(fig_class)
    
    # Module d'analyse comparative : tokens vs points
    if compare_token_points:
        st.markdown("### Analyse Comparative : Points vs Tokens")
        df_token = simulate_points(num_users, weeks, scenario, bonus_multiplier=1.0, 
                                   prob_success=prob_success, penalty=penalty, use_bonus_rules=False, use_incitation_mode=False)
        df_token["Total Tokens"] = df_token["Total Points"] * np.random.uniform(0.8, 1.2, size=num_users)
        st.write("Comparaison entre les distributions de Points et de Tokens")
        fig_compare = go.Figure()
        fig_compare.add_trace(go.Box(y=df_sim["Total Points"], name="Points"))
        fig_compare.add_trace(go.Box(y=df_token["Total Tokens"], name="Tokens"))
        fig_compare.update_layout(title="Distribution Points vs Tokens", yaxis_title="Valeur")
        st.plotly_chart(fig_compare)

st.markdown("---")
st.header("Simulation avec Quêtes")
st.markdown("""
Cette simulation utilise le tableau de quêtes défini pour attribuer un score à chaque utilisateur.  
Pour chaque semaine, la réussite de chaque quête (selon sa probabilité) ajoute un score spécifique.
""")
if st.button("Lancer la Simulation avec Quêtes"):
    st.write("Exécution de la simulation basée sur les quêtes...")
    df_sim_quests = simulate_with_quests(num_users, weeks, quests_df)
    st.subheader("Résultats de la Simulation avec Quêtes")
    st.dataframe(df_sim_quests)
    
    st.subheader("Histogramme de la Distribution des Points (Quêtes)")
    fig_quests = px.histogram(df_sim_quests, x="Total Points", nbins=20, title="Histogramme des Points Totaux (Quêtes)")
    st.plotly_chart(fig_quests)

st.markdown("""
---
### À Propos de ce Prototype
Ce code constitue une base complète pour simuler et analyser des systèmes de points dynamiques.  
Il intègre :
- Des scénarios multiples avec paramétrage en temps réel.
- Un module de gestion des quêtes accessible dans le menu principal.
- Des visualisations interactives (Histogramme, Courbe de Lorenz, Répartition par Classement).
- Une option d'analyse comparative entre un système basé sur les points et un système basé sur les tokens.

N'hésitez pas à adapter et enrichir ce prototype selon vos besoins métiers !
""")
