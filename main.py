import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go

# ------------------------
# Fonctions Utilitaires
# ------------------------
def compute_lorenz_curve(values):
    """Calcule la courbe de Lorenz à partir des valeurs."""
    sorted_vals = np.sort(values)
    cum_vals = np.cumsum(sorted_vals)
    cum_vals_norm = cum_vals / cum_vals[-1]
    # Ajout d'un point d'origine (0,0)
    lorenz = np.insert(cum_vals_norm, 0, 0)
    return lorenz

def gini_coefficient(values):
    """Calcule le coefficient de Gini pour mesurer l'inégalité."""
    sorted_vals = np.sort(values)
    n = len(values)
    index = np.arange(1, n+1)
    return (2*np.sum(index * sorted_vals)) / (n * np.sum(sorted_vals)) - (n + 1)/n

def classify_user(total_points):
    """Attribution d'un classement selon le total de points (exemple simple)."""
    if total_points < 500:
        return "Bronze"
    elif total_points < 1000:
        return "Argent"
    else:
        return "Or"

# ------------------------
# Fonction de Simulation
# ------------------------
def simulate_points(num_users, weeks, scenario, bonus_multiplier, prob_success, penalty, use_bonus_rules, use_incitation_mode):
    """
    Simule l'accumulation de points sur plusieurs semaines en appliquant différents scénarios.
    
    :param num_users: Nombre d'utilisateurs simulés.
    :param weeks: Nombre de semaines de simulation.
    :param scenario: Scénario choisi ("Optimiste", "Pessimiste", "Mixte").
    :param bonus_multiplier: Multiplicateur de bonus.
    :param prob_success: Probabilité de succès pour le scénario mixte.
    :param penalty: Valeur de la pénalité en cas d'inactivité.
    :param use_bonus_rules: Activation de bonus supplémentaires (bonus pour séries gagnantes, etc.).
    :param use_incitation_mode: Simulation d'un mode "non dilutif" influençant les points par la durée d'engagement.
    :return: DataFrame avec le total de points et classement par utilisateur, et liste des points hebdomadaires.
    """
    users_data = []
    
    for user in range(num_users):
        weekly_points = []
        total_points = 0
        # Exemple : durée d'engagement simulée aléatoirement pour simuler l'incitation non dilutive
        engagement_duration = np.random.randint(1, weeks+1) if use_incitation_mode else weeks
        
        for week in range(weeks):
            # Base de points aléatoire de la semaine
            base_points = np.random.randint(50, 150)
            # Application du scénario
            if scenario == "Optimiste":
                week_points = base_points * bonus_multiplier
            elif scenario == "Pessimiste":
                week_points = max(base_points - penalty, 0)
            else:  # Scénario Mixte
                if np.random.rand() < prob_success:
                    week_points = base_points * bonus_multiplier
                else:
                    week_points = max(base_points - penalty, 0)
            
            # Si activation de bonus (par exemple, série de 3 semaines consécutives au-dessus d'un seuil)
            if use_bonus_rules and week >= 2:
                if (weekly_points[-1] > 100) and (weekly_points[-2] > 100) and (week_points > 100):
                    week_points *= 1.2  # bonus de 20%
            
            # Si simulation d'incitation non dilutive, on pondère selon la durée d'engagement
            if use_incitation_mode:
                # Plus l'engagement est long, plus le score sera valorisé
                week_points *= np.sqrt(engagement_duration/float(weeks))
            
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
# Interface Streamlit
# ------------------------
st.title("Prototype de Simulation Points Autonomics")

st.markdown("""
Ce prototype permet de simuler diverses approches de répartition des points via différents scénarios et paramètres dynamiques.  
Vous pouvez ajuster en temps réel le nombre d'utilisateurs, la durée de la simulation, le scénario choisi ainsi que les règles bonus et incitations.
""")

# Sidebar – Configuration de la simulation
st.sidebar.header("Configuration de la Simulation")

num_users = st.sidebar.number_input("Nombre d'utilisateurs simulés", min_value=1, max_value=1000, value=100, step=1)
weeks = st.sidebar.slider("Durée de la simulation (semaines)", min_value=1, max_value=52, value=12)
scenario = st.sidebar.selectbox("Choix du scénario", options=["Optimiste", "Pessimiste", "Mixte"])
bonus_multiplier = st.sidebar.slider("Multiplicateur Bonus", min_value=1.0, max_value=5.0, value=1.5, step=0.1)
prob_success = st.sidebar.slider("Probabilité de succès (Mixte)", min_value=0.0, max_value=1.0, value=0.7, step=0.05)
penalty = st.sidebar.number_input("Pénalité d'inactivité", min_value=0, max_value=100, value=20, step=1)

st.sidebar.markdown("---")
use_bonus_rules = st.sidebar.checkbox("Activer Bonus/Multi pour séries", value=True)
use_incitation_mode = st.sidebar.checkbox("Activer simulation incitations non dilutives", value=False)
compare_token_points = st.sidebar.checkbox("Activer analyse comparative Token vs Points", value=False)

if st.sidebar.button("Lancer la Simulation"):
    st.write("Exécution de la simulation...")
    df_sim = simulate_points(num_users, weeks, scenario, bonus_multiplier,
                             prob_success, penalty, use_bonus_rules, use_incitation_mode)
    
    # Affichage du tableau de résultats
    st.subheader("Résultats de la Simulation")
    st.dataframe(df_sim[["Utilisateur", "Total Points", "Classement"]])
    
    # Histogramme de la distribution des Total Points
    st.subheader("Distribution des Points Totaux")
    fig_hist = px.histogram(df_sim, x="Total Points", nbins=20, title="Histogramme des Points Totaux")
    st.plotly_chart(fig_hist)
    
    # Calcul de la courbe de Lorenz et du coefficient de Gini
    total_points = df_sim["Total Points"].values
    lorenz = compute_lorenz_curve(total_points)
    gini = gini_coefficient(total_points)
    
    st.subheader("Indicateur de Fairness")
    st.write(f"Coefficient de Gini : {gini:.2f}")
    
    # Affichage de la courbe de Lorenz
    n = len(lorenz)
    x_vals = np.linspace(0.0, 1.0, n)
    fig_lorenz = go.Figure()
    fig_lorenz.add_trace(go.Scatter(x=x_vals, y=lorenz,
                                    mode='lines+markers',
                                    name='Courbe de Lorenz'))
    fig_lorenz.add_trace(go.Scatter(x=[0,1], y=[0,1],
                                    mode='lines',
                                    name='Égalité parfaite',
                                    line=dict(dash='dash')))
    fig_lorenz.update_layout(title="Courbe de Lorenz", xaxis_title="Part cumulée de la population", yaxis_title="Part cumulée des points")
    st.plotly_chart(fig_lorenz)
    
    # Visualisation du système de classement : répartition par niveau
    st.subheader("Répartition par Classement")
    classement_counts = df_sim["Classement"].value_counts().reset_index()
    classement_counts.columns = ["Classement", "Nombre d'utilisateurs"]
    fig_class = px.pie(classement_counts, names="Classement", values="Nombre d'utilisateurs", title="Distribution des Classements")
    st.plotly_chart(fig_class)
    
    # Section Comparative si activée
    if compare_token_points:
        st.markdown("### Analyse Comparative : Système de Récompense Token vs Points")
        # Pour cette démonstration, nous simulerons deux jeux de données :
        # 1. Simulation basée sur les points (déjà obtenue)
        # 2. Simulation simplifiée basée sur des tokens (valeurs légèrement différentes)
        df_token = simulate_points(num_users, weeks, scenario, bonus_multiplier=1.0, 
                                   prob_success=prob_success, penalty=penalty, use_bonus_rules=False, use_incitation_mode=False)
        df_token["Total Tokens"] = df_token["Total Points"] * np.random.uniform(0.8, 1.2, size=num_users)
        st.write("Comparaison entre récompenses basées sur les Points et les Tokens")
        
        fig_compare = go.Figure()
        fig_compare.add_trace(go.Box(y=df_sim["Total Points"], name="Points"))
        fig_compare.add_trace(go.Box(y=df_token["Total Tokens"], name="Tokens"))
        fig_compare.update_layout(title="Distribution Points vs Tokens", yaxis_title="Valeur")
        st.plotly_chart(fig_compare)

st.markdown("""
---
### À Propos de ce Prototype
Ce code sert de base pour un outil d'expérimentation évolutif dans lequel :
- **La simulation dynamique** offre plusieurs scénarios et règles ajustables en temps réel.
- **Les visualisations interactives** permettent d’analyser la répartition des points, l’équité et le classement des utilisateurs.
- **Les modules additionnels** (incitations non dilutives et comparaison Token vs Points) offrent des pistes pour évaluer et optimiser des modèles de récompense adaptés à vos objectifs stratégiques.

Vous pouvez adapter et enrichir ce code selon vos besoins métiers, en y ajoutant par exemple des fonctions plus avancées de simulation ou en affinant les critères de classement.
""")
