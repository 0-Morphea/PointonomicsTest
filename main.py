import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px

# Fonction de simulation basique pour illustrer différents scénarios
def simulate_points(num_users, weeks, scenario, bonus_multiplier, prob_success, penalty):
    results = []
    # Pour chaque utilisateur simulé
    for user in range(num_users):
        total_points = 0
        weekly_points = []
        # Itération sur le nombre de semaines
        for week in range(weeks):
            # Génère une base de points aléatoire pour la semaine
            base_points = np.random.randint(50, 150)
            # Application des règles selon le scénario choisi :
            if scenario == "Optimiste":
                points = base_points * bonus_multiplier
            elif scenario == "Pessimiste":
                points = max(base_points - penalty, 0)
            else:  # Scénario Mixte
                if np.random.rand() < prob_success:
                    points = base_points * bonus_multiplier
                else:
                    points = max(base_points - penalty, 0)
            total_points += points
            weekly_points.append(points)
        results.append({
            'Utilisateur': f'User_{user+1}',
            'Total Points': total_points,
            'Weekly Points': weekly_points
        })
    return pd.DataFrame(results)

# Sidebar : Configuration de la simulation
st.sidebar.title("Configuration de la Simulation")

num_users = st.sidebar.number_input("Nombre d'utilisateurs simulés", min_value=1, max_value=1000, value=100, step=1)
weeks = st.sidebar.slider("Durée de la simulation (en semaines)", min_value=1, max_value=52, value=12)
scenario = st.sidebar.selectbox("Choisir le scénario", ["Optimiste", "Pessimiste", "Mixte"])
bonus_multiplier = st.sidebar.slider("Multiplicateur Bonus", min_value=1.0, max_value=5.0, value=1.5, step=0.1)
prob_success = st.sidebar.slider("Probabilité de succès (scénario Mixte)", min_value=0.0, max_value=1.0, value=0.7, step=0.05)
penalty = st.sidebar.number_input("Pénalité d'inactivité", min_value=0, max_value=100, value=20, step=1)

# Bouton pour lancer la simulation
if st.sidebar.button("Lancer la simulation"):
    st.write("Exécution de la simulation, veuillez patienter...")
    df_results = simulate_points(num_users, weeks, scenario, bonus_multiplier, prob_success, penalty)
    
    # Affichage du tableau de résultats
    st.subheader("Résultats de la Simulation")
    st.dataframe(df_results)
    
    # Histogramme de la distribution des points totaux
    st.subheader("Distribution des Points Totaux par Utilisateur")
    fig_hist = px.histogram(df_results, x="Total Points", nbins=20, title="Histogramme des Points Totaux")
    st.plotly_chart(fig_hist)
    
    # Simulation d'un indicateur de fairness simple (exemple avec coefficient de variation)
    st.subheader("Indicateur de Fairness")
    mean_points = df_results["Total Points"].mean()
    std_points = df_results["Total Points"].std()
    coefficient_variation = std_points / mean_points if mean_points != 0 else 0
    st.write(f"Coefficient de variation (inversé pour indiquer l'équité) : {1 - coefficient_variation:.2f}")

# Section principale – introduction et affichage statique
st.title("Prototype de Simulation Points Autonomics")
st.markdown("""
Ce prototype vous permet de simuler divers scénarios de répartition des points en fonction de paramètres ajustables :
- **Scénarios Multiples** : Choix entre un scénario optimiste, pessimiste ou mixte.
- **Paramétrage Dynamique** : Ajustement en temps réel des probabilités de succès, bonus et pénalités.
- **Visualisation Interactive** : Graphiques interactifs pour observer la distribution des points et des indicateurs de fairness.
  
Vous pourrez par la suite compléter cette base pour intégrer :
- Des systèmes de classement et de ligues (ex : bronze, argent, or) inspirés des modèles Elo.
- Des dashboards plus complets incluant l'historique des points, l'évolution des bonus et une analyse comparative entre systèmes de récompense basés sur tokens versus points.
""")

