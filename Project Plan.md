

# Development of an Algorithm for Anomaly Detection in the Gold Market

## Introduction
The goal is to develop an algorithm capable of detecting anomalies in the gold market by analyzing historical financial metrics. The project is based on path signatures exploitation which allows to capture non-linear patterns and interactions within time series data.

## Step 1: Data Collection
Source: Historical gold price data collected on Yahoo Finance.
Timeframe: Data from January 2022 to October 2024 with a daily frequency is used to capture medium-term trends and dynamics.
Features Collected:
Gold spot price (Close, High, Low, and Open).
Trading volume (Volume).
Timestamp (Date).


## Step 2: Feature Engineering
Derived metrics are created to enrich the dataset, enabling deeper insights into market behavior:

-Daily Returns: Captures short-term price momentum as the percentage change in the closing price.
-20-Day Moving Average: Highlights longer-term trends by smoothing price fluctuations.
-Log Mid-Price: Logarithm of the mid-price (average of High and Low), which stabilizes variance and converts multiplicative price changes into additive ones.
-Spread: Difference between High and Low prices, indicating market volatility.
-Imbalance: Relative volume difference between consecutive intervals, providing insights into market sentiment.

## Step 3: Data Normalization
The derived metrics and raw features are normalized to ensure compatibility with machine learning models:

- Normalized Time: Scaled to [0, 1] to standardize temporal information.
- Normalized Log Mid-Price: Normalized to zero mean and unit variance, capturing central price dynamics.
- Normalized Spread, Imbalance, and Daily Returns: Standardized to remove scale effects and emphasize relative variability.
- Normalized Moving Average: Standardizes long-term trends for consistency across the dataset.

##Step 4: Lead-Lag Transformation
To better capture temporal dynamics, the lead-lag transformation is applied:

Each point is duplicated into "lead" (current observation) and "lag" (previous observation) values, creating a zigzag-like path.
This transformation ensures compatibility with path signature models, allowing them to process time-series data effectively.


### Étape 2 : Construction du Chemin Multidimensionnel

2.1. Représentation du Chemin
	•	Chemin en \mathbb{R}^n : Représentez les données prétraitées comme un chemin continu dans un espace multidimensionnel, où chaque dimension correspond à une variable.
	•	Variables à inclure dans le chemin :
	•	X_t = [\text{Temps normalisé}, \text{Prix de l{\prime}or}, \text{Volume}, \text{Autres variables}]

2.2. Transformation Lead-Lag
	•	Objectif : La transformation lead-lag permet de capturer les dépendances temporelles fines et les variations abruptes en doublant les dimensions du chemin.
	•	Procédure :
	•	Création des variables lead et lag : Pour chaque variable  X , créez  X_{\text{lead}}  (valeur avancée d’un pas de temps) et  X_{\text{lag}}  (valeur actuelle).
	•	Construction du chemin lead-lag : Le chemin final aura des dimensions  [X_{\text{lead}}, X_{\text{lag}}]  pour chaque variable.

2.3. Chemin Final
	•	Exemple de chemin final :

\text{Chemin}t = [\text{Temps}{\text{lead}}, \text{Temps}{\text{lag}}, \text{Prix}{\text{lead}}, \text{Prix}{\text{lag}}, \text{Volume}{\text{lead}}, \text{Volume}_{\text{lag}}, \dots]


Étape 3 : Calcul de la Signature du Chemin

3.1. Choix du Niveau de Troncature
	•	Niveau de signature : Décidez jusqu’à quel niveau  n  vous souhaitez calculer la signature. Un niveau plus élevé capture des interactions plus complexes mais augmente la dimensionnalité.
	•	Recommandation : Commencez par un niveau 4 ou 5 pour un bon compromis entre complexité et expressivité.

3.2. Calcul de la Signature
	•	Bibliothèques à utiliser :
	•	Python : Utilisez des bibliothèques comme esig (i.e., esig.tosig) ou iisignature.
	•	Procédure :
	•	Conversion du chemin en tableau NumPy : Assurez-vous que le chemin est dans un format compatible avec la bibliothèque choisie.
	•	Calcul :

import esig

# Chemin sous forme de tableau NumPy
path_array = chemin.values  # si chemin est un DataFrame pandas

# Calcul de la signature jusqu'au niveau n
signature = esig.stream2sig(path_array, n)



3.3. Gestion de la Dimensionnalité
	•	Nombre de termes de la signature : Le nombre de termes augmente exponentiellement avec le niveau  n  et le nombre de dimensions  d :

\text{Nombre de termes} = \sum_{k=0}^{n} d^k

	•	Réduction de dimension : Si nécessaire, utilisez des techniques comme l’analyse en composantes principales (ACP) pour réduire la dimensionnalité.

Étape 4 : Détection des Anomalies

4.1. Choix de la Méthode de Détection
	•	Approche non supervisée : Si vous n’avez pas de labels indiquant les anomalies passées.
	•	Algorithmes possibles :
	•	Isolation Forest : Détecte les anomalies en isolant les points de données.
	•	DBSCAN : Algorithme de clustering qui identifie les points qui ne font pas partie d’un cluster dense.
	•	Autoencodeurs : Réseaux de neurones qui apprennent une représentation comprimée des données et identifient les anomalies par reconstruction.
	•	Approche supervisée : Si vous disposez de labels (par exemple, anomalies historiques connues).
	•	Algorithmes possibles :
	•	Régression Logistique : Avec régularisation L1 (LASSO) pour la sélection de caractéristiques.
	•	Forêts Aléatoires : Robustes aux outliers et capables de gérer de nombreuses caractéristiques.
	•	Réseaux de Neurones : Pour capturer des relations non linéaires complexes.

4.2. Préparation des Données pour la Modélisation
	•	Création du jeu de données :
	•	Variables indépendantes : Les termes de la signature calculée.
	•	Variable dépendante : Indicateur binaire d’anomalie (pour les approches supervisées).
	•	Séparation des ensembles :
	•	Entraînement : Pour entraîner le modèle.
	•	Validation : Pour ajuster les hyperparamètres.
	•	Test : Pour évaluer la performance finale.

4.3. Entraînement du Modèle
	•	Modèle non supervisé (Isolation Forest) :

from sklearn.ensemble import IsolationForest

model = IsolationForest(contamination=0.01, random_state=42)
model.fit(X_train)


	•	Modèle supervisé (Régression Logistique avec LASSO) :

from sklearn.linear_model import LogisticRegressionCV

model = LogisticRegressionCV(cv=5, penalty='l1', solver='saga', random_state=42)
model.fit(X_train, y_train)



4.4. Détection des Anomalies
	•	Prédiction :

anomalies = model.predict(X_test)

	•	Pour IsolationForest, les anomalies sont généralement marquées par -1.
	•	Pour les modèles supervisés, les anomalies sont prédites en fonction du seuil de décision.

	•	Probabilité ou Score d’Anomalie :
	•	Utilisez decision_function ou predict_proba pour obtenir un score continu.

Étape 5 : Interprétation et Visualisation des Résultats

5.1. Identification des Périodes Anormales
	•	Marquage des anomalies : Associez les prédictions aux horodatages pour identifier les périodes spécifiques.
	•	Analyse temporelle : Visualisez les anomalies sur une ligne temporelle pour détecter des motifs ou des tendances.

5.2. Visualisation
	•	Graphiques :
	•	Cours de l’or avec anomalies : Tracez le prix de l’or et mettez en évidence les points où des anomalies ont été détectées.
	•	Scores d’anomalie : Visualisez les scores pour comprendre la sévérité des anomalies.
	•	Analyse des caractéristiques :
	•	Importance des termes de signature : Identifiez quels termes ont le plus contribué à la détection des anomalies.
	•	Interprétation : Comprenez quels types d’interactions entre variables sont associés aux anomalies.

5.3. Validation des Résultats
	•	Comparaison avec des événements réels :
	•	Actualités économiques : Vérifiez si les anomalies correspondent à des événements économiques connus (crises, annonces de politiques monétaires, etc.).
	•	Évaluation des performances :
	•	Métriques :
	•	AUC-ROC : Pour évaluer la capacité du modèle à distinguer les anomalies.
	•	Précision, Rappel, F1-score : Pour une évaluation plus détaillée.

Étape 6 : Optimisation et Améliorations

6.1. Ajustement des Hyperparamètres
	•	Recherche en grille : Utilisez GridSearchCV pour trouver les meilleurs hyperparamètres.
	•	Validation croisée : Assurez-vous que le modèle généralise bien aux nouvelles données.

6.2. Sélection de Caractéristiques
	•	Régularisation : Utilisez LASSO pour réduire le nombre de termes de signature et éviter le surapprentissage.
	•	Analyse de variance : Supprimez les termes avec une faible variance à travers les échantillons.

6.3. Traitement de la Dimensionnalité
	•	Réduction de dimension :
	•	ACP : Pour réduire la dimensionnalité tout en conservant la majorité de la variance.
	•	t-SNE : Pour visualiser les données en 2D ou 3D et détecter des clusters.

Étape 7 : Mise en Production

7.1. Intégration de l’Algorithme
	•	Pipeline automatisé :
	•	Collecte en temps réel : Intégrez une API pour obtenir les données en temps réel.
	•	Traitement par lots : Programmez des exécutions régulières de l’algorithme (par exemple, toutes les heures).

7.2. Alertes et Notifications
	•	Système d’alerte :
	•	Notifications : Configurez des alertes par e-mail ou SMS lorsque des anomalies sont détectées.
	•	Seuils : Définissez des seuils de score pour déclencher des alertes en fonction de la sévérité.

7.3. Tableau de Bord
	•	Visualisation en temps réel :
	•	Dashboards interactifs : Utilisez des outils comme Tableau, Power BI, ou Dash (Python) pour visualiser les anomalies et les données associées.

Étape 8 : Surveillance et Mise à Jour du Modèle

8.1. Surveillance de la Performance
	•	Drift du modèle : Surveillez les performances pour détecter une éventuelle dégradation due à des changements dans les données.
	•	Réentraînement : Préparez-vous à réentraîner le modèle régulièrement avec de nouvelles données.

8.2. Feedback et Boucle d’Amélioration
	•	Retour d’information : Si possible, incorporez le feedback des analystes ou des traders sur la pertinence des anomalies détectées.
	•	Affinement du modèle : Ajustez le modèle en fonction du feedback pour améliorer la précision.
