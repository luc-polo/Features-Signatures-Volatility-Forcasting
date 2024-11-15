Advice from a pro quant:

Make sure the code is clean and elegant. Input, model, and output should be separated and should each be runnable as one line of code. 
No absolute paths should exist in the input or output--only in a config file, stored as a variable.
In fact, it should be the only thing a user would have to change to run the code (as long as they have the data).



Other ideas of project:
https://quantnet.com/threads/projects-idea-for-students-going-into-quant-field.53603/

1. Statistical Arbitrage Trading Strategy Development
Develop a pairs trading or statistical arbitrage model to detect price inefficiencies between correlated assets. Use time-series analysis techniques like cointegration and mean reversion to generate signals, backtest them on historical data, and implement automated trading based on live market feeds.

Sujets à explorer :
Tests de cointegration (ex : méthode d’Engle-Granger ou test de Johansen).
Modèles de réversion à la moyenne (modèle Ornstein-Uhlenbeck).
Séries temporelles et autocorrélation.
Matériel d’apprentissage :
Documentation sur les modèles de séries temporelles (ARIMA, VAR).
Tutoriels sur les tests statistiques comme le test ADF (Augmented Dickey-Fuller).
Identifier des paires d’actifs corrélés en analysant leurs séries temporelles.
Tester la cointegration pour valider que les actifs reviennent à un niveau stable dans le temps.
Implémenter une stratégie basée sur les signaux de réversion (ex : acheter un actif sous-évalué et vendre celui surévalué).
Création d’un environnement de backtesting :
Mettre en place une simulation de la stratégie sur des données historiques.
Calculer les indicateurs de performance : Sharpe Ratio, drawdown maximum, rendement cumulatif.
Identifier les périodes où la stratégie est inefficace.


4. Portfolio Optimization Using Mean-Variance and Black-Litterman Models

Construct an optimized portfolio using traditional mean-variance optimization and the Black-Litterman model. Combine historical data with investor views to balance risk and returns, backtest performance, and account for transaction costs in rebalancing under varying market conditions.

Concepts clés :
Théorie Moderne du Portefeuille (Modern Portfolio Theory, MPT).
Optimisation Mean-Variance par Harry Markowitz.
Modèle Black-Litterman : intégration de données historiques et vues des investisseurs.
Frontière efficiente (efficient frontier).
Métriques : Sharpe Ratio, VaR (Value at Risk).
Concepts mathématiques :
Optimisation quadratique.
Relation entre rendement attendu, covariance, et volatilité.
Construire une fonction objectif pour minimiser le risque à rendement donné.
Inclure des contraintes :
Somme des poids = 100 %.
Poids minimum et maximum pour certains actifs.
Calculer la frontière efficiente et identifier les portefeuilles optimaux.


6. Volatility Trading Strategy with VIX Futures

Develop a volatility trading strategy using VIX futures and options to hedge or speculate on market volatility. Analyze historical VIX data and implement models to capture periods of rising or falling volatility, backtest the strategy, and measure its effectiveness under different market regimes.

Concepts clés :
Le VIX (Volatility Index), aussi appelé "indice de la peur", mesure la volatilité implicite des options sur le S&P 500.
Relation entre le VIX, les marchés actions et les conditions économiques.
Instruments à étudier :
Futures (contrats à terme) sur le VIX.
Options sur le VIX.
Produits négociés en bourse (ETN/ETF) basés sur le VIX.
Sujets complémentaires :
Contango et backwardation dans les contrats à terme.
Le rôle des coûts de roulement dans les stratégies sur les futures.
Modèles pour capturer les périodes de hausse ou de baisse de volatilité :
Moyenne mobile, RSI, Bollinger Bands sur le VIX.
Modèles statistiques comme ARIMA ou GARCH pour prédire la volatilité future.
Expérimentation avec des algorithmes de Machine Learning pour prédire les tendances :
Random Forest, Gradient Boosting.
Entrées et sorties basées sur :
Niveau du VIX (par exemple, acheter des futures si le VIX dépasse un certain seuil).
Pentes de la courbe des futures (contango/backwardation).
Stratégies long/short :
Acheter des options d’achat (calls) ou des futures en période de forte volatilité attendue.
Vendre des options ou acheter des futures short en période de volatilité décroissante.



### Martin
Iddes pour l'instant :

https://www.kaggle.com/competitions/jane-street-real-time-market-data-forecasting/overview

Pricing d'option avec transformees de fourier.

Prédiction de la volatilité avec des modèles hybrides

Les modèles traditionnels comme le GARCH sont largement utilisés pour modéliser la volatilité en capturant les dépendances temporelles et les clustering de volatilité.

Les modèles hybrides combinent les forces des modèles traditionnels avec celles des réseaux neuronaux. L’idée est d’utiliser le GARCH pour modéliser la composante linéaire et les dépendances temporelles, tandis que les réseaux neuronaux capturent les relations non linéaires et les schémas complexes que le GARCH ne peut pas appréhender seul.


KNN pour le stock trends prediction

### Personnes contactees : 
Cedric Viot



Bo



### Message:
Bonjour [Prénom Nom],

Etudiant à l'École des Mines de Saint-Étienne, je suis intéressé par les métiers de la finance quantitative. Je travaille actuellement sur le choix d’un sujet pour un projet académique portant sur des sujets d'analyse quantitative

J’hésite entre deux axes précis et me permets de solliciter votre avis d’expert :

Application des signatures pour prédire des anomalies, notamment des hausses de volatilité, en combinant des données financières multi-dimensionnelles (prix, volume, spread bid-ask). L’idée serait d’utiliser la transformation de signature pour capturer les interactions complexes entre ces dimensions et entraîner un modèle supervisé ou semi-supervisé.
Prédiction de la volatilité avec des modèles hybrides, combinant des approches traditionnelles comme GARCH avec des réseaux de neurones pour modéliser les composantes non linéaires.
Selon vous, lequel de ces deux sujets est le plus pertinent ou innovant dans le contexte actuel de la finance quantitative ?

Je serais ravi d’avoir votre retour ou vos conseils à ce sujet. Merci par avance pour votre aide et votre temps !

Bien à vous,
[Votre Prénom Nom
