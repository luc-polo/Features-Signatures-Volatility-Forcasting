# Identification of Atypical Market Behaviors in WTI Crude Oil Futures Using Path Signatures
The project focuses on detecting atypical market behaviors in WTI crude oil futures using advanced path signature techniques. I plan to examine 30-minute intervals throughout the trading day to identify behavioral anomalies, such as volatility spikes or unusual spreads between ask and bid prices.

### Origins and relevance of This Project
Using path signatures to detect atypical behaviors in WTI crude oil futures offers a powerful edge in market analysis and risk management. Given WTI’s sensitivity to geopolitical events, supply-demand shifts, and economic news, traditional volatility indicators often miss nuanced, short-term anomalies that signal upcoming market disruptions. Path signatures allow us to capture complex interactions within 30-minute intervals, including fluctuations in ask-bid spreads, volume imbalances, and micro-trends that standard models may overlook. This approach can enhance our ability to anticipate and respond to rapid price shifts, directly benefiting systematic trading strategies and improving our risk-adjusted returns in volatile market conditions

### Implementation Steps
Data Collection and Preprocessing

I will extract ask and bid prices along with the cumulative trading volume for WTI futures. Data will be collected at regular intervals in 30-minute segments throughout the trading day.
I will then normalize the data to standardize scales and remove trivial anomalies, ensuring a consistent basis for analysis.
Path Representation Construction and Lead-Lag Transformation

The next step involves converting the time series data into multidimensional paths, using the lead-lag transformation to capture complex temporal interactions.
These paths will include variables like average price, ask-bid spread, cumulative volume, and an imbalance indicator to represent the market data comprehensively.
Computation of Truncated Signatures (up to Level 4)

I will apply path signature techniques up to the fourth order to capture advanced features and complex interactions between variables.
Using level 4 allows me to capture subtle variations and higher-order interactions, which may reveal sudden changes or unusual patterns in the data.
Modeling and Anomaly Detection

I will utilize the signature terms as features in a machine learning model, such as LASSO regression, to select the most relevant terms.
The model will be trained to classify trading periods as normal or atypical, detecting anomalies like abnormal volatility increases.
Evaluation and Validation

I will evaluate the model’s accuracy in detecting atypical behaviors using metrics like the ROC curve and the Kolmogorov-Smirnov statistic.
To confirm the model's robustness and effectiveness in different market environments, I will validate it on test periods.

