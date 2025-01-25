import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import pandas as pd 
from sklearn.cross_decomposition import PLSRegression
from sklearn.preprocessing import StandardScaler


def evaluate_and_plot_model(grid_search, X_test, Y_test, title='Predicted vs Realized Volatility'):
    """
    Evaluate the best model from GridSearchCV and plot the predicted vs realized volatility and residuals.

    Parameters:
    - grid_search: The fitted GridSearchCV object.
    - X_test: Test features (numpy array or pandas DataFrame).
    - Y_test: Test target values (numpy array or pandas Series).
    - title: Title for the plot.

    Returns:
    - comparison_df: DataFrame containing realized and predicted volatility.
    - metrics: Dictionary containing RMSE, MAE, and the R² score.
    """
    # Best model
    best_model = grid_search.best_estimator_
    print(f"Best Parameters: {grid_search.best_params_} \n")

    # Retrieve the model name and its main parameters
    model = best_model.named_steps['regressor']
    model_name = model.__class__.__name__
    # Extract regressor-specific parameters
    model_params = {k.replace('regressor__', ''): v for k, v in grid_search.best_params_.items() if k.startswith('regressor__')}
    params_str = ', '.join([f"{key}={value}" for key, value in model_params.items()])

    # Update the title with the model name and its parameters
    full_title = f"{title} ({model_name}: {params_str})"

    # Initialize selected_features to None
    selected_features = None

    # Handle selected features by SelectKBest
    if 'feature_selector' in best_model.named_steps:
        selector = best_model.named_steps['feature_selector']
        if hasattr(X_test, 'columns'):
            selected_features = X_test.columns[selector.get_support()]
        else:
            selected_features = np.array([f"Feature_{i}" for i in range(X_test.shape[1])])[selector.get_support()]
        print(f"Selected Features: {selected_features.tolist()} \n")
    

    # Display coefficients for models with coefficients
    if hasattr(model, 'coef_'):
        feature_names = selected_features if selected_features is not None else [f"Feature_{i}" for i in range(len(model.coef_))]
        coef_df = pd.DataFrame({
            'Feature': feature_names,
            'Coefficient': model.coef_
        })
        print("Model Coefficients:")
        print(coef_df.sort_values(by='Coefficient', key=np.abs, ascending=False))

    # Predictions on the test set
    Y_pred = best_model.predict(X_test)

    # Create the comparison DataFrame
    comparison_df = pd.DataFrame({
        'Realized Volatility': Y_test,
        'Predicted Volatility': Y_pred
    }, index=X_test.index if hasattr(X_test, 'index') else None)

    # Compute metrics
    rmse = np.sqrt(mean_squared_error(Y_test, Y_pred))
    mae = mean_absolute_error(Y_test, Y_pred)
    r2 = r2_score(Y_test, Y_pred)

    metrics = {
        'Root Mean Squared Error (RMSE)': rmse,
        'Mean Absolute Error (MAE)': mae,
        'R² Score': r2
    }

    # Display metrics
    print("Evaluation Metrics:")
    for metric, value in metrics.items():
        print(f"{metric}: {value:.4f}")
    print("\n")

    # Plot functions
    def plot_predicted_vs_realized(df, title):
        plt.figure(figsize=(14, 7))
        plt.plot(df.index, df['Realized Volatility'], label='Realized Volatility', color='orange')
        plt.plot(df.index, df['Predicted Volatility'], label='Predicted Volatility', color='blue')
        plt.ylim(0, 0.35)
        plt.title(title)
        plt.xlabel('Date')
        plt.ylabel('Volatility')
        plt.legend()
        plt.show()

    def plot_residuals(df, title):
        residuals = df['Realized Volatility'] - df['Predicted Volatility']
        plt.figure(figsize=(14, 7))
        plt.plot(df.index, residuals, label='Residuals', color='red')
        plt.ylim(-0.175, 0.175)
        plt.axhline(0, color='black', linestyle='--')
        plt.title(f'Residuals {title}')
        plt.xlabel('Date')
        plt.ylabel('Residuals')
        plt.legend()
        plt.show()

    # Plot the results with the updated title
    plot_predicted_vs_realized(comparison_df, full_title)
    plot_residuals(comparison_df, full_title)

    return comparison_df, metrics



def pls_transform_and_plot(X, Y, n_components):
    """
    Performs PLS transformation, computes correlations between PLS components and the target (transformed by PLS),
    and returns the projected X as a DataFrame with preserved indices, along with the transformed Y.

    Parameters:
        X: DataFrame
            Predictor variables.
        Y: DataFrame or Series
            Target variable.
        n_components: int
            Number of components to calculate.

    Returns:
        X_projected_df: DataFrame
            Projected X in the latent space with n_components dimensions, with preserved indices.
    """
    # Normalize predictors and target
    scaler_X = StandardScaler()
    scaler_Y = StandardScaler()

    X_scaled = scaler_X.fit_transform(X)
    Y_scaled = scaler_Y.fit_transform(Y.values.reshape(-1, 1))

    # Initialize and fit PLS with the maximum number of components
    max_components = min(X.shape[1], X.shape[0])
    pls = PLSRegression(n_components=max_components)
    pls.fit(X_scaled, Y_scaled)

    # Project X into latent space (scores)
    X_scores = pls.x_scores_  # PLS scores for X
    Y_scores = pls.y_scores_  # PLS-transformed target

    # Compute absolute correlations of all scores with PLS-transformed Y
    correlations = [
        np.corrcoef(X_scores[:, i], Y_scores[:, 0])[0, 1]
        for i in range(X_scores.shape[1])
    ]
    correlations = np.abs(correlations)

    # Sort components by correlation
    sorted_indices = np.argsort(-correlations)
    sorted_correlations = correlations[sorted_indices]
    X_scores_sorted = X_scores[:, sorted_indices]

    # Create DataFrame for X scores
    X_projected_df = pd.DataFrame(
        X_scores_sorted[:, :n_components],
        index=X.index,
        columns=[f"PLS_Component_{i+1}" for i in range(n_components)]
    )

    # Print the score of the lowest-ranked component in X_projected_df
    print(f"Score of the least important component in X_projected_df: {sorted_correlations[n_components-1]:.4f}")

    # Plot correlations for all components
    plt.figure(figsize=(10, 6))
    plt.bar(range(1, len(sorted_correlations) + 1), sorted_correlations, color="skyblue")
    plt.title("Absolute Correlation of All PLS Components with PLS-Transformed Y")
    plt.xlabel("PLS Component (in order of correlation strength)")
    plt.ylabel("Absolute Correlation")
    plt.xticks(range(1, len(sorted_correlations) + 1))
    plt.show()

    return X_projected_df
