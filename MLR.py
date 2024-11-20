import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.datasets import load_diabetes
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score


def main():
    # Load the diabetes dataset
    diabetes = load_diabetes()
    X = diabetes.data  # Multiple features
    y = diabetes.target

    # Split the dataset into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=50
    )

    # Initialize and train the model
    model = LinearRegression()
    model.fit(X_train, y_train)

    # Make predictions on the testing set
    y_pred = model.predict(X_test)

    # Output the results
    print("Multiple Linear Regression Results")
    print("-------------------------------")
    print(f"Coefficient: {model.coef_}")
    print(f"Intercept: {model.intercept_:.2f}")
    print(f"Mean Squared Error: {mean_squared_error(y_test, y_pred):.2f}")
    print(f"Coefficient of Determination (R²): {r2_score(y_test, y_pred):.2f}")

    # Plotting Actual vs Predicted
    plt.scatter(y_test, y_pred)
    plt.plot([y.min(), y.max()], [y.min(), y.max()], color="red", linewidth=2)
    plt.xlabel("Actual Target")
    plt.ylabel("Predicted Target")
    plt.title("Actual vs Predicted Targets")
    plt.legend(["Perfect Prediction", "Predictions"])
    plt.show()


main()
