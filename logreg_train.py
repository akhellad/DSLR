import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

def predict_multiclass(X, all_theta):
    # Calculer les probabilités pour chaque classe
    probs = sigmoid(X.dot(all_theta.T))  # Note l'inversion pour all_theta.T
    # Choisir la classe avec la probabilité la plus élevée
    predictions = np.argmax(probs, axis=1)
    return predictions

def accuracy_multiclass(y, preds, houses):
    # Convertir les maisons en indices numériques pour la comparaison
    y_numeric = np.array([list(houses).index(house) for house in y])
    return np.mean(y_numeric == preds)

def sigmoid(z):
    return 1.0 / (1 + np.exp(-z))


def cost_function_logistic(X, y, theta):
    m = len(y)
    h = sigmoid(X.dot(theta))
    epsilon = 1e-5
    return -1/m * np.sum(y * np.log(h + epsilon) + (1 - y) * np.log(1 - h + epsilon))


def grad_logistic(X, y, theta):
    m = len(y)
    h = sigmoid(X.dot(theta))
    return 1/m * X.T.dot(h - y)

def gradient_descent(X, y, theta, learning_rate, n_iterations):
    cost_history = np.zeros(n_iterations)
    for i in range(n_iterations):
        theta -= learning_rate * grad_logistic(X, y, theta)
        cost_history[i] = cost_function_logistic(X, y, theta)
    return theta, cost_history

# Assure-toi que les fonctions sigmoid, cost_function_logistic, grad_logistic, gradient_descent, predict_multiclass et accuracy_multiclass sont bien définies.

def main():
    df = pd.read_csv('./datasets/dataset_train.csv')

    features = ['Herbology', 'Defense Against the Dark Arts', 'Ancient Runes', 'Potions', 'Charms']
    X = df[features].copy()

    X.fillna(X.mean(), inplace=True)

    y = df['Hogwarts House'].values
    houses = np.unique(y)
    all_theta = np.zeros((len(houses), X.shape[1] + 1))

    X = (X - X.mean()) / (X.std() + 1e-5)
    X = np.hstack((np.ones((X.shape[0], 1)), X))
    
    for i, house in enumerate(houses):
        print(f"Training model for {house}")
        y_binary = (y == house).astype(int)
        theta_i, cost_history = gradient_descent(X, y_binary.reshape(-1, 1), np.zeros((X.shape[1], 1)), 0.1, 3000)
        all_theta[i, :] = theta_i.T
        plt.plot(cost_history, label=house)

    plt.xlabel('Nombre d\'itérations')
    plt.ylabel('Coût')
    plt.title('Évolution du coût par maison')
    plt.legend()
    plt.show()

    preds = predict_multiclass(X, all_theta)
    print("Accuracy:", accuracy_multiclass(y, preds, houses))

    np.save('finalTheta.npy', all_theta)

if __name__ == "__main__":
    main()




