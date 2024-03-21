import numpy as np
import pandas as pd

def predict_multiclass(X, all_theta):
    # Calculer les probabilités pour chaque classe
    probs = sigmoid(X.dot(all_theta.T))  # Note l'inversion pour all_theta.T
    # Choisir la classe avec la probabilité la plus élevée
    predictions = np.argmax(probs, axis=1)
    return predictions

def sigmoid(z):
    return 1.0 / (1 + np.exp(-z))

def load_theta(path='finalTheta.npy'):
    return np.load(path)

def main():
    # Charger les données de test
    df_test = pd.read_csv('./datasets/dataset_test.csv')
    
    features = ['Herbology', 'Defense Against the Dark Arts', 'Ancient Runes', 'Potions', 'Charms']
    X_test = df_test[features].copy()
    
    # Gestion des valeurs manquantes
    X_test.fillna(X_test.mean(), inplace=True)
    
    # Normaliser X et ajouter une colonne pour le biais
    X_test = (X_test - X_test.mean()) / (X_test.std() + 1e-5)
    X_test = np.hstack((np.ones((X_test.shape[0], 1)), X_test))
    
    # Charger les poids entraînés
    all_theta = load_theta()
    
    # Faire des prédictions
    predictions = predict_multiclass(X_test, all_theta)
    
    # Convertir les indices de prédictions en noms de maisons
    houses = ['Gryffindor', 'Hufflepuff', 'Ravenclaw', 'Slytherin']
    predicted_houses = [houses[pred] for pred in predictions]
    
    # Créer et sauvegarder le fichier de prédictions
    output = pd.DataFrame({'Index': df_test.index, 'Hogwarts House': predicted_houses})
    output.to_csv('houses.csv', index=False, header=True)
    
    print("Predictions saved to houses.csv")

if __name__ == "__main__":
    main()