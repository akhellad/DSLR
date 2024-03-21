import numpy as np
import pandas as pd

def predict_multiclass(X, all_theta):
    probs = sigmoid(X.dot(all_theta.T))
    predictions = np.argmax(probs, axis=1)
    return predictions

def sigmoid(z):
    return 1.0 / (1 + np.exp(-z))

def load_theta(path='finalTheta.npy'):
    return np.load(path)

def main():
    print("Welcome to the Hogwarts House Prediction System !\n")
    name = input("What's your name ? ")
    age = input("How old are you ? ")
    print("Thanks, now i need your grades !\n")

    all_theta = load_theta()

    features = ['Herbology', 'Defense Against the Dark Arts', 'Ancient Runes', 'Potions', 'Charms']
    X_user = np.zeros((1, len(features) + 1))
    X_user[0, 0] = 1 

    for i, feature in enumerate(features, start=1):
        value = float(input(f"Grade in {feature} : "))
        X_user[0, i] = value

    X_user[0, 1:] = (X_user[0, 1:] - np.mean(X_user[0, 1:])) / (np.std(X_user[0, 1:]) + 1e-5)

    prediction = predict_multiclass(X_user, all_theta)
    
    houses = ['Gryffondor', 'Poufsouffle', 'Serdaigle', 'Serpentard']
    predicted_house = houses[prediction[0]]

    print(f"\n{name}, {age} years old, you belong to... {predicted_house} ! Congrats !")

if __name__ == "__main__":
    main()