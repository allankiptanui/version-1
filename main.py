from model import NeuralNetwork
import numpy as np
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import seaborn as sns

# Load data
def load_data(file_path):
    with open(file_path, 'r') as file:
        data = file.readlines()
    return [list(line.strip()) for line in data]

def encode_labels(labels):
    label_map = {'#': [1, -1, -1, -1], '.': [-1, 1, -1, -1], 'o': [-1, -1, 1, -1], '@': [-1, -1, -1, 1]}
    return [label_map[label] for label in labels]

def preprocess_data(train_file, test_file):
    # Load training data
    train_data = load_data(train_file)
    train_labels = encode_labels([pixel for line in train_data for pixel in line])
    train_inputs = np.array(train_labels)

    # Load testing data
    test_data = load_data(test_file)
    test_labels = encode_labels([pixel for line in test_data for pixel in line])
    test_inputs = np.array(test_labels)

    # Split data into training and testing sets
    X_train, X_test, y_train, y_test = train_inputs, test_inputs, np.array(train_labels), np.array(test_labels)

    return X_train, X_test, y_train, y_test

def plot_confusion_matrix(cm, classes):
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, cmap='Blues', fmt='g', xticklabels=classes, yticklabels=classes)
    plt.xlabel('Predicted')
    plt.ylabel('True')
    plt.title('Confusion Matrix')
    plt.show()

def main():
    # Data preprocessing
    train_file = 'HW3_Training-1.txt'
    test_file = 'HW3_Testing-1.txt'
    X_train, X_test, y_train, y_test = preprocess_data(train_file, test_file)

    # Model parameters
    input_size = len(X_train[0])
    hidden_size = 16
    output_size = len(y_train[0])
    learning_rate = 0.000001
    sigmoid_param = 1
    epochs = 100
    initial_weights = None  # Weights will be initialized randomly

    # Initialize neural network
    nn = NeuralNetwork(input_size, hidden_size, output_size, learning_rate)

    # Train the model
    nn.train(X_train, y_train, epochs)

    # Test the model
    predictions = nn.predict(X_test)
    predicted_labels = [np.argmax(prediction) for prediction in predictions]
    true_labels = [np.argmax(label) for label in y_test]

    # Calculate evaluation metrics
    accuracy = accuracy_score(true_labels, predicted_labels)
    precision = precision_score(true_labels, predicted_labels, average='weighted')
    recall = recall_score(true_labels, predicted_labels, average='weighted')
    f1 = f1_score(true_labels, predicted_labels, average='weighted')

    # Confusion matrix
    confusion = confusion_matrix(true_labels, predicted_labels)

    # Print evaluation metrics and confusion matrix
    print("Accuracy:", accuracy)
    print("Precision:", precision)
    print("Recall:", recall)
    print("F1-score:", f1)
    print("Confusion Matrix:")
    print(confusion)

    # Plot confusion matrix
    plot_confusion_matrix(confusion, classes=['#', '.', 'o', '@'])

if __name__ == "__main__":
    main()
