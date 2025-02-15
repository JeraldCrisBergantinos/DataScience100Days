# Import necessary libraries
from sklearn import tree
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

# [Height, weight, shoe size]
X = [
    [181, 80, 44],
    [177, 70, 43],
    [160, 60, 38],
    [154, 54, 37],
    [166, 65, 40],

    [190, 90, 47],
    [175, 64, 39],
    [177, 70, 40],
    [159, 55, 37],
    [171, 75, 42],

    [181, 85, 43]
]

Y = [
    'male',
    'female',
    'female',
    'female',
    'male',

    'male',
    'male',
    'female',
    'male',
    'female',

    'male'
]

# Split the dataset into training and testing sets
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, random_state=42)

# Initialize classifiers
models = {
    "Decision Tree": tree.DecisionTreeClassifier(),
    "K-Nearest Neighbors": KNeighborsClassifier(),
    "Support Vector Classifier": SVC()
}

# Train models and evaluate them
results = {}
for name, model in models.items():
    model.fit(X_train, Y_train) # Train the model
    predictions = model.predict(X_test) # Make predictions
    accuracy = accuracy_score(Y_test, predictions) # Calculate accuracy
    results[name] = accuracy

# Print the results
for name, accuracy in results.items():
    print(f"{name}: {accuracy:.2f}")

# Find and print the best model
best_model_name = max(results, key=results.get)
best_model_accuracy = results[best_model_name]
print(f"\nThe best model is: {best_model_name} with an accuracy of {best_model_accuracy:.2f}")
