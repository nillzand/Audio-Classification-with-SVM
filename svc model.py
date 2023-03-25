import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.model_selection import GridSearchCV
from sklearn.pipeline import Pipeline
from sklearn.metrics import precision_score, recall_score, accuracy_score, f1_score
from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns

# Load the data
df1 = pd.read_csv('audio_files.csv')
df2 = pd.read_csv('esc50.csv')

# Merge the data
df = pd.concat([df1, df2], axis=1)

# Remove unnecessary columns
del df['fold'], df['category'], df['filename'], df['esc10'], df['src_file'], df['take'], df['File Name']

# Split the data into training and testing sets
X = df.drop('target', axis=1)
y = df['target']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Define the model parameters
param_grid = {
    'svc__kernel': ['poly'],
    'svc__C': [0.1, 1, 10],
    'svc__degree': [2, 3, 4, 5],
    'svc__gamma': ['scale', 'auto']
}

# Build the pipeline
pipe = Pipeline([
    ('scaler', StandardScaler()),
    ('svc', SVC())
])

# Build the GridSearchCV object
grid_search = GridSearchCV(pipe, param_grid=param_grid, cv=5)

# Train the model using GridSearchCV
grid_search.fit(X_train, y_train)

# Display the best parameters
print(f"Best parameters: {grid_search.best_params_}")

# Calculate precision, recall, accuracy, and F1 score
y_pred = grid_search.predict(X_test)
precision = precision_score(y_test, y_pred, average='macro')
recall = recall_score(y_test, y_pred, average='macro')
accuracy = accuracy_score(y_test, y_pred)
f1 = f1_score(y_test, y_pred, average='macro')

print("Precision:", precision)
print("Recall:", recall)
print("Accuracy:", accuracy)
print("F1 Score:", f1)

# Plot the accuracy vs polynomial degree
degrees = [2, 3, 4, 5]
accuracies = [grid_search.cv_results_['mean_test_score'][i] for i in [0, 2, 4, 6]]
plt.plot(degrees, accuracies, 'o-')
plt.xlabel('Polynomial Degree')
plt.ylabel('Accuracy')
plt.title('Accuracy vs Polynomial Degree')
plt.show()

# Calculate and plot the confusion matrix
conf_matrix = confusion_matrix(y_test, y_pred)
print(conf_matrix)

sns.heatmap(conf_matrix, annot=True, fmt='d')
plt.title('Confusion Matrix')
plt.xlabel('Predicted Labels')
plt.ylabel('True Labels')
plt.show()