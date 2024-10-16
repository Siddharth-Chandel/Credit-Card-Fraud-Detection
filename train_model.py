import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report, confusion_matrix
from keras.models import Sequential
from keras.layers import Dense
from matplotlib import pyplot as plt
import seaborn as sns

# Load your dataset (replace with your actual dataset)
df = pd.read_csv('card_transdata.csv')

# Preprocess data
X = df.drop('fraud', axis=1)  # Features
y = df['fraud']  # Target (Fraud or not fraud)

# Split the dataset
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, stratify=y, random_state=42)

# Scale the data
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# Define FNN model
model = Sequential()
model.add(Dense(64, input_dim=X_train.shape[1], activation='relu'))
model.add(Dense(32, activation='relu'))
model.add(Dense(1, activation='sigmoid'))  # Output layer for binary classification

# Compile the model
model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])

# Train the model
model.fit(X_train, y_train, epochs=10, batch_size=32, validation_split=0.2)

# Evaluate the model
y_pred = model.predict(X_test)
y_pred = np.round(y_pred)  # Convert probabilities to binary outcomes

# Classification report
cm = confusion_matrix(y_test, y_pred)
print(classification_report(y_test, y_pred))

sns.heatmap(cm, annot=True, fmt='g')
plt.savefig('confusion_matrix.png')

model.save("detector.keras")