import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report, confusion_matrix, precision_recall_curve, roc_curve, auc
from imblearn.over_sampling import SMOTE  # For handling class imbalance
from keras.models import Sequential  # type: ignore
from keras.layers import Dense, Dropout  # type: ignore
from keras.callbacks import EarlyStopping  # type: ignore
from matplotlib import pyplot as plt
import seaborn as sns

# Load dataset (replace with your actual dataset)
df = pd.read_csv(r'data\card_transdata.csv')

# Data analysis: visualize fraud distribution before SMOTE
plt.figure(figsize=(6, 4))
sns.countplot(x='fraud', data=df)
plt.title('Distribution of Fraudulent vs Genuine Transactions (Before SMOTE)')
plt.xlabel('Fraudulent (1) vs Genuine (0)')
plt.ylabel('Count')
plt.savefig(r'img\fraud_distribution_before_smote.png')
plt.show()

# Preprocess data
X = df.drop('fraud', axis=1)  # Features
y = df['fraud']  # Target (Fraud or not fraud)

# Handle class imbalance (using SMOTE)
smote = SMOTE(random_state=42)
X_res, y_res = smote.fit_resample(X, y)

# Data analysis: visualize fraud distribution after SMOTE
plt.figure(figsize=(6, 4))
sns.countplot(x=y_res)  # Using the resampled target variable
plt.title('Distribution of Fraudulent vs Genuine Transactions (After SMOTE)')
plt.xlabel('Fraudulent (1) vs Genuine (0)')
plt.ylabel('Count')
plt.savefig(r'img\fraud_distribution_after_smote.png')
plt.show()

# Split the dataset
X_train, X_test, y_train, y_test = train_test_split(X_res, y_res, test_size=0.3, stratify=y_res, random_state=42)

# Scale the data
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# Visualizing feature correlations using heatmap
plt.figure(figsize=(6, 4))
corr = pd.DataFrame(X_train).corr()  # Correlation matrix of training features
sns.heatmap(corr, annot=True, cmap='coolwarm', fmt='.2f', linewidths=0.5)
plt.title('Feature Correlation Heatmap')
plt.savefig(r'img\feature_correlation_heatmap.png')
plt.show()

# Define FNN model
model = Sequential()
model.add(Dense(64, input_dim=X_train.shape[1], activation='relu'))
model.add(Dropout(0.5))  # Adding dropout layer to prevent overfitting
model.add(Dense(32, activation='relu'))
model.add(Dense(1, activation='sigmoid'))  # Output layer for binary classification

# Compile the model
model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])

# Define early stopping callback
early_stopping = EarlyStopping(monitor='val_loss', patience=3, restore_best_weights=True)

# Train the model with early stopping
history = model.fit(X_train, y_train, epochs=10, batch_size=32, validation_split=0.2, callbacks=[early_stopping])

# Evaluate the model
y_pred = model.predict(X_test)
y_pred = np.round(y_pred)  # Convert probabilities to binary outcomes

# Classification report
cm = confusion_matrix(y_test, y_pred)
print(classification_report(y_test, y_pred))

# Confusion matrix heatmap
plt.figure(figsize=(6, 5))
sns.heatmap(cm, annot=True, fmt='g', cmap='Blues')
plt.title('Confusion Matrix')
plt.xlabel('Predicted')
plt.ylabel('True')
plt.savefig(r'img\confusion_matrix.png')
plt.show()

# Plot training & validation accuracy
plt.figure(figsize=(12, 6))
plt.subplot(1, 2, 1)
plt.plot(history.history['accuracy'], label='Training Accuracy')
plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
plt.title('Training vs Validation Accuracy')
plt.xlabel('Epochs')
plt.ylabel('Accuracy')
plt.legend()

# Plot training & validation loss
plt.subplot(1, 2, 2)
plt.plot(history.history['loss'], label='Training Loss')
plt.plot(history.history['val_loss'], label='Validation Loss')
plt.title('Training vs Validation Loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()

plt.tight_layout()
plt.savefig(r'img\training_plot.png')

# Show the plot
plt.show()

# Save the model
model.save(r"model\detector.keras")