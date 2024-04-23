#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
from sklearn.metrics import mean_squared_error
import time


# In[2]:


data = pd.read_csv('GOOGL.csv')


# In[3]:


data_copy = data.copy()

data_copy.dropna(inplace=True)

selected_features = ['Open', 'High', 'Low', 'Close', 'Adj Close', 'Volume']
data_copy = data_copy[selected_features]

scaler = MinMaxScaler()
scaled_data = scaler.fit_transform(data_copy)


# In[4]:


df = pd.DataFrame(data)
df.head()


# In[5]:


missing_values = df.isnull().sum()
print("Missing Values:\n", missing_values)


# In[6]:


print("Dataset shape:", df.shape)
print("Columns:", df.columns)
print("Info:\n", df.info())
print("Summary statistics:\n", df.describe())


# In[7]:


plt.figure(figsize=(14, 6))
plt.plot(data['Close'], label='Close Price')
plt.title('Google Stock Prices (2016-2021)')
plt.xlabel('Date')
plt.ylabel('Price')
plt.legend()
plt.show()


# In[8]:


df['Date'] = pd.to_datetime(df['Date'], utc=True)
df.head()


# In[9]:


def prepare_data(data, time_steps):
    X, y = [], []
    for i in range(len(data) - time_steps):
        X.append(data[i:(i + time_steps)])
        y.append(data[i + time_steps])
    return np.array(X), np.array(y)

time_steps = 60  

X, y = prepare_data(scaled_data, time_steps)


# In[10]:


split_ratio = 0.8  # Train-test split ratio
split_index = int(split_ratio * len(X))
X_train, X_test = X[:split_index], X[split_index:]
y_train, y_test = y[:split_index], y[split_index:]


# In[11]:


model = Sequential([
    LSTM(units=100, return_sequences=True, input_shape=(X_train.shape[1], X_train.shape[2])),
    Dropout(0.2),
    LSTM(units=100, return_sequences=True),
    Dropout(0.2),
    LSTM(units=100),
    Dropout(0.2),
    Dense(units=len(selected_features))
])

# Compile model
model.compile(optimizer='adam', loss='mean_squared_error',metrics=['accuracy'])

# Display model
print(model.summary())


# In[12]:


# Measure training time
start_time = time.time()

epochs = 50 
history = model.fit(X_train, y_train, epochs=epochs, batch_size=32, verbose=1)

training_time = time.time() - start_time
print("Training Time:", training_time, "seconds")


# In[13]:


loss = history.history['loss']

epochs = range(len(loss))

plt.plot(epochs, loss, 'r', label='Training loss')

plt.title('Training loss', size=15, weight='bold')
plt.legend(loc=0)
plt.figure()

plt.show()
#model evaluasi
train_loss = model.evaluate(X_train, y_train, verbose=0)
test_loss = model.evaluate(X_test, y_test, verbose=0)

print(f"Train Loss: {train_loss}")
print(f"Test Loss: {test_loss}")


# In[14]:


# Measure prediction time
start_time = time.time()

predictions = model.predict(X_test)
predictions = scaler.inverse_transform(predictions)
y_test_inverse = scaler.inverse_transform(y_test)

prediction_time = time.time() - start_time
print("Prediction Time:", prediction_time, "seconds")

# predicted vs actual
plt.figure(figsize=(10, 6))
plt.plot(predictions[:,3], label='Predicted Close Price', color='r')
plt.plot(y_test_inverse[:,3], label='Actual Close Price', color='b')
plt.title('Google Stock Price Prediction')
plt.xlabel('Time')
plt.ylabel('Price')
plt.legend()
plt.show()


# In[ ]:




