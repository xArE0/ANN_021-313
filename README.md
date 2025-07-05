# Artificial Neural Networks: Perceptron  
**Name:** Avishek Shrestha  
**CRN:** 021-313  

## 🧠 Project Overview

This project implements a simple Artificial Neural Network (ANN) using the Perceptron learning algorithm from scratch in Python using NumPy only. It is designed for educational purposes to demonstrate:

- How a binary classifier works using a step activation function
- How a Perceptron can learn linear functions using a linear activation

---
## 🧠 Perceptron Learning Rule

For each input:

prediction = activation(w · x + b)  
error = actual - predicted  
weights = weights + α * error * input  
bias = bias + α * error

- α is the learning rate
- The step activation is used for classification:
  step(x) = 1 if x ≥ threshold else 0
- The linear activation is used for regression:
  linear(x) = x

Training continues until the error becomes zero or the maximum number of epochs is reached.

---

## 🔹 Task 1: Perceptron for 2-Input Basic Gates (AND / OR)

**Goal:** Implement a Perceptron to learn the 2-input AND and OR gates.

**Input:**  
- AND: [0,0→0], [0,1→0], [1,0→0], [1,1→1]  
- OR:  [0,0→0], [0,1→1], [1,0→1], [1,1→1]  

**Training Setup:**  
- Learning rate: 0.1  
- Max epochs: 100  

**Output:**  
- Learned weights and bias for each gate  
- Training progress (weights per epoch)  
- Final classification accuracy  
![ANN OR Gate](https://github.com/user-attachments/assets/ddce14f8-fef2-438f-bc4b-70212d93670e)

---

## 🔹 Task 2: Perceptron for n-Input Basic Gates (AND / OR)

**Goal:** Generalize the Perceptron to support n-input AND/OR gates (e.g., n = 3 or n = 4)

**Input:** Generated truth table for all 2ⁿ combinations of inputs

**Training Setup:**  
- Learning rate: 0.1  
- Max epochs: 100  

**Requirement:**  
- Test with n = 3 and n = 4  
- Display learned weights, bias, and classification accuracy  
![n input gates](https://github.com/user-attachments/assets/6eaa7020-19f5-425e-b283-c005db53c66d)

---

## 🔹 Task 3: Perceptron for Linear Function with 3 Features

**Goal:** Implement a Perceptron to learn the linear function:  
y = 2x₁ + 3x₂ - x₃ + 5  

**Input:**  
- 10 randomly generated samples where x₁, x₂, x₃ ∈ [0, 1]  

**Activation:** Linear (no step)  
**Training Setup:**  
- Learning rate: 0.01  
- Max epochs: 100  

**Output:**  
- Mean Squared Error after each epoch  
- Final weights and bias  
![linear function 34](https://github.com/user-attachments/assets/29509562-cf08-4eef-afad-ec67070dafd0)

---

## 🔹 Task 4: Perceptron for Linear Function with n Features

**Goal:** Generalize the linear Perceptron for any number of features n.

**Function:**  
y = w₁x₁ + w₂x₂ + ... + wₙxₙ + b  

**Setup:**  
- Generate 10 random samples  
- True weights randomly selected in [-1, 1]  
- Bias = 5  
- Learning rate: 0.01  
- Max epochs: 100  

**Requirement:**  
- Test with n = 4 and n = 5  
- Display MSE per epoch and final learned weights  
![Generalized LR](https://github.com/user-attachments/assets/d19b5cdb-44f4-4618-866f-614c0bac713e)

---
