def step(x, threshold=0):
    return 1 if x >= threshold else 0

def train_perceptron(gate_name, dataset, learning_rate=0.1, max_epochs=100):
    w1, w2, bias = 0.0, 0.0, 0.0
    print(f"\nTraining Perceptron for {gate_name} Gate")
    print("Initial weights: w1 = {:.2f}, w2 = {:.2f}, bias = {:.2f}".format(w1, w2, bias))
    
    for epoch in range(max_epochs):
        error_flag = False
        print(f"\nEpoch {epoch+1}")
        
        for x1, x2, target in dataset:
            y_in = x1 * w1 + x2 * w2 + bias
            output = step(y_in)
            error = target - output
            
            if error != 0:
                # Update weights and bias
                w1 += learning_rate * error * x1
                w2 += learning_rate * error * x2
                bias += learning_rate * error
                error_flag = True
            
            print(f"Input: [{x1}, {x2}] | Target: {target} | Output: {output} | Updated weights: w1 = {w1:.2f}, w2 = {w2:.2f}, bias = {bias:.2f}")
        
        if not error_flag:
            print("\nTraining converged.")
            break
    else:
        print("\nReached maximum epochs without full convergence.")
    
    # Final evaluation
    print("\nFinal weights:")
    print("w1 = {:.2f}, w2 = {:.2f}, bias = {:.2f}".format(w1, w2, bias))
    
    # Accuracy check
    correct = 0
    for x1, x2, target in dataset:
        output = step(x1 * w1 + x2 * w2 + bias)
        if output == target:
            correct += 1
    accuracy = correct / len(dataset) * 100
    print(f"Final Classification Accuracy: {accuracy:.2f}%\n")
    print("-" * 50)

# Define datasets
and_dataset = [
    (0, 0, 0),
    (0, 1, 0),
    (1, 0, 0),
    (1, 1, 1)
]

or_dataset = [
    (0, 0, 0),
    (0, 1, 1),
    (1, 0, 1),
    (1, 1, 1)
]

# Train on both gates
train_perceptron("AND", and_dataset)
train_perceptron("OR", or_dataset)
