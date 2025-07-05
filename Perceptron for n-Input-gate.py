import itertools

def step(x):
    return 1 if x >= 0 else 0

def generate_truth_table(n, gate_type):
    truth_table = []
    for bits in itertools.product([0, 1], repeat=n):
        if gate_type == "AND":
            output = int(all(bits))
        elif gate_type == "OR":
            output = int(any(bits))
        else:
            raise ValueError("Only AND/OR supported")
        truth_table.append((list(bits), output))
    return truth_table

def train_perceptron_n_input(n, gate_type, learning_rate=0.1, max_epochs=100):
    truth_table = generate_truth_table(n, gate_type)
    weights = [0.0] * n
    bias = 0.0

    print(f"\nTraining Perceptron for {gate_type} Gate with {n} inputs")
    print(f"Initial weights: {weights}, bias: {bias}\n")

    for epoch in range(max_epochs):
        error_flag = False
        print(f"Epoch {epoch + 1}")
        for inputs, target in truth_table:
            weighted_sum = sum(w * x for w, x in zip(weights, inputs)) + bias
            output = step(weighted_sum)
            error = target - output

            if error != 0:
                error_flag = True
                for i in range(n):
                    weights[i] += learning_rate * error * inputs[i]
                bias += learning_rate * error
            
            print(f"Input: {inputs} | Target: {target} | Output: {output} | Weights: {weights} | Bias: {bias:.2f}")
        
        if not error_flag:
            print("\nTraining converged.")
            break
        print()

    # Final Evaluation
    correct = 0
    for inputs, target in truth_table:
        output = step(sum(w * x for w, x in zip(weights, inputs)) + bias)
        if output == target:
            correct += 1
    accuracy = correct / len(truth_table) * 100

    print("\nFinal Results:")
    print(f"Weights: {weights}")
    print(f"Bias: {bias:.2f}")
    print(f"Accuracy: {accuracy:.2f}%")
    print("-" * 50)

# Run for both AND and OR gates with n = 3 and 4
train_perceptron_n_input(n=3, gate_type="AND")
train_perceptron_n_input(n=3, gate_type="OR")
train_perceptron_n_input(n=4, gate_type="AND")
train_perceptron_n_input(n=4, gate_type="OR")
