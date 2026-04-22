import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
import matplotlib.pyplot as plt

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# =========================
# Prunable Linear Layer
# =========================
class PrunableLinear(nn.Module):
    def __init__(self, in_features, out_features):
        super().__init__()
        self.weight = nn.Parameter(torch.randn(out_features, in_features) * 0.01)
        self.bias = nn.Parameter(torch.zeros(out_features))
        
        # Gate scores (learnable)
        self.gate_scores = nn.Parameter(torch.randn(out_features, in_features))

    def forward(self, x):
        gates = torch.sigmoid(self.gate_scores)
        pruned_weights = self.weight * gates
        return nn.functional.linear(x, pruned_weights, self.bias)

    def get_gates(self):
        return torch.sigmoid(self.gate_scores)


# =========================
# Neural Network
# =========================
class PruningNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = PrunableLinear(32*32*3, 512)
        self.fc2 = PrunableLinear(512, 256)
        self.fc3 = PrunableLinear(256, 10)
        self.relu = nn.ReLU()

    def forward(self, x):
        x = x.view(x.size(0), -1)
        x = self.relu(self.fc1(x))
        x = self.relu(self.fc2(x))
        x = self.fc3(x)
        return x

    def get_all_gates(self):
        gates = []
        for layer in [self.fc1, self.fc2, self.fc3]:
            gates.append(layer.get_gates().view(-1))
        return torch.cat(gates)


# =========================
# Data Loader
# =========================
def get_data():
    transform = transforms.Compose([
        transforms.ToTensor()
    ])

    trainset = torchvision.datasets.CIFAR10(root='./data', train=True,
                                            download=True, transform=transform)
    testset = torchvision.datasets.CIFAR10(root='./data', train=False,
                                           download=True, transform=transform)

    trainloader = torch.utils.data.DataLoader(trainset, batch_size=128, shuffle=True)
    testloader = torch.utils.data.DataLoader(testset, batch_size=128, shuffle=False)

    return trainloader, testloader


# =========================
# Sparsity Calculation
# =========================
def calculate_sparsity(gates, threshold=1e-2):
    total = gates.numel()
    pruned = (gates < threshold).sum().item()
    return (pruned / total) * 100


# =========================
# Training Function
# =========================
def train_model(lambda_val):
    trainloader, testloader = get_data()
    
    model = PruningNet().to(device)
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    criterion = nn.CrossEntropyLoss()

    epochs = 10

    for epoch in range(epochs):
        model.train()
        total_loss = 0

        for inputs, labels in trainloader:
            inputs, labels = inputs.to(device), labels.to(device)

            optimizer.zero_grad()

            outputs = model(inputs)
            classification_loss = criterion(outputs, labels)

            # Sparsity Loss (L1 on gates)
            gates = model.get_all_gates()
            sparsity_loss = torch.sum(gates)

            loss = classification_loss + lambda_val * sparsity_loss
            loss.backward()
            optimizer.step()

            total_loss += loss.item()

        print(f"Epoch {epoch+1}, Loss: {total_loss:.4f}")

    # =========================
    # Evaluation
    # =========================
    model.eval()
    correct = 0
    total = 0

    with torch.no_grad():
        for inputs, labels in testloader:
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            _, predicted = torch.max(outputs, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    accuracy = 100 * correct / total
    gates = model.get_all_gates().detach().cpu()
    sparsity = calculate_sparsity(gates)

    print(f"\nLambda: {lambda_val}")
    print(f"Accuracy: {accuracy:.2f}%")
    print(f"Sparsity: {sparsity:.2f}%")

    return gates, accuracy, sparsity


# =========================
# Main Experiment
# =========================
if __name__ == "__main__":
    lambdas = [0.0001, 0.001, 0.01]
    results = []

    for lam in lambdas:
        gates, acc, sparsity = train_model(lam)
        results.append((lam, acc, sparsity))

    # Plot gate distribution (last run)
    plt.hist(gates.numpy(), bins=50)
    plt.title("Gate Distribution")
    plt.xlabel("Gate Value")
    plt.ylabel("Frequency")
    plt.show()

    print("\nFinal Results:")
    for r in results:
        print(f"Lambda: {r[0]}, Accuracy: {r[1]:.2f}, Sparsity: {r[2]:.2f}")