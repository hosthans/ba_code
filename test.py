import torch
import torch.optim as optim
import torch.nn as nn

class SimpleModel(nn.Module):
    def __init__(self):
        super(SimpleModel, self).__init__()
        self.fc = nn.Linear(4, 3)
    
    def forward(self, x):
        return self.fc(x)
    
def getNoise(c0, ct, grad, step_size, epsilon, p=2):
    c = ct + step_size * torch.sign(grad)
    l = torch.argmax(c0)
    m = torch.relu(torch.max(c)-c[l])
    c[l] = c[l] + m
    n = c - c0
    
    # überprüfen, ob p-norm(epsilon) von n die Verzerrungsgrenze überschreitet
    norm_n = torch.norm(n, p)
    
    if norm_n > epsilon:
        n = (epsilon * n)/norm_n
    
    return n

def defenseAlgorithm(x, F, iterations, step_size, epsilon):
    # eig mit Inversionsmodell
    c0 = F(x)
    c = F(x)

    for i in range(iterations):
        recon = F(c)
        loss = nn.MSELoss()(x, recon)
        loss.backward()
        grad = c.grad
        c = c0 + getNoise(c0, c, grad, step_size, epsilon)
    
    adversarial_output = c
    return adversarial_output


def main():
    model = SimpleModel()
    optimizer = optim.SGD(model.parameters(), lr=0.01)
    print("Model builded")

    input_data = torch.randn(1, 4, requires_grad=False)
    print(f"Input data: {input_data}")

    conf_score = torch.softmax(model(input_data), dim=1)
    print(f"Confidence-Score: {conf_score}")

    new_conf = defenseAlgorithm(x=input_data, F=model, iterations=10, step_size=0.01, epsilon=0.1)
    print(f"new Conf-Score: {new_conf}")



if __name__ == '__main__':
    main()