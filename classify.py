import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

class Classifier(nn.Module):
    def __init__(self):
        super(Classifier, self).__init__()
        self.conv1 = nn.Conv2d(1, 32, 3, 1)
        self.conv2 = nn.Conv2d(32, 64, 3, 1)
        self.fc1 = nn.Linear(64*5*5, 128)
        self.fc2 = nn.Linear(128, 10)

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = F.max_pool2d(x, 2, 2)
        x = F.relu(self.conv2(x))
        x = F.max_pool2d(x, 2, 2)
        x = x.view(-1, 64*5*5)
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return F.log_softmax(x, dim=1)

def train_classifier(dl, classifier, classifier_optimizer, n_epochs, device):
    classifier.train()
    for epoch in range(n_epochs):
        for batch_idx, (images, labels) in enumerate(dl):
            images, labels = images.to(device), labels.to(device)

            classifier_optimizer.zero_grad()

            outputs = classifier(images)

            loss = F.nll_loss(outputs, labels)

            loss.backward()
            classifier_optimizer.step()

            if batch_idx % 200 == 0:
                print(f'Epoch [{epoch+1}/{n_epochs}], Step [{batch_idx}/{len(dl)}], Loss: {loss.item()}')


def show_generated_images(epoch, generator, device, n_cols=8, z_size=100):
    generator.eval()
    z = torch.randn(64, z_size, device=device)
    fake_images = generator(z)
    fake_images = fake_images.view(-1, 1, 28, 28).cpu().detach()

    # Reshape the generated images to match the standard image format
    # fake_images = fake_images.view(-1, 1, 28, 28)

    display_images(fake_images, n_cols=n_cols)

def evaluate_classifier(dl, classifier, device):
    classifier.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for images, labels in dl:
            images = images.view(-1, 1, 28, 28).to(device)  # Reshape images to match the expected input shape
            labels = labels.to(device)
            outputs = classifier(images)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
    accuracy = correct / total
    return accuracy

if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    classifier = Classifier().to(device)
    classifier_optimizer = optim.Adam(classifier.parameters(), lr=0.001)

    # Assuming train_ds and z_size are defined somewhere in your code
    dl = torch.utils.data.DataLoader(train_ds, batch_size=64, shuffle=True)

    n_classifier_epochs = 5
    train_classifier(dl, classifier, classifier_optimizer, n_classifier_epochs, device)

    real_accuracy = evaluate_classifier(dl, classifier, device)
    print(f'Accuracy on real images: {real_accuracy}')

    # Assuming g is your generator function
    z = torch.randn(10000, z_size, device=device)
    fake_images = g(z)
    fake_labels = torch.zeros(10000, dtype=torch.long) 

    fake_dataset = torch.utils.data.TensorDataset(fake_images, fake_labels)
    fake_dl = torch.utils.data.DataLoader(fake_dataset, batch_size=64, shuffle=True)

    fake_accuracy = evaluate_classifier(fake_dl, classifier, device)
    print(f'Accuracy on fake images: {fake_accuracy}')
