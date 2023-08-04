import os
import torch
import flwr as fl
from model import Net, get_parameters, set_parameters, test
from data_utils import load_datasets
from logging import INFO, DEBUG
from flwr.common.logger import log

# Load data and environment variables
trainloaders, valloaders, testloader = load_datasets()
DEVICE = torch.device(os.getenv("DEVICE", "cpu"))


class FlowerClient(fl.client.NumPyClient):
    def __init__(self, cid, net, trainloader, valloader):
        self.cid = cid
        self.net = net
        self.trainloader = trainloader
        self.valloader = valloader

    def train(self, epochs: int):
        criterion = torch.nn.CrossEntropyLoss()
        optimizer = torch.optim.Adam(self.net.parameters())
        self.net.train()
        for epoch in range(epochs):
            correct, total, epoch_loss = 0, 0, 0.0
            for images, labels in self.trainloader:
                images, labels = images.to(DEVICE), labels.to(DEVICE)
                optimizer.zero_grad()
                outputs = self.net(images)
                loss = criterion(outputs, labels)
                loss.backward()
                optimizer.step()
                epoch_loss += loss.item()
                total += labels.size(0)
                correct += (torch.max(outputs, 1)[1] == labels).sum().item()
            # Adjusted to divide by number of batches
            epoch_loss /= len(self.trainloader)
            epoch_acc = correct / total
            print(
                f"Epoch {epoch+1}: train loss {epoch_loss}, accuracy {epoch_acc}")

    def get_parameters(self, config):
        print(f"[Client {self.cid}] get_parameters")
        return get_parameters(self.net)

    def fit(self, parameters, config):
        server_round = config["server_round"]
        local_epochs = config["local_epochs"]
        print(
            f"[Client {self.cid}, round {server_round}] fit, config: {config}")
        log(INFO, f"Printing a custom INFO message at the start of fit() :)")
        set_parameters(self.net, parameters)
        log(DEBUG, f"Client {self.cid} is doing fit() with config: {config}")
        self.train(epochs=local_epochs)
        return get_parameters(self.net), len(self.trainloader), {}

    def evaluate(self, parameters, config):
        print(f"[Client {self.cid}] evaluate, config: {config}")
        set_parameters(self.net, parameters)
        loss, accuracy = test(self.net, self.valloader)
        return float(loss), len(self.valloader), {"accuracy": float(accuracy)}


def client_fn(cid) -> FlowerClient:
    net = Net().to(DEVICE)
    trainloader = trainloaders[int(cid)]
    valloader = valloaders[int(cid)]
    return FlowerClient(cid, net, trainloader, valloader)
