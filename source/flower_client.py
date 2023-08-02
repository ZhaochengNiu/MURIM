import torch
import flwr as fl
from model import Net, get_parameters, set_parameters
from data_utils import BATCH_SIZE, NUM_CLIENTS, load_datasets

DEVICE = torch.device("cpu")  # Adjust this if you want to use GPU


class FlowerClient(fl.client.NumPyClient):
    def __init__(self, net, trainloader, valloader):
        self.net = net
        self.trainloader = trainloader
        self.valloader = valloader

    def get_parameters(self, config):
        return get_parameters(self.net)

    def fit(self, parameters, config):
        set_parameters(self.net, parameters)
        # You might want to define the train function
        train(self.net, self.trainloader, epochs=1)
        return get_parameters(self.net), len(self.trainloader), {}

    def evaluate(self, parameters, config):
        set_parameters(self.net, parameters)
        # You might want to define the test function
        loss, accuracy = test(self.net, self.valloader)
        return float(loss), len(self.valloader), {"accuracy": float(accuracy)}


def client_fn(cid: str) -> FlowerClient:
    # Load model
    net = Net().to(DEVICE)

    # Note: each client gets a different trainloader/valloader
    trainloaders, valloaders, _ = load_datasets()
    trainloader = trainloaders[int(cid)]
    valloader = valloaders[int(cid)]

    # Create a single Flower client representing a single organization
    return FlowerClient(net, trainloader, valloader)
