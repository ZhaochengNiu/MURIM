import os
import torch
import flwr as fl
from typing import Dict, List, Tuple, Optional
from data_utils import display_sample, load_datasets
from flower_client import client_fn, DEVICE
from model import Net, get_parameters, test, set_parameters

# Constants
NUM_CLIENTS = int(os.getenv("NUM_CLIENTS", 10))
NUM_ROUNDS = int(os.getenv("NUM_ROUNDS", 3))
FRACTION_FIT = float(os.getenv("FRACTION_FIT", 0.3))
FRACTION_EVALUATE = float(os.getenv("FRACTION_EVALUATE", 0.3))
MIN_FIT_CLIENTS = int(os.getenv("MIN_FIT_CLIENTS", 3))
MIN_EVALUATE_CLIENTS = int(os.getenv("MIN_EVALUATE_CLIENTS", 3))

# Load data
trainloaders, valloaders, testloader = load_datasets()
display_sample(trainloaders)

# Federated Learning Strategy


def evaluate(server_round: int, parameters: fl.common.NDArrays, config: Dict[str, fl.common.Scalar]) -> Optional[Tuple[float, Dict[str, fl.common.Scalar]]]:
    net = Net().to(DEVICE)
    valloader = valloaders[0]
    set_parameters(net, parameters)
    loss, accuracy = test(net, valloader)
    print(f"Server-side evaluation loss {loss} / accuracy {accuracy}")
    return loss, {"accuracy": accuracy}


def fit_config(server_round: int):
    config = {
        "server_round": server_round,
        "local_epochs": 1 if server_round < 2 else 2,
    }
    return config


strategy = fl.server.strategy.FedAvg(
    fraction_fit=FRACTION_FIT,
    fraction_evaluate=FRACTION_EVALUATE,
    min_fit_clients=MIN_FIT_CLIENTS,
    min_evaluate_clients=MIN_EVALUATE_CLIENTS,
    min_available_clients=NUM_CLIENTS,
    initial_parameters=fl.common.ndarrays_to_parameters(get_parameters(Net())),
    evaluate_fn=evaluate,
    on_fit_config_fn=fit_config,
)

fl.simulation.start_simulation(
    client_fn=client_fn,  # This should be imported from flower_client
    num_clients=NUM_CLIENTS,
    config=fl.server.ServerConfig(num_rounds=NUM_ROUNDS),
    strategy=strategy
)
