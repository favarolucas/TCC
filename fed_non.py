from collections import OrderedDict
from typing import Dict, List, Tuple

import numpy as np
import torch

import model_non as mdl
import flwr as fl

#DEVICE: str = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
DEVICE: str = torch.device("cpu")


class ModelClient(fl.client.NumPyClient):
    """Flower client implementing model-10 image classification using
    PyTorch."""

    def __init__(
        self,
        model,
        trainloader,
        testloader,
        val_loader,
        test_labels,
        num_examples
    ):
        self.model = model
        self.trainloader = trainloader
        self.testloader = testloader
        self.val_loader = val_loader
        self.test_labels = test_labels
        self.num_examples = num_examples

    def get_parameters(self):
        # Return model parameters as a list of NumPy ndarrays
        return [val.cpu().numpy() for _, val in self.model.state_dict().items()]

    def set_parameters(self, parameters: List[np.ndarray]) -> None:
        # Set model parameters from a list of NumPy ndarrays
        params_dict = zip(self.model.state_dict().keys(), parameters)
        state_dict = OrderedDict({k: torch.tensor(v) for k, v in params_dict})
        self.model.load_state_dict(state_dict, strict=True)

    def fit(self, parameters, config):
        # Set model parameters, train model, return updated model parameters
        self.set_parameters(parameters)
        mdl.train(self.model, self.trainloader, self.val_loader)
        return self.get_parameters(), self.num_examples["trainset"], {}

    def evaluate(self, parameters, config):
        # Set model parameters, evaluate model on local test dataset, return result
        self.set_parameters(parameters)
        loss, accuracy = mdl.test(self.model, self.testloader, self.test_labels)
        return float(loss), self.num_examples["testset"], {"accuracy":float(accuracy)} 
        

def main() -> None:
    """Load data, start ModelClient."""

    # Load model and data
    model = mdl.Net()
    model.to(DEVICE)
    print("Load data")
    trainloader, val_loader,testloader, test_labels, num_examples = mdl.load_data()
    # Start client
    print("Client")
    client = ModelClient(model, trainloader, testloader,val_loader, test_labels, num_examples)
    fl.client.start_numpy_client("localhost:8080", client)


if __name__ == "__main__":
    main()