import flwr as fl

strategy = fl.server.strategy.FedAvg(
    min_fit_clients=5,  # Minimum number of clients to be sampled for the next round
    min_available_clients=5,  # Minimum number of clients that need to be connected to the server before a training round can start
    min_eval_clients = 1
)

if __name__ == "__main__":
    fl.server.start_server("localhost:8080", config={"num_rounds": 10}, strategy=strategy)
    
    