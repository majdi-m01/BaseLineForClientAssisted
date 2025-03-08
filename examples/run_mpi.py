import argparse
import os
import random
import time
import warnings
from mpi4py import MPI
from omegaconf import OmegaConf
from appfl.agent import ClientAgent, ServerAgent
from appfl.comm.mpi import MPIClientCommunicator, MPIServerCommunicator

# TODO import crypto library
import tenseal as ts

# controls precision of the fractional part
bits_scale = 26

def create_ckks_context():
    new_context = ts.context(ts.SCHEME_TYPE.CKKS, 8192, [31, bits_scale, bits_scale, bits_scale, bits_scale, bits_scale, bits_scale, 31])
    new_context.global_scale = pow(2, bits_scale)
    new_context.generate_relin_keys()
    new_context.generate_galois_keys()
    new_context.data.auto_relin = True
    new_context.data.auto_rescale = True
    new_context.data.auto_mod_switch = True
    return new_context

ckks_context = create_ckks_context()

serialized_context = ckks_context.serialize(save_secret_key=True)

with open('CKKS_keys_and_context.txt', 'wb') as f:
    f.write(serialized_context)

# TODO ignore warning
warnings.filterwarnings("ignore", category=FutureWarning, message=".*torch.load.*weights_only=False.*")
warnings.filterwarnings("ignore", message="The input does not fit in a single ciphertext, and some operations will be disabled.")

argparse = argparse.ArgumentParser()

warnings.filterwarnings("ignore", category=DeprecationWarning)
argparse.add_argument(
    "--server_config",
    type=str,
    default="./resources/configs/cifar10/server_fedasync.yaml",
)
argparse.add_argument(
    "--client_config", type=str, default="./resources/configs/cifar10/client_1.yaml"
)
args = argparse.parse_args()

comm = MPI.COMM_WORLD
rank = comm.Get_rank()
size = comm.Get_size()
num_clients = size - 1

with open('metrics_log.txt', 'w'):
    pass  # The file is now empty

with open('aggregation_steps.json', 'w'):
    pass  # The file is now empty

try:
    os.remove('latest_updated_global_model.pkl')
    print(f"File 'latest_updated_global_model.pkl' has been deleted.")
except FileNotFoundError:
    print(f"File 'latest_updated_global_model.pkl' does not exist.")

# TODO :D Define the mapping of rank to delay (in seconds)
rank_delay_mapping = {
    1: 0,
    2: 30,
    3: 60,
    4: 90,
    5: 120,
    6: 75,
    7: 90,
    8: 105,
    9: 120,
    10: 135,
    11: 150,
}

if rank == 0:
    # Load and set the server configurations
    server_agent_config = OmegaConf.load(args.server_config)
    server_agent_config.server_configs.scheduler_kwargs.num_clients = num_clients
    if hasattr(server_agent_config.server_configs.aggregator_kwargs, "num_clients"):
        server_agent_config.server_configs.aggregator_kwargs.num_clients = num_clients
    # Create the server agent and communicator
    server_agent = ServerAgent(server_agent_config=server_agent_config)
    server_communicator = MPIServerCommunicator(
        comm, server_agent, logger=server_agent.logger
    )
    # Start the server to serve the clients
    server_communicator.serve()
else:
    # Set the client configurations
    client_agent_config = OmegaConf.load(args.client_config)
    client_agent_config.train_configs.logging_id = f"Client{rank}"
    client_agent_config.data_configs.dataset_kwargs.num_clients = num_clients
    client_agent_config.data_configs.dataset_kwargs.client_id = rank - 1
    client_agent_config.data_configs.dataset_kwargs.visualization = (
        True if rank == 1 else False
    )

    # TODO Add rank to the client agent configuration
    client_agent_config.rank = rank  # Dynamically add the rank to the config

    # Create the client agent and communicator
    client_agent = ClientAgent(client_agent_config=client_agent_config)
    client_communicator = MPIClientCommunicator(comm, server_rank=0)
    # Load the configurations and initial global model
    client_config = client_communicator.get_configuration()
    client_agent.load_config(client_config)
    init_global_model = client_communicator.get_global_model(init_model=True)
    client_agent.load_initial_parameters(init_global_model) #TODO load initial parameters
    # Send the sample size to the server
    sample_size = client_agent.get_sample_size()
    client_communicator.invoke_custom_action(
        action="set_sample_size", sample_size=sample_size
    )
    init_global_model, total_sample_sizes = client_communicator.get_global_model(init_model=True)
    client_agent.load_total_sample_sizes(total_sample_sizes[0])

    # Generate data readiness report
    if (
        hasattr(client_config, "data_readiness_configs")
        and hasattr(client_config.data_readiness_configs, "generate_dr_report")
        and client_config.data_readiness_configs.generate_dr_report
    ):
        data_readiness = client_agent.generate_readiness_report(client_config)
        client_communicator.invoke_custom_action(
            action="get_data_readiness_report", **data_readiness
        )

    # Local training and global model update iterations
    while True:
        # TODO :D Introduce a delay based on the rank
        #delay = rank_delay_mapping.get(rank, 0)  # Default to 0 seconds if rank is not in the mapping
        #if delay > 0:
        #    time.sleep(delay)



        client_agent.train()
        local_model = client_agent.get_parameters()
        if isinstance(local_model, tuple):
            local_model, metadata = local_model[0], local_model[1]
        else:
            metadata = {}
        new_global_model, metadata = client_communicator.update_global_model(
            local_model, **metadata
        )
        if metadata["status"] == "DONE":
            break
        if "local_steps" in metadata:
            client_agent.trainer.train_configs.num_local_steps = metadata["local_steps"]
        client_agent.load_parameters(new_global_model)
    client_communicator.invoke_custom_action(action="close_connection")
