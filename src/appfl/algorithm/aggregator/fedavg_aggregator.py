import copy
import random

import torch
from omegaconf import DictConfig
from appfl.algorithm.aggregator import BaseAggregator
from typing import Union, Dict, OrderedDict, Any, Optional

# TODO :D import necessary libraries
import tenseal as ts


class FedAvgAggregator(BaseAggregator):
    """
    :param `model`: An optional instance of the model to be trained in the federated learning setup.
        This can be useful for aggregating parameters that does requires gradient, such as the batch
        normalization layers. If not provided, the aggregator will only aggregate the parameters
        sent by the clients.
    :param `aggregator_configs`: Configuration for the aggregator. It should be specified in the YAML
        configuration file under `aggregator_kwargs`.
    :param `logger`: An optional instance of the logger to be used for logging.
    """

    def __init__(
            self,
            model: Optional[torch.nn.Module] = None,
            aggregator_configs: DictConfig = DictConfig({}),
            logger: Optional[Any] = None
    ):
        self.model = model
        self.logger = logger
        self.aggregator_configs = aggregator_configs
        self.client_weights_mode = aggregator_configs.get("client_weights_mode", "sample_size")

        if self.model is not None:
            self.named_parameters = set()
            for name, _ in self.model.named_parameters():
                self.named_parameters.add(name)
        else:
            self.named_parameters = None

        # TODO removed this
        # self.global_state = None  # Models parameters that are used for aggregation, this is unknown at the beginning

        self.step = {}
        self.encryptedModelWeights = copy.deepcopy(self.model.state_dict())
        self.initRequest = 0
        self.global_step = 0

    def get_parameters(self, **kwargs) -> Any:
        if self.initRequest == 1:
            total_sample_sizes = [sum(self.client_sample_size.values())]
            return self.encryptedModelWeights, total_sample_sizes
        if self.initRequest == 0 or self.initRequest == 1:
            self.initRequest += 1
        return self.encryptedModelWeights

    def zeros_like_encrypted_model(self) -> Dict:
        zero_like_dict = {}
        for name in self.encryptedModelWeights:
            if not isinstance(self.encryptedModelWeights[name], ts.CKKSVector):
                zero_like_dict[name] = [0 for _ in self.encryptedModelWeights[name]]
            elif isinstance(self.encryptedModelWeights[name], ts.CKKSVector):
                shape = self.encryptedModelWeights[name].shape[0]
                zero_like_dict[name] = [0] * shape
        return zero_like_dict

    def aggregate(self, local_models: Dict[Union[str, int], Union[Dict, OrderedDict]], **kwargs) -> Dict:
        """
        Take the weighted average of local models from clients and return the global model.
        """
        self.logger.info("server is aggregating")
        if self.global_step == 0:
            self.global_step += 1
            for name in self.encryptedModelWeights:
                self.encryptedModelWeights[name] = self.encryptedModelWeights[name].flatten().tolist()
        global_state = self.zeros_like_encrypted_model()


        for client_id, model in local_models.items():
            for name in self.encryptedModelWeights:
                global_state[name] += model[name]
            self.logger.info(f"server is done aggregating client {client_id}")


        self.logger.info("server is starting serializing")
        for name, param in global_state.items():
            global_state[name] = param.serialize()
        self.logger.info("server done aggregating")
        return global_state