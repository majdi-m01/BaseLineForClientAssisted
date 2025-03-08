import copy
import torch
from omegaconf import DictConfig
from appfl.algorithm.aggregator import BaseAggregator
from typing import Union, Dict, OrderedDict, Any, Optional
import tenseal as ts

class FedAvgStepBasedAggregator(BaseAggregator):
    """
    :param `model`: An optional instance of the model to be trained in the federated learning setup.
        This can be useful for aggregating parameters that does require gradient, such as the batch
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
        logger: Optional[Any] = None,
    ):
        self.model = model
        self.logger = logger
        self.aggregator_configs = aggregator_configs
        self.client_weights_mode = aggregator_configs.get(
            "client_weights_mode", "sample_size"
        )

        if self.model is not None:
            self.named_parameters = set()
            for name, _ in self.model.named_parameters():
                self.named_parameters.add(name)
        else:
            self.named_parameters = None

        self.encryptedModelWeights = copy.deepcopy(self.model.state_dict())
        self.initRequest = 0
        self.global_step = 0

    def get_parameters(self, **kwargs) -> Any:
        if self.global_step == 0:
            if self.initRequest == 0:
                self.initRequest += 1
                return self.encryptedModelWeights
            if self.initRequest == 1:
                self.initRequest += 1
                total_sample_sizes = [sum(self.client_sample_size.values())]
                return self.encryptedModelWeights, total_sample_sizes
        else:
            for name in self.encryptedModelWeights:
                if isinstance(self.encryptedModelWeights[name], bytes):
                    continue
                else:
                    self.encryptedModelWeights[name] = self.encryptedModelWeights[name].serialize()
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

    def aggregate(
        self,
        client_id: Union[str, int],
        local_model: Union[Dict, OrderedDict],
        **kwargs,
    ) -> Dict:
        """
        Aggregate a single local model and return the updated global model.
        Aggregation rule: Wnew = Wold + weight_i * Wi
        """
        self.logger.info("server is starting aggregation")

        if self.global_step == 0:
            weight = self.client_sample_size[client_id] / sum(self.client_sample_size.values())
            self.logger.info("server is aggregating")
            for name in self.encryptedModelWeights:
                self.encryptedModelWeights[name] = self.encryptedModelWeights[name].flatten().tolist()
                self.encryptedModelWeights[name] += (local_model[name] - self.encryptedModelWeights[name]) * weight
            self.global_step += 1
        else:
            self.logger.info("server is aggregating")
            for name in self.encryptedModelWeights:
                self.encryptedModelWeights[name] += local_model[name]

        self.logger.info("server is finishing aggregation")

        return {name: param.serialize() for name, param in self.encryptedModelWeights.items()}
