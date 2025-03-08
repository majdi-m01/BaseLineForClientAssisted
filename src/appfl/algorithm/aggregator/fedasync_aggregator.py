import copy
import torch
from omegaconf import DictConfig
from appfl.algorithm.aggregator import BaseAggregator
from typing import Union, Dict, OrderedDict, Any, Optional

# Add file logging with file lock
from filelock import FileLock
import json
import os

class FedAsyncAggregator(BaseAggregator):
    """
    FedAsync Aggregator class for Federated Learning.
    For more details, check paper: https://arxiv.org/pdf/1903.03934.pdf
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
            "client_weights_mode", "equal"
        )

        if model is not None:
            self.named_parameters = set()
            for name, _ in self.model.named_parameters():
                self.named_parameters.add(name)
        else:
            self.named_parameters = None

        self.global_state = None  # Models parameters that are used for aggregation, this is unknown at the beginning

        self.staleness_fn = self.__staleness_fn_factory(
            staleness_fn_name=self.aggregator_configs.get("staleness_fn", "constant"),
            **self.aggregator_configs.get("staleness_fn_kwargs", {}),
        )
        self.alpha = self.aggregator_configs.get("alpha", 0.9)
        self.global_step = 0
        self.client_step = {}
        self.step = {}

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

    def aggregate(
        self,
        client_id: Union[str, int],
        local_model: Union[Dict, OrderedDict],
        **kwargs,
    ) -> Dict:
        self.logger.info("server is starting aggregation")

        if client_id not in self.client_step:
            self.client_step[client_id] = 0

        if self.global_step == 0:
            self.logger.info("at global step == 0")
            weight = self.client_sample_size[client_id] / sum(self.client_sample_size.values())
            alpha_t = (
                    self.alpha
                    * self.staleness_fn(self.global_step - self.client_step[client_id])
                    * weight
            )
            self.logger.info("alpha is {}".format(alpha_t))
            for name in self.encryptedModelWeights:
                self.encryptedModelWeights[name] = self.encryptedModelWeights[name].flatten().tolist()
                self.encryptedModelWeights[name] += (local_model[name] - self.encryptedModelWeights[name]) * alpha_t
        else:
            self.logger.info("at global step > 0")
            '''
            alpha_t_without_weight = (
                    self.alpha
                    * self.staleness_fn(self.global_step - self.client_step[client_id])
            )
            '''
            #self.logger.info("alpha_without_weight is {}".format(alpha_t_without_weight))
            for name in self.encryptedModelWeights:
                self.encryptedModelWeights[name] += local_model[name] #* alpha_t_without_weight

        self.global_step += 1
        self.client_step[client_id] = self.global_step

        # Define file path and lock file path
        log_file = "aggregation_steps.json"
        lock_file = log_file + ".lock"

        # Data to be written
        step_data = {
            "global_step": self.global_step,
            "client_step": dict(self.client_step)  # Convert to regular dict if it's a special type
        }

        # Use file lock to ensure thread-safe writing
        with FileLock(lock_file):
            # If file exists, read existing data and update it
            if os.path.exists(log_file):
                with open(log_file, 'r') as f:
                    try:
                        existing_data = json.load(f)
                    except json.JSONDecodeError:
                        existing_data = {}
                existing_data.update(step_data)
                step_data = existing_data

            # Write updated data to file
            with open(log_file, 'w') as f:
                json.dump(step_data, f, indent=4)

        self.logger.info("server is finishing aggregation")

        return {name: param.serialize() for name, param in self.encryptedModelWeights.items()}

    def __staleness_fn_factory(self, staleness_fn_name, **kwargs):
        if staleness_fn_name == "constant":
            return lambda u: 1
        elif staleness_fn_name == "polynomial":
            a = kwargs["a"]
            return lambda u: (u + 1) ** (-a)
        elif staleness_fn_name == "hinge":
            a = kwargs["a"]
            b = kwargs["b"]
            return lambda u: 1 if u <= b else 1.0 / (a * (u - b) + 1.0)
        else:
            raise NotImplementedError
