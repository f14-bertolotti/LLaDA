import models
import torch

class RandomRemaskingLLaDA(models.LLaDA):

    @torch.no_grad()
    def loglikelihood(self, requests):
        raise NotImplementedError()

    @torch.no_grad()
    def loglikelihood_rolling(self, requests):
        raise NotImplementedError()

    def generate_until(
        self,
        batch,
        steps=128,
        batch_size=4,
        **kwargs
    ):
        """
        Generate until the model has produced a complete sequence or until the maximum number of steps is reached.
        """
