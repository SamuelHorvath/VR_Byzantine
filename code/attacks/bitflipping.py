import torch

from worker import ByzantineWorker


class BitFlippingWorker(ByzantineWorker):
    def __str__(self) -> str:
        return "BitFlippingWorker"

    # def get_gradient(self):
    #     # Use self.simulator to get all other workers
    #     # Note that the byzantine worker does not modify the states directly.
    #     return -super().get_gradient()

    def get_gradient(self) -> torch.Tensor:
        return - self.scalar * self._get_saved_grad()
