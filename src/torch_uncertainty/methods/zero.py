import torch
from einops import rearrange
from torch import Tensor, nn
from torch.special import entr


class Zero(nn.Module):
    def __init__(
        self, core_model: nn.Module, num_tta: int, filter_views: float = 0.1, eps: float = 1e-8
    ) -> None:
        """Zero for test-time adaptation.

        Zero performs "0-temperature averaging" (i.e. majority voting) at evaluation. It starts
        by filtering the :attr:`filter_views` most confident predictions, and returns the majority vote
        as a prediction. If used during training, the predictions will be those of the inner-model
        passed as argument (:attr:`model`).

        Args:
            core_model (nn.Module): The inner model to train.
            num_tta (int): The number of views at evaluation time.
            filter_views (float): Filter out 1-:attr:`filter_views` of the predictions of the augmented views.
                Defaults to ``0.1``.
            eps (float): for computational stability. Defaults to ``1e-8``;
        """
        super().__init__()
        _zero_checks(num_tta, filter_views, eps)
        self.core_model = core_model
        self.filter = filter_views
        self.kept_views = int(filter_views * num_tta)
        self.num_tta = num_tta
        self.eps = eps

    def eval_forward(self, x: Tensor) -> Tensor:
        # predict and separate the views from the batch
        all_predictions = rearrange(self.core_model(x), "(b v) c -> b v c", v=self.num_tta)
        batch_size, _, num_classes = all_predictions.shape
        entropies = entr(all_predictions).sum(2)

        # Get the index of the most confident predictions on the views
        conf_idx = torch.argsort(entropies, dim=-1)
        votes = all_predictions.argmax(-1)

        # Count the votes
        predictions = torch.zeros((batch_size, num_classes), device=all_predictions.device)
        for img_id, img_votes in enumerate(votes):
            predictions[img_id, :] += torch.bincount(
                img_votes[conf_idx[img_id, : self.kept_views]], minlength=all_predictions.shape[-1]
            )
            maximum = predictions[img_id, :].max()
            i = 0
            # If the maximum is shared among two predictions, look at an additional one
            while (
                self.kept_views + i < self.num_tta
                and torch.sum(1 * (predictions[img_id, :] == maximum)) > 1
            ):
                predictions[img_id, img_votes[conf_idx[img_id, self.kept_views + i]]] += 1
                maximum = predictions[img_id, :].max()
                i += 1

        predictions /= self.num_tta
        # We will apply the softmax in the routine, so let's apply its inverse here
        return (predictions + self.eps).log()

    def forward(self, x: Tensor) -> Tensor:
        if self.training:
            return self.core_model.forward(x)
        return self.eval_forward(x)


def _zero_checks(num_tta: int, filter_views: float, eps: float) -> None:
    if filter_views <= 0.0 or filter_views > 1.0:
        raise ValueError(f"`filter_views` must be in the range ]0, 1]. Got {filter_views}.")
    if num_tta < 1 / filter_views:
        raise ValueError(
            f"`num_tta` should be greater than 1/filter_views to use Zero. Got {num_tta} < {1 / filter_views}."
        )
    if eps <= 0:
        raise ValueError(f"`eps` should be strictly positive. Got {eps}.")
