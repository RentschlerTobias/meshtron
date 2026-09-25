from dataclasses import dataclass
import math


@dataclass
class EpochMetrics:
    nll_per_token: float
    bits_per_token: float
    perplexity: float
    n_tokens: int
    n_batches: int


class TokenLossAccumulator:
    """Akkumuliert Cross-Entropy als Summe ueber valide (nicht-pad) Tokens.

    Liefert pro-Token gemittelte Metriken, die unabhaengig von Batch-Size,
    Sequenzlaenge und Padding-Anteil sind.
    """

    def __init__(self) -> None:
        self.loss_sum = 0.0
        self.n_tokens = 0
        self.n_batches = 0

    def update(self, loss_sum: float, n_tokens: int) -> None:
        if n_tokens <= 0:
            return
        self.loss_sum += float(loss_sum)
        self.n_tokens += int(n_tokens)
        self.n_batches += 1

    def compute(self) -> EpochMetrics:
        if self.n_tokens == 0:
            return EpochMetrics(
                nll_per_token=float("nan"),
                bits_per_token=float("nan"),
                perplexity=float("nan"),
                n_tokens=0,
                n_batches=self.n_batches,
            )
        nll = self.loss_sum / self.n_tokens
        return EpochMetrics(
            nll_per_token=nll,
            bits_per_token=nll / math.log(2),
            perplexity=math.exp(nll),
            n_tokens=self.n_tokens,
            n_batches=self.n_batches,
        )
