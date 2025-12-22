import numpy as np
from functools import partial
from typing import Optional
from dataclasses import dataclass

import torch
import math
from omegaconf import OmegaConf, DictConfig
from itertools import groupby

# Const values for T-1
BLANK_ID = 34
SPACE_ID = 33

class ConfidenceMethodConstants:
    NAMES = ("max_prob", "entropy")
    ENTROPY_TYPES = ("gibbs", "tsallis", "renyi")
    ENTROPY_NORMS = ("lin", "exp")

    @classmethod
    def print(cls):
        return (
            cls.__name__
            + ": "
            + str({"NAMES": cls.NAMES, "ENTROPY_TYPES": cls.ENTROPY_TYPES, "ENTROPY_NORMS": cls.ENTROPY_NORMS})
        )


class ConfidenceConstants:
    AGGREGATIONS = ("mean", "min", "max", "prod")

    @classmethod
    def print(cls):
        return cls.__name__ + ": " + str({"AGGREGATIONS": cls.AGGREGATIONS})


@dataclass
class ConfidenceMethodConfig:
    """A Config which contains the method name and settings to compute per-frame confidence scores.

    Args:
        name: The method name (str).
            Supported values:
                - 'max_prob' for using the maximum token probability as a confidence.
                - 'entropy' for using a normalized entropy of a log-likelihood vector.

        entropy_type: Which type of entropy to use (str).
            Used if confidence_method_cfg.name is set to `entropy`.
            Supported values:
                - 'gibbs' for the (standard) Gibbs entropy. If the alpha (α) is provided,
                    the formula is the following: H_α = -sum_i((p^α_i)*log(p^α_i)).
                    Note that for this entropy, the alpha should comply the following inequality:
                    (log(V)+2-sqrt(log^2(V)+4))/(2*log(V)) <= α <= (1+log(V-1))/log(V-1)
                    where V is the model vocabulary size.
                - 'tsallis' for the Tsallis entropy with the Boltzmann constant one.
                    Tsallis entropy formula is the following: H_α = 1/(α-1)*(1-sum_i(p^α_i)),
                    where α is a parameter. When α == 1, it works like the Gibbs entropy.
                    More: https://en.wikipedia.org/wiki/Tsallis_entropy
                - 'renyi' for the Rényi entropy.
                    Rényi entropy formula is the following: H_α = 1/(1-α)*log_2(sum_i(p^α_i)),
                    where α is a parameter. When α == 1, it works like the Gibbs entropy.
                    More: https://en.wikipedia.org/wiki/R%C3%A9nyi_entropy

        alpha: Power scale for logsoftmax (α for entropies). Here we restrict it to be > 0.
            When the alpha equals one, scaling is not applied to 'max_prob',
            and any entropy type behaves like the Shannon entropy: H = -sum_i(p_i*log(p_i))

        entropy_norm: A mapping of the entropy value to the interval [0,1].
            Supported values:
                - 'lin' for using the linear mapping.
                - 'exp' for using exponential mapping with linear shift.
    """

    name: str = "entropy"
    entropy_type: str = "tsallis"
    alpha: float = 0.33
    entropy_norm: str = "exp"
    temperature: str = "DEPRECATED"

    def __post_init__(self):
        if self.temperature != "DEPRECATED":
            # self.temperature has type str
            self.alpha = float(self.temperature)
            self.temperature = "DEPRECATED"
        if self.name not in ConfidenceMethodConstants.NAMES:
            raise ValueError(
                f"`name` must be one of the following: "
                f"{'`' + '`, `'.join(ConfidenceMethodConstants.NAMES) + '`'}. Provided: `{self.name}`"
            )
        if self.entropy_type not in ConfidenceMethodConstants.ENTROPY_TYPES:
            raise ValueError(
                f"`entropy_type` must be one of the following: "
                f"{'`' + '`, `'.join(ConfidenceMethodConstants.ENTROPY_TYPES) + '`'}. Provided: `{self.entropy_type}`"
            )
        if self.alpha <= 0.0:
            raise ValueError(f"`alpha` must be > 0. Provided: {self.alpha}")
        if self.entropy_norm not in ConfidenceMethodConstants.ENTROPY_NORMS:
            raise ValueError(
                f"`entropy_norm` must be one of the following: "
                f"{'`' + '`, `'.join(ConfidenceMethodConstants.ENTROPY_NORMS) + '`'}. Provided: `{self.entropy_norm}`"
            )
        

class ConfidenceMethod():
    """Confidence Method Mixin class.

    It initializes per-frame confidence method.
    """

    def _init_confidence_method(self, confidence_method_cfg: Optional[DictConfig] = None, blank_id: int = 1024):
        """Initialize per-frame confidence method from config.
        """
        # OmegaConf.structured ensures that post_init check is always executed
        confidence_method_cfg = OmegaConf.structured(
            ConfidenceMethodConfig()
            if confidence_method_cfg is None
            else confidence_method_cfg
        )

        # set confidence calculation method
        # we suppose that self.blank_id == len(vocabulary)
        self.num_tokens = blank_id + 1
        self.alpha = confidence_method_cfg.alpha

        # init confidence measure bank
        self.confidence_measure_bank = get_confidence_measure_bank()

        measure = None
        # construct measure_name
        measure_name = ""
        if confidence_method_cfg.name == "max_prob":
            measure_name = "max_prob"
        elif confidence_method_cfg.name == "entropy":
            measure_name = '_'.join(
                [confidence_method_cfg.name, confidence_method_cfg.entropy_type, confidence_method_cfg.entropy_norm]
            )
        else:
            raise ValueError(f"Unsupported `confidence_method_cfg.name`: `{confidence_method_cfg.name}`")
        if measure_name not in self.confidence_measure_bank:
            raise ValueError(f"Unsupported measure setup: `{measure_name}`")
        measure = partial(self.confidence_measure_bank[measure_name], v=self.num_tokens, t=self.alpha)

        self._confidence_measure = measure

    def _get_confidence(self, x: torch.Tensor) -> list[float]:
        """Compute confidence, return list of confidence items for each item in batch"""
        return self._get_confidence_tensor(x).tolist()

    def _get_confidence_tensor(self, x: torch.Tensor) -> torch.Tensor:
        """Compute confidence, return tensor"""
        return self._confidence_measure(torch.nan_to_num(x))


def get_confidence_measure_bank():
    """Generate a dictionary with confidence measure functionals.

    Supported confidence measures:
        max_prob: normalized maximum probability
        entropy_gibbs_lin: Gibbs entropy with linear normalization
        entropy_gibbs_exp: Gibbs entropy with exponential normalization
        entropy_tsallis_lin: Tsallis entropy with linear normalization
        entropy_tsallis_exp: Tsallis entropy with exponential normalization
        entropy_renyi_lin: Rényi entropy with linear normalization
        entropy_renyi_exp: Rényi entropy with exponential normalization

    Returns:
        dictionary with lambda functions.
    """
    # helper functions
    # Gibbs entropy is implemented without alpha
    neg_entropy_gibbs = lambda x: (x.exp() * x).sum(-1)
    neg_entropy_alpha = lambda x, t: (x * t).exp().sum(-1)
    neg_entropy_alpha_gibbs = lambda x, t: ((x * t).exp() * x).sum(-1)
    # too big for a lambda
    def entropy_tsallis_exp(x, v, t):
        exp_neg_max_ent = math.exp((1 - math.pow(v, 1 - t)) / (1 - t))
        return (((1 - neg_entropy_alpha(x, t)) / (1 - t)).exp() - exp_neg_max_ent) / (1 - exp_neg_max_ent)

    def entropy_gibbs_exp(x, v, t):
        exp_neg_max_ent = math.pow(v, -t * math.pow(v, 1 - t))
        return ((neg_entropy_alpha_gibbs(x, t) * t).exp() - exp_neg_max_ent) / (1 - exp_neg_max_ent)

    # use Gibbs entropies for Tsallis and Rényi with t == 1.0
    entropy_gibbs_lin_baseline = lambda x, v: 1 + neg_entropy_gibbs(x) / math.log(v)
    entropy_gibbs_exp_baseline = lambda x, v: (neg_entropy_gibbs(x).exp() * v - 1) / (v - 1)
    # fill the measure bank
    confidence_measure_bank = {}
    # Maximum probability measure is implemented without alpha
    confidence_measure_bank["max_prob"] = (
        lambda x, v, t: (x.max(dim=-1)[0].exp() * v - 1) / (v - 1)
        if t == 1.0
        else ((x.max(dim=-1)[0] * t).exp() * math.pow(v, t) - 1) / (math.pow(v, t) - 1)
    )
    confidence_measure_bank["entropy_gibbs_lin"] = (
        lambda x, v, t: entropy_gibbs_lin_baseline(x, v)
        if t == 1.0
        else 1 + neg_entropy_alpha_gibbs(x, t) / math.log(v) / math.pow(v, 1 - t)
    )
    confidence_measure_bank["entropy_gibbs_exp"] = (
        lambda x, v, t: entropy_gibbs_exp_baseline(x, v) if t == 1.0 else entropy_gibbs_exp(x, v, t)
    )
    confidence_measure_bank["entropy_tsallis_lin"] = (
        lambda x, v, t: entropy_gibbs_lin_baseline(x, v)
        if t == 1.0
        else 1 + (1 - neg_entropy_alpha(x, t)) / (math.pow(v, 1 - t) - 1)
    )
    confidence_measure_bank["entropy_tsallis_exp"] = (
        lambda x, v, t: entropy_gibbs_exp_baseline(x, v) if t == 1.0 else entropy_tsallis_exp(x, v, t)
    )
    confidence_measure_bank["entropy_renyi_lin"] = (
        lambda x, v, t: entropy_gibbs_lin_baseline(x, v)
        if t == 1.0
        else 1 + neg_entropy_alpha(x, t).log2() / (t - 1) / math.log(v, 2)
    )
    confidence_measure_bank["entropy_renyi_exp"] = (
        lambda x, v, t: entropy_gibbs_exp_baseline(x, v)
        if t == 1.0
        else (neg_entropy_alpha(x, t).pow(1 / (t - 1)) * v - 1) / (v - 1)
    )
    return confidence_measure_bank


def get_confidence_aggregation_bank():
    """Generate a dictionary with confidence aggregation functions.

    Supported confidence aggregation functions:
        min: minimum
        max: maximum
        mean: arithmetic mean
        prod: product

    Returns:
        dictionary with functions.
    """
    confidence_aggregation_bank = {"mean": lambda x: sum(x) / len(x), "min": min, "max": max}
    # python 3.7 and earlier do not have math.prod
    if hasattr(math, "prod"):
        confidence_aggregation_bank["prod"] = math.prod
    else:
        import operator
        from functools import reduce

        confidence_aggregation_bank["prod"] = lambda x: reduce(operator.mul, x, 1)
    return confidence_aggregation_bank


def round_confidence(confidence_number, ndigits=3):
    if isinstance(confidence_number, float):
        return round(confidence_number, ndigits)
    elif len(confidence_number.size()) == 0:  # torch.tensor with one element
        return round(confidence_number.item(), ndigits)
    elif len(confidence_number.size()) == 1:  # torch.tensor with a list if elements
        return [round(c.item(), ndigits) for c in confidence_number]
    else:
        raise RuntimeError(f"Unexpected confidence_number: `{confidence_number}`")


def compute_frame_confidence(decoder_output, decoder_lengths):
    confidence_method_cfg = ConfidenceMethodConfig( # Config for per-frame scores calculation (before aggregation)
        name="max_prob", # Or "entropy" (default), which usually works better
        entropy_type="gibbs", # Used only for name == "entropy". Recommended: "tsallis" (default) or "renyi"
        alpha=0.5, # Low values (<1) increase sensitivity, high values decrease sensitivity
        entropy_norm="lin" # How to normalize (map to [0,1]) entropy. Default: "exp"
    )
    confidence_method = ConfidenceMethod()
    confidence_method._init_confidence_method(confidence_method_cfg, blank_id=BLANK_ID)
    decoder_lengths = decoder_lengths.cpu() if decoder_lengths is not None else None
    frame_confidence = []
    for ind in range(len(decoder_output)):
        out_len = decoder_lengths[ind] if decoder_lengths is not None else None
        prediction_cpu_tensor = torch.empty(
            decoder_output[ind].shape, dtype=decoder_output[ind].dtype, device=torch.device("cpu"), pin_memory=True
        )
        prediction_cpu_tensor.copy_(decoder_output[ind], non_blocking=True)
        prediction = prediction_cpu_tensor[:out_len]
        frame_confidence.append(confidence_method._get_confidence(prediction))
    
    return frame_confidence

def compute_token_confidence(logprob_phrases, aggregation):
    confidence_aggregation_bank = get_confidence_aggregation_bank()
    aggregate_confidence = confidence_aggregation_bank[aggregation]
    non_blank_confidences = []
    phrases = []
    for logprobs in logprob_phrases:
        decoded_sequence = logprobs.logprobs.argmax(axis=-1).tolist()
        logpob_tensor = torch.Tensor(logprobs.logprobs)
        chunk_confidence = compute_frame_confidence(logpob_tensor, decoder_lengths=None)
        non_blank_confidence = []
        phrase = []
        i = 0
        for token, group in groupby(decoded_sequence):
            grouped = list(group)
            seq_len = len(grouped)
            
            if token != BLANK_ID:
                phrase.append(token)
            if token != BLANK_ID and token != SPACE_ID:
                confidence = aggregate_confidence(chunk_confidence[i : i + seq_len])
                non_blank_confidence.append(confidence)
            i += seq_len
        
        non_blank_confidences.extend(non_blank_confidence)
        phrases.extend(phrase)
    return non_blank_confidences

def compute_word_confidence(token_confidence, text, aggregation):
    confidence_aggregation_bank = get_confidence_aggregation_bank()
    aggregate_confidence = confidence_aggregation_bank[aggregation]
    word_confidence = []
    i = 0
    for word in text.split():
        word_len = len(word)
        word_confidence.append(aggregate_confidence(token_confidence[i : i + word_len]))
        i += word_len

    return word_confidence
