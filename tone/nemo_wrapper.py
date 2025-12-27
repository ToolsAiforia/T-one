"""NeMo-compatible wrapper for the T-one CTC model."""

from __future__ import annotations

from typing import Any

import torch
from nemo.collections.asr.parts.submodules.ctc_decoding import CTCDecoding, CTCDecodingConfig
from nemo.core.classes import ModelPT
from omegaconf import OmegaConf

from tone.training.model_wrapper import ToneConfig, ToneForCTC


class ToneCTCNemoModel(ModelPT):
    """Wrap T-one as a NeMo ModelPT for CTC decoding and save/restore."""

    tone_for_ctc: ToneForCTC
    ctc_decoding: CTCDecoding
    cur_decoder: str

    def __init__(self, cfg: Any, trainer: Any = None) -> None:
        super().__init__(cfg=cfg, trainer=trainer)
        tone_cfg = self._get_tone_config()
        self.tone_for_ctc = ToneForCTC(ToneConfig(**tone_cfg))
        self.ctc_decoding = self._build_ctc_decoding(self.cfg.get("decoding"))
        self.cur_decoder = "ctc"

    @classmethod
    def list_available_models(cls) -> list[dict[str, str]]:
        return []

    def setup_training_data(self, *args: Any, **kwargs: Any) -> None:
        pass

    def setup_validation_data(self, *args: Any, **kwargs: Any) -> None:
        pass

    def _get_tone_config(self) -> dict[str, Any]:
        tone_cfg = self.cfg.get("tone_config")
        if tone_cfg is None:
            raise ValueError("tone_config is required to initialize ToneCTCNemoModel.")
        if not isinstance(tone_cfg, dict):
            tone_cfg = OmegaConf.to_container(tone_cfg, resolve=True)
        return dict(tone_cfg)

    def _vocabulary(self) -> list[str]:
        tone_cfg = self._get_tone_config()
        decoder_params = tone_cfg.get("decoder_params") or {}
        vocab = decoder_params.get("vocabulary")
        if not vocab:
            raise ValueError("decoder_params.vocabulary is required for CTC decoding.")
        return list(vocab)

    def _build_ctc_decoding(self, decoding_cfg: Any | None) -> CTCDecoding:
        if decoding_cfg is None:
            decoding_cfg = CTCDecodingConfig(strategy="greedy")
        elif isinstance(decoding_cfg, dict):
            decoding_cfg = OmegaConf.create(decoding_cfg)
        return CTCDecoding(decoding_cfg=decoding_cfg, vocabulary=self._vocabulary())

    def change_decoding_strategy(self, decoding_cfg: Any | None, decoder_type: str = "ctc") -> None:
        if decoder_type != "ctc":
            raise ValueError(f"Unsupported decoder_type: {decoder_type}")
        self.ctc_decoding = self._build_ctc_decoding(decoding_cfg)
        self.cur_decoder = "ctc"

    def forward(
        self,
        input_values: torch.Tensor | None,
        input_lengths: torch.Tensor | None,
        attention_mask: torch.Tensor | None = None,
        labels: torch.Tensor | None = None,
    ) -> Any:
        return self.tone_for_ctc(
            input_values=input_values,
            input_lengths=input_lengths,
            attention_mask=attention_mask,
            labels=labels,
        )
