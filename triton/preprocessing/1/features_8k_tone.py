from typing import Any
import numpy as np

import nvidia.dali as dali
import nvidia.dali.fn as fn
import nvidia.dali.types as types
from nvidia.dali import pipeline_def
from nvidia.dali.plugin.triton import autoserialize

MAX_BATCH_SIZE = 16

LOG_ZERO_GUARD_VALUE = 2**-24
SAMPLE_RATE = 8000
WINDOW_STRIDE_SEC = 0.01
WINDOW_SIZE_SEC = 0.02

NFFT = 160
WIN_LENGTH = int(SAMPLE_RATE * WINDOW_SIZE_SEC)    # 160
HOP_LENGTH = int(SAMPLE_RATE * WINDOW_STRIDE_SEC)  # 80

WINDOW_FN = np.hanning(WIN_LENGTH).astype(np.float32).tolist()

@autoserialize
@pipeline_def(batch_size=MAX_BATCH_SIZE, num_threads=4, device_id=0)
def pipe() -> Any:
    # Assume that audio is already padded with 80 samples from prev window
    audio = fn.external_source(name="audio_data", no_copy=True).gpu()
    audio = fn.preemphasis_filter(audio, preemph_coeff=0.97, border="clamp")
    spec = fn.spectrogram(
        audio,
        nfft=NFFT,
        window_length=WIN_LENGTH,
        window_step=HOP_LENGTH,
        power=2,
        center_windows=False,
        window_fn=WINDOW_FN,
        layout="ft",
    )

    mel = fn.mel_filter_bank(
        spec,
        sample_rate=SAMPLE_RATE,
        nfilter=64,
        freq_high=SAMPLE_RATE / 2,
        freq_low=0.0,
        mel_formula="slaney",
        normalize=True,
    )

    mel_log = dali.math.log(mel + LOG_ZERO_GUARD_VALUE)
    # mel_log is (mel, time)
    ready_signal_shape = mel_log.shape(dtype=types.INT64, device="gpu")[-1:]
    # so ready signal shape is time
    return mel_log, ready_signal_shape
