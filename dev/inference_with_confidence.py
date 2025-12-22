from pathlib import Path
from typing import TYPE_CHECKING

import argparse
import numpy as np
import json
from tqdm import tqdm

from tone.pipeline import StreamingCTCPipeline
from tone.decoder import DecoderType

if TYPE_CHECKING:
    from collections.abc import Iterator

from tone import StreamingCTCPipeline, read_audio
from confidence import compute_token_confidence, compute_word_confidence


def read_stream_audio(*, path_to_file: str | Path) -> Iterator[StreamingCTCPipeline.InputType]:
    """Simple example of streaming audio source using example audio from the package."""
    chunk_size = StreamingCTCPipeline.CHUNK_SIZE # 300 ms
    audio = read_audio(path_to_file)
    # See description of PADDING in StreamingCTCPipeline
    audio = np.pad(audio, (StreamingCTCPipeline.PADDING, StreamingCTCPipeline.PADDING))

    for i in range(0, len(audio), chunk_size):
        audio_chunk = audio[i : i + chunk_size]
        audio_chunk = np.pad(audio_chunk, (0, -len(audio_chunk) % chunk_size))
        yield audio_chunk

def read_manifest(manifest_path):
    test_set = []
    with open(manifest_path, "r") as f:
        for line in f.readlines():
            row_dict = {}
            data = json.loads(line)
            row_dict.update({"audio_filepath": data['audio_filepath'], 
                             "text": data['text']
                             })
            test_set.append(row_dict)
    return test_set

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--model_path',
        type=str,
        help='Path to local model. If not, Hugging Face T-one is used',
    )
    parser.add_argument(
        '--manifest_path', type=str, help='Path to audio manifest',
        required=True,
    )
    parser.add_argument(
        '--output_file', type=str, help='Path to output predictions json',
        required=True,
    )
    args = parser.parse_args()
    if args.model_path:
        pipeline = StreamingCTCPipeline.from_local(args.model_path, decoder_type=DecoderType.GREEDY)
    else:
        pipeline = StreamingCTCPipeline.from_hugging_face(decoder_type=DecoderType.GREEDY)
    test_set = read_manifest(args.manifest_path)

    hyps = []
    for data in tqdm(test_set):
        audio_path = data['audio_filepath']
        ref_text = data['text']
        hyp = {}
        state = None  # Current state of the ASR pipeline (None - initial)
        chunk_phrases = []
        chunk_confidences = []
        for audio_chunk in read_stream_audio(path_to_file=audio_path):  # Use any source of audio chunks
            phrases, state, logprob_phrases = pipeline.forward(audio_chunk, state)
            chunk_confidence = compute_token_confidence(logprob_phrases, aggregation="mean")
            if chunk_confidence:
                chunk_confidences.extend(chunk_confidence)
            if phrases:
                chunk_phrases += phrases
        # Finalize the pipeline and get the remaining phrases
        new_phrases, _, logprob_phrases = pipeline.finalize(state)
        final_chunk_confidences = compute_token_confidence(logprob_phrases, aggregation="mean")
        if final_chunk_confidences:
            chunk_confidences += final_chunk_confidences
        output = chunk_phrases + new_phrases
        pred_text = " ".join([phrase.text for phrase in output])
        word_confidence = compute_word_confidence(chunk_confidences, pred_text, aggregation="mean")
        hyp.update({'audio_filepath': audio_path, 'ref_text': ref_text, 'pred_text': pred_text, 'confidence': round(np.mean(word_confidence), 3)})
        hyps.append(hyp)

    with open(args.output_file, 'w', encoding='utf-8') as f:
        for line in hyps:
            json.dump(line, f, ensure_ascii=False)
            f.write("\n")

if __name__ == '__main__':
    main()