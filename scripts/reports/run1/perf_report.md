# Triton Streaming ASR Perf Report
## Run
- Triton URL: `localhost:8001`
- Metrics URL: `http://localhost:8002/metrics`
- Model: `streaming_acoustic` (input `signal`, output `logprobs`)
- Dataset: `test_rec_support`
- Workers: `32`
- Decoder: `beam`

## Throughput
- Utterances: **0**
- Total audio: **0.000s**
- Wall time: **0.117s**
- RTF: **nan**
- Audio seconds / wall second: **0.000**
- Requests: **0**
- Requests/sec: **0.000**

## Client Request Latency (per chunk)
- Mean: **nan ms**
- P50: **nan ms**
- P90: **nan ms**
- P95: **nan ms**
- P99: **nan ms**

## Server-side Averages (Triton /metrics deltas)
- Success requests (delta): **0.0**
- Failed requests (delta): **0.0**
- Avg request: **nan ms**
- Avg queue: **nan ms**
- Avg compute_input: **nan ms**
- Avg compute_infer: **nan ms**
- Avg compute_output: **nan ms**

## Accuracy
- WER: **nan**
