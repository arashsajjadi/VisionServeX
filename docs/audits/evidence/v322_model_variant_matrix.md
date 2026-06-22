# Benchmark: v322_model_variant_matrix

- **gpu**: RTX 5080
- **video**: /home/arash/Downloads/lv_0_20260617224920.mp4
- **ladder**: [1, 2, 4, 8, 16]

| model | variant | task | batch_mode | true_batch | microbatch | frames | resolution | gpu_avg | gpu_peak | vram_peak_mb | forward_ms | preprocess_ms | postprocess_ms | throughput_fps | detections_per_frame | failure_reason |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| dfine-n | dfine-n | detect | true_tensor_batch | True | 1 | 1 | 1280x720 | 2.0 | 2 | 71.2 | 7.965 | 3.31 | 0.45 | 79.6 | 95.0 |  |
| dfine-n | dfine-n | detect | true_tensor_batch | True | 2 | 2 | 1280x720 | 2.0 | 2 | 112.2 | 22.456 | 5.782 | 0.582 | 67.9 | 80.5 |  |
| dfine-n | dfine-n | detect | true_tensor_batch | True | 4 | 4 | 1280x720 | 2.0 | 2 | 199.9 | 24.352 | 10.16 | 0.884 | 110.6 | 73.5 |  |
| dfine-n | dfine-n | detect | true_tensor_batch | True | 8 | 8 | 1280x720 | 2.0 | 2 | 374.5 | 12.864 | 23.6 | 1.704 | 203.8 | 69.6 |  |
| dfine-n | dfine-n | detect | true_tensor_batch | True | 16 | 16 | 1280x720 | 2.0 | 2 | 724.0 | 40.192 | 39.248 | 2.688 | 190.9 | 56.4 |  |
