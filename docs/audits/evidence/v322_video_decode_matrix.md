# Benchmark: v322_video_decode_matrix

- **video**: /home/arash/Downloads/lv_0_20260617224920.mp4
- **hwaccel**: True

| operation | detail | elapsed_ms | result | evidence |
| --- | --- | --- | --- | --- |
| detect_hwaccel | nvenc/nvdec |  | OK | nvenc=['h264_nvenc', 'hevc_nvenc', 'av1_nvenc'] nvdec=['h264_cuvid', 'hevc_cuvid', 'vp9_cuvid', 'av1_cuvid'] |
| ffprobe | h264/yuv420p/1280x720 | 37.0 | OK | faststart=False action=remux_faststart |
| remux_faststart | lossless -c copy | 74.2 | OK | out_faststart=True size=89921611 |
| transcode_browser_h264 | 720p/h264_nvenc | 4463.5 | OK | nvenc=True size=52024157 |
| extract_frames | sample_fps=1.0 | 1445.9 | OK | 89 frames, 61.6 fps decode |
| extract_frames | sample_fps=5.0 | 597.7 | OK | 120 frames, 200.8 fps decode |
| extract_frames | sample_fps=10.0 | 489.2 | OK | 120 frames, 245.3 fps decode |
