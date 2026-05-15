# About VisionServeX

VisionServeX is a Python framework for serving modern, permissively-licensed
computer vision models on local machines, workstations, and small servers.

## Author

**Arash Sajjadi**
PhD Candidate, Department of Computer Science, University of Saskatchewan
Email: arash.sajjadi@usask.ca

## Acknowledgment

This work is developed under the supervision of **Prof. Mark Eramian**,
Department of Computer Science, University of Saskatchewan, Computer Vision
Lab.

> VisionServeX is not an official product of the University of Saskatchewan
> and is not endorsed by the University unless stated otherwise in writing.

## License

VisionServeX is released under the Apache License 2.0
(`SPDX-License-Identifier: Apache-2.0`). See [`LICENSE`](../LICENSE).
Integrated upstream models retain their own licenses; see
[`docs/model_licenses.md`](model_licenses.md).

## Project goals

VisionServeX is designed to be:

- **Beginner-friendly.** A first-time user can run `doctor`, then
  `recommend`, then `pull`, then `predict`, without understanding CUDA,
  PyTorch internals, or HuggingFace Hub semantics.
- **Permissive-license aware.** Default models ship under Apache-2.0, MIT,
  or BSD licenses. Anything uncertain or commercially restricted is clearly
  labelled and disabled by default.
- **Predictable for automation.** Stable JSON shapes for both CLI (`--json`)
  and HTTP. Designed for LLM agents and CI pipelines.
- **Secure by default.** Binds to `127.0.0.1`, requires authentication for
  public-mode operation, enforces request size and image dimension limits,
  guards remote URL fetches against SSRF, redacts secrets from logs, and
  never executes arbitrary remote code.

## What VisionServeX is not

- A research training framework. Use upstream model repositories for
  training and evaluation.
- A benchmark suite. We do not claim that any specific model is the best
  in any task. Each entry in the registry includes honest status flags.
- A drop-in replacement for any specific commercial inference service.
  VisionServeX is a thin, permissively-licensed serving framework.

## Citation

If you use VisionServeX in academic work, please cite it via the
[`CITATION.cff`](../CITATION.cff) file at the repository root, or with the
following BibTeX entry:

```bibtex
@software{sajjadi2026visionservex,
  author       = {Arash Sajjadi},
  title        = {{VisionServeX: A permissive-license-aware framework for local computer vision model serving}},
  year         = {2026},
  url          = {https://github.com/example/visionservex},
  note         = {Developed under the supervision of Prof. Mark Eramian,
                  Department of Computer Science, University of Saskatchewan.}
}
```
