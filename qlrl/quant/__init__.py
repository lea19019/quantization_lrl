"""Quantizers. One file per method; every quantizer exposes the same interface.

Shared interface (Strategy pattern, no base class; a shared shape is enough):

    q = Method(bits=..., group_size=..., ...)   # config only, from the YAML
    q.add_batch(inputs)                        # calibrated methods only; inputs [tokens, in]
    weight_hat = q.quantize(weight)            # weight [out, in] -> fake-quantized [out, in]

`apply.quantize_model` walks the model, makes one fresh quantizer per nn.Linear (each
holds its own calibration state), feeds it inputs if it has `add_batch`, and replaces the
weight. Method sketches and equations: docs/papers/methods.md, "Algorithm sketches".
"""
