# Future learning goal: deployment-ready quantization

Build a system that converts a full-precision model into a real low-bit checkpoint
ready to serve with vLLM. It must produce packed weight codes, the per-group metadata
needed to reconstruct their approximate values, and the format information required by
the inference engine. This is separate from the project's fake-quantization evaluation
pipeline and will be pursued as future learning work.
