# Model Registry Notes

`model_registry.py` is the planner's static knowledge base. It contains researched architecture facts, component inventories, quantization policy, placement support, existing fast paths, and ordered candidate plans for every launcher model and auxiliary runtime model.

## How to read confidence fields

- `high` means the value is directly supported by a published configuration, repository inventory, model implementation, or the existing application code.
- `medium` means the value is derived from public architecture and checkpoint information with a reasonable engineering estimate.
- `estimated` means the value is deliberately approximate because allocator state, kernels, tensor liveness, workload shape, and backend versions determine the real peak.

The planner never treats an estimated peak as a promise that the next allocation will succeed. It uses the estimates to reject clearly impossible plans and order credible attempts, then records the result of real loading and inference.

## Memory values

Component weight estimates distinguish native, INT8, and INT4 storage. Quantized values account for the quantizable fraction of each component and leave protected or unsupported modules at higher precision. They do not imply that activations, KV caches, attention workspaces, latents, VAE decode tensors, or quantization temporaries use the same low-bit representation.

## Quantization policy

- BitsAndBytes INT8 is the normal automatic fallback for supported large transformer components.
- NF4 INT4 is model- and component-specific and never goes below four bits automatically.
- Large text encoders may use more aggressive quantization than image or video denoisers.
- VAEs, schedulers, tokenizers, processors, and small text encoders remain at native precision unless a model-specific native artifact says otherwise.
- Sensitive embeddings, normalization, modulation, input projections, output projections, and language-model heads are protected where the implementation supports skip lists.
- Native FP8, MXFP4, GGUF, or publisher-supplied low-bit checkpoints are treated as intended artifacts rather than generic dynamic quantization.

## Existing and portable loaders

The exact current application paths remain first-choice candidates when their hardware match is satisfied. Portable custom staged executors are implemented for FLUX.2, Qwen Image Edit, and Kandinsky I2V Pro. GLM Image, Qwen Image, and ChronoEdit preserve their exact hand-written paths and use their official pipeline implementations for portable plans.

## Runtime learning

A plan becomes validated only after one real inference completes. Successful and failed attempts are stored in `config/plan_history.json` using a signature that includes the hardware, relevant software versions, and configured memory reserves. A failed plan is skipped for the same or a heavier workload, while a successful plan is preferred for the same or a lighter workload.

## Updating the registry

When a checkpoint revision, model implementation, or library API changes:

1. Update the affected model profile and its source list.
2. Update component names or protected-module lists if the implementation changed.
3. Run `python validate_project.py`.
4. Remove the affected entries from `config/plan_history.json`, or remove the file entirely, so the planner validates the revised path again.
