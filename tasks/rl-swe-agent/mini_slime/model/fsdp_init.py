"""Layer-wise CPU initialization avoids ever moving the full model to one GPU.

FSDP2 moves each managed unit to the mesh device, shards it and offloads its
local parameters back to CPU. The full FP32 CPU model still exists on each rank
at load time. This lowers the GPU peak, not the host-memory requirement.
"""


def plan_units(model):
    layers = list(model.model.layers)
    if not layers:
        raise ValueError('Expected nonempty transformer layers')
    if any(t.device.type != 'cpu' for t in list(model.parameters()) + list(model.buffers())):
        raise ValueError('Load the model on CPU before layer-wise FSDP initialization')
    assigned = set()
    units = []
    for i, layer in enumerate(layers):
        params = list(layer.parameters())
        if assigned.intersection(id(p) for p in params):
            raise ValueError('Shared parameters across layer units need explicit grouping')
        assigned.update(id(p) for p in params)
        size = sum(p.numel() * p.element_size() for p in params)
        units.append({'name': f'layer_{i}', 'parameter_bytes': size})
    remaining = [p for p in model.parameters() if id(p) not in assigned]
    units.append({'name': 'root', 'parameter_bytes': sum(p.numel() * p.element_size() for p in remaining)})
    return units


def shard_cpu_model(model, *, fully_shard, mp_policy, offload_policy, max_unit_bytes):
    units = plan_units(model)
    if max(u['parameter_bytes'] for u in units) > max_unit_bytes:
        raise ValueError('An FSDP unit exceeds the initialization budget; use finer wrapping')
    kwargs = dict(mp_policy=mp_policy, offload_policy=offload_policy, reshard_after_forward=True)
    modules = list(model.model.layers) + [model]
    for module in modules:
        fully_shard(module, **kwargs)
        # Avoid extra all-gather prefetch peaks on 48 GB cards.
        module.set_modules_to_forward_prefetch([])
        module.set_modules_to_backward_prefetch([])
    return units
