"""Retrieve DEME positions into a caller-owned NVIDIA Warp array."""

import warp as wp


def retrieve_positions(tracker, owner_count: int, device_id: int = 0) -> wp.array:
    """Return tracked positions as a contiguous ``wp.vec3`` CUDA array."""
    device = f"cuda:{device_id}"
    positions = wp.empty(owner_count, dtype=wp.vec3, device=device)

    # Warp may use stream-ordered allocation. Make the allocation visible
    # before DEME accesses it from DEME's own CUDA stream.
    wp.synchronize_device(device)

    tracker.PositionsToDevice(
        int(positions.ptr),  # pybind11 converts this Python int to std::uintptr_t.
        owner_count,  # Capacity is float3/wp.vec3 elements, not bytes.
        device_id,
    )
    return positions
