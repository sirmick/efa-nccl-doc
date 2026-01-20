# NCCL on AWS EFA Documentation

Comprehensive technical documentation for NCCL (NVIDIA Collective Communications Library) using the OFI plugin with libfabric and AWS EFA (Elastic Fabric Adapter).

## Purpose

This documentation provides in-depth coverage of the complete stack used for GPU-to-GPU communication on AWS EC2 instances with EFA. It's designed for engineers working on optimization, debugging, and understanding the performance characteristics of distributed deep learning workloads.

## Documentation Structure

### Core Concepts

- **[overview.md](overview.md)** - Architecture overview and component interactions
- **[nccl-core.md](nccl-core.md)** - NCCL fundamentals, topology, communicators, channels
- **[nccl-collectives.md](nccl-collectives.md)** - Collective operations, algorithms (Ring/Tree), protocols (Simple/LL/LL128)
- **[nccl-tuner.md](nccl-tuner.md)** - Cost-based algorithm/protocol selection, AWS region-based tuner, channel count tuning
- **[nccl-datapath.md](nccl-datapath.md)** - Complete data path from GPU through network

### Collective Algorithms (Detailed Analysis)

- **[algorithms/ring-algorithm.md](algorithms/ring-algorithm.md)** - Ring AllReduce with complete latency derivation, bandwidth analysis, O(N) scaling
- **[algorithms/tree-algorithm.md](algorithms/tree-algorithm.md)** - Double binary tree with O(log N) latency derivation, ~67% bandwidth utilization
- **[algorithms/nvls-tree-algorithm.md](algorithms/nvls-tree-algorithm.md)** - NVLS Tree with NVSwitch multicast acceleration, 2-10× speedup for small messages
- **[algorithms/pat-algorithm.md](algorithms/pat-algorithm.md)** - Parallel Aggregated Trees (PAT) with adaptive logarithmic-to-linear communication

### EFA Hardware and Protocol

- **[efa-hardware-architecture.md](efa-hardware-architecture.md)** - EFA queue pairs, completion queues, work queue entries, memory layout, hardware capabilities
- **[srd-protocol.md](srd-protocol.md)** - Scalable Reliable Datagram protocol: multipath routing, congestion control, reliability mechanisms
- **[kernel-efa-driver.md](kernel-efa-driver.md)** - Linux kernel EFA driver internals

### Transport Layer

- **[ofi-plugin.md](ofi-plugin.md)** - OFI NCCL plugin architecture and implementation
- **[libfabric-overview.md](libfabric-overview.md)** - Libfabric API, capabilities, data transfer operations
- **[efa-provider.md](efa-provider.md)** - EFA provider specifics, eager/rendezvous protocols

### System Level

- **[threading-model.md](threading-model.md)** - Threading across the stack, proxy threads, concurrency
- **[rdma-memreg.md](rdma-memreg.md)** - RDMA operations, memory registration, caching strategies

### Performance

- **[optimizations.md](optimizations.md)** - Kernel bypass, caching, tuning parameters, best practices
- **[optimization-opportunities.md](optimization-opportunities.md)** - Detailed analysis of optimization opportunities with priorities

### Deep Dives

- **[lkey-rkey-explained.md](lkey-rkey-explained.md)** - Understanding local and remote keys in RDMA
- **[ofi-plugin-protocols.md](ofi-plugin-protocols.md)** - Connection establishment and send/recv protocols
- **[topology-and-binding.md](topology-and-binding.md)** - Topology detection and interface binding mechanisms
- **[freelist-allocator.md](freelist-allocator.md)** - Freelist allocator for high-performance object pooling (estimated 20-100x faster than malloc)
- **[mr-cache-implementation.md](mr-cache-implementation.md)** - Memory registration cache implementation details (approximately 25x speedup)
- **[rdma-core-and-verbs.md](rdma-core-and-verbs.md)** - rdma-core library and libibverbs API

### GPU and Accelerator Memory

- **[cuda-memory.md](cuda-memory.md)** - NVIDIA CUDA memory: dynamic loading, cross-version compatibility, GPUDirect, GDRCopy
- **[dmabuf-gpu-memory.md](dmabuf-gpu-memory.md)** - DMA-BUF framework for vendor-neutral GPU memory registration
- **[neuron-memory.md](neuron-memory.md)** - AWS Trainium/Inferentia with P2P registration (not dmabuf)
- **[rocm-memory.md](rocm-memory.md)** - AMD ROCm with HIP API (CUDA-compatible)

## Interactive Tools

- **[Region-Based Tuner Visualizer](region-visualizer/)** - Interactive visualization of AWS OFI NCCL tuner algorithm/protocol selection regions for p5/p5en instances ([Live Demo](https://sirmick.github.io/efa-nccl-doc/region-visualizer/))

## Quick Start

### Understanding the Stack

Start with these documents in order:

1. [overview.md](overview.md) - Get the big picture
2. [nccl-core.md](nccl-core.md) - Understand NCCL basics
3. [nccl-datapath.md](nccl-datapath.md) - Follow a message through the system
4. [optimizations.md](optimizations.md) - Apply optimizations

### For Specific Topics

- **Performance tuning** → [optimizations.md](optimizations.md), [optimization-opportunities.md](optimization-opportunities.md)
- **Debugging network issues** → [ofi-plugin.md](ofi-plugin.md), [efa-provider.md](efa-provider.md)
- **Algorithm selection** → [nccl-collectives.md](nccl-collectives.md), [algorithms/](algorithms/) for detailed analysis
- **Collective algorithms deep dive** → [algorithms/ring-algorithm.md](algorithms/ring-algorithm.md), [algorithms/tree-algorithm.md](algorithms/tree-algorithm.md), [algorithms/nvls-tree-algorithm.md](algorithms/nvls-tree-algorithm.md), [algorithms/pat-algorithm.md](algorithms/pat-algorithm.md)
- **Memory registration problems** → [rdma-memreg.md](rdma-memreg.md), [mr-cache-implementation.md](mr-cache-implementation.md)
- **Threading issues** → [threading-model.md](threading-model.md)
- **Connection protocols** → [ofi-plugin-protocols.md](ofi-plugin-protocols.md)
- **Topology and multi-NIC** → [topology-and-binding.md](topology-and-binding.md)
- **Understanding RDMA keys** → [lkey-rkey-explained.md](lkey-rkey-explained.md)
- **Memory allocators & caching** → [freelist-allocator.md](freelist-allocator.md), [mr-cache-implementation.md](mr-cache-implementation.md)
- **GPU memory registration** → [cuda-memory.md](cuda-memory.md), [dmabuf-gpu-memory.md](dmabuf-gpu-memory.md), [neuron-memory.md](neuron-memory.md), [rocm-memory.md](rocm-memory.md)
- **Kernel driver details** → [kernel-efa-driver.md](kernel-efa-driver.md)
- **Userspace RDMA API** → [rdma-core-and-verbs.md](rdma-core-and-verbs.md)

## Key Concepts

### Architecture Stack

```
Application (PyTorch, TensorFlow)
    ↓
NCCL Library (Collectives, Algorithms)
    ↓
OFI NCCL Plugin (Network Transport)
    ↓
Libfabric (Fabric Abstraction)
    ↓
EFA Provider (Hardware-Specific)
    ↓
rdma-core (libibverbs)
    ↓
EFA Kernel Driver (efa.ko)
    ↓
EFA Hardware (p4d: 4×100G, p5+: 32×100G EFAv3)
```

### Data Flow

1. **Application** calls NCCL collective (e.g., AllReduce)
2. **NCCL** breaks into algorithm steps (Ring/Tree)
3. **Proxy threads** handle network I/O
4. **OFI plugin** translates to libfabric operations
5. **Libfabric** posts to EFA provider
6. **EFA provider** builds WQEs, rings doorbells
7. **EFA driver** DMAs from GPU memory
8. **EFA hardware** transmits over network with SRD reliability
9. **Remote side** receives and completes in reverse

### Critical Performance Factors

1. **Memory Registration Caching** - Estimated 100-500x speedup for repeated transfers
2. **Algorithm Selection** - Ring for large messages (>1 MB), Tree for medium (100 KB-1 MB), PAT for small-medium at scale (32 KB-1 MB), NVLS Tree for intra-node with NVSwitch. See [algorithms/](algorithms/) for complete mathematical analysis
3. **Protocol Selection** - LL for latency (<32 KB), LL128 for balanced (32 KB-1 MB), Simple for bandwidth (>1 MB)
4. **Multi-rail** - Utilize all EFA adapters (4-8 per instance)
5. **CPU Affinity** - Pin proxy threads to avoid NUMA remote access
6. **Kernel Bypass** - Direct user-space access to hardware

## Performance Targets

### Latency (Single Message)

| Message Size | Target Latency | Notes |
|--------------|----------------|-------|
| 8 bytes      | 10-15 μs       | Protocol overhead dominates |
| 1 KB         | 12-20 μs       | LL protocol optimal |
| 64 KB        | 20-30 μs       | LL128 protocol |
| 1 MB         | 100-200 μs     | Simple protocol |

### Bandwidth (Sustained)

| Instance Type | EFA per GPU | Total GPU | Target per GPU | Instance Total | Notes |
|---------------|-------------|-----------|----------------|----------------|-------|
| p4d.24xlarge  | 4×100G      | 8 A100    | ~50 GB/s       | ~400 GB/s      | Send/recv only |
| p4de.24xlarge | 4×100G      | 8 A100    | ~50 GB/s       | ~400 GB/s      | Send/recv only |
| p5.48xlarge   | 4×100G      | 8 H100    | ~50 GB/s       | ~400 GB/s      | RDMA supported |
| p5e.48xlarge  | 32 EFAs (400G per GPU) | 8 H100 | ~50 GB/s | ~400 GB/s | RDMA supported |
| p5en.48xlarge | 32 EFAs (400G per GPU) | 8 H200 | ~50 GB/s | ~400 GB/s | RDMA + EFAv3, 35% lower latency |

**Note**: Total instance bandwidth: p4d/p4de = 400 Gbps, p5/p5e/p5en = 3200 Gbps. Each GPU gets ~400 Gbps (50 GB/s) of EFA bandwidth for collective operations.

### AllReduce (8 GPUs, Ring Algorithm)

| Data Size | Latency | Bus Bandwidth | Notes |
|-----------|---------|---------------|-------|
| 128 MB    | 5-10 ms | 100+ GB/s     | Typical gradient size |
| 1 GB      | 40-80 ms| 100+ GB/s     | Large model |

## Benchmarking

### NCCL Tests

```bash
git clone https://github.com/NVIDIA/nccl-tests.git
cd nccl-tests && make

# All-reduce
./build/all_reduce_perf -b 8 -e 1G -f 2 -g 8

# All-gather
./build/all_gather_perf -b 8 -e 1G -f 2 -g 8

# Broadcast
./build/broadcast_perf -b 8 -e 1G -f 2 -g 8
```

### Interpreting Results

- **AlgBW**: Algorithm bandwidth (useful for comparison)
- **BusBW**: Actual network bandwidth utilized
- **Time**: Average latency per operation

For AllReduce with Ring algorithm:
```
BusBW = AlgBW × 2 × (N-1) / N
```

## Additional Resources

### Official Documentation

- NCCL: https://docs.nvidia.com/deeplearning/nccl/
- OFI Plugin: https://github.com/aws/aws-ofi-nccl
- Libfabric: https://ofiwg.github.io/libfabric/
- AWS EFA: https://docs.aws.amazon.com/AWSEC2/latest/UserGuide/efa.html

### Repositories

- NCCL: https://github.com/NVIDIA/nccl
- aws-ofi-nccl: https://github.com/aws/aws-ofi-nccl
- libfabric: https://github.com/ofiwg/libfabric
- nccl-tests: https://github.com/NVIDIA/nccl-tests

## Contributing

This documentation is focused on understanding the existing implementation for optimization purposes. For improvements to the actual software components:

- NCCL issues/PRs: https://github.com/NVIDIA/nccl
- OFI plugin issues/PRs: https://github.com/aws/aws-ofi-nccl
- Libfabric issues/PRs: https://github.com/ofiwg/libfabric

## Changelog

### 2026-01-19

#### Algorithm Documentation
- Added **[algorithms/ring-algorithm.md](algorithms/ring-algorithm.md)** with complete mathematical latency derivation (`T = 2(N-1)×α + 2β×S×(N-1)/N`), bandwidth analysis proving ~100% utilization, O(N) scaling analysis, protocol-specific equations (Simple/LL128/LL), and concrete p5 instance examples
- Added **[algorithms/tree-algorithm.md](algorithms/tree-algorithm.md)** with double binary tree algorithm, O(log N) latency derivation (`T = 2×log₂(N)×α + log₂(N)×β×S`), ~67% bandwidth utilization analysis for N=8, tree topology construction, and comparison with Ring showing log vs linear scaling
- Added **[algorithms/nvls-tree-algorithm.md](algorithms/nvls-tree-algorithm.md)** documenting NVLink SHARP hardware acceleration, NVSwitch multicast mechanisms, in-network reduction, intra-node latency (`T = 2×α_nvls + 2×β_nvlink×S`), multi-node extensions, 2-10× speedup for small messages, and p5/p5en specific performance on Hopper/Blackwell GPUs
- Added **[algorithms/pat-algorithm.md](algorithms/pat-algorithm.md)** for Parallel Aggregated Trees (NCCL 2.23+), adaptive logarithmic-to-linear communication strategy, AllGather derivation (`T = log₂(N)×α + β×S×(1-1/N)`), achieving O(log N) latency with ~100% bandwidth, optimal for 32 KB-1 MB messages at scale, and LLM training use cases
- All algorithm files include complete α-β model derivations, step-by-step walkthroughs, protocol analysis, crossover point calculations, and AWS instance benchmarks

#### Hardware and Protocol Documentation
- Added comprehensive **[efa-hardware-architecture.md](efa-hardware-architecture.md)** documenting EFA queue pairs, completion queues, work queue entries, memory layout, and programming model
- Added **[srd-protocol.md](srd-protocol.md)** documenting AWS's Scalable Reliable Datagram protocol with multipath routing (64 paths), hardware-based congestion control, and reliability mechanisms
- Added **[nccl-tuner.md](nccl-tuner.md)** documenting NCCL's cost-based tuner, algorithm/protocol selection, channel count tuning, and AWS OFI region-based tuner with geometric selection

#### Improvements
- Updated all numerical performance claims to be properly qualified as estimates or approximations
- Removed file numbering from all documentation files to ease maintenance
- Removed Troubleshooting section from README
- Added algorithm documentation section to Quick Start guide

### 2026-01-17
- Initial comprehensive documentation
- Covered full stack: NCCL → EFA hardware
- Included optimization guide

---

**Audience**: Engineers optimizing distributed deep learning workloads on AWS EC2 with EFA
**Focus**: Technical depth, performance optimization, debugging
**Scope**: NCCL + OFI plugin + libfabric + EFA driver + EFA hardware
