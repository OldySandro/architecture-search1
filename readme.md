# ARC Search: Hardware-Aware LLM Architecture Explorer

ARC Search is a Neural Architecture Search (NAS) framework for exploring Large Language Model (LLM) architectures under real hardware constraints.

Its primary goal is to identify architectures that best match a target GPU in terms of efficiency, compatibility, and practical resource usage.

ARC Search does **not** claim that the selected architecture is the most stable or universally best model.  
Instead, it searches for configurations that are the most suitable fit for the selected hardware profile.

## What ARC Search Optimizes

ARC Search evaluates candidates based on hardware-fit metrics such as:

- VRAM compatibility
- Throughput (tokens/sec)
- Memory efficiency
- Tensor Core alignment
- Attention kernel suitability
- FLOPs utilization
- Compute vs memory bottlenecks
- Overall deployment practicality

The reported score reflects how well an architecture matches the target hardware environment — **not long-term training stability, convergence quality, or benchmark superiority**.

---

## Preview Interface

The system uses a modern terminal UI powered by `Rich` for architecture search and profiling output.

<img width="707" height="1304" alt="Live Architecture Ranking" src="https://github.com/user-attachments/assets/b98cb001-e5d2-468e-a1fa-7790c6e2ac5a" />

*Figure 1: Live architecture ranking during adaptive search across multiple candidates.*

<img width="696" height="257" alt="Adaptif RL Log" src="https://github.com/user-attachments/assets/22746c4f-b324-4069-9283-b5fca4f99d40" />

*Figure 2: Reinforcement-based exploration and candidate refinement.*

<img width="720" height="937" alt="tabel RL" src="https://github.com/user-attachments/assets/cdaf07e7-2dc8-449c-b3ab-39545ce1f705" />

*Figure 3: Candidate comparison table across throughput, VRAM, and efficiency metrics.*

---

## Case Study: NVIDIA T4 Search Result

For NVIDIA T4, ARC Search selected **ARC-2142 (Long-Horizon)** as the best hardware-fit candidate among explored architectures.

This means ARC-2142 achieved the strongest balance of:

- memory usage
- throughput
- compute efficiency
- architecture compatibility
- usable context length

It does **not** automatically imply highest model quality or best training convergence.

<img width="720" height="404" alt="Final Rekomendasi" src="https://github.com/user-attachments/assets/d5c80b61-aaa0-4c7f-a9d1-2048cacb6934" />

*Figure 4: Final recommendation report for ARC-2142.*

### ARC-2142 Summary

- Parameters: **444M**
- Layers / Hidden Size: **20 × 1280**
- Attention: **Grouped-Query Attention (GQA)**
- FFN: **GeGLU**
- Context Length: **4096**
- Throughput: **~4,209 tokens/sec**
- VRAM Usage: **11.11 GB / 16 GB**
- Fit Score: High suitability for NVIDIA T4

---

## Key Features

- **Hardware-Aware Search**  
  Finds architectures that fit real GPU constraints.

- **Adaptive Exploration**  
  Iteratively improves candidates through guided search.

- **Efficiency Scoring**  
  Measures practical speed and memory tradeoffs.

- **Explainable Results**  
  Shows why one architecture fits better than another.

- **GPU-Specific Recommendations**  
  Different GPUs may receive different top candidates.

---

## Installation
```bash
git clone https://github.com/OldySandro/architecture-search.git
cd architecture-search
pip install -r requirements.txt
```

---

## Run
```bash
cd search
python pipeline.py
```

---

### Important Hardware Note

GPU architectures evolve rapidly. Performance scores and recommendations in this repository are based on hardware profiles available at the time of release.

An architecture that performs well on NVIDIA T4 (Turing) may rank differently on newer generations such as Ada, Hopper, Blackwell, or future designs due to changes in:

- Tensor Core structure
- Memory bandwidth
- Shared memory capacity
- Cache hierarchy
- Scheduling behavior
- Kernel efficiency

For the most accurate results, always run ARC Search using the latest hardware profiles and current version of the pipeline.

---

#### Release Baseline :April 18, 2026
#### Author :Oldy Sandro
#### License :MIT License