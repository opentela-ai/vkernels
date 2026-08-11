Iron Rules:

- One kernel a directory.
- Inside the directory, one file per architecture (GPU Setup, e.g., gb10, GH200, A100, gfx942, ...).
- Inside the file, headline with the description of the kernel, followed by pytorch or pseudocode implementation of the kernel, then summary of what has been implemented, then the workload of the benchmark,then the table of performance results:

| Implementation | Time (ms) | Workload | Achieved GFLOPS | Delta to Baseline | Reproduce Script | Note |
| Baseline Pytorch | ...

- Then a list of caveats and notes about the implementation, including any known issues or limitations.
- At the end, write a journal entry summarizing the work done, any challenges faced, and any future work or improvements that could be made.