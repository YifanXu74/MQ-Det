**fatal error: THC/THC.h: No such file or directory**
It is because pytorch removed THC/THC.h after its version 1.11. One solution is to downgrade the torch version, but this may be incompatible with the system  dependencies (e.g., GPUs, CUDA, ...)
Another solution is to modify the cuda file:
1. remove all #include <THC/THC.h>
2. replace all
   ```
   THCudaCheck(...);
   ```
   with
   ```
   AT_CUDA_CHECK(...);
   ```
**THCCeilDiv is undefined**
1. #include <ATen/ceil_div.h>
2. replace all
   ```
   THCCeilDiv(...)
   ```
   with
   ```
   at::ceil_div(...)
   ```
**THCudaMalloc/THCudaFree/THCState  is undefined**
1. #include <ATen/cuda/ThrustAllocator.h>
2. remove the line with THCState
3. replace
   ```
   THCudaMalloc(param1, param2)
   ```
   with
   ```
   c10::cuda::CUDACachingAllocator::raw_alloc(param2)
   ```
4. replace
   ```
   THCudaFree(param1, param2)
   ```
   with
   ```
   c10::cuda::CUDACachingAllocator::raw_delete(param2)
   ```

**unrecognized arguments: --local-rank=5**
This is because torch with a high version receive ``--local-rank`` rather than ``--local_rank``.
Replace ``--local-rank`` with ``--local_rank`` in coresponding code, and vice versa.

**ImportError: libGL.so.1: cannot open shared object file: No such file or directory**
Solved by this [link](https://stackoverflow.com/questions/55313610/importerror-libgl-so-1-cannot-open-shared-object-file-no-such-file-or-directo)

**Error in dataloader**
Try to pass:
```
DATALOADER.NUM_WORKERS 0
```