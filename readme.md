# A implementation of "Optimizing Parallel Reduction in CUDA"

- Please refer to the nVidia webinar sildes [Optimizing Parallel Reduction in CUDA](https://developer.download.nvidia.com/assets/cuda/files/reduction.pdf)
- Code is implement strcitly according to these slides (e.g., no template to apply for different types of variables)
- It seems that by given the test case mentioned in the sildes (1+2+...+2^22), using "int" as the temporary variable is not sufficient enough (it get "overflowed"). But since both CPU and GPU sum function are doing the same thing, the result should still be the same.
- Seems that using float to test is not a good idea. The error accumulated in a different way, leads to a slightly different result.
- To measure the efficiency of different reductions, please refer to [How to implement performance metrics in CUDA C/C++](https://developer.nvidia.com/blog/how-implement-performance-metrics-cuda-cc/)