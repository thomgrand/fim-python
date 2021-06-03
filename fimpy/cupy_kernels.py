"""This file contains some custom CUDA kernels, used in the CUDA implementation of FIMPY
"""

compute_perm_kernel_str = (r'''
extern "C" __global__
void perm_kernel(const int* active_elems_perm, const int* active_inds,
            bool* perm_mask,
            const unsigned int active_elems_size, 
            const unsigned int active_inds_size) {
    const int nr_perms = {active_perms};
    const int bidx = blockIdx.x;
    const int block_size = blockDim.x;
    const int tidx = block_size * bidx + threadIdx.x;
      
    for(int offset = tidx; offset < active_elems_size*nr_perms; offset+=block_size*{parallel_blocks})
    {
        const int point_i = active_elems_perm[offset];
        bool match = false;
        for(int active_i = 0; !match && (active_i < active_inds_size); active_i++)
        {
            if(active_inds[active_i] == point_i)
            {
              perm_mask[offset] = true;
              match = true;
            }
        }
    }
}
''') #: CUDA kernel to compute a mask of all element permutations containing at least one active index. Old, less inefficient version not using shared memory.

compute_perm_kernel_shared = (r'''
extern "C"{
  
  //https://en.cppreference.com/w/cpp/algorithm/upper_bound
  /**
  * @brief Similar to https://en.cppreference.com/w/cpp/algorithm/upper_bound , but also works on 
  *        the CUDA device. Assumes the range to be sorted, but has O(log n) runtime in return.
  * 
  * @param first Beginning of the range where to find the upper bound
  * @param last End (exlusive) of the range where to find the upper bound
  * @param value The value for which we want to find the upper bound
  * @return int* The upper bound location. =End if outside
  */
  __device__ int* upper_bound(int* first, int* last, const int& value)
  {
      int* it;
      int count, step;
      count = last - first;
  
      while (count > 0) {
          it = first; 
          step = count / 2; 
          it += step;
          if (value >= *it) {
              first = ++it;
              count -= step + 1;
          } 
          else
              count = step;
      }
      return first;
  }
  
  __global__
  void perm_kernel(const int* active_elems_perm, const int* active_inds,
              bool* perm_mask,
              const unsigned int active_elems_size, 
              const unsigned int active_inds_size) {
      const int nr_perms = {active_perms};
      const int bidx = blockIdx.x;
      const int block_size = blockDim.x;
      const int tidx_global = block_size * bidx + threadIdx.x;
      const int tidx_local = threadIdx.x;
      __shared__ int active_inds_buf[{shared_buf_size}];
      const int nr_shared_bufs_needed = static_cast<int>(ceil(static_cast<float>(active_inds_size) / {shared_buf_size}));

      for(int shared_buf_run = 0; shared_buf_run < nr_shared_bufs_needed; shared_buf_run++)
      {
        const int shared_buf_offset = {shared_buf_size} * shared_buf_run;
        const int current_active_inds_size = min({shared_buf_size}, active_inds_size - {shared_buf_size}*shared_buf_run);
        if(shared_buf_run > 0)
          __syncthreads();
        
        //Fill shared memory
        for(int active_i = tidx_local; active_i < current_active_inds_size; active_i += block_size)
          active_inds_buf[active_i] = active_inds[shared_buf_offset + active_i];

        __syncthreads();
        
        for(int elem_offset = tidx_global; elem_offset < active_elems_size*nr_perms; elem_offset+=block_size*{parallel_blocks})
        {
          const int point_i = active_elems_perm[elem_offset];
          bool match = perm_mask[elem_offset]; //Maybe already set in the last shared buffer run
          if(!match)
          {
            const int* bound = upper_bound(active_inds_buf, active_inds_buf + current_active_inds_size, point_i);
            const int idx = max(0, (int)((bound-1) - active_inds_buf));
            if(active_inds_buf[idx] == point_i)
              perm_mask[elem_offset] = true;
          }
        }
      }
  }
}''') #: CUDA kernel to compute a mask of all element permutations containing at least one active index. New, more inefficient version using shared memory.
