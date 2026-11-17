#define WP_NO_BFLOAT16

#define WP_TILE_BLOCK_DIM 256
#define WP_NO_CRT
#include "builtin.h"
#include "deterministic.h"

// Map wp.breakpoint() to a device brkpt at the call site so cuda-gdb attributes the stop to the generated .cu line
#if defined(__CUDACC__) && !defined(_MSC_VER)
#define __debugbreak() __brkpt()
#endif

#define builtin_tid1d() wp::tid(_idx, dim)
#define builtin_tid2d(x, y) wp::tid(x, y, _idx, dim)
#define builtin_tid3d(x, y, z) wp::tid(x, y, z, _idx, dim)
#define builtin_tid4d(x, y, z, w) wp::tid(x, y, z, w, _idx, dim)

#define builtin_block_dim() wp::block_dim()

// CUDA Thread Block Cluster shape declaration. Expands to __cluster_dims__
// only on devices that support clusters (compute capability 9.0+); otherwise
// expands to nothing so the same source compiles cleanly for any target arch.
#if defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 900)
#define WP_CLUSTER_DIMS(x, y, z) __cluster_dims__(x, y, z)
#else
#define WP_CLUSTER_DIMS(x, y, z)
#endif

// Maximum registers per thread. __maxnreg__ was added in CUDA Toolkit 12.4;
// older toolkits ignore the opt-in so the same source remains compilable.
#if defined(__CUDACC_VER_MAJOR__) && (__CUDACC_VER_MAJOR__ > 12 || (__CUDACC_VER_MAJOR__ == 12 && __CUDACC_VER_MINOR__ >= 4))
#define WP_MAXNREG(n) __maxnreg__(n)
#else
#define WP_MAXNREG(n)
#endif

// Allow PTXAS to spill registers into shared memory. CUDA Toolkit 13.0
// introduced the pragma, which is unavailable in device-debug compilation.
#if defined(__CUDACC_VER_MAJOR__) && (__CUDACC_VER_MAJOR__ >= 13) && !defined(_DEBUG)
#define WP_ENABLE_SMEM_SPILLING() asm volatile(".pragma \"enable_smem_spilling\";");
#else
#define WP_ENABLE_SMEM_SPILLING()
#endif


// avoid namespacing of float type for casting to float type, this is to avoid wp::float(x), which is not valid in C++
#define float(x) cast_float(x)
#define adj_float(x, adj_x, adj_ret) adj_cast_float(x, adj_x, adj_ret)

#define int(x) cast_int(x)
#define adj_int(x, adj_x, adj_ret) adj_cast_int(x, adj_x, adj_ret)



extern "C" __global__ void _solve_init_dof__locals__kernel_a1349e36_cuda_kernel_forward(
    wp::launch_bounds_t<2> dim,
    wp::array_t<wp::int32> var_nefc_in,
    wp::array_t<wp::float32> var_qacc_warmstart_in,
    wp::array_t<wp::float32> var_qacc_smooth_in,
    wp::array_t<wp::float32> var_qacc_out,
    wp::array_t<wp::float32> var_qfrc_constraint_out)
{
    wp::tile_shared_storage_t tile_mem;

    const size_t _idx = static_cast<size_t>(blockIdx.z * gridDim.y + blockIdx.y) * static_cast<size_t>(gridDim.x * blockDim.x) + static_cast<size_t>(blockIdx.x * blockDim.x + threadIdx.x);
    if (_idx >= dim.size) return;
            // reset shared memory allocator
    wp::tile_shared_storage_t::init();

        //---------
        // primal vars
        wp::int32 var_0;
        wp::int32 var_1;
        const bool var_2 = true;
        wp::int32* var_3;
        const wp::int32 var_4 = 0;
        bool var_5;
        wp::int32 var_6;
        wp::float32* var_7;
        wp::float32 var_8;
        wp::float32 var_9;
        wp::float32* var_10;
        wp::float32 var_11;
        wp::float32 var_12;
        const bool var_13 = true;
        wp::int32* var_14;
        const wp::int32 var_15 = 0;
        bool var_16;
        wp::int32 var_17;
        const wp::float32 var_18 = 0.0;
        //---------
        // forward
        // def kernel(                                                                            <L 1638>
        // worldid, dofid = wp.tid()                                                              <L 1647>
        builtin_tid2d(var_0, var_1);
        // if wp.static(WARMSTART):                                                               <L 1649>
        // if nefc_in[worldid] > 0:                                                               <L 1650>
        var_3 = wp::address(var_nefc_in, var_0);
        var_6 = wp::load(var_3);
        var_5 = (var_6 > var_4);
        if (var_5) {
            // qacc_out[worldid, dofid] = qacc_warmstart_in[worldid, dofid]                       <L 1651>
            var_7 = wp::address(var_qacc_warmstart_in, var_0, var_1);
            var_9 = wp::load(var_7);
            var_8 = wp::copy(var_9);
            wp::array_store(var_qacc_out, var_0, var_1, var_8);
        }
        if (!var_5) {
            // qacc_out[worldid, dofid] = qacc_smooth_in[worldid, dofid]                          <L 1653>
            var_10 = wp::address(var_qacc_smooth_in, var_0, var_1);
            var_12 = wp::load(var_10);
            var_11 = wp::copy(var_12);
            wp::array_store(var_qacc_out, var_0, var_1, var_11);
        }
        // if wp.static(SPARSE):                                                                  <L 1657>
        // if nefc_in[worldid] == 0:                                                              <L 1658>
        var_14 = wp::address(var_nefc_in, var_0);
        var_17 = wp::load(var_14);
        var_16 = (var_17 == var_15);
        if (var_16) {
            // qfrc_constraint_out[worldid, dofid] = 0.0                                          <L 1659>
            wp::array_store(var_qfrc_constraint_out, var_0, var_1, var_18);
        }
}

