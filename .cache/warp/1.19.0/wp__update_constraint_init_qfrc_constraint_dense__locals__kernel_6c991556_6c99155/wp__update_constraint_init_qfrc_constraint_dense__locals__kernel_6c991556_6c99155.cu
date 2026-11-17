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



extern "C" __global__ void _update_constraint_init_qfrc_constraint_dense__locals__kernel_70c42511_cuda_kernel_forward(
    wp::launch_bounds_t<2> dim,
    wp::array_t<wp::int32> var_nefc_in,
    wp::array_t<wp::float32> var_efc_J_in,
    wp::array_t<wp::float32> var_efc_force_in,
    wp::int32 var_njmax_in,
    wp::array_t<wp::int32> var_state_changed_count_in,
    wp::array_t<bool> var_ctx_done_in,
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
        bool* var_2;
        bool var_3;
        bool var_4;
        const bool var_5 = false;
        const wp::float32 var_6 = 0.0;
        wp::float32 var_7;
        wp::int32* var_8;
        wp::int32 var_9;
        wp::int32 var_10;
        wp::range_t var_11;
        wp::int32 var_12;
        wp::float32* var_13;
        wp::float32 var_14;
        wp::float32 var_15;
        wp::float32* var_16;
        wp::float32 var_17;
        wp::float32 var_18;
        wp::float32 var_19;
        wp::float32 var_20;
        //---------
        // forward
        // def kernel(                                                                            <L 2003>
        // worldid, dofid = wp.tid()                                                              <L 2015>
        builtin_tid2d(var_0, var_1);
        // if ctx_done_in[worldid]:                                                               <L 2017>
        var_2 = wp::address(var_ctx_done_in, var_0);
        var_3 = wp::load(var_2);
        if (var_3) {
            // return                                                                             <L 2018>
            return;
        }
        var_4 = wp::load(var_2);
        // if wp.static(STABLE_FAST):                                                             <L 2021>
        // sum_qfrc = float(0.0)                                                                  <L 2025>
        var_7 = wp::float(var_6);
        // for efcid in range(min(njmax_in, nefc_in[worldid])):                                   <L 2026>
        var_8 = wp::address(var_nefc_in, var_0);
        var_10 = wp::load(var_8);
        var_9 = wp::min(var_njmax_in, var_10);
        var_11 = wp::range(var_9);
        start_for_1:;
            if (iter_cmp(var_11) == 0) goto end_for_1;
            var_12 = wp::iter_next(var_11);
            // efc_J = efc_J_in[worldid, efcid, dofid]                                            <L 2027>
            var_13 = wp::address(var_efc_J_in, var_0, var_12, var_1);
            var_15 = wp::load(var_13);
            var_14 = wp::copy(var_15);
            // force = efc_force_in[worldid, efcid]                                               <L 2028>
            var_16 = wp::address(var_efc_force_in, var_0, var_12);
            var_18 = wp::load(var_16);
            var_17 = wp::copy(var_18);
            // sum_qfrc += efc_J * force                                                          <L 2029>
            var_19 = wp::mul(var_14, var_17);
            var_20 = wp::add(var_7, var_19);
            wp::assign(var_7, var_20);
            goto start_for_1;
        end_for_1:;
        // qfrc_constraint_out[worldid, dofid] = sum_qfrc                                         <L 2031>
        wp::array_store(var_qfrc_constraint_out, var_0, var_1, var_7);
}

