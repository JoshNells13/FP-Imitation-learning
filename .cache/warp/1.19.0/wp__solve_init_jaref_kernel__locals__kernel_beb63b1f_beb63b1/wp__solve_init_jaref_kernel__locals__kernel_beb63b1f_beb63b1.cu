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



extern "C" __global__ void _solve_init_jaref_kernel__locals__kernel_7eae8057_cuda_kernel_forward(
    wp::launch_bounds_t<3> dim,
    wp::array_t<wp::int32> var_nefc_in,
    wp::array_t<wp::float32> var_qacc_in,
    wp::array_t<wp::int32> var_efc_J_rownnz_in,
    wp::array_t<wp::int32> var_efc_J_rowadr_in,
    wp::array_t<wp::int32> var_efc_J_colind_in,
    wp::array_t<wp::float32> var_efc_J_in,
    wp::array_t<wp::float32> var_efc_aref_in,
    wp::array_t<wp::int32> var_dof_cdof_in,
    wp::array_t<wp::float32> var_ctx_Jaref_out)
{
    wp::tile_shared_storage_t tile_mem;

    for (size_t _idx = static_cast<size_t>(blockDim.x) * static_cast<size_t>(blockIdx.x) + static_cast<size_t>(threadIdx.x);
         _idx < dim.size;
         _idx += static_cast<size_t>(blockDim.x) * static_cast<size_t>(gridDim.x))
    {
            // reset shared memory allocator
        wp::tile_shared_storage_t::init();

        //---------
        // primal vars
        wp::int32 var_0;
        wp::int32 var_1;
        wp::int32 var_2;
        wp::int32* var_3;
        bool var_4;
        wp::int32 var_5;
        const wp::float32 var_6 = 0.0;
        wp::float32 var_7;
        const bool var_8 = false;
        const bool var_9 = true;
        wp::int32 var_10;
        wp::int32 var_11;
        wp::float32* var_12;
        wp::float32* var_13;
        wp::float32 var_14;
        wp::float32 var_15;
        wp::float32 var_16;
        wp::float32 var_17;
        wp::float32* var_18;
        wp::float32 var_19;
        wp::float32 var_20;
        //---------
        // forward
        // def kernel(                                                                            <L 1683>
        // worldid, efcid, dofstart = wp.tid()                                                    <L 1696>
        builtin_tid3d(var_0, var_1, var_2);
        // if efcid >= nefc_in[worldid]:                                                          <L 1698>
        var_3 = wp::address(var_nefc_in, var_0);
        var_5 = wp::load(var_3);
        var_4 = (var_1 >= var_5);
        if (var_4) {
            // return                                                                             <L 1699>
            continue;
        }
        // jaref = float(0.0)                                                                     <L 1701>
        var_7 = wp::float(var_6);
        // if wp.static(is_sparse):                                                               <L 1702>
        // if wp.static(dofs_per_thread >= nv):                                                   <L 1715>
        // for i in range(wp.static(min(dofs_per_thread, nv))):                                   <L 1716>
        var_10 = 0;
        start_for_1:;
            if (var_10 >= 28) goto end_for_1;
            var_11 = var_10;
            var_10 += 1;
            // jaref += efc_J_in[worldid, efcid, i] * qacc_in[worldid, i]                         <L 1717>
            var_12 = wp::address(var_efc_J_in, var_0, var_1, var_11);
            var_13 = wp::address(var_qacc_in, var_0, var_11);
            var_15 = wp::load(var_12);
            var_16 = wp::load(var_13);
            var_14 = wp::mul(var_15, var_16);
            var_17 = wp::add(var_7, var_14);
            wp::assign(var_7, var_17);
            goto start_for_1;
        end_for_1:;
        // ctx_Jaref_out[worldid, efcid] = jaref - efc_aref_in[worldid, efcid]                    <L 1718>
        var_18 = wp::address(var_efc_aref_in, var_0, var_1);
        var_20 = wp::load(var_18);
        var_19 = wp::sub(var_7, var_20);
        wp::array_store(var_ctx_Jaref_out, var_0, var_1, var_19);
    }
}

