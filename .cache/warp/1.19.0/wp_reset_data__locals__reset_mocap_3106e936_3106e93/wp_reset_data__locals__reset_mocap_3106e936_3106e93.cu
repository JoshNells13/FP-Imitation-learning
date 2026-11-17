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



extern "C" __global__ void reset_data__locals__reset_mocap_9e83f44d_cuda_kernel_forward(
    wp::launch_bounds_t<2> dim,
    wp::array_t<wp::int32> var_body_mocapid,
    wp::array_t<wp::vec_t<3, wp::float32>> var_body_pos,
    wp::array_t<wp::quat_t<wp::float32>> var_body_quat,
    wp::array_t<bool> var_reset_in,
    wp::array_t<wp::vec_t<3, wp::float32>> var_mocap_pos_out,
    wp::array_t<wp::quat_t<wp::float32>> var_mocap_quat_out)
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
        bool* var_3;
        bool var_4;
        bool var_5;
        wp::int32* var_6;
        wp::int32 var_7;
        wp::int32 var_8;
        const wp::int32 var_9 = 0;
        bool var_10;
        wp::shape_t* var_11;
        const wp::int32 var_12 = 0;
        wp::int32 var_13;
        wp::shape_t var_14;
        wp::int32 var_15;
        wp::vec_t<3, wp::float32>* var_16;
        wp::vec_t<3, wp::float32> var_17;
        wp::vec_t<3, wp::float32> var_18;
        wp::shape_t* var_19;
        const wp::int32 var_20 = 0;
        wp::int32 var_21;
        wp::shape_t var_22;
        wp::int32 var_23;
        wp::quat_t<wp::float32>* var_24;
        wp::quat_t<wp::float32> var_25;
        wp::quat_t<wp::float32> var_26;
        //---------
        // forward
        // def reset_mocap(                                                                       <L 2798>
        // worldid, bodyid = wp.tid()                                                             <L 2809>
        builtin_tid2d(var_0, var_1);
        // if wp.static(reset is not None):                                                       <L 2811>
        // if not reset_in[worldid]:                                                              <L 2812>
        var_3 = wp::address(var_reset_in, var_0);
        var_5 = wp::load(var_3);
        var_4 = wp::unot(var_5);
        if (var_4) {
            // return                                                                             <L 2813>
            return;
        }
        // mocapid = body_mocapid[bodyid]                                                         <L 2815>
        var_6 = wp::address(var_body_mocapid, var_1);
        var_8 = wp::load(var_6);
        var_7 = wp::copy(var_8);
        // if mocapid >= 0:                                                                       <L 2817>
        var_10 = (var_7 >= var_9);
        if (var_10) {
            // mocap_pos_out[worldid, mocapid] = body_pos[worldid % body_pos.shape[0], bodyid]       <L 2818>
            var_11 = &(var_body_pos.shape);
            var_14 = wp::load(var_11);
            var_13 = wp::extract(var_14, var_12);
            var_15 = wp::mod(var_0, var_13);
            var_16 = wp::address(var_body_pos, var_15, var_1);
            var_18 = wp::load(var_16);
            var_17 = wp::copy(var_18);
            wp::array_store(var_mocap_pos_out, var_0, var_7, var_17);
            // mocap_quat_out[worldid, mocapid] = body_quat[worldid % body_quat.shape[0], bodyid]       <L 2819>
            var_19 = &(var_body_quat.shape);
            var_22 = wp::load(var_19);
            var_21 = wp::extract(var_22, var_20);
            var_23 = wp::mod(var_0, var_21);
            var_24 = wp::address(var_body_quat, var_23, var_1);
            var_26 = wp::load(var_24);
            var_25 = wp::copy(var_26);
            wp::array_store(var_mocap_quat_out, var_0, var_7, var_25);
        }
}

