#define WP_NO_BFLOAT16

#define WP_TILE_BLOCK_DIM 64
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



extern "C" __global__ void _contact_sort__locals__contact_sort_a69e9756_cuda_kernel_forward(
    wp::launch_bounds_t<2> dim,
    wp::array_t<wp::int32> var_sensor_intprm,
    wp::array_t<wp::int32> var_sensor_contact_adr_in,
    wp::array_t<wp::int32> var_sensor_contact_nmatch_in,
    wp::array_t<wp::int32> var_sensor_contact_matchid_in,
    wp::array_t<wp::float32> var_sensor_contact_criteria_in,
    wp::array_t<wp::int32> var_sensor_contact_matchid_out)
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
        wp::int32* var_2;
        wp::int32 var_3;
        wp::int32 var_4;
        const wp::int32 var_5 = 1;
        wp::int32* var_6;
        wp::int32 var_7;
        wp::int32 var_8;
        bool var_9;
        const wp::int32 var_10 = 0;
        bool var_11;
        const wp::int32 var_12 = 3;
        bool var_13;
        wp::int32* var_14;
        wp::int32 var_15;
        wp::int32 var_16;
        const wp::int32 var_17 = 1;
        bool var_18;
        wp::array_t<wp::float32> var_19;
        const wp::int32 var_20 = 64;
        wp::tile_shared_t<wp::float32,wp::tile_layout_strided_t<wp::tile_shape_t<64>, wp::tile_stride_t<1>>, true> var_21 = wp::tile_alloc_empty<wp::float32,wp::tile_shape_t<64>,wp::tile_stride_t<1>,false>();
        const wp::int32 var_22 = 0;
        wp::array_t<wp::int32> var_23;
        wp::tile_shared_t<wp::int32,wp::tile_layout_strided_t<wp::tile_shape_t<64>, wp::tile_stride_t<1>>, true> var_24 = wp::tile_alloc_empty<wp::int32,wp::tile_shape_t<64>,wp::tile_stride_t<1>,false>();
        const wp::int32 var_25 = 0;
        wp::array_t<wp::int32> var_26;
        const wp::int32 var_27 = 0;
        //---------
        // forward
        // def contact_sort(                                                                      <L 2402>
        // worldid, contactsensorid = wp.tid()                                                    <L 2413>
        builtin_tid2d(var_0, var_1);
        // sensorid = sensor_contact_adr_in[contactsensorid]                                      <L 2414>
        var_2 = wp::address(var_sensor_contact_adr_in, var_1);
        var_4 = wp::load(var_2);
        var_3 = wp::copy(var_4);
        // reduce = sensor_intprm[sensorid, 1]                                                    <L 2416>
        var_6 = wp::address(var_sensor_intprm, var_3, var_5);
        var_8 = wp::load(var_6);
        var_7 = wp::copy(var_8);
        // if reduce == 0 or reduce == 3:  # none or netforce                                     <L 2417>
        var_11 = (var_7 == var_10);
        var_9 = var_11;
        if (!var_9) {
            var_13 = (var_7 == var_12);
            var_9 = var_9 || var_13;
        }
        if (var_9) {
            // return                                                                             <L 2418>
            return;
        }
        // nmatch = sensor_contact_nmatch_in[worldid, contactsensorid]                            <L 2420>
        var_14 = wp::address(var_sensor_contact_nmatch_in, var_0, var_1);
        var_16 = wp::load(var_14);
        var_15 = wp::copy(var_16);
        // if nmatch <= 1:                                                                        <L 2423>
        var_18 = (var_15 <= var_17);
        if (var_18) {
            // return                                                                             <L 2424>
            return;
        }
        // criteria_tile = wp.tile_load(sensor_contact_criteria_in[worldid, contactsensorid], shape=maxmatch)       <L 2426>
        var_19 = wp::view(var_sensor_contact_criteria_in, var_0, var_1);
        var_21 = wp::tile_load<wp::float32, true, false, 64>(var_19, var_22);
        // matchid_tile = wp.tile_load(sensor_contact_matchid_in[worldid, contactsensorid], shape=maxmatch)       <L 2427>
        var_23 = wp::view(var_sensor_contact_matchid_in, var_0, var_1);
        var_24 = wp::tile_load<wp::int32, true, false, 64>(var_23, var_25);
        // wp.tile_sort(criteria_tile, matchid_tile)                                              <L 2428>
        wp::tile_sort(var_21, var_24);
        // wp.tile_store(sensor_contact_matchid_out[worldid, contactsensorid], matchid_tile)       <L 2429>
        var_26 = wp::view(var_sensor_contact_matchid_out, var_0, var_1);
        wp::tile_store<wp::int32, true, false>(var_26, var_27, var_24);
}

