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


// C:\Users\ASUS\Documents\human-imitation-learning\.venv\Lib\site-packages\mujoco_warp\_src\solver.py:2329
static CUDA_CALLABLE void _upper_triangle_index_0(
    wp::int32 var_elementid,
    wp::int32 var_n,
    wp::int32 & ret_0,
    wp::int32 & ret_1)
{
    //---------
    // primal vars
    const wp::int32 var_0 = 2;
    wp::int32 var_1;
    const wp::int32 var_2 = 1;
    wp::int32 var_3;
    wp::float32 var_4;
    wp::float32 var_5;
    const wp::float32 var_6 = 8.0;
    wp::float32 var_7;
    wp::float32 var_8;
    wp::float32 var_9;
    wp::float32 var_10;
    wp::float32 var_11;
    const wp::float32 var_12 = 0.5;
    wp::float32 var_13;
    wp::int32 var_14;
    wp::int32 var_15;
    const wp::int32 var_16 = 1;
    wp::int32 var_17;
    wp::int32 var_18;
    const wp::int32 var_19 = 2;
    wp::int32 var_20;
    wp::int32 var_21;
    bool var_22;
    const wp::int32 var_23 = 1;
    wp::int32 var_24;
    const wp::int32 var_25 = 1;
    wp::int32 var_26;
    wp::int32 var_27;
    const wp::int32 var_28 = 1;
    wp::int32 var_29;
    wp::int32 var_30;
    const wp::int32 var_31 = 2;
    wp::int32 var_32;
    wp::int32 var_33;
    bool var_34;
    const wp::int32 var_35 = 1;
    wp::int32 var_36;
    wp::int32 var_37;
    wp::int32 var_38;
    wp::int32 var_39;
    wp::int32 var_40;
    const wp::int32 var_41 = 1;
    wp::int32 var_42;
    wp::int32 var_43;
    const wp::int32 var_44 = 2;
    wp::int32 var_45;
    wp::int32 var_46;
    wp::int32 var_47;
    //---------
    // forward
    // def _upper_triangle_index(elementid: int, n: int):                                     <L 2330>
    // b = float(2 * n + 1)                                                                   <L 2333>
    var_1 = wp::mul(var_0, var_n);
    var_3 = wp::add(var_1, var_2);
    var_4 = wp::float(var_3);
    // i = int((b - wp.sqrt(b * b - 8.0 * float(elementid))) * 0.5)                           <L 2334>
    var_5 = wp::mul(var_4, var_4);
    var_7 = wp::float(var_elementid);
    var_8 = wp::mul(var_6, var_7);
    var_9 = wp::sub(var_5, var_8);
    var_10 = wp::sqrt(var_9);
    var_11 = wp::sub(var_4, var_10);
    var_13 = wp::mul(var_11, var_12);
    var_14 = wp::int(var_13);
    // if i * n - i * (i - 1) // 2 > elementid:                                               <L 2335>
    var_15 = wp::mul(var_14, var_n);
    var_17 = wp::sub(var_14, var_16);
    var_18 = wp::mul(var_14, var_17);
    var_20 = wp::floordiv(var_18, var_19);
    var_21 = wp::sub(var_15, var_20);
    var_22 = (var_21 > var_elementid);
    if (var_22) {
        // i -= 1                                                                             <L 2336>
        var_24 = wp::sub(var_14, var_23);
    }
    if (!var_22) {
        // elif (i + 1) * n - (i + 1) * i // 2 <= elementid:                                  <L 2337>
        var_26 = wp::add(var_14, var_25);
        var_27 = wp::mul(var_26, var_n);
        var_29 = wp::add(var_14, var_28);
        var_30 = wp::mul(var_29, var_14);
        var_32 = wp::floordiv(var_30, var_31);
        var_33 = wp::sub(var_27, var_32);
        var_34 = (var_33 <= var_elementid);
        if (var_34) {
            // i += 1                                                                         <L 2338>
            var_36 = wp::add(var_14, var_35);
        }
        var_37 = wp::where(var_34, var_36, var_14);
    }
    var_38 = wp::where(var_22, var_24, var_37);
    // return i, i + elementid - (i * n - i * (i - 1) // 2)                                   <L 2339>
    var_39 = wp::add(var_38, var_elementid);
    var_40 = wp::mul(var_38, var_n);
    var_42 = wp::sub(var_38, var_41);
    var_43 = wp::mul(var_38, var_42);
    var_45 = wp::floordiv(var_43, var_44);
    var_46 = wp::sub(var_40, var_45);
    var_47 = wp::sub(var_39, var_46);
    ret_0 = var_38;
    ret_1 = var_47;
    return;
}


// C:\Users\ASUS\Documents\human-imitation-learning\.venv\Lib\site-packages\mujoco_warp\_src\solver.py:2329
static CUDA_CALLABLE void adj__upper_triangle_index_0(
    wp::int32 var_elementid,
    wp::int32 var_n,
    wp::int32 & ret_0,
    wp::int32 & ret_1,
    wp::int32 & adj_elementid,
    wp::int32 & adj_n,
    wp::int32 & adj_ret_0,
    wp::int32 & adj_ret_1)
{
	// reverse mode disabled (no backward-enabled kernel depends on this function)
}



extern "C" __global__ void _update_gradient_init_h_sparse__locals__kernel_cb87950b_cuda_kernel_forward(
    wp::launch_bounds_t<2> dim,
    wp::int32 var_nv,
    wp::array_t<wp::int32> var_M_elemid,
    wp::array_t<wp::float32> var_M_in,
    wp::array_t<wp::int32> var_cdof_dof_in,
    wp::array_t<bool> var_ctx_done_in,
    wp::array_t<wp::float32> var_ctx_h_out)
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
        bool* var_2;
        bool var_3;
        bool var_4;
        wp::shape_t* var_5;
        const wp::int32 var_6 = 1;
        wp::int32 var_7;
        wp::shape_t var_8;
        wp::int32 var_9;
        wp::int32 var_10;
        const bool var_11 = false;
        wp::int32 var_12;
        bool var_13;
        bool var_14;
        const wp::float32 var_15 = 1.0;
        const wp::float32 var_16 = 0.0;
        wp::float32 var_17;
        wp::int32 var_18;
        const bool var_19 = false;
        wp::int32* var_20;
        wp::int32 var_21;
        wp::int32 var_22;
        const wp::int32 var_23 = 0;
        bool var_24;
        wp::float32* var_25;
        wp::float32 var_26;
        wp::float32 var_27;
        const wp::float32 var_28 = 0.0;
        //---------
        // forward
        // def kernel(                                                                            <L 2347>
        // worldid, elementid = wp.tid()                                                          <L 2359>
        builtin_tid2d(var_0, var_1);
        // if ctx_done_in[worldid]:                                                               <L 2361>
        var_2 = wp::address(var_ctx_done_in, var_0);
        var_3 = wp::load(var_2);
        if (var_3) {
            // return                                                                             <L 2362>
            continue;
        }
        var_4 = wp::load(var_2);
        // i, j = _upper_triangle_index(elementid, ctx_h_out.shape[1])                            <L 2365>
        var_5 = &(var_ctx_h_out.shape);
        var_8 = wp::load(var_5);
        var_7 = wp::extract(var_8, var_6);
        _upper_triangle_index_0(var_1, var_7, var_9, var_10);
        // if wp.static(COMPACT):                                                                 <L 2368>
        // dof_j = j                                                                              <L 2372>
        var_12 = wp::copy(var_10);
        // padding = j >= nv                                                                      <L 2373>
        var_13 = (var_10 >= var_nv);
        // if padding:                                                                            <L 2376>
        if (var_13) {
            // ctx_h_out[worldid, i, j] = wp.where(i == j, 1.0, 0.0)                              <L 2377>
            var_14 = (var_9 == var_10);
            var_17 = wp::where(var_14, var_15, var_16);
            wp::array_store(var_ctx_h_out, var_0, var_9, var_10, var_17);
            // return                                                                             <L 2378>
            continue;
        }
        // dof_i = i                                                                              <L 2380>
        var_18 = wp::copy(var_9);
        // if wp.static(COMPACT):                                                                 <L 2381>
        // elemid = M_elemid[dof_j, dof_i]                                                        <L 2385>
        var_20 = wp::address(var_M_elemid, var_12, var_18);
        var_22 = wp::load(var_20);
        var_21 = wp::copy(var_22);
        // if elemid >= 0:                                                                        <L 2386>
        var_24 = (var_21 >= var_23);
        if (var_24) {
            // ctx_h_out[worldid, i, j] = M_in[worldid, elemid]                                   <L 2387>
            var_25 = wp::address(var_M_in, var_0, var_21);
            var_27 = wp::load(var_25);
            var_26 = wp::copy(var_27);
            wp::array_store(var_ctx_h_out, var_0, var_9, var_10, var_26);
        }
        if (!var_24) {
            // ctx_h_out[worldid, i, j] = 0.0                                                     <L 2389>
            wp::array_store(var_ctx_h_out, var_0, var_9, var_10, var_28);
        }
    }
}

