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



extern "C" __global__ void reset_data__locals__reset_nworld_a721e0a6_cuda_kernel_forward(
    wp::launch_bounds_t<1> dim,
    wp::int32 var_nq,
    wp::int32 var_nv,
    wp::int32 var_nu,
    wp::int32 var_na,
    wp::int32 var_nbody,
    wp::int32 var_ntree,
    wp::int32 var_nflex,
    wp::int32 var_neq,
    wp::int32 var_nuserdata,
    wp::int32 var_nsensordata,
    wp::array_t<wp::float32> var_qpos0,
    wp::array_t<bool> var_eq_active0,
    wp::int32 var_nworld_in,
    wp::array_t<bool> var_reset_in,
    wp::array_t<wp::int32> var_solver_niter_out,
    wp::array_t<wp::int32> var_ne_out,
    wp::array_t<wp::int32> var_nf_out,
    wp::array_t<wp::int32> var_nl_out,
    wp::array_t<wp::int32> var_nefc_out,
    wp::array_t<wp::int32> var_ntree_awake_out,
    wp::array_t<wp::int32> var_nbody_awake_out,
    wp::array_t<wp::int32> var_nv_awake_out,
    wp::array_t<wp::float32> var_time_out,
    wp::array_t<wp::vec_t<2, wp::float32>> var_energy_out,
    wp::array_t<wp::float32> var_qpos_out,
    wp::array_t<wp::float32> var_qvel_out,
    wp::array_t<wp::float32> var_act_out,
    wp::array_t<wp::float32> var_qacc_warmstart_out,
    wp::array_t<wp::float32> var_ctrl_out,
    wp::array_t<wp::float32> var_qfrc_applied_out,
    wp::array_t<bool> var_eq_active_out,
    wp::array_t<wp::float32> var_qacc_out,
    wp::array_t<wp::float32> var_act_dot_out,
    wp::array_t<wp::float32> var_userdata_out,
    wp::array_t<wp::float32> var_sensordata_out,
    wp::array_t<bool> var_flex_hessian_valid_out,
    wp::array_t<wp::int32> var_nacon_out,
    wp::array_t<wp::int32> var_overflow_out)
{
    wp::tile_shared_storage_t tile_mem;

    const size_t _idx = static_cast<size_t>(blockIdx.z * gridDim.y + blockIdx.y) * static_cast<size_t>(gridDim.x * blockDim.x) + static_cast<size_t>(blockIdx.x * blockDim.x + threadIdx.x);
    if (_idx >= dim.size) return;
            // reset shared memory allocator
    wp::tile_shared_storage_t::init();

        //---------
        // primal vars
        wp::int32 var_0;
        const bool var_1 = true;
        bool* var_2;
        bool var_3;
        bool var_4;
        const wp::int32 var_5 = 0;
        const wp::int32 var_6 = 0;
        bool var_7;
        const wp::int32 var_8 = 0;
        const wp::int32 var_9 = 0;
        const wp::int32 var_10 = 0;
        const wp::int32 var_11 = 0;
        const wp::int32 var_12 = 0;
        const wp::int32 var_13 = 0;
        const wp::float32 var_14 = 0.0;
        const wp::float32 var_15 = 0.0;
        const wp::float32 var_16 = 0.0;
        wp::vec_t<2, wp::float32> var_17;
        wp::shape_t* var_18;
        const wp::int32 var_19 = 0;
        wp::int32 var_20;
        wp::shape_t var_21;
        wp::int32 var_22;
        wp::range_t var_23;
        wp::int32 var_24;
        wp::float32* var_25;
        wp::float32 var_26;
        wp::float32 var_27;
        bool var_28;
        const wp::float32 var_29 = 0.0;
        const wp::float32 var_30 = 0.0;
        const wp::float32 var_31 = 0.0;
        const wp::float32 var_32 = 0.0;
        wp::range_t var_33;
        wp::int32 var_34;
        const wp::float32 var_35 = 0.0;
        bool var_36;
        const wp::float32 var_37 = 0.0;
        const wp::float32 var_38 = 0.0;
        wp::range_t var_39;
        wp::int32 var_40;
        bool* var_41;
        bool var_42;
        bool var_43;
        wp::range_t var_44;
        wp::int32 var_45;
        const bool var_46 = false;
        wp::range_t var_47;
        wp::int32 var_48;
        const wp::float32 var_49 = 0.0;
        wp::range_t var_50;
        wp::int32 var_51;
        const wp::float32 var_52 = 0.0;
        const wp::int32 var_53 = 0;
        //---------
        // forward
        // def reset_nworld(                                                                      <L 2712>
        // worldid = wp.tid()                                                                     <L 2756>
        var_0 = builtin_tid1d();
        // if wp.static(reset is not None):                                                       <L 2758>
        // if not reset_in[worldid]:                                                              <L 2759>
        var_2 = wp::address(var_reset_in, var_0);
        var_4 = wp::load(var_2);
        var_3 = wp::unot(var_4);
        if (var_3) {
            // return                                                                             <L 2760>
            return;
        }
        // solver_niter_out[worldid] = 0                                                          <L 2762>
        wp::array_store(var_solver_niter_out, var_0, var_5);
        // if worldid == 0:                                                                       <L 2763>
        var_7 = (var_0 == var_6);
        if (var_7) {
            // nacon_out[0] = 0                                                                   <L 2764>
            wp::array_store(var_nacon_out, var_9, var_8);
        }
        // ne_out[worldid] = 0                                                                    <L 2765>
        wp::array_store(var_ne_out, var_0, var_10);
        // nf_out[worldid] = 0                                                                    <L 2766>
        wp::array_store(var_nf_out, var_0, var_11);
        // nl_out[worldid] = 0                                                                    <L 2767>
        wp::array_store(var_nl_out, var_0, var_12);
        // nefc_out[worldid] = 0                                                                  <L 2768>
        wp::array_store(var_nefc_out, var_0, var_13);
        // time_out[worldid] = 0.0                                                                <L 2769>
        wp::array_store(var_time_out, var_0, var_14);
        // energy_out[worldid] = wp.vec2(0.0, 0.0)                                                <L 2770>
        var_17 = wp::vec_t<2, wp::float32>(var_15, var_16);
        wp::array_store(var_energy_out, var_0, var_17);
        // ntree_awake_out[worldid] = ntree                                                       <L 2771>
        wp::array_store(var_ntree_awake_out, var_0, var_ntree);
        // nbody_awake_out[worldid] = nbody                                                       <L 2772>
        wp::array_store(var_nbody_awake_out, var_0, var_nbody);
        // nv_awake_out[worldid] = nv                                                             <L 2773>
        wp::array_store(var_nv_awake_out, var_0, var_nv);
        // qpos0_id = worldid % qpos0.shape[0]                                                    <L 2774>
        var_18 = &(var_qpos0.shape);
        var_21 = wp::load(var_18);
        var_20 = wp::extract(var_21, var_19);
        var_22 = wp::mod(var_0, var_20);
        // for i in range(nq):                                                                    <L 2775>
        var_23 = wp::range(var_nq);
        start_for_1:;
            if (iter_cmp(var_23) == 0) goto end_for_1;
            var_24 = wp::iter_next(var_23);
            // qpos_out[worldid, i] = qpos0[qpos0_id, i]                                          <L 2776>
            var_25 = wp::address(var_qpos0, var_22, var_24);
            var_27 = wp::load(var_25);
            var_26 = wp::copy(var_27);
            wp::array_store(var_qpos_out, var_0, var_24, var_26);
            // if i < nv:                                                                         <L 2777>
            var_28 = (var_24 < var_nv);
            if (var_28) {
                // qvel_out[worldid, i] = 0.0                                                     <L 2778>
                wp::array_store(var_qvel_out, var_0, var_24, var_29);
                // qacc_warmstart_out[worldid, i] = 0.0                                           <L 2779>
                wp::array_store(var_qacc_warmstart_out, var_0, var_24, var_30);
                // qfrc_applied_out[worldid, i] = 0.0                                             <L 2780>
                wp::array_store(var_qfrc_applied_out, var_0, var_24, var_31);
                // qacc_out[worldid, i] = 0.0                                                     <L 2781>
                wp::array_store(var_qacc_out, var_0, var_24, var_32);
            }
            goto start_for_1;
        end_for_1:;
        // for i in range(nu):                                                                    <L 2782>
        var_33 = wp::range(var_nu);
        start_for_3:;
            if (iter_cmp(var_33) == 0) goto end_for_3;
            var_34 = wp::iter_next(var_33);
            // ctrl_out[worldid, i] = 0.0                                                         <L 2783>
            wp::array_store(var_ctrl_out, var_0, var_34, var_35);
            // if i < na:                                                                         <L 2784>
            var_36 = (var_34 < var_na);
            if (var_36) {
                // act_out[worldid, i] = 0.0                                                      <L 2785>
                wp::array_store(var_act_out, var_0, var_34, var_37);
                // act_dot_out[worldid, i] = 0.0                                                  <L 2786>
                wp::array_store(var_act_dot_out, var_0, var_34, var_38);
            }
            goto start_for_3;
        end_for_3:;
        // for i in range(neq):                                                                   <L 2787>
        var_39 = wp::range(var_neq);
        start_for_5:;
            if (iter_cmp(var_39) == 0) goto end_for_5;
            var_40 = wp::iter_next(var_39);
            // eq_active_out[worldid, i] = eq_active0[i]                                          <L 2788>
            var_41 = wp::address(var_eq_active0, var_40);
            var_43 = wp::load(var_41);
            var_42 = wp::copy(var_43);
            wp::array_store(var_eq_active_out, var_0, var_40, var_42);
            goto start_for_5;
        end_for_5:;
        // for i in range(nflex):                                                                 <L 2789>
        var_44 = wp::range(var_nflex);
        start_for_7:;
            if (iter_cmp(var_44) == 0) goto end_for_7;
            var_45 = wp::iter_next(var_44);
            // flex_hessian_valid_out[worldid, i] = False                                         <L 2790>
            wp::array_store(var_flex_hessian_valid_out, var_0, var_45, var_46);
            goto start_for_7;
        end_for_7:;
        // for i in range(nsensordata):                                                           <L 2791>
        var_47 = wp::range(var_nsensordata);
        start_for_9:;
            if (iter_cmp(var_47) == 0) goto end_for_9;
            var_48 = wp::iter_next(var_47);
            // sensordata_out[worldid, i] = 0.0                                                   <L 2792>
            wp::array_store(var_sensordata_out, var_0, var_48, var_49);
            goto start_for_9;
        end_for_9:;
        // for i in range(nuserdata):                                                             <L 2793>
        var_50 = wp::range(var_nuserdata);
        start_for_11:;
            if (iter_cmp(var_50) == 0) goto end_for_11;
            var_51 = wp::iter_next(var_50);
            // userdata_out[worldid, i] = 0.0                                                     <L 2794>
            wp::array_store(var_userdata_out, var_0, var_51, var_52);
            goto start_for_11;
        end_for_11:;
        // overflow_out[worldid] = 0                                                              <L 2795>
        wp::array_store(var_overflow_out, var_0, var_53);
}

