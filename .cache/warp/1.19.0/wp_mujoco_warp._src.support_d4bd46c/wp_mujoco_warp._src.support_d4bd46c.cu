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


// C:\Users\ASUS\Documents\human-imitation-learning\.venv\Lib\site-packages\mujoco_warp\_src\support.py:320
static CUDA_CALLABLE wp::vec_t<6, wp::float32> _decode_pyramid_0(
    wp::int32 var_njmax_in,
    wp::array_t<wp::float32> var_pyramid,
    wp::int32 var_efc_address,
    wp::vec_t<5, wp::float32> var_mu,
    wp::int32 var_condim)
{
    //---------
    // primal vars
    wp::vec_t<6, wp::float32> var_0;
    const wp::int32 var_1 = 1;
    bool var_2;
    wp::float32* var_3;
    wp::float32 var_4;
    wp::float32 var_5;
    const wp::int32 var_6 = 0;
    const wp::float32 var_7 = 0.0;
    wp::float32 var_8;
    const wp::int32 var_9 = 0;
    const wp::int32 var_10 = 1;
    wp::int32 var_11;
    wp::range_t var_12;
    wp::int32 var_13;
    const wp::int32 var_14 = 2;
    wp::int32 var_15;
    wp::int32 var_16;
    bool var_17;
    wp::float32* var_18;
    wp::float32 var_19;
    wp::float32 var_20;
    const wp::float32 var_21 = 0.0;
    wp::float32 var_22;
    const wp::int32 var_23 = 1;
    wp::int32 var_24;
    bool var_25;
    const wp::int32 var_26 = 1;
    wp::int32 var_27;
    wp::float32* var_28;
    wp::float32 var_29;
    wp::float32 var_30;
    const wp::float32 var_31 = 0.0;
    wp::float32 var_32;
    const wp::int32 var_33 = 0;
    wp::float32 var_34;
    wp::float32 var_35;
    wp::float32 var_36;
    wp::float32 var_37;
    const wp::int32 var_38 = 1;
    wp::int32 var_39;
    //---------
    // forward
    // def _decode_pyramid(njmax_in: int, pyramid: wp.array[float], efc_address: int, mu: vec5, condim: int) -> wp.spatial_vector:       <L 321>
    // force = wp.spatial_vector()                                                            <L 323>
    var_0 = wp::vec_t<6, wp::float32>();
    // if condim == 1:                                                                        <L 325>
    var_2 = (var_condim == var_1);
    if (var_2) {
        // force[0] = pyramid[efc_address]                                                    <L 326>
        var_3 = wp::address(var_pyramid, var_efc_address);
        var_5 = wp::load(var_3);
        var_4 = wp::copy(var_5);
        wp::assign_inplace(var_0, var_6, var_4);
        // return force                                                                       <L 327>
        return var_0;
    }
    // force[0] = float(0.0)                                                                  <L 329>
    var_8 = wp::float(var_7);
    wp::assign_inplace(var_0, var_9, var_8);
    // for i in range(condim - 1):                                                            <L 330>
    var_11 = wp::sub(var_condim, var_10);
    var_12 = wp::range(var_11);
    start_for_1:;
        if (iter_cmp(var_12) == 0) goto end_for_1;
        var_13 = wp::iter_next(var_12);
        // adr = 2 * i + efc_address                                                          <L 331>
        var_15 = wp::mul(var_14, var_13);
        var_16 = wp::add(var_15, var_efc_address);
        // if adr < njmax_in:                                                                 <L 332>
        var_17 = (var_16 < var_njmax_in);
        if (var_17) {
            // dir1 = pyramid[adr]                                                            <L 333>
            var_18 = wp::address(var_pyramid, var_16);
            var_20 = wp::load(var_18);
            var_19 = wp::copy(var_20);
        }
        if (!var_17) {
            // dir1 = 0.0                                                                     <L 335>
        }
        var_22 = wp::where(var_17, var_19, var_21);
        // if adr + 1 < njmax_in:                                                             <L 336>
        var_24 = wp::add(var_16, var_23);
        var_25 = (var_24 < var_njmax_in);
        if (var_25) {
            // dir2 = pyramid[adr + 1]                                                        <L 337>
            var_27 = wp::add(var_16, var_26);
            var_28 = wp::address(var_pyramid, var_27);
            var_30 = wp::load(var_28);
            var_29 = wp::copy(var_30);
        }
        if (!var_25) {
            // dir2 = 0.0                                                                     <L 339>
        }
        var_32 = wp::where(var_25, var_29, var_31);
        // force[0] += dir1 + dir2                                                            <L 340>
        var_34 = wp::add(var_22, var_32);
        wp::add_inplace(var_0, var_33, var_34);
        // force[i + 1] = (dir1 - dir2) * mu[i]                                               <L 341>
        var_35 = wp::sub(var_22, var_32);
        var_36 = wp::extract(var_mu, var_13);
        var_37 = wp::mul(var_35, var_36);
        var_39 = wp::add(var_13, var_38);
        wp::assign_inplace(var_0, var_39, var_37);
        goto start_for_1;
    end_for_1:;
    // return force                                                                           <L 343>
    return var_0;
}


// C:\Users\ASUS\Documents\human-imitation-learning\.venv\Lib\site-packages\mujoco_warp\_src\support.py:346
static CUDA_CALLABLE wp::vec_t<6, wp::float32> contact_force_fn_0(
    wp::int32 var_opt_cone,
    wp::array_t<wp::mat_t<3, 3, wp::float32>> var_contact_frame_in,
    wp::array_t<wp::vec_t<5, wp::float32>> var_contact_friction_in,
    wp::array_t<wp::int32> var_contact_dim_in,
    wp::array_t<wp::int32> var_contact_efc_address_in,
    wp::array_t<wp::float32> var_contact_adhesion_in,
    wp::array_t<wp::float32> var_efc_force_in,
    wp::int32 var_njmax_in,
    wp::array_t<wp::int32> var_nacon_in,
    wp::int32 var_worldid,
    wp::int32 var_contact_id,
    bool var_to_world_frame)
{
    //---------
    // primal vars
    const wp::float32 var_0 = 0.0;
    const wp::float32 var_1 = 0.0;
    const wp::float32 var_2 = 0.0;
    const wp::float32 var_3 = 0.0;
    const wp::float32 var_4 = 0.0;
    const wp::float32 var_5 = 0.0;
    wp::vec_t<6, wp::float32> var_6;
    wp::int32* var_7;
    wp::int32 var_8;
    wp::int32 var_9;
    const wp::int32 var_10 = 0;
    wp::int32* var_11;
    wp::int32 var_12;
    wp::int32 var_13;
    bool var_14;
    const wp::int32 var_15 = 0;
    bool var_16;
    const wp::int32 var_17 = 0;
    wp::int32* var_18;
    bool var_19;
    wp::int32 var_20;
    const wp::int32 var_21 = 0;
    bool var_22;
    const wp::int32 var_23 = 0;
    bool var_24;
    wp::array_t<wp::float32> var_25;
    wp::vec_t<5, wp::float32>* var_26;
    wp::vec_t<6, wp::float32> var_27;
    wp::vec_t<5, wp::float32> var_28;
    wp::range_t var_29;
    wp::int32 var_30;
    wp::int32* var_31;
    bool var_32;
    wp::int32 var_33;
    wp::int32* var_34;
    wp::int32 var_35;
    wp::float32* var_36;
    wp::float32 var_37;
    wp::float32 var_38;
    wp::vec_t<6, wp::float32> var_39;
    const wp::int32 var_40 = 0;
    wp::float32* var_41;
    wp::float32 var_42;
    wp::vec_t<6, wp::float32> var_43;
    wp::vec_t<3, wp::float32> var_44;
    wp::mat_t<3, 3, wp::float32>* var_45;
    wp::vec_t<3, wp::float32> var_46;
    wp::mat_t<3, 3, wp::float32> var_47;
    wp::vec_t<3, wp::float32> var_48;
    wp::mat_t<3, 3, wp::float32>* var_49;
    wp::vec_t<3, wp::float32> var_50;
    wp::mat_t<3, 3, wp::float32> var_51;
    wp::vec_t<6, wp::float32> var_52;
    wp::vec_t<6, wp::float32> var_53;
    //---------
    // forward
    // def contact_force_fn(                                                                  <L 347>
    // force = wp.spatial_vector(0.0, 0.0, 0.0, 0.0, 0.0, 0.0)                                <L 365>
    var_6 = wp::vec_t<6, wp::float32>({var_0, var_1, var_2, var_3, var_4, var_5});
    // condim = contact_dim_in[contact_id]                                                    <L 366>
    var_7 = wp::address(var_contact_dim_in, var_contact_id);
    var_9 = wp::load(var_7);
    var_8 = wp::copy(var_9);
    // efc_address = contact_efc_address_in[contact_id, 0]                                    <L 367>
    var_11 = wp::address(var_contact_efc_address_in, var_contact_id, var_10);
    var_13 = wp::load(var_11);
    var_12 = wp::copy(var_13);
    // if contact_id >= 0 and contact_id <= nacon_in[0] and efc_address >= 0:                 <L 369>
    var_16 = (var_contact_id >= var_15);
    var_14 = var_16;
    if (var_14) {
        var_18 = wp::address(var_nacon_in, var_17);
        var_20 = wp::load(var_18);
        var_19 = (var_contact_id <= var_20);
        var_14 = var_14 && var_19;
    }
    if (var_14) {
        var_22 = (var_12 >= var_21);
        var_14 = var_14 && var_22;
    }
    if (var_14) {
        // if opt_cone == ConeType.PYRAMIDAL:                                                 <L 370>
        var_24 = (var_opt_cone == var_23);
        if (var_24) {
            // force = _decode_pyramid(                                                       <L 371>
            // njmax_in,                                                                      <L 372>
            // efc_force_in[worldid],                                                         <L 373>
            var_25 = wp::view(var_efc_force_in, var_worldid);
            // efc_address,                                                                   <L 374>
            // contact_friction_in[contact_id],                                               <L 375>
            var_26 = wp::address(var_contact_friction_in, var_contact_id);
            // condim,                                                                        <L 376>
            var_28 = wp::load(var_26);
            var_27 = _decode_pyramid_0(var_njmax_in, var_25, var_12, var_28, var_8);
        }
        if (!var_24) {
            // for i in range(condim):                                                        <L 379>
            var_29 = wp::range(var_8);
            start_for_0:;
                if (iter_cmp(var_29) == 0) goto end_for_0;
                var_30 = wp::iter_next(var_29);
                // if contact_efc_address_in[contact_id, i] < njmax_in:                       <L 380>
                var_31 = wp::address(var_contact_efc_address_in, var_contact_id, var_30);
                var_33 = wp::load(var_31);
                var_32 = (var_33 < var_njmax_in);
                if (var_32) {
                    // force[i] = efc_force_in[worldid, contact_efc_address_in[contact_id, i]]       <L 381>
                    var_34 = wp::address(var_contact_efc_address_in, var_contact_id, var_30);
                    var_35 = wp::load(var_34);
                    var_36 = wp::address(var_efc_force_in, var_worldid, var_35);
                    var_38 = wp::load(var_36);
                    var_37 = wp::copy(var_38);
                    wp::assign_inplace(var_6, var_30, var_37);
                }
                goto start_for_0;
            end_for_0:;
        }
        var_39 = wp::where(var_24, var_27, var_6);
        // force[0] -= contact_adhesion_in[contact_id]                                        <L 384>
        var_41 = wp::address(var_contact_adhesion_in, var_contact_id);
        var_42 = wp::load(var_41);
        wp::sub_inplace(var_39, var_40, var_42);
    }
    var_43 = wp::where(var_14, var_39, var_6);
    // if to_world_frame:                                                                     <L 386>
    if (var_to_world_frame) {
        // t = wp.spatial_top(force) @ contact_frame_in[contact_id]                           <L 388>
        var_44 = wp::spatial_top(var_43);
        var_45 = wp::address(var_contact_frame_in, var_contact_id);
        var_47 = wp::load(var_45);
        var_46 = wp::mul(var_44, var_47);
        // b = wp.spatial_bottom(force) @ contact_frame_in[contact_id]                        <L 389>
        var_48 = wp::spatial_bottom(var_43);
        var_49 = wp::address(var_contact_frame_in, var_contact_id);
        var_51 = wp::load(var_49);
        var_50 = wp::mul(var_48, var_51);
        // force = wp.spatial_vector(t, b)                                                    <L 390>
        var_52 = wp::vec_t<6, wp::float32>(var_46, var_50);
    }
    var_53 = wp::where(var_to_world_frame, var_52, var_43);
    // return force                                                                           <L 392>
    return var_53;
}


// C:\Users\ASUS\Documents\human-imitation-learning\.venv\Lib\site-packages\mujoco_warp\_src\support.py:320
static CUDA_CALLABLE void adj__decode_pyramid_0(
    wp::int32 var_njmax_in,
    wp::array_t<wp::float32> var_pyramid,
    wp::int32 var_efc_address,
    wp::vec_t<5, wp::float32> var_mu,
    wp::int32 var_condim,
    wp::int32 & adj_njmax_in,
    wp::array_t<wp::float32> & adj_pyramid,
    wp::int32 & adj_efc_address,
    wp::vec_t<5, wp::float32> & adj_mu,
    wp::int32 & adj_condim,
    wp::vec_t<6, wp::float32> & adj_ret)
{
	// reverse mode disabled (no backward-enabled kernel depends on this function)
}


// C:\Users\ASUS\Documents\human-imitation-learning\.venv\Lib\site-packages\mujoco_warp\_src\support.py:346
static CUDA_CALLABLE void adj_contact_force_fn_0(
    wp::int32 var_opt_cone,
    wp::array_t<wp::mat_t<3, 3, wp::float32>> var_contact_frame_in,
    wp::array_t<wp::vec_t<5, wp::float32>> var_contact_friction_in,
    wp::array_t<wp::int32> var_contact_dim_in,
    wp::array_t<wp::int32> var_contact_efc_address_in,
    wp::array_t<wp::float32> var_contact_adhesion_in,
    wp::array_t<wp::float32> var_efc_force_in,
    wp::int32 var_njmax_in,
    wp::array_t<wp::int32> var_nacon_in,
    wp::int32 var_worldid,
    wp::int32 var_contact_id,
    bool var_to_world_frame,
    wp::int32 & adj_opt_cone,
    wp::array_t<wp::mat_t<3, 3, wp::float32>> & adj_contact_frame_in,
    wp::array_t<wp::vec_t<5, wp::float32>> & adj_contact_friction_in,
    wp::array_t<wp::int32> & adj_contact_dim_in,
    wp::array_t<wp::int32> & adj_contact_efc_address_in,
    wp::array_t<wp::float32> & adj_contact_adhesion_in,
    wp::array_t<wp::float32> & adj_efc_force_in,
    wp::int32 & adj_njmax_in,
    wp::array_t<wp::int32> & adj_nacon_in,
    wp::int32 & adj_worldid,
    wp::int32 & adj_contact_id,
    bool & adj_to_world_frame,
    wp::vec_t<6, wp::float32> & adj_ret)
{
	// reverse mode disabled (no backward-enabled kernel depends on this function)
}



extern "C" __global__ void contact_force_kernel_0710b6fa_cuda_kernel_forward(
    wp::launch_bounds_t<1> dim,
    wp::int32 var_opt_cone,
    wp::array_t<wp::mat_t<3, 3, wp::float32>> var_contact_frame_in,
    wp::array_t<wp::vec_t<5, wp::float32>> var_contact_friction_in,
    wp::array_t<wp::int32> var_contact_dim_in,
    wp::array_t<wp::int32> var_contact_efc_address_in,
    wp::array_t<wp::int32> var_contact_worldid_in,
    wp::array_t<wp::float32> var_contact_adhesion_in,
    wp::array_t<wp::float32> var_efc_force_in,
    wp::int32 var_njmax_in,
    wp::array_t<wp::int32> var_nacon_in,
    wp::array_t<wp::int32> var_contact_ids,
    bool var_to_world_frame,
    wp::array_t<wp::vec_t<6, wp::float32>> var_out)
{
    wp::tile_shared_storage_t tile_mem;

    const size_t _idx = static_cast<size_t>(blockIdx.z * gridDim.y + blockIdx.y) * static_cast<size_t>(gridDim.x * blockDim.x) + static_cast<size_t>(blockIdx.x * blockDim.x + threadIdx.x);
    if (_idx >= dim.size) return;
            // reset shared memory allocator
    wp::tile_shared_storage_t::init();

        //---------
        // primal vars
        wp::int32 var_0;
        wp::int32* var_1;
        wp::int32 var_2;
        wp::int32 var_3;
        const wp::int32 var_4 = 0;
        wp::int32* var_5;
        bool var_6;
        wp::int32 var_7;
        wp::int32* var_8;
        wp::int32 var_9;
        wp::int32 var_10;
        wp::vec_t<6, wp::float32> var_11;
        //---------
        // forward
        // def contact_force_kernel(                                                              <L 396>
        // tid = wp.tid()                                                                         <L 415>
        var_0 = builtin_tid1d();
        // contactid = contact_ids[tid]                                                           <L 417>
        var_1 = wp::address(var_contact_ids, var_0);
        var_3 = wp::load(var_1);
        var_2 = wp::copy(var_3);
        // if contactid >= nacon_in[0]:                                                           <L 419>
        var_5 = wp::address(var_nacon_in, var_4);
        var_7 = wp::load(var_5);
        var_6 = (var_2 >= var_7);
        if (var_6) {
            // return                                                                             <L 420>
            return;
        }
        // worldid = contact_worldid_in[contactid]                                                <L 422>
        var_8 = wp::address(var_contact_worldid_in, var_2);
        var_10 = wp::load(var_8);
        var_9 = wp::copy(var_10);
        // out[tid] = contact_force_fn(                                                           <L 424>
        // opt_cone,                                                                              <L 425>
        // contact_frame_in,                                                                      <L 426>
        // contact_friction_in,                                                                   <L 427>
        // contact_dim_in,                                                                        <L 428>
        // contact_efc_address_in,                                                                <L 429>
        // contact_adhesion_in,                                                                   <L 430>
        // efc_force_in,                                                                          <L 431>
        // njmax_in,                                                                              <L 432>
        // nacon_in,                                                                              <L 433>
        // worldid,                                                                               <L 434>
        // contactid,                                                                             <L 435>
        // to_world_frame,                                                                        <L 436>
        var_11 = contact_force_fn_0(var_opt_cone, var_contact_frame_in, var_contact_friction_in, var_contact_dim_in, var_contact_efc_address_in, var_contact_adhesion_in, var_efc_force_in, var_njmax_in, var_nacon_in, var_9, var_2, var_to_world_frame);
        // out[tid] = contact_force_fn(                                                           <L 424>
        wp::array_store(var_out, var_0, var_11);
}



extern "C" __global__ void _apply_ft_9dc677d0_cuda_kernel_forward(
    wp::launch_bounds_t<2> dim,
    wp::array_t<wp::int32> var_body_rootid,
    wp::array_t<wp::int32> var_dof_bodyid,
    wp::array_t<wp::int32> var_body_subtreenum,
    wp::array_t<wp::vec_t<3, wp::float32>> var_xipos_in,
    wp::array_t<wp::vec_t<3, wp::float32>> var_subtree_com_in,
    wp::array_t<wp::vec_t<6, wp::float32>> var_cdof_in,
    wp::array_t<wp::vec_t<6, wp::float32>> var_ft_in,
    bool var_flg_add,
    wp::array_t<wp::float32> var_qfrc_out)
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
        wp::vec_t<6, wp::float32>* var_2;
        wp::vec_t<6, wp::float32> var_3;
        wp::vec_t<6, wp::float32> var_4;
        const wp::int32 var_5 = 0;
        wp::float32 var_6;
        const wp::int32 var_7 = 1;
        wp::float32 var_8;
        const wp::int32 var_9 = 2;
        wp::float32 var_10;
        wp::vec_t<3, wp::float32> var_11;
        const wp::int32 var_12 = 3;
        wp::float32 var_13;
        const wp::int32 var_14 = 4;
        wp::float32 var_15;
        const wp::int32 var_16 = 5;
        wp::float32 var_17;
        const wp::int32 var_18 = 0;
        wp::float32 var_19;
        const wp::int32 var_20 = 1;
        wp::float32 var_21;
        const wp::int32 var_22 = 2;
        wp::float32 var_23;
        wp::vec_t<6, wp::float32> var_24;
        wp::int32* var_25;
        wp::int32 var_26;
        wp::int32 var_27;
        wp::int32* var_28;
        wp::int32 var_29;
        wp::vec_t<3, wp::float32>* var_30;
        wp::vec_t<3, wp::float32> var_31;
        wp::vec_t<3, wp::float32> var_32;
        const wp::float32 var_33 = 0.0;
        wp::float32 var_34;
        wp::int32* var_35;
        wp::int32 var_36;
        wp::int32 var_37;
        wp::range_t var_38;
        wp::int32 var_39;
        wp::vec_t<6, wp::float32>* var_40;
        wp::vec_t<6, wp::float32> var_41;
        wp::vec_t<6, wp::float32> var_42;
        wp::vec_t<6, wp::float32> var_43;
        bool var_44;
        wp::vec_t<3, wp::float32>* var_45;
        wp::vec_t<3, wp::float32> var_46;
        wp::vec_t<3, wp::float32> var_47;
        wp::vec_t<3, wp::float32> var_48;
        wp::float32 var_49;
        wp::vec_t<3, wp::float32> var_50;
        wp::float32 var_51;
        wp::float32 var_52;
        wp::float32 var_53;
        wp::float32 var_54;
        //---------
        // forward
        // def _apply_ft(                                                                         <L 260>
        // worldid, dofid = wp.tid()                                                              <L 275>
        builtin_tid2d(var_0, var_1);
        // cdof = cdof_in[worldid, dofid]                                                         <L 276>
        var_2 = wp::address(var_cdof_in, var_0, var_1);
        var_4 = wp::load(var_2);
        var_3 = wp::copy(var_4);
        // rotational_cdof = wp.vec3(cdof[0], cdof[1], cdof[2])                                   <L 277>
        var_6 = wp::extract(var_3, var_5);
        var_8 = wp::extract(var_3, var_7);
        var_10 = wp::extract(var_3, var_9);
        var_11 = wp::vec_t<3, wp::float32>(var_6, var_8, var_10);
        // jac = wp.spatial_vector(cdof[3], cdof[4], cdof[5], cdof[0], cdof[1], cdof[2])          <L 278>
        var_13 = wp::extract(var_3, var_12);
        var_15 = wp::extract(var_3, var_14);
        var_17 = wp::extract(var_3, var_16);
        var_19 = wp::extract(var_3, var_18);
        var_21 = wp::extract(var_3, var_20);
        var_23 = wp::extract(var_3, var_22);
        var_24 = wp::vec_t<6, wp::float32>({var_13, var_15, var_17, var_19, var_21, var_23});
        // dofbodyid = dof_bodyid[dofid]                                                          <L 280>
        var_25 = wp::address(var_dof_bodyid, var_1);
        var_27 = wp::load(var_25);
        var_26 = wp::copy(var_27);
        // subtree_com = subtree_com_in[worldid, body_rootid[dofbodyid]]                          <L 281>
        var_28 = wp::address(var_body_rootid, var_26);
        var_29 = wp::load(var_28);
        var_30 = wp::address(var_subtree_com_in, var_0, var_29);
        var_32 = wp::load(var_30);
        var_31 = wp::copy(var_32);
        // accumul = float(0.0)                                                                   <L 282>
        var_34 = wp::float(var_33);
        // for bodyid in range(dofbodyid, dofbodyid + body_subtreenum[dofbodyid]):                <L 285>
        var_35 = wp::address(var_body_subtreenum, var_26);
        var_37 = wp::load(var_35);
        var_36 = wp::add(var_26, var_37);
        var_38 = wp::range(var_26, var_36);
        start_for_0:;
            if (iter_cmp(var_38) == 0) goto end_for_0;
            var_39 = wp::iter_next(var_38);
            // ft_body = ft_in[worldid, bodyid]                                                   <L 286>
            var_40 = wp::address(var_ft_in, var_0, var_39);
            var_42 = wp::load(var_40);
            var_41 = wp::copy(var_42);
            // if ft_body == wp.spatial_vector():                                                 <L 287>
            var_43 = wp::vec_t<6, wp::float32>();
            var_44 = (var_41 == var_43);
            if (var_44) {
                // continue                                                                       <L 288>
                goto start_for_0;
            }
            // offset = xipos_in[worldid, bodyid] - subtree_com                                   <L 289>
            var_45 = wp::address(var_xipos_in, var_0, var_39);
            var_47 = wp::load(var_45);
            var_46 = wp::sub(var_47, var_31);
            // cross_term = wp.cross(rotational_cdof, offset)                                     <L 290>
            var_48 = wp::cross(var_11, var_46);
            // accumul += wp.dot(jac, ft_body) + wp.dot(cross_term, wp.spatial_top(ft_body))       <L 291>
            var_49 = wp::dot(var_24, var_41);
            var_50 = wp::spatial_top(var_41);
            var_51 = wp::dot(var_48, var_50);
            var_52 = wp::add(var_49, var_51);
            var_53 = wp::add(var_34, var_52);
            wp::assign(var_34, var_53);
            goto start_for_0;
        end_for_0:;
        // if flg_add:                                                                            <L 293>
        if (var_flg_add) {
            // qfrc_out[worldid, dofid] += accumul                                                <L 294>
            var_54 = wp::atomic_add(var_qfrc_out, var_0, var_1, var_34);
        }
        if (!var_flg_add) {
            // qfrc_out[worldid, dofid] = accumul                                                 <L 296>
            wp::array_store(var_qfrc_out, var_0, var_1, var_34);
        }
}

