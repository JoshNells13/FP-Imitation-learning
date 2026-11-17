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


// C:\Users\ASUS\Documents\human-imitation-learning\.venv\Lib\site-packages\mujoco_warp\_src\math.py:115
static CUDA_CALLABLE wp::quat_t<wp::float32> quat_inv_0(
    wp::quat_t<wp::float32> var_quat)
{
    //---------
    // primal vars
    const wp::int32 var_0 = 0;
    wp::float32 var_1;
    const wp::int32 var_2 = 1;
    wp::float32 var_3;
    wp::float32 var_4;
    const wp::int32 var_5 = 2;
    wp::float32 var_6;
    wp::float32 var_7;
    const wp::int32 var_8 = 3;
    wp::float32 var_9;
    wp::float32 var_10;
    wp::quat_t<wp::float32> var_11;
    //---------
    // forward
    // def quat_inv(quat: wp.quat) -> wp.quat:                                                <L 116>
    // return wp.quat(quat[0], -quat[1], -quat[2], -quat[3])                                  <L 117>
    var_1 = wp::extract(var_quat, var_0);
    var_3 = wp::extract(var_quat, var_2);
    var_4 = wp::neg(var_3);
    var_6 = wp::extract(var_quat, var_5);
    var_7 = wp::neg(var_6);
    var_9 = wp::extract(var_quat, var_8);
    var_10 = wp::neg(var_9);
    var_11 = wp::quat_t<wp::float32>(var_1, var_4, var_7, var_10);
    return var_11;
}


// C:\Users\ASUS\Documents\human-imitation-learning\.venv\Lib\site-packages\mujoco_warp\_src\math.py:23
static CUDA_CALLABLE wp::quat_t<wp::float32> mul_quat_0(
    wp::quat_t<wp::float32> var_u,
    wp::quat_t<wp::float32> var_v)
{
    //---------
    // primal vars
    const wp::int32 var_0 = 0;
    wp::float32 var_1;
    const wp::int32 var_2 = 0;
    wp::float32 var_3;
    wp::float32 var_4;
    const wp::int32 var_5 = 1;
    wp::float32 var_6;
    const wp::int32 var_7 = 1;
    wp::float32 var_8;
    wp::float32 var_9;
    wp::float32 var_10;
    const wp::int32 var_11 = 2;
    wp::float32 var_12;
    const wp::int32 var_13 = 2;
    wp::float32 var_14;
    wp::float32 var_15;
    wp::float32 var_16;
    const wp::int32 var_17 = 3;
    wp::float32 var_18;
    const wp::int32 var_19 = 3;
    wp::float32 var_20;
    wp::float32 var_21;
    wp::float32 var_22;
    const wp::int32 var_23 = 0;
    wp::float32 var_24;
    const wp::int32 var_25 = 1;
    wp::float32 var_26;
    wp::float32 var_27;
    const wp::int32 var_28 = 1;
    wp::float32 var_29;
    const wp::int32 var_30 = 0;
    wp::float32 var_31;
    wp::float32 var_32;
    wp::float32 var_33;
    const wp::int32 var_34 = 2;
    wp::float32 var_35;
    const wp::int32 var_36 = 3;
    wp::float32 var_37;
    wp::float32 var_38;
    wp::float32 var_39;
    const wp::int32 var_40 = 3;
    wp::float32 var_41;
    const wp::int32 var_42 = 2;
    wp::float32 var_43;
    wp::float32 var_44;
    wp::float32 var_45;
    const wp::int32 var_46 = 0;
    wp::float32 var_47;
    const wp::int32 var_48 = 2;
    wp::float32 var_49;
    wp::float32 var_50;
    const wp::int32 var_51 = 1;
    wp::float32 var_52;
    const wp::int32 var_53 = 3;
    wp::float32 var_54;
    wp::float32 var_55;
    wp::float32 var_56;
    const wp::int32 var_57 = 2;
    wp::float32 var_58;
    const wp::int32 var_59 = 0;
    wp::float32 var_60;
    wp::float32 var_61;
    wp::float32 var_62;
    const wp::int32 var_63 = 3;
    wp::float32 var_64;
    const wp::int32 var_65 = 1;
    wp::float32 var_66;
    wp::float32 var_67;
    wp::float32 var_68;
    const wp::int32 var_69 = 0;
    wp::float32 var_70;
    const wp::int32 var_71 = 3;
    wp::float32 var_72;
    wp::float32 var_73;
    const wp::int32 var_74 = 1;
    wp::float32 var_75;
    const wp::int32 var_76 = 2;
    wp::float32 var_77;
    wp::float32 var_78;
    wp::float32 var_79;
    const wp::int32 var_80 = 2;
    wp::float32 var_81;
    const wp::int32 var_82 = 1;
    wp::float32 var_83;
    wp::float32 var_84;
    wp::float32 var_85;
    const wp::int32 var_86 = 3;
    wp::float32 var_87;
    const wp::int32 var_88 = 0;
    wp::float32 var_89;
    wp::float32 var_90;
    wp::float32 var_91;
    wp::quat_t<wp::float32> var_92;
    //---------
    // forward
    // def mul_quat(u: wp.quat, v: wp.quat) -> wp.quat:                                       <L 24>
    // return wp.quat(                                                                        <L 25>
    // u[0] * v[0] - u[1] * v[1] - u[2] * v[2] - u[3] * v[3],                                 <L 26>
    var_1 = wp::extract(var_u, var_0);
    var_3 = wp::extract(var_v, var_2);
    var_4 = wp::mul(var_1, var_3);
    var_6 = wp::extract(var_u, var_5);
    var_8 = wp::extract(var_v, var_7);
    var_9 = wp::mul(var_6, var_8);
    var_10 = wp::sub(var_4, var_9);
    var_12 = wp::extract(var_u, var_11);
    var_14 = wp::extract(var_v, var_13);
    var_15 = wp::mul(var_12, var_14);
    var_16 = wp::sub(var_10, var_15);
    var_18 = wp::extract(var_u, var_17);
    var_20 = wp::extract(var_v, var_19);
    var_21 = wp::mul(var_18, var_20);
    var_22 = wp::sub(var_16, var_21);
    // u[0] * v[1] + u[1] * v[0] + u[2] * v[3] - u[3] * v[2],                                 <L 27>
    var_24 = wp::extract(var_u, var_23);
    var_26 = wp::extract(var_v, var_25);
    var_27 = wp::mul(var_24, var_26);
    var_29 = wp::extract(var_u, var_28);
    var_31 = wp::extract(var_v, var_30);
    var_32 = wp::mul(var_29, var_31);
    var_33 = wp::add(var_27, var_32);
    var_35 = wp::extract(var_u, var_34);
    var_37 = wp::extract(var_v, var_36);
    var_38 = wp::mul(var_35, var_37);
    var_39 = wp::add(var_33, var_38);
    var_41 = wp::extract(var_u, var_40);
    var_43 = wp::extract(var_v, var_42);
    var_44 = wp::mul(var_41, var_43);
    var_45 = wp::sub(var_39, var_44);
    // u[0] * v[2] - u[1] * v[3] + u[2] * v[0] + u[3] * v[1],                                 <L 28>
    var_47 = wp::extract(var_u, var_46);
    var_49 = wp::extract(var_v, var_48);
    var_50 = wp::mul(var_47, var_49);
    var_52 = wp::extract(var_u, var_51);
    var_54 = wp::extract(var_v, var_53);
    var_55 = wp::mul(var_52, var_54);
    var_56 = wp::sub(var_50, var_55);
    var_58 = wp::extract(var_u, var_57);
    var_60 = wp::extract(var_v, var_59);
    var_61 = wp::mul(var_58, var_60);
    var_62 = wp::add(var_56, var_61);
    var_64 = wp::extract(var_u, var_63);
    var_66 = wp::extract(var_v, var_65);
    var_67 = wp::mul(var_64, var_66);
    var_68 = wp::add(var_62, var_67);
    // u[0] * v[3] + u[1] * v[2] - u[2] * v[1] + u[3] * v[0],                                 <L 29>
    var_70 = wp::extract(var_u, var_69);
    var_72 = wp::extract(var_v, var_71);
    var_73 = wp::mul(var_70, var_72);
    var_75 = wp::extract(var_u, var_74);
    var_77 = wp::extract(var_v, var_76);
    var_78 = wp::mul(var_75, var_77);
    var_79 = wp::add(var_73, var_78);
    var_81 = wp::extract(var_u, var_80);
    var_83 = wp::extract(var_v, var_82);
    var_84 = wp::mul(var_81, var_83);
    var_85 = wp::sub(var_79, var_84);
    var_87 = wp::extract(var_u, var_86);
    var_89 = wp::extract(var_v, var_88);
    var_90 = wp::mul(var_87, var_89);
    var_91 = wp::add(var_85, var_90);
    var_92 = wp::quat_t<wp::float32>(var_22, var_45, var_68, var_91);
    return var_92;
}


// C:\Users\ASUS\Documents\human-imitation-learning\.venv\Lib\site-packages\mujoco_warp\_src\math.py:115
static CUDA_CALLABLE void adj_quat_inv_0(
    wp::quat_t<wp::float32> var_quat,
    wp::quat_t<wp::float32> & adj_quat,
    wp::quat_t<wp::float32> & adj_ret)
{
	// reverse mode disabled (no backward-enabled kernel depends on this function)
}


// C:\Users\ASUS\Documents\human-imitation-learning\.venv\Lib\site-packages\mujoco_warp\_src\math.py:23
static CUDA_CALLABLE void adj_mul_quat_0(
    wp::quat_t<wp::float32> var_u,
    wp::quat_t<wp::float32> var_v,
    wp::quat_t<wp::float32> & adj_u,
    wp::quat_t<wp::float32> & adj_v,
    wp::quat_t<wp::float32> & adj_ret)
{
	// reverse mode disabled (no backward-enabled kernel depends on this function)
}



extern "C" __global__ void _resolve_dampratio_0420eac5_cuda_kernel_forward(
    wp::launch_bounds_t<1> dim,
    wp::array_t<wp::int32> var_actuator_biastype,
    wp::array_t<wp::vec_t<10, wp::float32>> var_actuator_gainprm,
    wp::int32 var_actid,
    wp::int32 var_nv,
    wp::array_t<wp::float32> var_act_moment_vec_in,
    wp::array_t<wp::float32> var_act_result_vec_in,
    wp::array_t<wp::vec_t<10, wp::float32>> var_actuator_biasprm_out)
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
        const wp::int32 var_4 = 1;
        bool var_5;
        wp::shape_t* var_6;
        const wp::int32 var_7 = 0;
        wp::int32 var_8;
        wp::shape_t var_9;
        wp::int32 var_10;
        wp::shape_t* var_11;
        const wp::int32 var_12 = 0;
        wp::int32 var_13;
        wp::shape_t var_14;
        wp::int32 var_15;
        wp::vec_t<10, wp::float32>* var_16;
        const wp::int32 var_17 = 0;
        wp::float32 var_18;
        wp::vec_t<10, wp::float32> var_19;
        wp::vec_t<10, wp::float32>* var_20;
        wp::vec_t<10, wp::float32> var_21;
        wp::vec_t<10, wp::float32> var_22;
        const wp::int32 var_23 = 1;
        wp::float32 var_24;
        wp::float32 var_25;
        wp::float32 var_26;
        const wp::float32 var_27 = 1e-15;
        bool var_28;
        const wp::int32 var_29 = 2;
        wp::float32 var_30;
        const wp::float32 var_31 = 0.0;
        bool var_32;
        const wp::int32 var_33 = 2;
        wp::float32 var_34;
        const wp::float32 var_35 = 0.0;
        wp::float32 var_36;
        wp::range_t var_37;
        wp::int32 var_38;
        wp::float32* var_39;
        wp::float32* var_40;
        wp::float32 var_41;
        wp::float32 var_42;
        wp::float32 var_43;
        wp::float32 var_44;
        const wp::float32 var_45 = 0.0;
        wp::float32 var_46;
        bool var_47;
        const wp::float32 var_48 = 1.0;
        wp::float32 var_49;
        wp::float32 var_50;
        const wp::float32 var_51 = 2.0;
        wp::float32 var_52;
        wp::float32 var_53;
        wp::float32 var_54;
        wp::float32 var_55;
        wp::float32 var_56;
        const wp::int32 var_57 = 2;
        //---------
        // forward
        // def _resolve_dampratio(                                                                <L 507>
        // worldid = wp.tid()                                                                     <L 519>
        var_0 = builtin_tid1d();
        // biastype = actuator_biastype[actid]                                                    <L 520>
        var_1 = wp::address(var_actuator_biastype, var_actid);
        var_3 = wp::load(var_1);
        var_2 = wp::copy(var_3);
        // if biastype != BiasType.AFFINE:                                                        <L 523>
        var_5 = (var_2 != var_4);
        if (var_5) {
            // return                                                                             <L 524>
            return;
        }
        // gainprm_id = worldid % actuator_gainprm.shape[0]                                       <L 526>
        var_6 = &(var_actuator_gainprm.shape);
        var_9 = wp::load(var_6);
        var_8 = wp::extract(var_9, var_7);
        var_10 = wp::mod(var_0, var_8);
        // biasprm_id = worldid % actuator_biasprm_out.shape[0]                                   <L 527>
        var_11 = &(var_actuator_biasprm_out.shape);
        var_14 = wp::load(var_11);
        var_13 = wp::extract(var_14, var_12);
        var_15 = wp::mod(var_0, var_13);
        // kp = actuator_gainprm[gainprm_id, actid][0]                                            <L 528>
        var_16 = wp::address(var_actuator_gainprm, var_10, var_actid);
        var_19 = wp::load(var_16);
        var_18 = wp::extract(var_19, var_17);
        // biasprm = actuator_biasprm_out[biasprm_id, actid]                                      <L 530>
        var_20 = wp::address(var_actuator_biasprm_out, var_15, var_actid);
        var_22 = wp::load(var_20);
        var_21 = wp::copy(var_22);
        // if wp.abs(kp + biasprm[1]) > MJ_MINVAL:                                                <L 532>
        var_24 = wp::extract(var_21, var_23);
        var_25 = wp::add(var_18, var_24);
        var_26 = wp::abs(var_25);
        var_28 = (var_26 > var_27);
        if (var_28) {
            // return                                                                             <L 533>
            return;
        }
        // if biasprm[2] <= 0.0:                                                                  <L 534>
        var_30 = wp::extract(var_21, var_29);
        var_32 = (var_30 <= var_31);
        if (var_32) {
            // return                                                                             <L 535>
            return;
        }
        // dampratio = biasprm[2]                                                                 <L 537>
        var_34 = wp::extract(var_21, var_33);
        // invweight = float(0.0)                                                                 <L 540>
        var_36 = wp::float(var_35);
        // for i in range(nv):                                                                    <L 541>
        var_37 = wp::range(var_nv);
        start_for_3:;
            if (iter_cmp(var_37) == 0) goto end_for_3;
            var_38 = wp::iter_next(var_37);
            // invweight += act_moment_vec_in[worldid, i] * act_result_vec_in[worldid, i]         <L 542>
            var_39 = wp::address(var_act_moment_vec_in, var_0, var_38);
            var_40 = wp::address(var_act_result_vec_in, var_0, var_38);
            var_42 = wp::load(var_39);
            var_43 = wp::load(var_40);
            var_41 = wp::mul(var_42, var_43);
            var_44 = wp::add(var_36, var_41);
            wp::assign(var_36, var_44);
            goto start_for_3;
        end_for_3:;
        // mass = float(0.0)                                                                      <L 544>
        var_46 = wp::float(var_45);
        // if invweight > MJ_MINVAL:                                                              <L 545>
        var_47 = (var_36 > var_27);
        if (var_47) {
            // mass = 1.0 / invweight                                                             <L 546>
            var_49 = wp::div(var_48, var_36);
        }
        var_50 = wp::where(var_47, var_49, var_46);
        // damping = dampratio * 2.0 * wp.sqrt(kp * mass)                                         <L 548>
        var_52 = wp::mul(var_34, var_51);
        var_53 = wp::mul(var_18, var_50);
        var_54 = wp::sqrt(var_53);
        var_55 = wp::mul(var_52, var_54);
        // biasprm[2] = -damping                                                                  <L 551>
        var_56 = wp::neg(var_55);
        wp::assign_inplace(var_21, var_57, var_56);
        // actuator_biasprm_out[biasprm_id, actid] = biasprm                                      <L 552>
        wp::array_store(var_actuator_biasprm_out, var_15, var_actid, var_21);
}



extern "C" __global__ void _copy_qpos0_to_qpos_32739b15_cuda_kernel_forward(
    wp::launch_bounds_t<2> dim,
    wp::array_t<wp::float32> var_qpos0,
    wp::array_t<wp::float32> var_qpos_out)
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
        wp::shape_t* var_2;
        const wp::int32 var_3 = 0;
        wp::int32 var_4;
        wp::shape_t var_5;
        wp::int32 var_6;
        wp::float32* var_7;
        wp::float32 var_8;
        wp::float32 var_9;
        //---------
        // forward
        // def _copy_qpos0_to_qpos(                                                               <L 59>
        // worldid, i = wp.tid()                                                                  <L 63>
        builtin_tid2d(var_0, var_1);
        // qpos0_id = worldid % qpos0.shape[0]                                                    <L 64>
        var_2 = &(var_qpos0.shape);
        var_5 = wp::load(var_2);
        var_4 = wp::extract(var_5, var_3);
        var_6 = wp::mod(var_0, var_4);
        // qpos_out[worldid, i] = qpos0[qpos0_id, i]                                              <L 65>
        var_7 = wp::address(var_qpos0, var_6, var_1);
        var_9 = wp::load(var_7);
        var_8 = wp::copy(var_9);
        wp::array_store(var_qpos_out, var_0, var_1, var_8);
}



extern "C" __global__ void _copy_actuator_moment_4af7079d_cuda_kernel_forward(
    wp::launch_bounds_t<1> dim,
    wp::int32 var_actid_target,
    wp::array_t<wp::int32> var_moment_rownnz_in,
    wp::array_t<wp::int32> var_moment_rowadr_in,
    wp::array_t<wp::int32> var_moment_colind_in,
    wp::array_t<wp::float32> var_actuator_moment_in,
    wp::array_t<wp::float32> var_act_moment_vec_out)
{
    wp::tile_shared_storage_t tile_mem;

    const size_t _idx = static_cast<size_t>(blockIdx.z * gridDim.y + blockIdx.y) * static_cast<size_t>(gridDim.x * blockDim.x) + static_cast<size_t>(blockIdx.x * blockDim.x + threadIdx.x);
    if (_idx >= dim.size) return;
            // reset shared memory allocator
    wp::tile_shared_storage_t::init();

        //---------
        // primal vars
        wp::int32 var_0;
        wp::shape_t* var_1;
        const wp::int32 var_2 = 1;
        wp::int32 var_3;
        wp::shape_t var_4;
        wp::range_t var_5;
        wp::int32 var_6;
        const wp::float32 var_7 = 0.0;
        wp::int32* var_8;
        wp::int32 var_9;
        wp::int32 var_10;
        wp::int32* var_11;
        wp::int32 var_12;
        wp::int32 var_13;
        wp::range_t var_14;
        wp::int32 var_15;
        wp::int32 var_16;
        wp::int32* var_17;
        wp::int32 var_18;
        wp::int32 var_19;
        wp::float32* var_20;
        wp::float32 var_21;
        wp::float32 var_22;
        //---------
        // forward
        // def _copy_actuator_moment(                                                             <L 472>
        // worldid = wp.tid()                                                                     <L 480>
        var_0 = builtin_tid1d();
        // nv = act_moment_vec_out.shape[1]                                                       <L 481>
        var_1 = &(var_act_moment_vec_out.shape);
        var_4 = wp::load(var_1);
        var_3 = wp::extract(var_4, var_2);
        // for i in range(nv):                                                                    <L 482>
        var_5 = wp::range(var_3);
        start_for_0:;
            if (iter_cmp(var_5) == 0) goto end_for_0;
            var_6 = wp::iter_next(var_5);
            // act_moment_vec_out[worldid, i] = 0.0                                               <L 483>
            wp::array_store(var_act_moment_vec_out, var_0, var_6, var_7);
            goto start_for_0;
        end_for_0:;
        // rownnz = moment_rownnz_in[worldid, actid_target]                                       <L 484>
        var_8 = wp::address(var_moment_rownnz_in, var_0, var_actid_target);
        var_10 = wp::load(var_8);
        var_9 = wp::copy(var_10);
        // rowadr = moment_rowadr_in[worldid, actid_target]                                       <L 485>
        var_11 = wp::address(var_moment_rowadr_in, var_0, var_actid_target);
        var_13 = wp::load(var_11);
        var_12 = wp::copy(var_13);
        // for i in range(rownnz):                                                                <L 486>
        var_14 = wp::range(var_9);
        start_for_2:;
            if (iter_cmp(var_14) == 0) goto end_for_2;
            var_15 = wp::iter_next(var_14);
            // sparseid = rowadr + i                                                              <L 487>
            var_16 = wp::add(var_12, var_15);
            // col = moment_colind_in[worldid, sparseid]                                          <L 488>
            var_17 = wp::address(var_moment_colind_in, var_0, var_16);
            var_19 = wp::load(var_17);
            var_18 = wp::copy(var_19);
            // act_moment_vec_out[worldid, col] = actuator_moment_in[worldid, sparseid]           <L 489>
            var_20 = wp::address(var_actuator_moment_in, var_0, var_16);
            var_22 = wp::load(var_20);
            var_21 = wp::copy(var_22);
            wp::array_store(var_act_moment_vec_out, var_0, var_18, var_21);
            goto start_for_2;
        end_for_2:;
}



extern "C" __global__ void _copy_tendon_length0_5230a976_cuda_kernel_forward(
    wp::launch_bounds_t<2> dim,
    wp::array_t<wp::float32> var_ten_length_in,
    wp::array_t<wp::float32> var_tendon_length0_out)
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
        wp::shape_t* var_2;
        const wp::int32 var_3 = 0;
        wp::int32 var_4;
        wp::shape_t var_5;
        wp::int32 var_6;
        wp::float32* var_7;
        wp::float32 var_8;
        wp::float32 var_9;
        //---------
        // forward
        // def _copy_tendon_length0(                                                              <L 69>
        // worldid, tenid = wp.tid()                                                              <L 73>
        builtin_tid2d(var_0, var_1);
        // tendon_length0_id = worldid % tendon_length0_out.shape[0]                              <L 74>
        var_2 = &(var_tendon_length0_out.shape);
        var_5 = wp::load(var_2);
        var_4 = wp::extract(var_5, var_3);
        var_6 = wp::mod(var_0, var_4);
        // tendon_length0_out[tendon_length0_id, tenid] = ten_length_in[worldid, tenid]           <L 75>
        var_7 = wp::address(var_ten_length_in, var_0, var_1);
        var_9 = wp::load(var_7);
        var_8 = wp::copy(var_9);
        wp::array_store(var_tendon_length0_out, var_6, var_1, var_8);
}



extern "C" __global__ void _compute_cam_pos0_5993818e_cuda_kernel_forward(
    wp::launch_bounds_t<2> dim,
    wp::array_t<wp::int32> var_cam_bodyid,
    wp::array_t<wp::int32> var_cam_targetbodyid,
    wp::array_t<wp::vec_t<3, wp::float32>> var_cam_xpos_in,
    wp::array_t<wp::mat_t<3, 3, wp::float32>> var_cam_xmat_in,
    wp::array_t<wp::vec_t<3, wp::float32>> var_xpos_in,
    wp::array_t<wp::vec_t<3, wp::float32>> var_subtree_com_in,
    wp::array_t<wp::vec_t<3, wp::float32>> var_cam_pos0_out,
    wp::array_t<wp::vec_t<3, wp::float32>> var_cam_poscom0_out,
    wp::array_t<wp::mat_t<3, 3, wp::float32>> var_cam_mat0_out)
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
        wp::shape_t* var_2;
        const wp::int32 var_3 = 0;
        wp::int32 var_4;
        wp::shape_t var_5;
        wp::int32 var_6;
        wp::int32* var_7;
        wp::int32 var_8;
        wp::int32 var_9;
        wp::int32* var_10;
        wp::int32 var_11;
        wp::int32 var_12;
        wp::vec_t<3, wp::float32>* var_13;
        wp::vec_t<3, wp::float32> var_14;
        wp::vec_t<3, wp::float32> var_15;
        wp::vec_t<3, wp::float32>* var_16;
        wp::vec_t<3, wp::float32> var_17;
        wp::vec_t<3, wp::float32> var_18;
        const wp::int32 var_19 = 0;
        bool var_20;
        wp::vec_t<3, wp::float32>* var_21;
        wp::vec_t<3, wp::float32> var_22;
        wp::vec_t<3, wp::float32> var_23;
        wp::vec_t<3, wp::float32>* var_24;
        wp::vec_t<3, wp::float32> var_25;
        wp::vec_t<3, wp::float32> var_26;
        wp::mat_t<3, 3, wp::float32>* var_27;
        wp::mat_t<3, 3, wp::float32> var_28;
        wp::mat_t<3, 3, wp::float32> var_29;
        //---------
        // forward
        // def _compute_cam_pos0(                                                                 <L 420>
        // worldid, camid = wp.tid()                                                              <L 431>
        builtin_tid2d(var_0, var_1);
        // cam_pos0_id = worldid % cam_pos0_out.shape[0]                                          <L 432>
        var_2 = &(var_cam_pos0_out.shape);
        var_5 = wp::load(var_2);
        var_4 = wp::extract(var_5, var_3);
        var_6 = wp::mod(var_0, var_4);
        // bodyid = cam_bodyid[camid]                                                             <L 433>
        var_7 = wp::address(var_cam_bodyid, var_1);
        var_9 = wp::load(var_7);
        var_8 = wp::copy(var_9);
        // targetid = cam_targetbodyid[camid]                                                     <L 434>
        var_10 = wp::address(var_cam_targetbodyid, var_1);
        var_12 = wp::load(var_10);
        var_11 = wp::copy(var_12);
        // cam_xpos = cam_xpos_in[worldid, camid]                                                 <L 435>
        var_13 = wp::address(var_cam_xpos_in, var_0, var_1);
        var_15 = wp::load(var_13);
        var_14 = wp::copy(var_15);
        // cam_pos0_out[cam_pos0_id, camid] = cam_xpos - xpos_in[worldid, bodyid]                 <L 437>
        var_16 = wp::address(var_xpos_in, var_0, var_8);
        var_18 = wp::load(var_16);
        var_17 = wp::sub(var_14, var_18);
        wp::array_store(var_cam_pos0_out, var_6, var_1, var_17);
        // if targetid >= 0:                                                                      <L 438>
        var_20 = (var_11 >= var_19);
        if (var_20) {
            // cam_poscom0_out[cam_pos0_id, camid] = cam_xpos - subtree_com_in[worldid, targetid]       <L 439>
            var_21 = wp::address(var_subtree_com_in, var_0, var_11);
            var_23 = wp::load(var_21);
            var_22 = wp::sub(var_14, var_23);
            wp::array_store(var_cam_poscom0_out, var_6, var_1, var_22);
        }
        if (!var_20) {
            // cam_poscom0_out[cam_pos0_id, camid] = cam_xpos - subtree_com_in[worldid, bodyid]       <L 441>
            var_24 = wp::address(var_subtree_com_in, var_0, var_8);
            var_26 = wp::load(var_24);
            var_25 = wp::sub(var_14, var_26);
            wp::array_store(var_cam_poscom0_out, var_6, var_1, var_25);
        }
        // cam_mat0_out[cam_pos0_id, camid] = cam_xmat_in[worldid, camid]                         <L 442>
        var_27 = wp::address(var_cam_xmat_in, var_0, var_1);
        var_29 = wp::load(var_27);
        var_28 = wp::copy(var_29);
        wp::array_store(var_cam_mat0_out, var_6, var_1, var_28);
}



extern "C" __global__ void _finalize_body_invweight0_64175ec4_cuda_kernel_forward(
    wp::launch_bounds_t<2> dim,
    wp::array_t<wp::int32> var_body_weldid,
    wp::array_t<wp::float32> var_body_A_diag_in,
    wp::array_t<wp::vec_t<2, wp::float32>> var_body_invweight0_out)
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
        wp::shape_t* var_2;
        const wp::int32 var_3 = 0;
        wp::int32 var_4;
        wp::shape_t var_5;
        wp::int32 var_6;
        wp::shape_t* var_7;
        const wp::int32 var_8 = 0;
        wp::int32 var_9;
        wp::shape_t var_10;
        wp::int32 var_11;
        bool var_12;
        const wp::int32 var_13 = 0;
        bool var_14;
        wp::int32* var_15;
        const wp::int32 var_16 = 0;
        bool var_17;
        wp::int32 var_18;
        const wp::float32 var_19 = 0.0;
        const wp::float32 var_20 = 0.0;
        wp::vec_t<2, wp::float32> var_21;
        const wp::float32 var_22 = 0.3333333333333333;
        const wp::int32 var_23 = 0;
        wp::float32* var_24;
        const wp::int32 var_25 = 1;
        wp::float32* var_26;
        wp::float32 var_27;
        wp::float32 var_28;
        wp::float32 var_29;
        const wp::int32 var_30 = 2;
        wp::float32* var_31;
        wp::float32 var_32;
        wp::float32 var_33;
        wp::float32 var_34;
        const wp::float32 var_35 = 0.3333333333333333;
        const wp::int32 var_36 = 3;
        wp::float32* var_37;
        const wp::int32 var_38 = 4;
        wp::float32* var_39;
        wp::float32 var_40;
        wp::float32 var_41;
        wp::float32 var_42;
        const wp::int32 var_43 = 5;
        wp::float32* var_44;
        wp::float32 var_45;
        wp::float32 var_46;
        wp::float32 var_47;
        bool var_48;
        const wp::float32 var_49 = 1e-15;
        bool var_50;
        const wp::float32 var_51 = 1e-15;
        bool var_52;
        wp::float32 var_53;
        bool var_54;
        const wp::float32 var_55 = 1e-15;
        bool var_56;
        const wp::float32 var_57 = 1e-15;
        bool var_58;
        wp::float32 var_59;
        wp::float32 var_60;
        wp::float32 var_61;
        wp::float32 var_62;
        wp::vec_t<2, wp::float32> var_63;
        //---------
        // forward
        // def _finalize_body_invweight0(                                                         <L 339>
        // worldid, bodyid = wp.tid()                                                             <L 344>
        builtin_tid2d(var_0, var_1);
        // body_invweight0_id = worldid % body_invweight0_out.shape[0]                            <L 345>
        var_2 = &(var_body_invweight0_out.shape);
        var_5 = wp::load(var_2);
        var_4 = wp::extract(var_5, var_3);
        var_6 = wp::mod(var_0, var_4);
        // body_A_diag_id = worldid % body_A_diag_in.shape[0]                                     <L 346>
        var_7 = &(var_body_A_diag_in.shape);
        var_10 = wp::load(var_7);
        var_9 = wp::extract(var_10, var_8);
        var_11 = wp::mod(var_0, var_9);
        // if bodyid == 0 or body_weldid[bodyid] == 0:                                            <L 349>
        var_14 = (var_1 == var_13);
        var_12 = var_14;
        if (!var_12) {
            var_15 = wp::address(var_body_weldid, var_1);
            var_18 = wp::load(var_15);
            var_17 = (var_18 == var_16);
            var_12 = var_12 || var_17;
        }
        if (var_12) {
            // body_invweight0_out[body_invweight0_id, bodyid] = wp.vec2(0.0, 0.0)                <L 350>
            var_21 = wp::vec_t<2, wp::float32>(var_19, var_20);
            wp::array_store(var_body_invweight0_out, var_6, var_1, var_21);
            // return                                                                             <L 351>
            return;
        }
        // inv_trans = wp.static(1.0 / 3.0) * (                                                   <L 354>
        // body_A_diag_in[body_A_diag_id, bodyid, 0]                                              <L 355>
        var_24 = wp::address(var_body_A_diag_in, var_11, var_1, var_23);
        // + body_A_diag_in[body_A_diag_id, bodyid, 1]                                            <L 356>
        var_26 = wp::address(var_body_A_diag_in, var_11, var_1, var_25);
        var_28 = wp::load(var_24);
        var_29 = wp::load(var_26);
        var_27 = wp::add(var_28, var_29);
        // + body_A_diag_in[body_A_diag_id, bodyid, 2]                                            <L 357>
        var_31 = wp::address(var_body_A_diag_in, var_11, var_1, var_30);
        var_33 = wp::load(var_31);
        var_32 = wp::add(var_27, var_33);
        var_34 = wp::mul(var_22, var_32);
        // inv_rot = wp.static(1.0 / 3.0) * (                                                     <L 359>
        // body_A_diag_in[body_A_diag_id, bodyid, 3]                                              <L 360>
        var_37 = wp::address(var_body_A_diag_in, var_11, var_1, var_36);
        // + body_A_diag_in[body_A_diag_id, bodyid, 4]                                            <L 361>
        var_39 = wp::address(var_body_A_diag_in, var_11, var_1, var_38);
        var_41 = wp::load(var_37);
        var_42 = wp::load(var_39);
        var_40 = wp::add(var_41, var_42);
        // + body_A_diag_in[body_A_diag_id, bodyid, 5]                                            <L 362>
        var_44 = wp::address(var_body_A_diag_in, var_11, var_1, var_43);
        var_46 = wp::load(var_44);
        var_45 = wp::add(var_40, var_46);
        var_47 = wp::mul(var_35, var_45);
        // if inv_trans < mujoco.mjMINVAL and inv_rot > mujoco.mjMINVAL:                          <L 366>
        var_50 = (var_34 < var_49);
        var_48 = var_50;
        if (var_48) {
            var_52 = (var_47 > var_51);
            var_48 = var_48 && var_52;
        }
        if (var_48) {
            // inv_trans = inv_rot  # use rotation as fallback for translation                    <L 367>
            var_53 = wp::copy(var_47);
        }
        if (!var_48) {
            // elif inv_rot < mujoco.mjMINVAL and inv_trans > mujoco.mjMINVAL:                    <L 368>
            var_56 = (var_47 < var_55);
            var_54 = var_56;
            if (var_54) {
                var_58 = (var_34 > var_57);
                var_54 = var_54 && var_58;
            }
            if (var_54) {
                // inv_rot = inv_trans  # use translation as fallback for rotation                <L 369>
                var_59 = wp::copy(var_34);
            }
            var_60 = wp::where(var_54, var_59, var_47);
        }
        var_61 = wp::where(var_48, var_53, var_34);
        var_62 = wp::where(var_48, var_47, var_60);
        // body_invweight0_out[body_invweight0_id, bodyid] = wp.vec2(inv_trans, inv_rot)          <L 371>
        var_63 = wp::vec_t<2, wp::float32>(var_61, var_62);
        wp::array_store(var_body_invweight0_out, var_6, var_1, var_63);
}



extern "C" __global__ void _compute_body_jac_row_6eb5882a_cuda_kernel_forward(
    wp::launch_bounds_t<1> dim,
    wp::int32 var_nv,
    wp::int32 var_bodyid_target,
    wp::int32 var_row_idx,
    wp::array_t<wp::int32> var_body_parentid,
    wp::array_t<wp::int32> var_body_rootid,
    wp::array_t<wp::int32> var_body_dofadr,
    wp::array_t<wp::int32> var_body_dofnum,
    wp::array_t<wp::int32> var_dof_parentid,
    wp::array_t<wp::vec_t<3, wp::float32>> var_subtree_com_in,
    wp::array_t<wp::vec_t<3, wp::float32>> var_xipos_in,
    wp::array_t<wp::vec_t<6, wp::float32>> var_cdof_in,
    wp::array_t<wp::float32> var_body_jac_row_out)
{
    wp::tile_shared_storage_t tile_mem;

    const size_t _idx = static_cast<size_t>(blockIdx.z * gridDim.y + blockIdx.y) * static_cast<size_t>(gridDim.x * blockDim.x) + static_cast<size_t>(blockIdx.x * blockDim.x + threadIdx.x);
    if (_idx >= dim.size) return;
            // reset shared memory allocator
    wp::tile_shared_storage_t::init();

        //---------
        // primal vars
        wp::int32 var_0;
        wp::range_t var_1;
        wp::int32 var_2;
        const wp::float32 var_3 = 0.0;
        wp::int32 var_4;
        bool var_5;
        const wp::int32 var_6 = 0;
        bool var_7;
        wp::int32* var_8;
        const wp::int32 var_9 = 0;
        bool var_10;
        wp::int32 var_11;
        wp::int32* var_12;
        wp::int32 var_13;
        wp::int32 var_14;
        const wp::int32 var_15 = 0;
        bool var_16;
        wp::vec_t<3, wp::float32>* var_17;
        wp::vec_t<3, wp::float32> var_18;
        wp::vec_t<3, wp::float32> var_19;
        wp::int32* var_20;
        wp::int32 var_21;
        wp::vec_t<3, wp::float32>* var_22;
        wp::vec_t<3, wp::float32> var_23;
        wp::vec_t<3, wp::float32> var_24;
        wp::int32* var_25;
        wp::int32* var_26;
        wp::int32 var_27;
        wp::int32 var_28;
        wp::int32 var_29;
        const wp::int32 var_30 = 1;
        wp::int32 var_31;
        const wp::int32 var_32 = 0;
        bool var_33;
        wp::vec_t<6, wp::float32>* var_34;
        wp::vec_t<6, wp::float32> var_35;
        wp::vec_t<6, wp::float32> var_36;
        wp::vec_t<3, wp::float32> var_37;
        wp::vec_t<3, wp::float32> var_38;
        const wp::int32 var_39 = 3;
        bool var_40;
        wp::vec_t<3, wp::float32> var_41;
        const wp::int32 var_42 = 0;
        bool var_43;
        const wp::int32 var_44 = 0;
        wp::float32 var_45;
        const wp::int32 var_46 = 0;
        wp::float32 var_47;
        wp::float32 var_48;
        const wp::int32 var_49 = 1;
        bool var_50;
        const wp::int32 var_51 = 1;
        wp::float32 var_52;
        const wp::int32 var_53 = 1;
        wp::float32 var_54;
        wp::float32 var_55;
        const wp::int32 var_56 = 2;
        wp::float32 var_57;
        const wp::int32 var_58 = 2;
        wp::float32 var_59;
        wp::float32 var_60;
        const wp::int32 var_61 = 3;
        bool var_62;
        const wp::int32 var_63 = 0;
        wp::float32 var_64;
        const wp::int32 var_65 = 4;
        bool var_66;
        const wp::int32 var_67 = 1;
        wp::float32 var_68;
        const wp::int32 var_69 = 2;
        wp::float32 var_70;
        wp::int32* var_71;
        wp::int32 var_72;
        wp::int32 var_73;
        //---------
        // forward
        // def _compute_body_jac_row(                                                             <L 262>
        // worldid = wp.tid()                                                                     <L 276>
        var_0 = builtin_tid1d();
        // for i in range(nv):                                                                    <L 278>
        var_1 = wp::range(var_nv);
        start_for_0:;
            if (iter_cmp(var_1) == 0) goto end_for_0;
            var_2 = wp::iter_next(var_1);
            // body_jac_row_out[worldid, i] = 0.0                                                 <L 279>
            wp::array_store(var_body_jac_row_out, var_0, var_2, var_3);
            goto start_for_0;
        end_for_0:;
        // bodyid = bodyid_target                                                                 <L 281>
        var_4 = wp::copy(var_bodyid_target);
        // while bodyid > 0 and body_dofnum[bodyid] == 0:                                         <L 282>
        start_while_2:;
        var_7 = (var_4 > var_6);
        var_5 = var_7;
        if (var_5) {
            var_8 = wp::address(var_body_dofnum, var_4);
            var_11 = wp::load(var_8);
            var_10 = (var_11 == var_9);
            var_5 = var_5 && var_10;
        }
        if ((var_5) == false) goto end_while_2;
            // bodyid = body_parentid[bodyid]                                                     <L 283>
            var_12 = wp::address(var_body_parentid, var_4);
            var_14 = wp::load(var_12);
            var_13 = wp::copy(var_14);
            wp::assign(var_4, var_13);
        goto start_while_2;
        end_while_2:;
        // if bodyid == 0:                                                                        <L 285>
        var_16 = (var_4 == var_15);
        if (var_16) {
            // return                                                                             <L 286>
            return;
        }
        // point = xipos_in[worldid, bodyid_target]                                               <L 289>
        var_17 = wp::address(var_xipos_in, var_0, var_bodyid_target);
        var_19 = wp::load(var_17);
        var_18 = wp::copy(var_19);
        // offset = point - subtree_com_in[worldid, body_rootid[bodyid_target]]                   <L 290>
        var_20 = wp::address(var_body_rootid, var_bodyid_target);
        var_21 = wp::load(var_20);
        var_22 = wp::address(var_subtree_com_in, var_0, var_21);
        var_24 = wp::load(var_22);
        var_23 = wp::sub(var_18, var_24);
        // dofid = body_dofadr[bodyid] + body_dofnum[bodyid] - 1                                  <L 293>
        var_25 = wp::address(var_body_dofadr, var_4);
        var_26 = wp::address(var_body_dofnum, var_4);
        var_28 = wp::load(var_25);
        var_29 = wp::load(var_26);
        var_27 = wp::add(var_28, var_29);
        var_31 = wp::sub(var_27, var_30);
        // while dofid >= 0:                                                                      <L 296>
        start_while_5:;
        var_33 = (var_31 >= var_32);
        if ((var_33) == false) goto end_while_5;
            // cdof = cdof_in[worldid, dofid]                                                     <L 297>
            var_34 = wp::address(var_cdof_in, var_0, var_31);
            var_36 = wp::load(var_34);
            var_35 = wp::copy(var_36);
            // cdof_ang = wp.spatial_top(cdof)                                                    <L 298>
            var_37 = wp::spatial_top(var_35);
            // cdof_lin = wp.spatial_bottom(cdof)                                                 <L 299>
            var_38 = wp::spatial_bottom(var_35);
            // if row_idx < 3:                                                                    <L 301>
            var_40 = (var_row_idx < var_39);
            if (var_40) {
                // tmp = wp.cross(cdof_ang, offset)                                               <L 302>
                var_41 = wp::cross(var_37, var_23);
                // if row_idx == 0:                                                               <L 303>
                var_43 = (var_row_idx == var_42);
                if (var_43) {
                    // body_jac_row_out[worldid, dofid] = cdof_lin[0] + tmp[0]                    <L 304>
                    var_45 = wp::extract(var_38, var_44);
                    var_47 = wp::extract(var_41, var_46);
                    var_48 = wp::add(var_45, var_47);
                    wp::array_store(var_body_jac_row_out, var_0, var_31, var_48);
                }
                if (!var_43) {
                    // elif row_idx == 1:                                                         <L 305>
                    var_50 = (var_row_idx == var_49);
                    if (var_50) {
                        // body_jac_row_out[worldid, dofid] = cdof_lin[1] + tmp[1]                <L 306>
                        var_52 = wp::extract(var_38, var_51);
                        var_54 = wp::extract(var_41, var_53);
                        var_55 = wp::add(var_52, var_54);
                        wp::array_store(var_body_jac_row_out, var_0, var_31, var_55);
                    }
                    if (!var_50) {
                        // body_jac_row_out[worldid, dofid] = cdof_lin[2] + tmp[2]                <L 308>
                        var_57 = wp::extract(var_38, var_56);
                        var_59 = wp::extract(var_41, var_58);
                        var_60 = wp::add(var_57, var_59);
                        wp::array_store(var_body_jac_row_out, var_0, var_31, var_60);
                    }
                }
            }
            if (!var_40) {
                // if row_idx == 3:                                                               <L 310>
                var_62 = (var_row_idx == var_61);
                if (var_62) {
                    // body_jac_row_out[worldid, dofid] = cdof_ang[0]                             <L 311>
                    var_64 = wp::extract(var_37, var_63);
                    wp::array_store(var_body_jac_row_out, var_0, var_31, var_64);
                }
                if (!var_62) {
                    // elif row_idx == 4:                                                         <L 312>
                    var_66 = (var_row_idx == var_65);
                    if (var_66) {
                        // body_jac_row_out[worldid, dofid] = cdof_ang[1]                         <L 313>
                        var_68 = wp::extract(var_37, var_67);
                        wp::array_store(var_body_jac_row_out, var_0, var_31, var_68);
                    }
                    if (!var_66) {
                        // body_jac_row_out[worldid, dofid] = cdof_ang[2]                         <L 315>
                        var_70 = wp::extract(var_37, var_69);
                        wp::array_store(var_body_jac_row_out, var_0, var_31, var_70);
                    }
                }
            }
            // dofid = dof_parentid[dofid]                                                        <L 317>
            var_71 = wp::address(var_dof_parentid, var_31);
            var_73 = wp::load(var_71);
            var_72 = wp::copy(var_73);
            wp::assign(var_31, var_72);
        goto start_while_5;
        end_while_5:;
}



extern "C" __global__ void _extract_dof_A_diag_917a07a1_cuda_kernel_forward(
    wp::launch_bounds_t<1> dim,
    wp::int32 var_dofid,
    wp::array_t<wp::float32> var_result_vec_in,
    wp::array_t<wp::float32> var_dof_A_diag_out)
{
    wp::tile_shared_storage_t tile_mem;

    const size_t _idx = static_cast<size_t>(blockIdx.z * gridDim.y + blockIdx.y) * static_cast<size_t>(gridDim.x * blockDim.x) + static_cast<size_t>(blockIdx.x * blockDim.x + threadIdx.x);
    if (_idx >= dim.size) return;
            // reset shared memory allocator
    wp::tile_shared_storage_t::init();

        //---------
        // primal vars
        wp::int32 var_0;
        wp::shape_t* var_1;
        const wp::int32 var_2 = 0;
        wp::int32 var_3;
        wp::shape_t var_4;
        wp::int32 var_5;
        wp::float32* var_6;
        wp::float32 var_7;
        wp::float32 var_8;
        //---------
        // forward
        // def _extract_dof_A_diag(                                                               <L 207>
        // worldid = wp.tid()                                                                     <L 212>
        var_0 = builtin_tid1d();
        // dof_A_diag_id = worldid % dof_A_diag_out.shape[0]                                      <L 213>
        var_1 = &(var_dof_A_diag_out.shape);
        var_4 = wp::load(var_1);
        var_3 = wp::extract(var_4, var_2);
        var_5 = wp::mod(var_0, var_3);
        // dof_A_diag_out[dof_A_diag_id, dofid] = result_vec_in[worldid, dofid]                   <L 214>
        var_6 = wp::address(var_result_vec_in, var_0, var_dofid);
        var_8 = wp::load(var_6);
        var_7 = wp::copy(var_8);
        wp::array_store(var_dof_A_diag_out, var_5, var_dofid, var_7);
}



extern "C" __global__ void _finalize_dof_invweight0_966ce0d5_cuda_kernel_forward(
    wp::launch_bounds_t<2> dim,
    wp::array_t<wp::int32> var_dof_jntid,
    wp::array_t<wp::int32> var_jnt_type,
    wp::array_t<wp::int32> var_jnt_dofadr,
    wp::array_t<wp::float32> var_dof_A_diag_in,
    wp::array_t<wp::float32> var_dof_invweight0_out)
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
        wp::shape_t* var_2;
        const wp::int32 var_3 = 0;
        wp::int32 var_4;
        wp::shape_t var_5;
        wp::int32 var_6;
        wp::shape_t* var_7;
        const wp::int32 var_8 = 0;
        wp::int32 var_9;
        wp::shape_t var_10;
        wp::int32 var_11;
        wp::int32* var_12;
        wp::int32 var_13;
        wp::int32 var_14;
        wp::int32* var_15;
        wp::int32 var_16;
        wp::int32 var_17;
        wp::int32* var_18;
        wp::int32 var_19;
        wp::int32 var_20;
        const wp::int32 var_21 = 0;
        const wp::int32 var_22 = 0;
        wp::int32 var_23;
        bool var_24;
        const wp::int32 var_25 = 3;
        wp::int32 var_26;
        bool var_27;
        const wp::float32 var_28 = 0.3333333333333333;
        const wp::int32 var_29 = 0;
        wp::int32 var_30;
        wp::float32* var_31;
        const wp::int32 var_32 = 1;
        wp::int32 var_33;
        wp::float32* var_34;
        wp::float32 var_35;
        wp::float32 var_36;
        wp::float32 var_37;
        const wp::int32 var_38 = 2;
        wp::int32 var_39;
        wp::float32* var_40;
        wp::float32 var_41;
        wp::float32 var_42;
        wp::float32 var_43;
        const wp::float32 var_44 = 0.3333333333333333;
        const wp::int32 var_45 = 3;
        wp::int32 var_46;
        wp::float32* var_47;
        const wp::int32 var_48 = 4;
        wp::int32 var_49;
        wp::float32* var_50;
        wp::float32 var_51;
        wp::float32 var_52;
        wp::float32 var_53;
        const wp::int32 var_54 = 5;
        wp::int32 var_55;
        wp::float32* var_56;
        wp::float32 var_57;
        wp::float32 var_58;
        wp::float32 var_59;
        wp::float32 var_60;
        const wp::int32 var_61 = 1;
        const wp::int32 var_62 = 1;
        wp::int32 var_63;
        bool var_64;
        const wp::float32 var_65 = 0.3333333333333333;
        const wp::int32 var_66 = 0;
        wp::int32 var_67;
        wp::float32* var_68;
        const wp::int32 var_69 = 1;
        wp::int32 var_70;
        wp::float32* var_71;
        wp::float32 var_72;
        wp::float32 var_73;
        wp::float32 var_74;
        const wp::int32 var_75 = 2;
        wp::int32 var_76;
        wp::float32* var_77;
        wp::float32 var_78;
        wp::float32 var_79;
        wp::float32 var_80;
        wp::float32* var_81;
        wp::float32 var_82;
        wp::float32 var_83;
        wp::float32 var_84;
        //---------
        // forward
        // def _finalize_dof_invweight0(                                                          <L 218>
        // worldid, dofid = wp.tid()                                                              <L 225>
        builtin_tid2d(var_0, var_1);
        // dof_invweight0_id = worldid % dof_invweight0_out.shape[0]                              <L 226>
        var_2 = &(var_dof_invweight0_out.shape);
        var_5 = wp::load(var_2);
        var_4 = wp::extract(var_5, var_3);
        var_6 = wp::mod(var_0, var_4);
        // dof_A_diag_id = worldid % dof_A_diag_in.shape[0]                                       <L 227>
        var_7 = &(var_dof_A_diag_in.shape);
        var_10 = wp::load(var_7);
        var_9 = wp::extract(var_10, var_8);
        var_11 = wp::mod(var_0, var_9);
        // jntid = dof_jntid[dofid]                                                               <L 229>
        var_12 = wp::address(var_dof_jntid, var_1);
        var_14 = wp::load(var_12);
        var_13 = wp::copy(var_14);
        // jtype = jnt_type[jntid]                                                                <L 230>
        var_15 = wp::address(var_jnt_type, var_13);
        var_17 = wp::load(var_15);
        var_16 = wp::copy(var_17);
        // dofadr = jnt_dofadr[jntid]                                                             <L 231>
        var_18 = wp::address(var_jnt_dofadr, var_13);
        var_20 = wp::load(var_18);
        var_19 = wp::copy(var_20);
        // if jtype == int(types.JointType.FREE.value):                                           <L 233>
        var_23 = wp::int(var_22);
        var_24 = (var_16 == var_23);
        if (var_24) {
            // if dofid < dofadr + 3:                                                             <L 235>
            var_26 = wp::add(var_19, var_25);
            var_27 = (var_1 < var_26);
            if (var_27) {
                // avg = wp.static(1.0 / 3.0) * (                                                 <L 236>
                // dof_A_diag_in[dof_A_diag_id, dofadr + 0]                                       <L 237>
                var_30 = wp::add(var_19, var_29);
                var_31 = wp::address(var_dof_A_diag_in, var_11, var_30);
                // + dof_A_diag_in[dof_A_diag_id, dofadr + 1]                                     <L 238>
                var_33 = wp::add(var_19, var_32);
                var_34 = wp::address(var_dof_A_diag_in, var_11, var_33);
                var_36 = wp::load(var_31);
                var_37 = wp::load(var_34);
                var_35 = wp::add(var_36, var_37);
                // + dof_A_diag_in[dof_A_diag_id, dofadr + 2]                                     <L 239>
                var_39 = wp::add(var_19, var_38);
                var_40 = wp::address(var_dof_A_diag_in, var_11, var_39);
                var_42 = wp::load(var_40);
                var_41 = wp::add(var_35, var_42);
                var_43 = wp::mul(var_28, var_41);
            }
            if (!var_27) {
                // avg = wp.static(1.0 / 3.0) * (                                                 <L 242>
                // dof_A_diag_in[dof_A_diag_id, dofadr + 3]                                       <L 243>
                var_46 = wp::add(var_19, var_45);
                var_47 = wp::address(var_dof_A_diag_in, var_11, var_46);
                // + dof_A_diag_in[dof_A_diag_id, dofadr + 4]                                     <L 244>
                var_49 = wp::add(var_19, var_48);
                var_50 = wp::address(var_dof_A_diag_in, var_11, var_49);
                var_52 = wp::load(var_47);
                var_53 = wp::load(var_50);
                var_51 = wp::add(var_52, var_53);
                // + dof_A_diag_in[dof_A_diag_id, dofadr + 5]                                     <L 245>
                var_55 = wp::add(var_19, var_54);
                var_56 = wp::address(var_dof_A_diag_in, var_11, var_55);
                var_58 = wp::load(var_56);
                var_57 = wp::add(var_51, var_58);
                var_59 = wp::mul(var_44, var_57);
            }
            var_60 = wp::where(var_27, var_43, var_59);
            // dof_invweight0_out[dof_invweight0_id, dofid] = avg                                 <L 247>
            wp::array_store(var_dof_invweight0_out, var_6, var_1, var_60);
        }
        if (!var_24) {
            // elif jtype == int(types.JointType.BALL.value):                                     <L 248>
            var_63 = wp::int(var_62);
            var_64 = (var_16 == var_63);
            if (var_64) {
                // avg = wp.static(1.0 / 3.0) * (                                                 <L 250>
                // dof_A_diag_in[dof_A_diag_id, dofadr + 0]                                       <L 251>
                var_67 = wp::add(var_19, var_66);
                var_68 = wp::address(var_dof_A_diag_in, var_11, var_67);
                // + dof_A_diag_in[dof_A_diag_id, dofadr + 1]                                     <L 252>
                var_70 = wp::add(var_19, var_69);
                var_71 = wp::address(var_dof_A_diag_in, var_11, var_70);
                var_73 = wp::load(var_68);
                var_74 = wp::load(var_71);
                var_72 = wp::add(var_73, var_74);
                // + dof_A_diag_in[dof_A_diag_id, dofadr + 2]                                     <L 253>
                var_76 = wp::add(var_19, var_75);
                var_77 = wp::address(var_dof_A_diag_in, var_11, var_76);
                var_79 = wp::load(var_77);
                var_78 = wp::add(var_72, var_79);
                var_80 = wp::mul(var_65, var_78);
                // dof_invweight0_out[dof_invweight0_id, dofid] = avg                             <L 255>
                wp::array_store(var_dof_invweight0_out, var_6, var_1, var_80);
            }
            if (!var_64) {
                // dof_invweight0_out[dof_invweight0_id, dofid] = dof_A_diag_in[dof_A_diag_id, dofid]       <L 258>
                var_81 = wp::address(var_dof_A_diag_in, var_11, var_1);
                var_83 = wp::load(var_81);
                var_82 = wp::copy(var_83);
                wp::array_store(var_dof_invweight0_out, var_6, var_1, var_82);
            }
        }
        var_84 = wp::where(var_24, var_60, var_80);
}



extern "C" __global__ void _compute_actuator_acc0_a0c3fa27_cuda_kernel_forward(
    wp::launch_bounds_t<1> dim,
    wp::int32 var_actid_target,
    wp::int32 var_nv,
    wp::array_t<wp::float32> var_result_vec_in,
    wp::array_t<wp::float32> var_actuator_acc0_out)
{
    wp::tile_shared_storage_t tile_mem;

    const size_t _idx = static_cast<size_t>(blockIdx.z * gridDim.y + blockIdx.y) * static_cast<size_t>(gridDim.x * blockDim.x) + static_cast<size_t>(blockIdx.x * blockDim.x + threadIdx.x);
    if (_idx >= dim.size) return;
            // reset shared memory allocator
    wp::tile_shared_storage_t::init();

        //---------
        // primal vars
        wp::int32 var_0;
        const wp::float32 var_1 = 0.0;
        wp::float32 var_2;
        wp::range_t var_3;
        wp::int32 var_4;
        wp::float32* var_5;
        wp::float32* var_6;
        wp::float32 var_7;
        wp::float32 var_8;
        wp::float32 var_9;
        wp::float32 var_10;
        wp::float32 var_11;
        //---------
        // forward
        // def _compute_actuator_acc0(                                                            <L 493>
        // worldid = wp.tid()                                                                     <L 499>
        var_0 = builtin_tid1d();
        // norm_sq = float(0.0)                                                                   <L 500>
        var_2 = wp::float(var_1);
        // for i in range(nv):                                                                    <L 501>
        var_3 = wp::range(var_nv);
        start_for_0:;
            if (iter_cmp(var_3) == 0) goto end_for_0;
            var_4 = wp::iter_next(var_3);
            // norm_sq += result_vec_in[worldid, i] * result_vec_in[worldid, i]                   <L 502>
            var_5 = wp::address(var_result_vec_in, var_0, var_4);
            var_6 = wp::address(var_result_vec_in, var_0, var_4);
            var_8 = wp::load(var_5);
            var_9 = wp::load(var_6);
            var_7 = wp::mul(var_8, var_9);
            var_10 = wp::add(var_2, var_7);
            wp::assign(var_2, var_10);
            goto start_for_0;
        end_for_0:;
        // actuator_acc0_out[worldid, actid_target] = wp.sqrt(norm_sq)                            <L 503>
        var_11 = wp::sqrt(var_2);
        wp::array_store(var_actuator_acc0_out, var_0, var_actid_target, var_11);
}



extern "C" __global__ void _set_length_range_a26be645_cuda_kernel_forward(
    wp::launch_bounds_t<2> dim,
    wp::array_t<wp::int32> var_actuator_trntype,
    wp::array_t<wp::vec_t<2, wp::int32>> var_actuator_trnid,
    wp::array_t<wp::vec_t<6, wp::float32>> var_actuator_gear,
    wp::array_t<wp::int32> var_jnt_limited,
    wp::array_t<wp::vec_t<2, wp::float32>> var_jnt_range,
    wp::array_t<wp::int32> var_tendon_limited,
    wp::array_t<wp::vec_t<2, wp::float32>> var_tendon_range,
    wp::int32 var_ntendon,
    wp::array_t<wp::vec_t<2, wp::float32>> var_actuator_lengthrange_out)
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
        wp::vec_t<2, wp::int32>* var_5;
        const wp::int32 var_6 = 0;
        wp::int32 var_7;
        wp::vec_t<2, wp::int32> var_8;
        wp::shape_t* var_9;
        const wp::int32 var_10 = 0;
        wp::int32 var_11;
        wp::shape_t var_12;
        wp::int32 var_13;
        wp::vec_t<6, wp::float32>* var_14;
        const wp::int32 var_15 = 0;
        wp::float32 var_16;
        wp::vec_t<6, wp::float32> var_17;
        const wp::float32 var_18 = 0.0;
        const wp::float32 var_19 = 0.0;
        wp::vec_t<2, wp::float32> var_20;
        bool var_21;
        const wp::int32 var_22 = 0;
        bool var_23;
        const wp::int32 var_24 = 1;
        bool var_25;
        wp::int32* var_26;
        wp::int32 var_27;
        wp::shape_t* var_28;
        const wp::int32 var_29 = 0;
        wp::int32 var_30;
        wp::shape_t var_31;
        wp::int32 var_32;
        wp::vec_t<2, wp::float32>* var_33;
        wp::vec_t<2, wp::float32> var_34;
        wp::vec_t<2, wp::float32> var_35;
        const wp::float32 var_36 = 0.0;
        bool var_37;
        const wp::int32 var_38 = 0;
        wp::float32 var_39;
        wp::float32 var_40;
        const wp::int32 var_41 = 1;
        wp::float32 var_42;
        wp::float32 var_43;
        wp::vec_t<2, wp::float32> var_44;
        const wp::int32 var_45 = 1;
        wp::float32 var_46;
        wp::float32 var_47;
        const wp::int32 var_48 = 0;
        wp::float32 var_49;
        wp::float32 var_50;
        wp::vec_t<2, wp::float32> var_51;
        wp::vec_t<2, wp::float32> var_52;
        wp::int32 var_53;
        wp::vec_t<2, wp::float32> var_54;
        wp::int32 var_55;
        const wp::int32 var_56 = 3;
        bool var_57;
        bool var_58;
        const wp::int32 var_59 = 0;
        bool var_60;
        wp::int32* var_61;
        wp::int32 var_62;
        wp::shape_t* var_63;
        const wp::int32 var_64 = 0;
        wp::int32 var_65;
        wp::shape_t var_66;
        wp::int32 var_67;
        wp::vec_t<2, wp::float32>* var_68;
        wp::vec_t<2, wp::float32> var_69;
        wp::vec_t<2, wp::float32> var_70;
        const wp::float32 var_71 = 0.0;
        bool var_72;
        const wp::int32 var_73 = 0;
        wp::float32 var_74;
        wp::float32 var_75;
        const wp::int32 var_76 = 1;
        wp::float32 var_77;
        wp::float32 var_78;
        wp::vec_t<2, wp::float32> var_79;
        const wp::int32 var_80 = 1;
        wp::float32 var_81;
        wp::float32 var_82;
        const wp::int32 var_83 = 0;
        wp::float32 var_84;
        wp::float32 var_85;
        wp::vec_t<2, wp::float32> var_86;
        wp::vec_t<2, wp::float32> var_87;
        wp::vec_t<2, wp::float32> var_88;
        wp::vec_t<2, wp::float32> var_89;
        wp::vec_t<2, wp::float32> var_90;
        wp::vec_t<2, wp::float32> var_91;
        //---------
        // forward
        // def _set_length_range(                                                                 <L 556>
        // worldid, actid = wp.tid()                                                              <L 567>
        builtin_tid2d(var_0, var_1);
        // trntype = actuator_trntype[actid]                                                      <L 568>
        var_2 = wp::address(var_actuator_trntype, var_1);
        var_4 = wp::load(var_2);
        var_3 = wp::copy(var_4);
        // id0 = actuator_trnid[actid][0]                                                         <L 569>
        var_5 = wp::address(var_actuator_trnid, var_1);
        var_8 = wp::load(var_5);
        var_7 = wp::extract(var_8, var_6);
        // gear0 = actuator_gear[worldid % actuator_gear.shape[0], actid][0]                      <L 570>
        var_9 = &(var_actuator_gear.shape);
        var_12 = wp::load(var_9);
        var_11 = wp::extract(var_12, var_10);
        var_13 = wp::mod(var_0, var_11);
        var_14 = wp::address(var_actuator_gear, var_13, var_1);
        var_17 = wp::load(var_14);
        var_16 = wp::extract(var_17, var_15);
        // lr = wp.vec2(0.0, 0.0)                                                                 <L 572>
        var_20 = wp::vec_t<2, wp::float32>(var_18, var_19);
        // if trntype == TrnType.JOINT or trntype == TrnType.JOINTINPARENT:                       <L 574>
        var_23 = (var_3 == var_22);
        var_21 = var_23;
        if (!var_21) {
            var_25 = (var_3 == var_24);
            var_21 = var_21 || var_25;
        }
        if (var_21) {
            // if jnt_limited[id0]:                                                               <L 575>
            var_26 = wp::address(var_jnt_limited, var_7);
            var_27 = wp::load(var_26);
            if (var_27) {
                // rng = jnt_range[worldid % jnt_range.shape[0], id0]                             <L 576>
                var_28 = &(var_jnt_range.shape);
                var_31 = wp::load(var_28);
                var_30 = wp::extract(var_31, var_29);
                var_32 = wp::mod(var_0, var_30);
                var_33 = wp::address(var_jnt_range, var_32, var_7);
                var_35 = wp::load(var_33);
                var_34 = wp::copy(var_35);
                // if gear0 > 0.0:                                                                <L 577>
                var_37 = (var_16 > var_36);
                if (var_37) {
                    // lr = wp.vec2(rng[0] * gear0, rng[1] * gear0)                               <L 578>
                    var_39 = wp::extract(var_34, var_38);
                    var_40 = wp::mul(var_39, var_16);
                    var_42 = wp::extract(var_34, var_41);
                    var_43 = wp::mul(var_42, var_16);
                    var_44 = wp::vec_t<2, wp::float32>(var_40, var_43);
                }
                if (!var_37) {
                    // lr = wp.vec2(rng[1] * gear0, rng[0] * gear0)                               <L 580>
                    var_46 = wp::extract(var_34, var_45);
                    var_47 = wp::mul(var_46, var_16);
                    var_49 = wp::extract(var_34, var_48);
                    var_50 = wp::mul(var_49, var_16);
                    var_51 = wp::vec_t<2, wp::float32>(var_47, var_50);
                }
                var_52 = wp::where(var_37, var_44, var_51);
            }
            var_53 = wp::load(var_26);
            var_55 = wp::load(var_26);
            var_54 = wp::where(var_55, var_52, var_20);
        }
        if (!var_21) {
            // elif trntype == TrnType.TENDON:                                                    <L 581>
            var_57 = (var_3 == var_56);
            if (var_57) {
                // if ntendon > 0 and tendon_limited[id0]:                                        <L 582>
                var_60 = (var_ntendon > var_59);
                var_58 = var_60;
                if (var_58) {
                    var_61 = wp::address(var_tendon_limited, var_7);
                    var_62 = wp::load(var_61);
                    var_58 = var_58 && var_62;
                }
                if (var_58) {
                    // rng = tendon_range[worldid % tendon_range.shape[0], id0]                   <L 583>
                    var_63 = &(var_tendon_range.shape);
                    var_66 = wp::load(var_63);
                    var_65 = wp::extract(var_66, var_64);
                    var_67 = wp::mod(var_0, var_65);
                    var_68 = wp::address(var_tendon_range, var_67, var_7);
                    var_70 = wp::load(var_68);
                    var_69 = wp::copy(var_70);
                    // if gear0 > 0.0:                                                            <L 584>
                    var_72 = (var_16 > var_71);
                    if (var_72) {
                        // lr = wp.vec2(rng[0] * gear0, rng[1] * gear0)                           <L 585>
                        var_74 = wp::extract(var_69, var_73);
                        var_75 = wp::mul(var_74, var_16);
                        var_77 = wp::extract(var_69, var_76);
                        var_78 = wp::mul(var_77, var_16);
                        var_79 = wp::vec_t<2, wp::float32>(var_75, var_78);
                    }
                    if (!var_72) {
                        // lr = wp.vec2(rng[1] * gear0, rng[0] * gear0)                           <L 587>
                        var_81 = wp::extract(var_69, var_80);
                        var_82 = wp::mul(var_81, var_16);
                        var_84 = wp::extract(var_69, var_83);
                        var_85 = wp::mul(var_84, var_16);
                        var_86 = wp::vec_t<2, wp::float32>(var_82, var_85);
                    }
                    var_87 = wp::where(var_72, var_79, var_86);
                }
                var_88 = wp::where(var_58, var_87, var_20);
            }
            var_89 = wp::where(var_57, var_88, var_20);
        }
        var_90 = wp::where(var_21, var_54, var_89);
        var_91 = wp::where(var_21, var_34, var_69);
        // actuator_lengthrange_out[worldid, actid] = lr                                          <L 589>
        wp::array_store(var_actuator_lengthrange_out, var_0, var_1, var_90);
}



extern "C" __global__ void _compute_light_pos0_a2f2ed64_cuda_kernel_forward(
    wp::launch_bounds_t<2> dim,
    wp::array_t<wp::int32> var_light_bodyid,
    wp::array_t<wp::int32> var_light_targetbodyid,
    wp::array_t<wp::vec_t<3, wp::float32>> var_light_xpos_in,
    wp::array_t<wp::vec_t<3, wp::float32>> var_light_xdir_in,
    wp::array_t<wp::vec_t<3, wp::float32>> var_xpos_in,
    wp::array_t<wp::vec_t<3, wp::float32>> var_subtree_com_in,
    wp::array_t<wp::vec_t<3, wp::float32>> var_light_pos0_out,
    wp::array_t<wp::vec_t<3, wp::float32>> var_light_poscom0_out,
    wp::array_t<wp::vec_t<3, wp::float32>> var_light_dir0_out)
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
        wp::shape_t* var_2;
        const wp::int32 var_3 = 0;
        wp::int32 var_4;
        wp::shape_t var_5;
        wp::int32 var_6;
        wp::int32* var_7;
        wp::int32 var_8;
        wp::int32 var_9;
        wp::int32* var_10;
        wp::int32 var_11;
        wp::int32 var_12;
        wp::vec_t<3, wp::float32>* var_13;
        wp::vec_t<3, wp::float32> var_14;
        wp::vec_t<3, wp::float32> var_15;
        wp::vec_t<3, wp::float32>* var_16;
        wp::vec_t<3, wp::float32> var_17;
        wp::vec_t<3, wp::float32> var_18;
        const wp::int32 var_19 = 0;
        bool var_20;
        wp::vec_t<3, wp::float32>* var_21;
        wp::vec_t<3, wp::float32> var_22;
        wp::vec_t<3, wp::float32> var_23;
        wp::vec_t<3, wp::float32>* var_24;
        wp::vec_t<3, wp::float32> var_25;
        wp::vec_t<3, wp::float32> var_26;
        wp::vec_t<3, wp::float32>* var_27;
        wp::vec_t<3, wp::float32> var_28;
        wp::vec_t<3, wp::float32> var_29;
        //---------
        // forward
        // def _compute_light_pos0(                                                               <L 446>
        // worldid, lightid = wp.tid()                                                            <L 457>
        builtin_tid2d(var_0, var_1);
        // light_pos0_id = worldid % light_pos0_out.shape[0]                                      <L 458>
        var_2 = &(var_light_pos0_out.shape);
        var_5 = wp::load(var_2);
        var_4 = wp::extract(var_5, var_3);
        var_6 = wp::mod(var_0, var_4);
        // bodyid = light_bodyid[lightid]                                                         <L 459>
        var_7 = wp::address(var_light_bodyid, var_1);
        var_9 = wp::load(var_7);
        var_8 = wp::copy(var_9);
        // targetid = light_targetbodyid[lightid]                                                 <L 460>
        var_10 = wp::address(var_light_targetbodyid, var_1);
        var_12 = wp::load(var_10);
        var_11 = wp::copy(var_12);
        // light_xpos = light_xpos_in[worldid, lightid]                                           <L 461>
        var_13 = wp::address(var_light_xpos_in, var_0, var_1);
        var_15 = wp::load(var_13);
        var_14 = wp::copy(var_15);
        // light_pos0_out[light_pos0_id, lightid] = light_xpos - xpos_in[worldid, bodyid]         <L 463>
        var_16 = wp::address(var_xpos_in, var_0, var_8);
        var_18 = wp::load(var_16);
        var_17 = wp::sub(var_14, var_18);
        wp::array_store(var_light_pos0_out, var_6, var_1, var_17);
        // if targetid >= 0:                                                                      <L 464>
        var_20 = (var_11 >= var_19);
        if (var_20) {
            // light_poscom0_out[light_pos0_id, lightid] = light_xpos - subtree_com_in[worldid, targetid]       <L 465>
            var_21 = wp::address(var_subtree_com_in, var_0, var_11);
            var_23 = wp::load(var_21);
            var_22 = wp::sub(var_14, var_23);
            wp::array_store(var_light_poscom0_out, var_6, var_1, var_22);
        }
        if (!var_20) {
            // light_poscom0_out[light_pos0_id, lightid] = light_xpos - subtree_com_in[worldid, bodyid]       <L 467>
            var_24 = wp::address(var_subtree_com_in, var_0, var_8);
            var_26 = wp::load(var_24);
            var_25 = wp::sub(var_14, var_26);
            wp::array_store(var_light_poscom0_out, var_6, var_1, var_25);
        }
        // light_dir0_out[light_pos0_id, lightid] = light_xdir_in[worldid, lightid]               <L 468>
        var_27 = wp::address(var_light_xdir_in, var_0, var_1);
        var_29 = wp::load(var_27);
        var_28 = wp::copy(var_29);
        wp::array_store(var_light_dir0_out, var_6, var_1, var_28);
}



extern "C" __global__ void _accumulate_subtreemass_a3369753_cuda_kernel_forward(
    wp::launch_bounds_t<2> dim,
    wp::array_t<wp::int32> var_body_parentid,
    wp::array_t<wp::float32> var_body_subtreemass_io,
    wp::array_t<wp::int32> var_body_tree_)
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
        wp::shape_t* var_2;
        const wp::int32 var_3 = 0;
        wp::int32 var_4;
        wp::shape_t var_5;
        wp::int32 var_6;
        wp::int32* var_7;
        wp::int32 var_8;
        wp::int32 var_9;
        wp::int32* var_10;
        wp::int32 var_11;
        wp::int32 var_12;
        const wp::int32 var_13 = 0;
        bool var_14;
        wp::float32* var_15;
        wp::float32 var_16;
        wp::float32 var_17;
        //---------
        // forward
        // def _accumulate_subtreemass(                                                           <L 45>
        // worldid, nodeid = wp.tid()                                                             <L 50>
        builtin_tid2d(var_0, var_1);
        // body_subtreemass_id = worldid % body_subtreemass_io.shape[0]                           <L 51>
        var_2 = &(var_body_subtreemass_io.shape);
        var_5 = wp::load(var_2);
        var_4 = wp::extract(var_5, var_3);
        var_6 = wp::mod(var_0, var_4);
        // bodyid = body_tree_[nodeid]                                                            <L 52>
        var_7 = wp::address(var_body_tree_, var_1);
        var_9 = wp::load(var_7);
        var_8 = wp::copy(var_9);
        // parentid = body_parentid[bodyid]                                                       <L 53>
        var_10 = wp::address(var_body_parentid, var_8);
        var_12 = wp::load(var_10);
        var_11 = wp::copy(var_12);
        // if bodyid != 0:                                                                        <L 54>
        var_14 = (var_8 != var_13);
        if (var_14) {
            // wp.atomic_add(body_subtreemass_io, body_subtreemass_id, parentid, body_subtreemass_io[body_subtreemass_id, bodyid])       <L 55>
            var_15 = wp::address(var_body_subtreemass_io, var_6, var_8);
            var_17 = wp::load(var_15);
            var_16 = wp::atomic_add(var_body_subtreemass_io, var_6, var_11, var_17);
        }
}



extern "C" __global__ void _init_subtreemass_a34720d1_cuda_kernel_forward(
    wp::launch_bounds_t<2> dim,
    wp::array_t<wp::float32> var_body_mass_in,
    wp::array_t<wp::float32> var_body_subtreemass_out)
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
        wp::shape_t* var_2;
        const wp::int32 var_3 = 0;
        wp::int32 var_4;
        wp::shape_t var_5;
        wp::int32 var_6;
        wp::shape_t* var_7;
        const wp::int32 var_8 = 0;
        wp::int32 var_9;
        wp::shape_t var_10;
        wp::int32 var_11;
        wp::float32* var_12;
        wp::float32 var_13;
        wp::float32 var_14;
        //---------
        // forward
        // def _init_subtreemass(                                                                 <L 34>
        // worldid, bodyid = wp.tid()                                                             <L 38>
        builtin_tid2d(var_0, var_1);
        // body_mass_id = worldid % body_mass_in.shape[0]                                         <L 39>
        var_2 = &(var_body_mass_in.shape);
        var_5 = wp::load(var_2);
        var_4 = wp::extract(var_5, var_3);
        var_6 = wp::mod(var_0, var_4);
        // body_subtreemass_id = worldid % body_subtreemass_out.shape[0]                          <L 40>
        var_7 = &(var_body_subtreemass_out.shape);
        var_10 = wp::load(var_7);
        var_9 = wp::extract(var_10, var_8);
        var_11 = wp::mod(var_0, var_9);
        // body_subtreemass_out[body_subtreemass_id, bodyid] = body_mass_in[body_mass_id, bodyid]       <L 41>
        var_12 = wp::address(var_body_mass_in, var_6, var_1);
        var_14 = wp::load(var_12);
        var_13 = wp::copy(var_14);
        wp::array_store(var_body_subtreemass_out, var_11, var_1, var_13);
}



extern "C" __global__ void _resolve_tendon_lengthspring_a988a292_cuda_kernel_forward(
    wp::launch_bounds_t<2> dim,
    wp::array_t<wp::float32> var_ten_length_in,
    wp::array_t<wp::vec_t<2, wp::float32>> var_tendon_lengthspring_out)
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
        wp::shape_t* var_2;
        const wp::int32 var_3 = 0;
        wp::int32 var_4;
        wp::shape_t var_5;
        wp::int32 var_6;
        wp::vec_t<2, wp::float32>* var_7;
        wp::vec_t<2, wp::float32> var_8;
        wp::vec_t<2, wp::float32> var_9;
        bool var_10;
        const wp::int32 var_11 = 0;
        wp::float32 var_12;
        const wp::float32 var_13 = -1.0;
        bool var_14;
        const wp::int32 var_15 = 1;
        wp::float32 var_16;
        const wp::float32 var_17 = -1.0;
        bool var_18;
        wp::float32* var_19;
        wp::float32 var_20;
        wp::float32 var_21;
        wp::vec_t<2, wp::float32> var_22;
        //---------
        // forward
        // def _resolve_tendon_lengthspring(                                                      <L 156>
        // worldid, tenid = wp.tid()                                                              <L 160>
        builtin_tid2d(var_0, var_1);
        // tendon_lengthspring_id = worldid % tendon_lengthspring_out.shape[0]                    <L 161>
        var_2 = &(var_tendon_lengthspring_out.shape);
        var_5 = wp::load(var_2);
        var_4 = wp::extract(var_5, var_3);
        var_6 = wp::mod(var_0, var_4);
        // val = tendon_lengthspring_out[tendon_lengthspring_id, tenid]                           <L 162>
        var_7 = wp::address(var_tendon_lengthspring_out, var_6, var_1);
        var_9 = wp::load(var_7);
        var_8 = wp::copy(var_9);
        // if val[0] == -1.0 and val[1] == -1.0:                                                  <L 163>
        var_12 = wp::extract(var_8, var_11);
        var_14 = (var_12 == var_13);
        var_10 = var_14;
        if (var_10) {
            var_16 = wp::extract(var_8, var_15);
            var_18 = (var_16 == var_17);
            var_10 = var_10 && var_18;
        }
        if (var_10) {
            // l = ten_length_in[worldid, tenid]                                                  <L 164>
            var_19 = wp::address(var_ten_length_in, var_0, var_1);
            var_21 = wp::load(var_19);
            var_20 = wp::copy(var_21);
            // tendon_lengthspring_out[tendon_lengthspring_id, tenid] = wp.vec2(l, l)             <L 165>
            var_22 = wp::vec_t<2, wp::float32>(var_20, var_20);
            wp::array_store(var_tendon_lengthspring_out, var_6, var_1, var_22);
        }
}



extern "C" __global__ void _compute_meaninertia_af08211b_cuda_kernel_forward(
    wp::launch_bounds_t<1> dim,
    wp::int32 var_nv,
    wp::array_t<wp::int32> var_M_rownnz_in,
    wp::array_t<wp::int32> var_M_rowadr_in,
    wp::array_t<wp::float32> var_M_in,
    wp::array_t<wp::float32> var_meaninertia_out)
{
    wp::tile_shared_storage_t tile_mem;

    const size_t _idx = static_cast<size_t>(blockIdx.z * gridDim.y + blockIdx.y) * static_cast<size_t>(gridDim.x * blockDim.x) + static_cast<size_t>(blockIdx.x * blockDim.x + threadIdx.x);
    if (_idx >= dim.size) return;
            // reset shared memory allocator
    wp::tile_shared_storage_t::init();

        //---------
        // primal vars
        wp::int32 var_0;
        const wp::int32 var_1 = 0;
        bool var_2;
        const wp::float32 var_3 = 1.0;
        wp::shape_t* var_4;
        const wp::int32 var_5 = 0;
        wp::int32 var_6;
        wp::shape_t var_7;
        wp::int32 var_8;
        const wp::float32 var_9 = 0.0;
        wp::float32 var_10;
        wp::range_t var_11;
        wp::int32 var_12;
        wp::int32* var_13;
        wp::int32* var_14;
        wp::int32 var_15;
        wp::int32 var_16;
        wp::int32 var_17;
        const wp::int32 var_18 = 1;
        wp::int32 var_19;
        wp::float32* var_20;
        wp::float32 var_21;
        wp::float32 var_22;
        wp::float32 var_23;
        wp::float32 var_24;
        wp::shape_t* var_25;
        const wp::int32 var_26 = 0;
        wp::int32 var_27;
        wp::shape_t var_28;
        wp::int32 var_29;
        //---------
        // forward
        // def _compute_meaninertia(                                                              <L 169>
        // worldid = wp.tid()                                                                     <L 177>
        var_0 = builtin_tid1d();
        // if nv == 0:                                                                            <L 179>
        var_2 = (var_nv == var_1);
        if (var_2) {
            // meaninertia_out[worldid % meaninertia_out.shape[0]] = 1.0  # Default from MuJoCo       <L 180>
            var_4 = &(var_meaninertia_out.shape);
            var_7 = wp::load(var_4);
            var_6 = wp::extract(var_7, var_5);
            var_8 = wp::mod(var_0, var_6);
            wp::array_store(var_meaninertia_out, var_8, var_3);
            // return                                                                             <L 181>
            return;
        }
        // total = float(0.0)                                                                     <L 183>
        var_10 = wp::float(var_9);
        // for i in range(nv):                                                                    <L 184>
        var_11 = wp::range(var_nv);
        start_for_1:;
            if (iter_cmp(var_11) == 0) goto end_for_1;
            var_12 = wp::iter_next(var_11);
            // madr = M_rowadr_in[i] + M_rownnz_in[i] - 1                                         <L 186>
            var_13 = wp::address(var_M_rowadr_in, var_12);
            var_14 = wp::address(var_M_rownnz_in, var_12);
            var_16 = wp::load(var_13);
            var_17 = wp::load(var_14);
            var_15 = wp::add(var_16, var_17);
            var_19 = wp::sub(var_15, var_18);
            // total += M_in[worldid, madr]                                                       <L 187>
            var_20 = wp::address(var_M_in, var_0, var_19);
            var_22 = wp::load(var_20);
            var_21 = wp::add(var_10, var_22);
            wp::assign(var_10, var_21);
            goto start_for_1;
        end_for_1:;
        // meaninertia_out[worldid % meaninertia_out.shape[0]] = total / float(nv)                <L 189>
        var_23 = wp::float(var_nv);
        var_24 = wp::div(var_10, var_23);
        var_25 = &(var_meaninertia_out.shape);
        var_28 = wp::load(var_25);
        var_27 = wp::extract(var_28, var_26);
        var_29 = wp::mod(var_0, var_27);
        wp::array_store(var_meaninertia_out, var_29, var_24);
}



extern "C" __global__ void _compute_tendon_dot_product_be0bc0e6_cuda_kernel_forward(
    wp::launch_bounds_t<1> dim,
    wp::array_t<wp::int32> var_ten_J_rownnz,
    wp::array_t<wp::int32> var_ten_J_rowadr,
    wp::array_t<wp::int32> var_ten_J_colind,
    wp::int32 var_tenid_target,
    wp::array_t<wp::float32> var_ten_J_in,
    wp::array_t<wp::float32> var_result_vec_in,
    wp::array_t<wp::float32> var_tendon_invweight0_out)
{
    wp::tile_shared_storage_t tile_mem;

    const size_t _idx = static_cast<size_t>(blockIdx.z * gridDim.y + blockIdx.y) * static_cast<size_t>(gridDim.x * blockDim.x) + static_cast<size_t>(blockIdx.x * blockDim.x + threadIdx.x);
    if (_idx >= dim.size) return;
            // reset shared memory allocator
    wp::tile_shared_storage_t::init();

        //---------
        // primal vars
        wp::int32 var_0;
        wp::shape_t* var_1;
        const wp::int32 var_2 = 0;
        wp::int32 var_3;
        wp::shape_t var_4;
        wp::int32 var_5;
        const wp::float32 var_6 = 0.0;
        wp::float32 var_7;
        wp::int32* var_8;
        wp::int32 var_9;
        wp::int32 var_10;
        wp::int32* var_11;
        wp::int32 var_12;
        wp::int32 var_13;
        wp::range_t var_14;
        wp::int32 var_15;
        wp::int32 var_16;
        wp::int32* var_17;
        wp::int32 var_18;
        wp::int32 var_19;
        wp::float32* var_20;
        wp::float32* var_21;
        wp::float32 var_22;
        wp::float32 var_23;
        wp::float32 var_24;
        wp::float32 var_25;
        //---------
        // forward
        // def _compute_tendon_dot_product(                                                       <L 393>
        // worldid = wp.tid()                                                                     <L 405>
        var_0 = builtin_tid1d();
        // tendon_invweight0_id = worldid % tendon_invweight0_out.shape[0]                        <L 406>
        var_1 = &(var_tendon_invweight0_out.shape);
        var_4 = wp::load(var_1);
        var_3 = wp::extract(var_4, var_2);
        var_5 = wp::mod(var_0, var_3);
        // dot_prod = float(0.0)                                                                  <L 407>
        var_7 = wp::float(var_6);
        // rownnz = ten_J_rownnz[tenid_target]                                                    <L 409>
        var_8 = wp::address(var_ten_J_rownnz, var_tenid_target);
        var_10 = wp::load(var_8);
        var_9 = wp::copy(var_10);
        // rowadr = ten_J_rowadr[tenid_target]                                                    <L 410>
        var_11 = wp::address(var_ten_J_rowadr, var_tenid_target);
        var_13 = wp::load(var_11);
        var_12 = wp::copy(var_13);
        // for i in range(rownnz):                                                                <L 411>
        var_14 = wp::range(var_9);
        start_for_0:;
            if (iter_cmp(var_14) == 0) goto end_for_0;
            var_15 = wp::iter_next(var_14);
            // sparseid = rowadr + i                                                              <L 412>
            var_16 = wp::add(var_12, var_15);
            // colind = ten_J_colind[sparseid]                                                    <L 413>
            var_17 = wp::address(var_ten_J_colind, var_16);
            var_19 = wp::load(var_17);
            var_18 = wp::copy(var_19);
            // dot_prod += ten_J_in[worldid, sparseid] * result_vec_in[worldid, colind]           <L 414>
            var_20 = wp::address(var_ten_J_in, var_0, var_16);
            var_21 = wp::address(var_result_vec_in, var_0, var_18);
            var_23 = wp::load(var_20);
            var_24 = wp::load(var_21);
            var_22 = wp::mul(var_23, var_24);
            var_25 = wp::add(var_7, var_22);
            wp::assign(var_7, var_25);
            goto start_for_0;
        end_for_0:;
        // tendon_invweight0_out[tendon_invweight0_id, tenid_target] = dot_prod                   <L 416>
        wp::array_store(var_tendon_invweight0_out, var_5, var_tenid_target, var_7);
}



extern "C" __global__ void _copy_tendon_jacobian_cf465fd5_cuda_kernel_forward(
    wp::launch_bounds_t<1> dim,
    wp::int32 var_tenid_target,
    wp::array_t<wp::int32> var_ten_J_rownnz,
    wp::array_t<wp::int32> var_ten_J_rowadr,
    wp::array_t<wp::int32> var_ten_J_colind,
    wp::array_t<wp::float32> var_ten_J_in,
    wp::array_t<wp::float32> var_ten_J_vec_out)
{
    wp::tile_shared_storage_t tile_mem;

    const size_t _idx = static_cast<size_t>(blockIdx.z * gridDim.y + blockIdx.y) * static_cast<size_t>(gridDim.x * blockDim.x) + static_cast<size_t>(blockIdx.x * blockDim.x + threadIdx.x);
    if (_idx >= dim.size) return;
            // reset shared memory allocator
    wp::tile_shared_storage_t::init();

        //---------
        // primal vars
        wp::int32 var_0;
        wp::shape_t* var_1;
        const wp::int32 var_2 = 2;
        wp::int32 var_3;
        wp::shape_t var_4;
        wp::int32* var_5;
        wp::int32 var_6;
        wp::int32 var_7;
        wp::int32* var_8;
        wp::int32 var_9;
        wp::int32 var_10;
        wp::range_t var_11;
        wp::int32 var_12;
        wp::int32 var_13;
        wp::int32* var_14;
        wp::int32 var_15;
        wp::int32 var_16;
        wp::int32 var_17;
        wp::float32* var_18;
        wp::float32 var_19;
        wp::float32 var_20;
        //---------
        // forward
        // def _copy_tendon_jacobian(                                                             <L 375>
        // worldid = wp.tid()                                                                     <L 383>
        var_0 = builtin_tid1d();
        // nv = ten_J_in.shape[2]                                                                 <L 384>
        var_1 = &(var_ten_J_in.shape);
        var_4 = wp::load(var_1);
        var_3 = wp::extract(var_4, var_2);
        // rownnz = ten_J_rownnz[tenid_target]                                                    <L 385>
        var_5 = wp::address(var_ten_J_rownnz, var_tenid_target);
        var_7 = wp::load(var_5);
        var_6 = wp::copy(var_7);
        // rowadr = ten_J_rowadr[tenid_target]                                                    <L 386>
        var_8 = wp::address(var_ten_J_rowadr, var_tenid_target);
        var_10 = wp::load(var_8);
        var_9 = wp::copy(var_10);
        // for i in range(rownnz):                                                                <L 387>
        var_11 = wp::range(var_6);
        start_for_0:;
            if (iter_cmp(var_11) == 0) goto end_for_0;
            var_12 = wp::iter_next(var_11);
            // colind = ten_J_colind[rowadr + i]                                                  <L 388>
            var_13 = wp::add(var_9, var_12);
            var_14 = wp::address(var_ten_J_colind, var_13);
            var_16 = wp::load(var_14);
            var_15 = wp::copy(var_16);
            // ten_J_vec_out[worldid, colind] = ten_J_in[worldid, rowadr + i]                     <L 389>
            var_17 = wp::add(var_9, var_12);
            var_18 = wp::address(var_ten_J_in, var_0, var_17);
            var_20 = wp::load(var_18);
            var_19 = wp::copy(var_20);
            wp::array_store(var_ten_J_vec_out, var_0, var_15, var_19);
            goto start_for_0;
        end_for_0:;
}



extern "C" __global__ void _compute_eq_data0_d31148f7_cuda_kernel_forward(
    wp::launch_bounds_t<2> dim,
    wp::array_t<wp::int32> var_eq_type,
    wp::array_t<wp::int32> var_eq_obj1id,
    wp::array_t<wp::int32> var_eq_obj2id,
    wp::array_t<wp::int32> var_eq_objtype,
    wp::array_t<wp::vec_t<3, wp::float32>> var_xpos_in,
    wp::array_t<wp::quat_t<wp::float32>> var_xquat_in,
    wp::array_t<wp::mat_t<3, 3, wp::float32>> var_xmat_in,
    wp::array_t<wp::vec_t<11, wp::float32>> var_eq_data_out)
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
        wp::shape_t* var_2;
        const wp::int32 var_3 = 0;
        wp::int32 var_4;
        wp::shape_t var_5;
        wp::int32 var_6;
        wp::int32* var_7;
        wp::int32 var_8;
        wp::int32 var_9;
        wp::int32* var_10;
        wp::int32 var_11;
        wp::int32 var_12;
        wp::vec_t<11, wp::float32>* var_13;
        wp::vec_t<11, wp::float32> var_14;
        wp::vec_t<11, wp::float32> var_15;
        const wp::int32 var_16 = 0;
        const wp::int32 var_17 = 0;
        wp::int32 var_18;
        bool var_19;
        const wp::int32 var_20 = 1;
        const wp::int32 var_21 = 1;
        wp::int32 var_22;
        bool var_23;
        wp::int32* var_24;
        wp::int32 var_25;
        wp::int32 var_26;
        wp::int32* var_27;
        wp::int32 var_28;
        wp::int32 var_29;
        const wp::int32 var_30 = 0;
        wp::float32 var_31;
        const wp::int32 var_32 = 1;
        wp::float32 var_33;
        const wp::int32 var_34 = 2;
        wp::float32 var_35;
        wp::vec_t<3, wp::float32> var_36;
        wp::vec_t<3, wp::float32>* var_37;
        wp::mat_t<3, 3, wp::float32>* var_38;
        wp::vec_t<3, wp::float32> var_39;
        wp::mat_t<3, 3, wp::float32> var_40;
        wp::vec_t<3, wp::float32> var_41;
        wp::vec_t<3, wp::float32> var_42;
        wp::mat_t<3, 3, wp::float32>* var_43;
        wp::mat_t<3, 3, wp::float32> var_44;
        wp::mat_t<3, 3, wp::float32> var_45;
        wp::vec_t<3, wp::float32>* var_46;
        wp::vec_t<3, wp::float32> var_47;
        wp::vec_t<3, wp::float32> var_48;
        wp::vec_t<3, wp::float32> var_49;
        const wp::int32 var_50 = 0;
        wp::float32 var_51;
        const wp::int32 var_52 = 3;
        const wp::int32 var_53 = 1;
        wp::float32 var_54;
        const wp::int32 var_55 = 4;
        const wp::int32 var_56 = 2;
        wp::float32 var_57;
        const wp::int32 var_58 = 5;
        const wp::int32 var_59 = 6;
        const wp::int32 var_60 = 6;
        wp::int32 var_61;
        bool var_62;
        const wp::float32 var_63 = 0.0;
        wp::vec_t<11, wp::float32> var_64;
        const wp::int32 var_65 = 1;
        const wp::int32 var_66 = 1;
        wp::int32 var_67;
        bool var_68;
        const wp::int32 var_69 = 1;
        const wp::int32 var_70 = 1;
        wp::int32 var_71;
        bool var_72;
        const wp::int32 var_73 = 6;
        wp::float32 var_74;
        const wp::int32 var_75 = 7;
        wp::float32 var_76;
        const wp::int32 var_77 = 8;
        wp::float32 var_78;
        const wp::int32 var_79 = 9;
        wp::float32 var_80;
        wp::quat_t<wp::float32> var_81;
        wp::float32 var_82;
        const wp::float32 var_83 = 0.0;
        bool var_84;
        wp::quat_t<wp::float32> var_85;
        const wp::int32 var_86 = 0;
        wp::float32 var_87;
        const wp::int32 var_88 = 6;
        const wp::int32 var_89 = 1;
        wp::float32 var_90;
        const wp::int32 var_91 = 7;
        const wp::int32 var_92 = 2;
        wp::float32 var_93;
        const wp::int32 var_94 = 8;
        const wp::int32 var_95 = 3;
        wp::float32 var_96;
        const wp::int32 var_97 = 9;
        wp::int32* var_98;
        wp::int32 var_99;
        wp::int32 var_100;
        wp::int32* var_101;
        wp::int32 var_102;
        wp::int32 var_103;
        const wp::int32 var_104 = 0;
        wp::float32 var_105;
        const wp::int32 var_106 = 1;
        wp::float32 var_107;
        const wp::int32 var_108 = 2;
        wp::float32 var_109;
        wp::vec_t<3, wp::float32> var_110;
        wp::vec_t<3, wp::float32>* var_111;
        wp::mat_t<3, 3, wp::float32>* var_112;
        wp::vec_t<3, wp::float32> var_113;
        wp::mat_t<3, 3, wp::float32> var_114;
        wp::vec_t<3, wp::float32> var_115;
        wp::vec_t<3, wp::float32> var_116;
        wp::mat_t<3, 3, wp::float32>* var_117;
        wp::mat_t<3, 3, wp::float32> var_118;
        wp::mat_t<3, 3, wp::float32> var_119;
        wp::vec_t<3, wp::float32>* var_120;
        wp::vec_t<3, wp::float32> var_121;
        wp::vec_t<3, wp::float32> var_122;
        wp::vec_t<3, wp::float32> var_123;
        const wp::int32 var_124 = 0;
        wp::float32 var_125;
        const wp::int32 var_126 = 3;
        const wp::int32 var_127 = 1;
        wp::float32 var_128;
        const wp::int32 var_129 = 4;
        const wp::int32 var_130 = 2;
        wp::float32 var_131;
        const wp::int32 var_132 = 5;
        wp::quat_t<wp::float32>* var_133;
        wp::quat_t<wp::float32> var_134;
        wp::quat_t<wp::float32> var_135;
        wp::quat_t<wp::float32>* var_136;
        wp::quat_t<wp::float32> var_137;
        wp::quat_t<wp::float32> var_138;
        const wp::int32 var_139 = 0;
        wp::float32 var_140;
        const wp::int32 var_141 = 6;
        const wp::int32 var_142 = 1;
        wp::float32 var_143;
        const wp::int32 var_144 = 7;
        const wp::int32 var_145 = 2;
        wp::float32 var_146;
        const wp::int32 var_147 = 8;
        const wp::int32 var_148 = 3;
        wp::float32 var_149;
        const wp::int32 var_150 = 9;
        wp::quat_t<wp::float32> var_151;
        wp::int32 var_152;
        wp::int32 var_153;
        wp::vec_t<3, wp::float32> var_154;
        wp::vec_t<3, wp::float32> var_155;
        wp::vec_t<3, wp::float32> var_156;
        //---------
        // forward
        // def _compute_eq_data0(                                                                 <L 79>
        // worldid, eqid = wp.tid()                                                               <L 96>
        builtin_tid2d(var_0, var_1);
        // eq_data_id = worldid % eq_data_out.shape[0]                                            <L 97>
        var_2 = &(var_eq_data_out.shape);
        var_5 = wp::load(var_2);
        var_4 = wp::extract(var_5, var_3);
        var_6 = wp::mod(var_0, var_4);
        // eqtype = eq_type[eqid]                                                                 <L 99>
        var_7 = wp::address(var_eq_type, var_1);
        var_9 = wp::load(var_7);
        var_8 = wp::copy(var_9);
        // objtype = eq_objtype[eqid]                                                             <L 100>
        var_10 = wp::address(var_eq_objtype, var_1);
        var_12 = wp::load(var_10);
        var_11 = wp::copy(var_12);
        // data = eq_data_out[eq_data_id, eqid]                                                   <L 101>
        var_13 = wp::address(var_eq_data_out, var_6, var_1);
        var_15 = wp::load(var_13);
        var_14 = wp::copy(var_15);
        // if eqtype == int(types.EqType.CONNECT.value):                                          <L 103>
        var_18 = wp::int(var_17);
        var_19 = (var_8 == var_18);
        if (var_19) {
            // if objtype == int(types.ObjType.BODY.value):                                       <L 104>
            var_22 = wp::int(var_21);
            var_23 = (var_11 == var_22);
            if (var_23) {
                // obj1id = eq_obj1id[eqid]                                                       <L 105>
                var_24 = wp::address(var_eq_obj1id, var_1);
                var_26 = wp::load(var_24);
                var_25 = wp::copy(var_26);
                // obj2id = eq_obj2id[eqid]                                                       <L 106>
                var_27 = wp::address(var_eq_obj2id, var_1);
                var_29 = wp::load(var_27);
                var_28 = wp::copy(var_29);
                // anchor1 = wp.vec3(data[0], data[1], data[2])                                   <L 109>
                var_31 = wp::extract(var_14, var_30);
                var_33 = wp::extract(var_14, var_32);
                var_35 = wp::extract(var_14, var_34);
                var_36 = wp::vec_t<3, wp::float32>(var_31, var_33, var_35);
                // pos = xpos_in[worldid, obj1id] + xmat_in[worldid, obj1id] @ anchor1            <L 110>
                var_37 = wp::address(var_xpos_in, var_0, var_25);
                var_38 = wp::address(var_xmat_in, var_0, var_25);
                var_40 = wp::load(var_38);
                var_39 = wp::mul(var_40, var_36);
                var_42 = wp::load(var_37);
                var_41 = wp::add(var_42, var_39);
                // anchor2 = wp.transpose(xmat_in[worldid, obj2id]) @ (pos - xpos_in[worldid, obj2id])       <L 113>
                var_43 = wp::address(var_xmat_in, var_0, var_28);
                var_45 = wp::load(var_43);
                var_44 = wp::transpose(var_45);
                var_46 = wp::address(var_xpos_in, var_0, var_28);
                var_48 = wp::load(var_46);
                var_47 = wp::sub(var_41, var_48);
                var_49 = wp::mul(var_44, var_47);
                // data[3] = anchor2[0]                                                           <L 114>
                var_51 = wp::extract(var_49, var_50);
                wp::assign_inplace(var_14, var_52, var_51);
                // data[4] = anchor2[1]                                                           <L 115>
                var_54 = wp::extract(var_49, var_53);
                wp::assign_inplace(var_14, var_55, var_54);
                // data[5] = anchor2[2]                                                           <L 116>
                var_57 = wp::extract(var_49, var_56);
                wp::assign_inplace(var_14, var_58, var_57);
                // eq_data_out[eq_data_id, eqid] = data                                           <L 117>
                wp::array_store(var_eq_data_out, var_6, var_1, var_14);
            }
            if (!var_23) {
                // elif objtype == int(types.ObjType.SITE.value):                                 <L 118>
                var_61 = wp::int(var_60);
                var_62 = (var_11 == var_61);
                if (var_62) {
                    // eq_data_out[eq_data_id, eqid] = types.vec11(0.0)                           <L 120>
                    var_64 = wp::vec_t<11, wp::float32>(var_63);
                    wp::array_store(var_eq_data_out, var_6, var_1, var_64);
                }
            }
        }
        if (!var_19) {
            // elif eqtype == int(types.EqType.WELD.value):                                       <L 121>
            var_67 = wp::int(var_66);
            var_68 = (var_8 == var_67);
            if (var_68) {
                // if objtype == int(types.ObjType.BODY.value):                                   <L 122>
                var_71 = wp::int(var_70);
                var_72 = (var_11 == var_71);
                if (var_72) {
                    // quat = wp.quat(data[6], data[7], data[8], data[9])                         <L 123>
                    var_74 = wp::extract(var_14, var_73);
                    var_76 = wp::extract(var_14, var_75);
                    var_78 = wp::extract(var_14, var_77);
                    var_80 = wp::extract(var_14, var_79);
                    var_81 = wp::quat_t<wp::float32>(var_74, var_76, var_78, var_80);
                    // if wp.length_sq(quat) > 0.0:                                               <L 124>
                    var_82 = wp::length_sq(var_81);
                    var_84 = (var_82 > var_83);
                    if (var_84) {
                        // quat = wp.normalize(quat)                                              <L 126>
                        var_85 = wp::normalize(var_81);
                        // data[6] = quat[0]                                                      <L 127>
                        var_87 = wp::extract(var_85, var_86);
                        wp::assign_inplace(var_14, var_88, var_87);
                        // data[7] = quat[1]                                                      <L 128>
                        var_90 = wp::extract(var_85, var_89);
                        wp::assign_inplace(var_14, var_91, var_90);
                        // data[8] = quat[2]                                                      <L 129>
                        var_93 = wp::extract(var_85, var_92);
                        wp::assign_inplace(var_14, var_94, var_93);
                        // data[9] = quat[3]                                                      <L 130>
                        var_96 = wp::extract(var_85, var_95);
                        wp::assign_inplace(var_14, var_97, var_96);
                        // eq_data_out[eq_data_id, eqid] = data                                   <L 131>
                        wp::array_store(var_eq_data_out, var_6, var_1, var_14);
                    }
                    if (!var_84) {
                        // obj1id = eq_obj1id[eqid]                                               <L 133>
                        var_98 = wp::address(var_eq_obj1id, var_1);
                        var_100 = wp::load(var_98);
                        var_99 = wp::copy(var_100);
                        // obj2id = eq_obj2id[eqid]                                               <L 134>
                        var_101 = wp::address(var_eq_obj2id, var_1);
                        var_103 = wp::load(var_101);
                        var_102 = wp::copy(var_103);
                        // anchor2 = wp.vec3(data[0], data[1], data[2])                           <L 137>
                        var_105 = wp::extract(var_14, var_104);
                        var_107 = wp::extract(var_14, var_106);
                        var_109 = wp::extract(var_14, var_108);
                        var_110 = wp::vec_t<3, wp::float32>(var_105, var_107, var_109);
                        // pos = xpos_in[worldid, obj2id] + xmat_in[worldid, obj2id] @ anchor2       <L 138>
                        var_111 = wp::address(var_xpos_in, var_0, var_102);
                        var_112 = wp::address(var_xmat_in, var_0, var_102);
                        var_114 = wp::load(var_112);
                        var_113 = wp::mul(var_114, var_110);
                        var_116 = wp::load(var_111);
                        var_115 = wp::add(var_116, var_113);
                        // anchor1 = wp.transpose(xmat_in[worldid, obj1id]) @ (pos - xpos_in[worldid, obj1id])       <L 141>
                        var_117 = wp::address(var_xmat_in, var_0, var_99);
                        var_119 = wp::load(var_117);
                        var_118 = wp::transpose(var_119);
                        var_120 = wp::address(var_xpos_in, var_0, var_99);
                        var_122 = wp::load(var_120);
                        var_121 = wp::sub(var_115, var_122);
                        var_123 = wp::mul(var_118, var_121);
                        // data[3] = anchor1[0]                                                   <L 142>
                        var_125 = wp::extract(var_123, var_124);
                        wp::assign_inplace(var_14, var_126, var_125);
                        // data[4] = anchor1[1]                                                   <L 143>
                        var_128 = wp::extract(var_123, var_127);
                        wp::assign_inplace(var_14, var_129, var_128);
                        // data[5] = anchor1[2]                                                   <L 144>
                        var_131 = wp::extract(var_123, var_130);
                        wp::assign_inplace(var_14, var_132, var_131);
                        // relquat = mjmath.mul_quat(mjmath.quat_inv(xquat_in[worldid, obj1id]), xquat_in[worldid, obj2id])       <L 147>
                        var_133 = wp::address(var_xquat_in, var_0, var_99);
                        var_135 = wp::load(var_133);
                        var_134 = quat_inv_0(var_135);
                        var_136 = wp::address(var_xquat_in, var_0, var_102);
                        var_138 = wp::load(var_136);
                        var_137 = mul_quat_0(var_134, var_138);
                        // data[6] = relquat[0]                                                   <L 148>
                        var_140 = wp::extract(var_137, var_139);
                        wp::assign_inplace(var_14, var_141, var_140);
                        // data[7] = relquat[1]                                                   <L 149>
                        var_143 = wp::extract(var_137, var_142);
                        wp::assign_inplace(var_14, var_144, var_143);
                        // data[8] = relquat[2]                                                   <L 150>
                        var_146 = wp::extract(var_137, var_145);
                        wp::assign_inplace(var_14, var_147, var_146);
                        // data[9] = relquat[3]                                                   <L 151>
                        var_149 = wp::extract(var_137, var_148);
                        wp::assign_inplace(var_14, var_150, var_149);
                        // eq_data_out[eq_data_id, eqid] = data                                   <L 152>
                        wp::array_store(var_eq_data_out, var_6, var_1, var_14);
                    }
                    var_151 = wp::where(var_84, var_85, var_81);
                }
            }
        }
        var_152 = wp::where(var_19, var_25, var_99);
        var_153 = wp::where(var_19, var_28, var_102);
        var_154 = wp::where(var_19, var_36, var_123);
        var_155 = wp::where(var_19, var_41, var_115);
        var_156 = wp::where(var_19, var_49, var_110);
}



extern "C" __global__ void _compute_body_A_diag_entry_d84c280f_cuda_kernel_forward(
    wp::launch_bounds_t<1> dim,
    wp::int32 var_nv,
    wp::int32 var_bodyid_target,
    wp::int32 var_row_idx,
    wp::array_t<wp::float32> var_body_jac_row_in,
    wp::array_t<wp::float32> var_result_vec_in,
    wp::array_t<wp::float32> var_body_A_diag_out)
{
    wp::tile_shared_storage_t tile_mem;

    const size_t _idx = static_cast<size_t>(blockIdx.z * gridDim.y + blockIdx.y) * static_cast<size_t>(gridDim.x * blockDim.x) + static_cast<size_t>(blockIdx.x * blockDim.x + threadIdx.x);
    if (_idx >= dim.size) return;
            // reset shared memory allocator
    wp::tile_shared_storage_t::init();

        //---------
        // primal vars
        wp::int32 var_0;
        wp::shape_t* var_1;
        const wp::int32 var_2 = 0;
        wp::int32 var_3;
        wp::shape_t var_4;
        wp::int32 var_5;
        const wp::float32 var_6 = 0.0;
        wp::float32 var_7;
        wp::range_t var_8;
        wp::int32 var_9;
        wp::float32* var_10;
        wp::float32* var_11;
        wp::float32 var_12;
        wp::float32 var_13;
        wp::float32 var_14;
        wp::float32 var_15;
        //---------
        // forward
        // def _compute_body_A_diag_entry(                                                        <L 321>
        // worldid = wp.tid()                                                                     <L 329>
        var_0 = builtin_tid1d();
        // body_A_diag_id = worldid % body_A_diag_out.shape[0]                                    <L 330>
        var_1 = &(var_body_A_diag_out.shape);
        var_4 = wp::load(var_1);
        var_3 = wp::extract(var_4, var_2);
        var_5 = wp::mod(var_0, var_3);
        // dot_prod = float(0.0)                                                                  <L 332>
        var_7 = wp::float(var_6);
        // for i in range(nv):                                                                    <L 333>
        var_8 = wp::range(var_nv);
        start_for_0:;
            if (iter_cmp(var_8) == 0) goto end_for_0;
            var_9 = wp::iter_next(var_8);
            // dot_prod += body_jac_row_in[worldid, i] * result_vec_in[worldid, i]                <L 334>
            var_10 = wp::address(var_body_jac_row_in, var_0, var_9);
            var_11 = wp::address(var_result_vec_in, var_0, var_9);
            var_13 = wp::load(var_10);
            var_14 = wp::load(var_11);
            var_12 = wp::mul(var_13, var_14);
            var_15 = wp::add(var_7, var_12);
            wp::assign(var_7, var_15);
            goto start_for_0;
        end_for_0:;
        // body_A_diag_out[body_A_diag_id, bodyid_target, row_idx] = dot_prod                     <L 335>
        wp::array_store(var_body_A_diag_out, var_5, var_bodyid_target, var_row_idx, var_7);
}



extern "C" __global__ void _set_unit_vector_f74e5cf9_cuda_kernel_forward(
    wp::launch_bounds_t<1> dim,
    wp::int32 var_dofid_target,
    wp::array_t<wp::float32> var_unit_vec_out)
{
    wp::tile_shared_storage_t tile_mem;

    const size_t _idx = static_cast<size_t>(blockIdx.z * gridDim.y + blockIdx.y) * static_cast<size_t>(gridDim.x * blockDim.x) + static_cast<size_t>(blockIdx.x * blockDim.x + threadIdx.x);
    if (_idx >= dim.size) return;
            // reset shared memory allocator
    wp::tile_shared_storage_t::init();

        //---------
        // primal vars
        wp::int32 var_0;
        wp::shape_t* var_1;
        const wp::int32 var_2 = 1;
        wp::int32 var_3;
        wp::shape_t var_4;
        wp::range_t var_5;
        wp::int32 var_6;
        bool var_7;
        const wp::float32 var_8 = 1.0;
        const wp::float32 var_9 = 0.0;
        //---------
        // forward
        // def _set_unit_vector(                                                                  <L 193>
        // worldid = wp.tid()                                                                     <L 197>
        var_0 = builtin_tid1d();
        // nv = unit_vec_out.shape[1]                                                             <L 198>
        var_1 = &(var_unit_vec_out.shape);
        var_4 = wp::load(var_1);
        var_3 = wp::extract(var_4, var_2);
        // for i in range(nv):                                                                    <L 199>
        var_5 = wp::range(var_3);
        start_for_0:;
            if (iter_cmp(var_5) == 0) goto end_for_0;
            var_6 = wp::iter_next(var_5);
            // if i == dofid_target:                                                              <L 200>
            var_7 = (var_6 == var_dofid_target);
            if (var_7) {
                // unit_vec_out[worldid, i] = 1.0                                                 <L 201>
                wp::array_store(var_unit_vec_out, var_0, var_6, var_8);
            }
            if (!var_7) {
                // unit_vec_out[worldid, i] = 0.0                                                 <L 203>
                wp::array_store(var_unit_vec_out, var_0, var_6, var_9);
            }
            goto start_for_0;
        end_for_0:;
}

