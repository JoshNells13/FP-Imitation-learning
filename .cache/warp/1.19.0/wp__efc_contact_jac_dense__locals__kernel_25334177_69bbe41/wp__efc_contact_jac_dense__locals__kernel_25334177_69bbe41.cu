#define WP_NO_BFLOAT16

#define WP_TILE_BLOCK_DIM 32
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


// C:\Users\ASUS\Documents\human-imitation-learning\.venv\Lib\site-packages\mujoco_warp\_src\support.py:483
static CUDA_CALLABLE wp::vec_t<3, wp::float32> _compute_jacp_0(
    wp::vec_t<6, wp::float32> var_cdof_clip,
    wp::vec_t<3, wp::float32> var_offset,
    wp::int32 var_affect)
{
    //---------
    // primal vars
    const wp::int32 var_0 = 0;
    bool var_1;
    const wp::float32 var_2 = 0.0;
    wp::vec_t<3, wp::float32> var_3;
    wp::vec_t<3, wp::float32> var_4;
    wp::vec_t<3, wp::float32> var_5;
    wp::vec_t<3, wp::float32> var_6;
    wp::vec_t<3, wp::float32> var_7;
    //---------
    // forward
    // def _compute_jacp(cdof_clip: wp.spatial_vector, offset: wp.vec3, affect: int) -> wp.vec3:       <L 484>
    // if affect == 0:                                                                        <L 485>
    var_1 = (var_affect == var_0);
    if (var_1) {
        // return wp.vec3(0.0)                                                                <L 486>
        var_3 = wp::vec_t<3, wp::float32>(var_2);
        return var_3;
    }
    // cdof_lin = wp.spatial_bottom(cdof_clip)                                                <L 487>
    var_4 = wp::spatial_bottom(var_cdof_clip);
    // cdof_ang = wp.spatial_top(cdof_clip)                                                   <L 488>
    var_5 = wp::spatial_top(var_cdof_clip);
    // return cdof_lin + wp.cross(cdof_ang, offset)                                           <L 489>
    var_6 = wp::cross(var_5, var_offset);
    var_7 = wp::add(var_4, var_6);
    return var_7;
}


// C:\Users\ASUS\Documents\human-imitation-learning\.venv\Lib\site-packages\mujoco_warp\_src\support.py:492
static CUDA_CALLABLE wp::vec_t<3, wp::float32> _compute_jacr_0(
    wp::vec_t<6, wp::float32> var_cdof_clip,
    wp::int32 var_affect)
{
    //---------
    // primal vars
    const wp::int32 var_0 = 0;
    bool var_1;
    const wp::float32 var_2 = 0.0;
    wp::vec_t<3, wp::float32> var_3;
    wp::vec_t<3, wp::float32> var_4;
    //---------
    // forward
    // def _compute_jacr(cdof_clip: wp.spatial_vector, affect: int) -> wp.vec3:               <L 493>
    // if affect == 0:                                                                        <L 494>
    var_1 = (var_affect == var_0);
    if (var_1) {
        // return wp.vec3(0.0)                                                                <L 495>
        var_3 = wp::vec_t<3, wp::float32>(var_2);
        return var_3;
    }
    // return wp.spatial_top(cdof_clip)                                                       <L 496>
    var_4 = wp::spatial_top(var_cdof_clip);
    return var_4;
}


// C:\Users\ASUS\Documents\human-imitation-learning\.venv\Lib\site-packages\mujoco_warp\_src\support.py:483
static CUDA_CALLABLE void adj__compute_jacp_0(
    wp::vec_t<6, wp::float32> var_cdof_clip,
    wp::vec_t<3, wp::float32> var_offset,
    wp::int32 var_affect,
    wp::vec_t<6, wp::float32> & adj_cdof_clip,
    wp::vec_t<3, wp::float32> & adj_offset,
    wp::int32 & adj_affect,
    wp::vec_t<3, wp::float32> & adj_ret)
{
	// reverse mode disabled (no backward-enabled kernel depends on this function)
}


// C:\Users\ASUS\Documents\human-imitation-learning\.venv\Lib\site-packages\mujoco_warp\_src\support.py:492
static CUDA_CALLABLE void adj__compute_jacr_0(
    wp::vec_t<6, wp::float32> var_cdof_clip,
    wp::int32 var_affect,
    wp::vec_t<6, wp::float32> & adj_cdof_clip,
    wp::int32 & adj_affect,
    wp::vec_t<3, wp::float32> & adj_ret)
{
	// reverse mode disabled (no backward-enabled kernel depends on this function)
}



extern "C" __global__ void _efc_contact_jac_dense__locals__kernel_4179614b_cuda_kernel_forward(
    wp::launch_bounds_t<3> dim,
    wp::array_t<wp::int32> var_body_rootid,
    wp::array_t<wp::int32> var_geom_bodyid,
    wp::array_t<wp::int32> var_body_isdofancestor,
    wp::array_t<wp::int32> var_ne_in,
    wp::array_t<wp::int32> var_nf_in,
    wp::array_t<wp::int32> var_nl_in,
    wp::array_t<wp::int32> var_nefc_in,
    wp::array_t<wp::float32> var_qvel_in,
    wp::array_t<wp::vec_t<3, wp::float32>> var_subtree_com_in,
    wp::array_t<wp::vec_t<6, wp::float32>> var_cdof_in,
    wp::array_t<wp::int32> var_contact_efc_address_in,
    wp::array_t<wp::int32> var_efc_id_in,
    wp::int32 var_njmax_in,
    wp::int32 var_nv_padded,
    wp::array_t<wp::int32> var_condim_in,
    wp::array_t<wp::vec_t<2, wp::int32>> var_geom_in,
    wp::array_t<wp::vec_t<3, wp::float32>> var_pos_in,
    wp::array_t<wp::vec_t<3, wp::float32>> var_frame_in,
    wp::array_t<wp::float32> var_friction_in,
    wp::array_t<wp::float32> var_efc_J_out,
    wp::array_t<wp::float32> var_efc_Jqvel_out)
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
        wp::int32 var_2;
        const wp::int32 var_3 = 32;
        wp::int32 var_4;
        bool var_5;
        wp::array_t<wp::vec_t<6, wp::float32>> var_6;
        const wp::int32 var_7 = 32;
        const bool var_8 = true;
        wp::tile_register_t<wp::vec_t<6, wp::float32>,wp::tile_layout_register_t<wp::tile_shape_t<32>>> var_9 = wp::tile_register_t<wp::vec_t<6, wp::float32>,wp::tile_layout_register_t<wp::tile_shape_t<32>>>{};
        wp::array_t<wp::float32> var_10;
        const bool var_11 = true;
        wp::tile_register_t<wp::float32,wp::tile_layout_register_t<wp::tile_shape_t<32>>> var_12 = wp::tile_register_t<wp::float32,wp::tile_layout_register_t<wp::tile_shape_t<32>>>{};
        wp::int32* var_13;
        wp::int32* var_14;
        wp::int32 var_15;
        wp::int32 var_16;
        wp::int32 var_17;
        wp::int32* var_18;
        wp::int32 var_19;
        wp::int32 var_20;
        wp::int32* var_21;
        wp::int32 var_22;
        wp::int32 var_23;
        const wp::int32 var_24 = -1;
        wp::int32 var_25;
        const wp::int32 var_26 = 0;
        wp::int32 var_27;
        wp::range_t var_28;
        wp::int32 var_29;
        wp::int32* var_30;
        wp::int32 var_31;
        wp::int32 var_32;
        bool var_33;
        wp::int32 var_34;
        wp::int32* var_35;
        wp::int32 var_36;
        wp::int32 var_37;
        wp::vec_t<2, wp::int32>* var_38;
        wp::vec_t<2, wp::int32> var_39;
        wp::vec_t<2, wp::int32> var_40;
        const wp::int32 var_41 = 0;
        wp::int32 var_42;
        wp::int32* var_43;
        wp::int32 var_44;
        wp::int32 var_45;
        const wp::int32 var_46 = 1;
        wp::int32 var_47;
        wp::int32* var_48;
        wp::int32 var_49;
        wp::int32 var_50;
        wp::vec_t<3, wp::float32>* var_51;
        wp::vec_t<3, wp::float32> var_52;
        wp::vec_t<3, wp::float32> var_53;
        wp::int32* var_54;
        wp::int32 var_55;
        wp::vec_t<3, wp::float32>* var_56;
        wp::vec_t<3, wp::float32> var_57;
        wp::vec_t<3, wp::float32> var_58;
        wp::array_t<wp::int32> var_59;
        const bool var_60 = true;
        wp::tile_register_t<wp::int32,wp::tile_layout_register_t<wp::tile_shape_t<32>>> var_61 = wp::tile_register_t<wp::int32,wp::tile_layout_register_t<wp::tile_shape_t<32>>>{};
        wp::tile_register_t<wp::vec_t<3, wp::float32>,wp::tile_layout_register_t<wp::tile_shape_t<32>>> var_62 = wp::tile_register_t<wp::vec_t<3, wp::float32>,wp::tile_layout_register_t<wp::tile_shape_t<32>>>{};
        wp::tile_register_t<wp::vec_t<3, wp::float32>,wp::tile_layout_register_t<wp::tile_shape_t<32>>> var_63 = wp::tile_register_t<wp::vec_t<3, wp::float32>,wp::tile_layout_register_t<wp::tile_shape_t<32>>>{};
        wp::int32* var_64;
        wp::int32 var_65;
        wp::vec_t<3, wp::float32>* var_66;
        wp::vec_t<3, wp::float32> var_67;
        wp::vec_t<3, wp::float32> var_68;
        wp::array_t<wp::int32> var_69;
        const bool var_70 = true;
        wp::tile_register_t<wp::int32,wp::tile_layout_register_t<wp::tile_shape_t<32>>> var_71 = wp::tile_register_t<wp::int32,wp::tile_layout_register_t<wp::tile_shape_t<32>>>{};
        wp::tile_register_t<wp::vec_t<3, wp::float32>,wp::tile_layout_register_t<wp::tile_shape_t<32>>> var_72 = wp::tile_register_t<wp::vec_t<3, wp::float32>,wp::tile_layout_register_t<wp::tile_shape_t<32>>>{};
        wp::tile_register_t<wp::vec_t<3, wp::float32>,wp::tile_layout_register_t<wp::tile_shape_t<32>>> var_73 = wp::tile_register_t<wp::vec_t<3, wp::float32>,wp::tile_layout_register_t<wp::tile_shape_t<32>>>{};
        wp::tile_register_t<wp::vec_t<3, wp::float32>,wp::tile_layout_register_t<wp::tile_shape_t<32>>> var_74 = wp::tile_register_t<wp::vec_t<3, wp::float32>,wp::tile_layout_register_t<wp::tile_shape_t<32>>>{};
        wp::tile_register_t<wp::vec_t<3, wp::float32>,wp::tile_layout_register_t<wp::tile_shape_t<32>>> var_75 = wp::tile_register_t<wp::vec_t<3, wp::float32>,wp::tile_layout_register_t<wp::tile_shape_t<32>>>{};
        const bool var_76 = false;
        bool var_77;
        const wp::int32 var_78 = 0;
        wp::vec_t<3, wp::float32>* var_79;
        wp::vec_t<3, wp::float32> var_80;
        wp::vec_t<3, wp::float32> var_81;
        wp::tile_register_t<wp::float32,wp::tile_layout_register_t<wp::tile_shape_t<32>>> var_82 = wp::tile_register_t<wp::float32,wp::tile_layout_register_t<wp::tile_shape_t<32>>>{};
        const wp::int32 var_83 = 1;
        bool var_84;
        wp::tile_register_t<wp::float32,wp::tile_layout_register_t<wp::tile_shape_t<32>>> var_85 = wp::tile_register_t<wp::float32,wp::tile_layout_register_t<wp::tile_shape_t<32>>>{};
        const wp::int32 var_86 = 1;
        wp::vec_t<3, wp::float32>* var_87;
        wp::vec_t<3, wp::float32> var_88;
        wp::vec_t<3, wp::float32> var_89;
        wp::tile_register_t<wp::float32,wp::tile_layout_register_t<wp::tile_shape_t<32>>> var_90 = wp::tile_register_t<wp::float32,wp::tile_layout_register_t<wp::tile_shape_t<32>>>{};
        wp::tile_register_t<wp::float32,wp::tile_layout_register_t<wp::tile_shape_t<32>>> var_91 = wp::tile_register_t<wp::float32,wp::tile_layout_register_t<wp::tile_shape_t<32>>>{};
        const wp::int32 var_92 = 2;
        wp::vec_t<3, wp::float32>* var_93;
        wp::vec_t<3, wp::float32> var_94;
        wp::vec_t<3, wp::float32> var_95;
        wp::tile_register_t<wp::float32,wp::tile_layout_register_t<wp::tile_shape_t<32>>> var_96 = wp::tile_register_t<wp::float32,wp::tile_layout_register_t<wp::tile_shape_t<32>>>{};
        wp::tile_register_t<wp::float32,wp::tile_layout_register_t<wp::tile_shape_t<32>>> var_97 = wp::tile_register_t<wp::float32,wp::tile_layout_register_t<wp::tile_shape_t<32>>>{};
        wp::int32 var_98;
        wp::int32 var_99;
        const bool var_100 = false;
        wp::tile_register_t<wp::float32,wp::tile_layout_register_t<wp::tile_shape_t<32>>> var_101 = wp::tile_register_t<wp::float32,wp::tile_layout_register_t<wp::tile_shape_t<32>>>{};
        const wp::int32 var_102 = 1;
        bool var_103;
        const wp::int32 var_104 = 0;
        wp::int32* var_105;
        wp::int32 var_106;
        wp::int32 var_107;
        const wp::int32 var_108 = 2;
        wp::int32 var_109;
        const wp::int32 var_110 = 1;
        wp::int32 var_111;
        const wp::int32 var_112 = 1;
        wp::int32 var_113;
        wp::float32* var_114;
        wp::float32 var_115;
        wp::float32 var_116;
        const wp::float32 var_117 = 1.0;
        const wp::float32 var_118 = 2.0;
        const wp::int32 var_119 = 1;
        wp::int32 var_120;
        wp::float32 var_121;
        wp::float32 var_122;
        wp::float32 var_123;
        wp::float32 var_124;
        const wp::int32 var_125 = 1;
        bool var_126;
        wp::tile_register_t<wp::float32,wp::tile_layout_register_t<wp::tile_shape_t<32>>> var_127 = wp::tile_register_t<wp::float32,wp::tile_layout_register_t<wp::tile_shape_t<32>>>{};
        wp::tile_register_t<wp::float32,wp::tile_layout_register_t<wp::tile_shape_t<32>>> var_128 = wp::tile_register_t<wp::float32,wp::tile_layout_register_t<wp::tile_shape_t<32>>>{};
        const wp::int32 var_129 = 2;
        bool var_130;
        wp::tile_register_t<wp::float32,wp::tile_layout_register_t<wp::tile_shape_t<32>>> var_131 = wp::tile_register_t<wp::float32,wp::tile_layout_register_t<wp::tile_shape_t<32>>>{};
        wp::tile_register_t<wp::float32,wp::tile_layout_register_t<wp::tile_shape_t<32>>> var_132 = wp::tile_register_t<wp::float32,wp::tile_layout_register_t<wp::tile_shape_t<32>>>{};
        const wp::int32 var_133 = 3;
        bool var_134;
        wp::tile_register_t<wp::float32,wp::tile_layout_register_t<wp::tile_shape_t<32>>> var_135 = wp::tile_register_t<wp::float32,wp::tile_layout_register_t<wp::tile_shape_t<32>>>{};
        wp::tile_register_t<wp::float32,wp::tile_layout_register_t<wp::tile_shape_t<32>>> var_136 = wp::tile_register_t<wp::float32,wp::tile_layout_register_t<wp::tile_shape_t<32>>>{};
        const wp::int32 var_137 = 4;
        bool var_138;
        wp::tile_register_t<wp::float32,wp::tile_layout_register_t<wp::tile_shape_t<32>>> var_139 = wp::tile_register_t<wp::float32,wp::tile_layout_register_t<wp::tile_shape_t<32>>>{};
        wp::tile_register_t<wp::float32,wp::tile_layout_register_t<wp::tile_shape_t<32>>> var_140 = wp::tile_register_t<wp::float32,wp::tile_layout_register_t<wp::tile_shape_t<32>>>{};
        wp::tile_register_t<wp::float32,wp::tile_layout_register_t<wp::tile_shape_t<32>>> var_141 = wp::tile_register_t<wp::float32,wp::tile_layout_register_t<wp::tile_shape_t<32>>>{};
        wp::tile_register_t<wp::float32,wp::tile_layout_register_t<wp::tile_shape_t<32>>> var_142 = wp::tile_register_t<wp::float32,wp::tile_layout_register_t<wp::tile_shape_t<32>>>{};
        wp::tile_register_t<wp::float32,wp::tile_layout_register_t<wp::tile_shape_t<32>>> var_143 = wp::tile_register_t<wp::float32,wp::tile_layout_register_t<wp::tile_shape_t<32>>>{};
        wp::tile_register_t<wp::float32,wp::tile_layout_register_t<wp::tile_shape_t<32>>> var_144 = wp::tile_register_t<wp::float32,wp::tile_layout_register_t<wp::tile_shape_t<32>>>{};
        wp::tile_register_t<wp::float32,wp::tile_layout_register_t<wp::tile_shape_t<32>>> var_145 = wp::tile_register_t<wp::float32,wp::tile_layout_register_t<wp::tile_shape_t<32>>>{};
        wp::tile_register_t<wp::float32,wp::tile_layout_register_t<wp::tile_shape_t<32>>> var_146 = wp::tile_register_t<wp::float32,wp::tile_layout_register_t<wp::tile_shape_t<32>>>{};
        wp::tile_register_t<wp::float32,wp::tile_layout_register_t<wp::tile_shape_t<32>>> var_147 = wp::tile_register_t<wp::float32,wp::tile_layout_register_t<wp::tile_shape_t<32>>>{};
        wp::array_t<wp::float32> var_148;
        const bool var_149 = true;
        wp::tile_register_t<wp::float32,wp::tile_layout_register_t<wp::tile_shape_t<32>>> var_150 = wp::tile_register_t<wp::float32,wp::tile_layout_register_t<wp::tile_shape_t<32>>>{};
        wp::tile_shared_t<wp::float32,wp::tile_layout_strided_t<wp::tile_shape_t<1>, wp::tile_stride_t<1>>, true> var_151 = wp::tile_alloc_empty<wp::float32,wp::tile_shape_t<1>,wp::tile_stride_t<1>,false>();
        const wp::int32 var_152 = 0;
        bool var_153;
        wp::array_t<wp::float32> var_154;
        const wp::int32 var_155 = 0;
        wp::float32 var_156;
        wp::float32 var_157;
        //---------
        // forward
        // def kernel(                                                                            <L 3877>
        // worldid, dof_block_id, tid = wp.tid()                                                  <L 3904>
        builtin_tid3d(var_0, var_1, var_2);
        // dof_start = dof_block_id * wp.static(TILE_SIZE)                                        <L 3906>
        var_4 = wp::mul(var_1, var_3);
        // if dof_start >= nv_padded:                                                             <L 3907>
        var_5 = (var_4 >= var_nv_padded);
        if (var_5) {
            // return                                                                             <L 3908>
            return;
        }
        // cdof_tile = wp.tile_load(cdof_in[worldid], shape=TILE_SIZE, offset=dof_start, bounds_check=True)       <L 3910>
        var_6 = wp::view(var_cdof_in, var_0);
        var_9 = wp::tile_load<wp::vec_t<6, wp::float32>, true, false, 32>(var_6, var_4);
        // qvel_tile = wp.tile_load(qvel_in[worldid], shape=TILE_SIZE, offset=dof_start, bounds_check=True)       <L 3911>
        var_10 = wp::view(var_qvel_in, var_0);
        var_12 = wp::tile_load<wp::float32, true, false, 32>(var_10, var_4);
        // efcid_start = ne_in[worldid] + nf_in[worldid] + nl_in[worldid]                         <L 3913>
        var_13 = wp::address(var_ne_in, var_0);
        var_14 = wp::address(var_nf_in, var_0);
        var_16 = wp::load(var_13);
        var_17 = wp::load(var_14);
        var_15 = wp::add(var_16, var_17);
        var_18 = wp::address(var_nl_in, var_0);
        var_20 = wp::load(var_18);
        var_19 = wp::add(var_15, var_20);
        // efcid_end = wp.min(nefc_in[worldid], njmax_in)                                         <L 3914>
        var_21 = wp::address(var_nefc_in, var_0);
        var_23 = wp::load(var_21);
        var_22 = wp::min(var_23, var_njmax_in);
        // prev_conid = int(-1)                                                                   <L 3916>
        var_25 = wp::int(var_24);
        // condim = int(0)                                                                        <L 3917>
        var_27 = wp::int(var_26);
        // for efcid in range(efcid_start, efcid_end):                                            <L 3919>
        var_28 = wp::range(var_19, var_22);
        start_for_1:;
            if (iter_cmp(var_28) == 0) goto end_for_1;
            var_29 = wp::iter_next(var_28);
            // conid = efc_id_in[worldid, efcid]                                                  <L 3920>
            var_30 = wp::address(var_efc_id_in, var_0, var_29);
            var_32 = wp::load(var_30);
            var_31 = wp::copy(var_32);
            // if conid != prev_conid:                                                            <L 3923>
            var_33 = (var_31 != var_25);
            if (var_33) {
                // prev_conid = conid                                                             <L 3924>
                var_34 = wp::copy(var_31);
                // condim = condim_in[conid]                                                      <L 3925>
                var_35 = wp::address(var_condim_in, var_31);
                var_37 = wp::load(var_35);
                var_36 = wp::copy(var_37);
                // geom = geom_in[conid]                                                          <L 3927>
                var_38 = wp::address(var_geom_in, var_31);
                var_40 = wp::load(var_38);
                var_39 = wp::copy(var_40);
                // body1 = geom_bodyid[geom[0]]                                                   <L 3928>
                var_42 = wp::extract(var_39, var_41);
                var_43 = wp::address(var_geom_bodyid, var_42);
                var_45 = wp::load(var_43);
                var_44 = wp::copy(var_45);
                // body2 = geom_bodyid[geom[1]]                                                   <L 3929>
                var_47 = wp::extract(var_39, var_46);
                var_48 = wp::address(var_geom_bodyid, var_47);
                var_50 = wp::load(var_48);
                var_49 = wp::copy(var_50);
                // con_pos = pos_in[conid]                                                        <L 3931>
                var_51 = wp::address(var_pos_in, var_31);
                var_53 = wp::load(var_51);
                var_52 = wp::copy(var_53);
                // offset1 = con_pos - subtree_com_in[worldid, body_rootid[body1]]                <L 3932>
                var_54 = wp::address(var_body_rootid, var_44);
                var_55 = wp::load(var_54);
                var_56 = wp::address(var_subtree_com_in, var_0, var_55);
                var_58 = wp::load(var_56);
                var_57 = wp::sub(var_52, var_58);
                // affects1_tile = wp.tile_load(body_isdofancestor[body1], shape=TILE_SIZE, offset=dof_start, bounds_check=True)       <L 3933>
                var_59 = wp::view(var_body_isdofancestor, var_44);
                var_61 = wp::tile_load<wp::int32, true, false, 32>(var_59, var_4);
                // jacp1_tile = wp.tile_map(support._compute_jacp, cdof_tile, offset1, affects1_tile)       <L 3934>
                var_62 = wp::tile_map(_compute_jacp_0, var_9, var_57, var_61);
                // jacr1_tile = wp.tile_map(support._compute_jacr, cdof_tile, affects1_tile)       <L 3935>
                var_63 = wp::tile_binary_map(_compute_jacr_0, var_9, var_61);
                // offset2 = con_pos - subtree_com_in[worldid, body_rootid[body2]]                <L 3937>
                var_64 = wp::address(var_body_rootid, var_49);
                var_65 = wp::load(var_64);
                var_66 = wp::address(var_subtree_com_in, var_0, var_65);
                var_68 = wp::load(var_66);
                var_67 = wp::sub(var_52, var_68);
                // affects2_tile = wp.tile_load(body_isdofancestor[body2], shape=TILE_SIZE, offset=dof_start, bounds_check=True)       <L 3938>
                var_69 = wp::view(var_body_isdofancestor, var_49);
                var_71 = wp::tile_load<wp::int32, true, false, 32>(var_69, var_4);
                // jacp2_tile = wp.tile_map(support._compute_jacp, cdof_tile, offset2, affects2_tile)       <L 3939>
                var_72 = wp::tile_map(_compute_jacp_0, var_9, var_67, var_71);
                // jacr2_tile = wp.tile_map(support._compute_jacr, cdof_tile, affects2_tile)       <L 3940>
                var_73 = wp::tile_binary_map(_compute_jacr_0, var_9, var_71);
                // jacp_dif_tile = wp.tile_map(wp.sub, jacp2_tile, jacp1_tile)                    <L 3942>
                var_74 = wp::tile_binary_map(wp::sub, var_72, var_62);
                // jacr_dif_tile = wp.tile_map(wp.sub, jacr2_tile, jacr1_tile)                    <L 3943>
                var_75 = wp::tile_binary_map(wp::sub, var_73, var_63);
                // if not wp.static(IS_ELLIPTIC):                                                 <L 3945>
                var_77 = wp::unot(var_76);
                if (var_77) {
                    // frame_0 = frame_in[conid, 0]                                               <L 3946>
                    var_79 = wp::address(var_frame_in, var_31, var_78);
                    var_81 = wp::load(var_79);
                    var_80 = wp::copy(var_81);
                    // Ji_0p_tile = wp.tile_map(wp.dot, jacp_dif_tile, frame_0)                   <L 3947>
                    var_82 = wp::tile_binary_map(wp::dot, var_74, var_80);
                    // if condim > 1:                                                             <L 3949>
                    var_84 = (var_36 > var_83);
                    if (var_84) {
                        // Ji_0r_tile = wp.tile_map(wp.dot, jacr_dif_tile, frame_0)               <L 3950>
                        var_85 = wp::tile_binary_map(wp::dot, var_75, var_80);
                        // frame_1 = frame_in[conid, 1]                                           <L 3951>
                        var_87 = wp::address(var_frame_in, var_31, var_86);
                        var_89 = wp::load(var_87);
                        var_88 = wp::copy(var_89);
                        // Ji_1p_tile = wp.tile_map(wp.dot, jacp_dif_tile, frame_1)               <L 3952>
                        var_90 = wp::tile_binary_map(wp::dot, var_74, var_88);
                        // Ji_1r_tile = wp.tile_map(wp.dot, jacr_dif_tile, frame_1)               <L 3953>
                        var_91 = wp::tile_binary_map(wp::dot, var_75, var_88);
                        // frame_2 = frame_in[conid, 2]                                           <L 3954>
                        var_93 = wp::address(var_frame_in, var_31, var_92);
                        var_95 = wp::load(var_93);
                        var_94 = wp::copy(var_95);
                        // Ji_2p_tile = wp.tile_map(wp.dot, jacp_dif_tile, frame_2)               <L 3955>
                        var_96 = wp::tile_binary_map(wp::dot, var_74, var_94);
                        // Ji_2r_tile = wp.tile_map(wp.dot, jacr_dif_tile, frame_2)               <L 3956>
                        var_97 = wp::tile_binary_map(wp::dot, var_75, var_94);
                    }
                }
            }
            var_98 = wp::where(var_33, var_34, var_25);
            var_99 = wp::where(var_33, var_36, var_27);
            // if wp.static(IS_ELLIPTIC):                                                         <L 3958>
            // J_tile = Ji_0p_tile                                                                <L 3972>
            var_101 = wp::copy(var_82);
            // if condim > 1:                                                                     <L 3973>
            var_103 = (var_99 > var_102);
            if (var_103) {
                // dimid = efcid - contact_efc_address_in[conid, 0]                               <L 3974>
                var_105 = wp::address(var_contact_efc_address_in, var_31, var_104);
                var_107 = wp::load(var_105);
                var_106 = wp::sub(var_29, var_107);
                // dimid2 = dimid / 2 + 1                                                         <L 3975>
                var_109 = wp::div(var_106, var_108);
                var_111 = wp::add(var_109, var_110);
                // frii = friction_in[conid, dimid2 - 1]                                          <L 3976>
                var_113 = wp::sub(var_111, var_112);
                var_114 = wp::address(var_friction_in, var_31, var_113);
                var_116 = wp::load(var_114);
                var_115 = wp::copy(var_116);
                // frii_sign = frii * (1.0 - 2.0 * float(dimid & 1))                              <L 3977>
                var_120 = wp::bit_and(var_106, var_119);
                var_121 = wp::float(var_120);
                var_122 = wp::mul(var_118, var_121);
                var_123 = wp::sub(var_117, var_122);
                var_124 = wp::mul(var_115, var_123);
                // if dimid2 == 1:                                                                <L 3979>
                var_126 = (var_111 == var_125);
                if (var_126) {
                    // J_tile = wp.tile_map(wp.add, J_tile, wp.tile_map(wp.mul, Ji_1p_tile, frii_sign))       <L 3980>
                    var_127 = wp::tile_binary_map(wp::mul, var_90, var_124);
                    var_128 = wp::tile_binary_map(wp::add, var_101, var_127);
                }
                if (!var_126) {
                    // elif dimid2 == 2:                                                          <L 3981>
                    var_130 = (var_111 == var_129);
                    if (var_130) {
                        // J_tile = wp.tile_map(wp.add, J_tile, wp.tile_map(wp.mul, Ji_2p_tile, frii_sign))       <L 3982>
                        var_131 = wp::tile_binary_map(wp::mul, var_96, var_124);
                        var_132 = wp::tile_binary_map(wp::add, var_101, var_131);
                    }
                    if (!var_130) {
                        // elif dimid2 == 3:                                                      <L 3983>
                        var_134 = (var_111 == var_133);
                        if (var_134) {
                            // J_tile = wp.tile_map(wp.add, J_tile, wp.tile_map(wp.mul, Ji_0r_tile, frii_sign))       <L 3984>
                            var_135 = wp::tile_binary_map(wp::mul, var_85, var_124);
                            var_136 = wp::tile_binary_map(wp::add, var_101, var_135);
                        }
                        if (!var_134) {
                            // elif dimid2 == 4:                                                  <L 3985>
                            var_138 = (var_111 == var_137);
                            if (var_138) {
                                // J_tile = wp.tile_map(wp.add, J_tile, wp.tile_map(wp.mul, Ji_1r_tile, frii_sign))       <L 3986>
                                var_139 = wp::tile_binary_map(wp::mul, var_91, var_124);
                                var_140 = wp::tile_binary_map(wp::add, var_101, var_139);
                            }
                            if (!var_138) {
                                // J_tile = wp.tile_map(wp.add, J_tile, wp.tile_map(wp.mul, Ji_2r_tile, frii_sign))       <L 3988>
                                var_141 = wp::tile_binary_map(wp::mul, var_97, var_124);
                                var_142 = wp::tile_binary_map(wp::add, var_101, var_141);
                            }
                            var_143 = wp::where(var_138, var_140, var_142);
                        }
                        var_144 = wp::where(var_134, var_136, var_143);
                    }
                    var_145 = wp::where(var_130, var_132, var_144);
                }
                var_146 = wp::where(var_126, var_128, var_145);
            }
            var_147 = wp::where(var_103, var_146, var_101);
            // wp.tile_store(efc_J_out[worldid, efcid], J_tile, offset=dof_start, bounds_check=True)       <L 3990>
            var_148 = wp::view(var_efc_J_out, var_0, var_29);
            wp::tile_store<wp::float32, true, false>(var_148, var_4, var_147);
            // Jqvel_tile = wp.tile_map(wp.mul, J_tile, qvel_tile)                                <L 3992>
            var_150 = wp::tile_binary_map(wp::mul, var_147, var_12);
            // Jqvel_sum = wp.tile_reduce(wp.add, Jqvel_tile)                                     <L 3993>
            var_151 = wp::tile_reduce(wp::add, var_150);
            // if tid == 0:                                                                       <L 3994>
            var_153 = (var_2 == var_152);
            if (var_153) {
                // wp.atomic_add(efc_Jqvel_out[worldid], efcid, wp.tile_extract(Jqvel_sum, 0))       <L 3995>
                var_154 = wp::view(var_efc_Jqvel_out, var_0);
                var_156 = wp::tile_extract(var_151, var_155);
                var_157 = wp::atomic_add(var_154, var_29, var_156);
            }
            wp::assign(var_25, var_98);
            wp::assign(var_27, var_99);
            goto start_for_1;
        end_for_1:;
}

