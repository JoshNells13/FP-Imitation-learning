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



extern "C" __global__ void _efc_contact_init__locals__kernel_4a133af1_cuda_kernel_forward(
    wp::launch_bounds_t<1> dim,
    wp::array_t<wp::int32> var_body_weldid,
    wp::array_t<wp::int32> var_body_dofnum,
    wp::array_t<wp::int32> var_body_dofadr,
    wp::array_t<wp::int32> var_dof_parentid,
    wp::array_t<wp::int32> var_geom_bodyid,
    wp::int32 var_njmax_in,
    wp::int32 var_njmax_nnz_in,
    wp::array_t<wp::int32> var_nacon_in,
    wp::array_t<wp::float32> var_dist_in,
    wp::array_t<wp::int32> var_condim_in,
    wp::array_t<wp::float32> var_includemargin_in,
    wp::array_t<wp::float32> var_adhesion_in,
    wp::array_t<wp::int32> var_worldid_in,
    wp::array_t<wp::vec_t<2, wp::int32>> var_geom_in,
    wp::array_t<wp::int32> var_type_in,
    wp::array_t<wp::int32> var_nefc_out,
    wp::array_t<wp::int32> var_contact_efc_address_out,
    wp::array_t<wp::int32> var_efc_id_out,
    wp::array_t<wp::int32> var_efc_jtdaj_adr_out,
    wp::array_t<wp::int32> var_efc_jtdaj_nrow_out,
    wp::array_t<wp::int32> var_efc_jtdaj_nblock_out,
    wp::array_t<wp::int32> var_efc_J_rownnz_out,
    wp::array_t<wp::int32> var_efc_J_rowadr_out,
    wp::array_t<wp::int32> var_efc_nnz_out)
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
        const wp::int32 var_1 = 0;
        wp::int32* var_2;
        bool var_3;
        wp::int32 var_4;
        wp::int32* var_5;
        const wp::int32 var_6 = 1;
        wp::int32 var_7;
        wp::int32 var_8;
        bool var_9;
        wp::int32* var_10;
        wp::int32 var_11;
        wp::int32 var_12;
        wp::float32* var_13;
        wp::float32 var_14;
        wp::float32 var_15;
        wp::float32* var_16;
        wp::float32 var_17;
        wp::float32 var_18;
        const bool var_19 = false;
        const wp::float32 var_20 = 0.0;
        bool var_21;
        bool var_22;
        const bool var_23 = false;
        const wp::int32 var_24 = 1;
        bool var_25;
        const wp::int32 var_26 = 1;
        const wp::int32 var_27 = 2;
        const wp::int32 var_28 = 1;
        wp::int32 var_29;
        wp::int32 var_30;
        wp::int32 var_31;
        wp::int32* var_32;
        wp::int32 var_33;
        wp::int32 var_34;
        wp::int32 var_35;
        wp::range_t var_36;
        wp::int32 var_37;
        wp::int32 var_38;
        bool var_39;
        const wp::int32 var_40 = -1;
        const bool var_41 = true;
        bool var_42;
        const wp::int32 var_43 = 1;
        wp::int32 var_44;
        wp::int32 var_45;
        wp::int32 var_46;
        const bool var_47 = true;
        wp::vec_t<2, wp::int32>* var_48;
        wp::vec_t<2, wp::int32> var_49;
        wp::vec_t<2, wp::int32> var_50;
        const wp::int32 var_51 = 0;
        wp::int32 var_52;
        wp::int32* var_53;
        wp::int32 var_54;
        wp::int32* var_55;
        wp::int32 var_56;
        wp::int32 var_57;
        const wp::int32 var_58 = 1;
        wp::int32 var_59;
        wp::int32* var_60;
        wp::int32 var_61;
        wp::int32* var_62;
        wp::int32 var_63;
        wp::int32 var_64;
        wp::int32* var_65;
        wp::int32* var_66;
        wp::int32 var_67;
        wp::int32 var_68;
        wp::int32 var_69;
        const wp::int32 var_70 = 1;
        wp::int32 var_71;
        wp::int32 var_72;
        wp::int32* var_73;
        wp::int32* var_74;
        wp::int32 var_75;
        wp::int32 var_76;
        wp::int32 var_77;
        const wp::int32 var_78 = 1;
        wp::int32 var_79;
        wp::int32 var_80;
        const wp::int32 var_81 = 0;
        wp::int32 var_82;
        bool var_83;
        const wp::int32 var_84 = 0;
        bool var_85;
        const wp::int32 var_86 = 0;
        bool var_87;
        wp::int32 var_88;
        bool var_89;
        bool var_90;
        bool var_91;
        bool var_92;
        wp::int32* var_93;
        wp::int32 var_94;
        wp::int32 var_95;
        wp::int32 var_96;
        bool var_97;
        wp::int32* var_98;
        wp::int32 var_99;
        wp::int32 var_100;
        wp::int32 var_101;
        const wp::int32 var_102 = 1;
        wp::int32 var_103;
        wp::int32 var_104;
        wp::int32 var_105;
        wp::int32 var_106;
        wp::int32 var_107;
        bool var_108;
        wp::range_t var_109;
        wp::int32 var_110;
        wp::int32 var_111;
        bool var_112;
        wp::int32 var_113;
        wp::int32 var_114;
        //---------
        // forward
        // def kernel(                                                                            <L 2767>
        // conid = wp.tid()                                                                       <L 2798>
        var_0 = builtin_tid1d();
        // if conid >= nacon_in[0]:                                                               <L 2800>
        var_2 = wp::address(var_nacon_in, var_1);
        var_4 = wp::load(var_2);
        var_3 = (var_0 >= var_4);
        if (var_3) {
            // return                                                                             <L 2801>
            continue;
        }
        // if not (type_in[conid] & ContactType.CONSTRAINT):                                      <L 2803>
        var_5 = wp::address(var_type_in, var_0);
        var_8 = wp::load(var_5);
        var_7 = wp::bit_and(var_8, var_6);
        var_9 = wp::unot(var_7);
        if (var_9) {
            // return                                                                             <L 2804>
            continue;
        }
        // condim = condim_in[conid]                                                              <L 2806>
        var_10 = wp::address(var_condim_in, var_0);
        var_12 = wp::load(var_10);
        var_11 = wp::copy(var_12);
        // includemargin = includemargin_in[conid]                                                <L 2808>
        var_13 = wp::address(var_includemargin_in, var_0);
        var_15 = wp::load(var_13);
        var_14 = wp::copy(var_15);
        // pos = dist_in[conid] - includemargin                                                   <L 2809>
        var_16 = wp::address(var_dist_in, var_0);
        var_18 = wp::load(var_16);
        var_17 = wp::sub(var_18, var_14);
        // if wp.static(flg_adhesion):                                                            <L 2810>
        // active = pos < 0.0                                                                     <L 2813>
        var_21 = (var_17 < var_20);
        // if not active:                                                                         <L 2815>
        var_22 = wp::unot(var_21);
        if (var_22) {
            // return                                                                             <L 2816>
            continue;
        }
        // if wp.static(IS_ELLIPTIC):                                                             <L 2818>
        // if condim == 1:                                                                        <L 2821>
        var_25 = (var_11 == var_24);
        if (var_25) {
            // ndim = 1                                                                           <L 2822>
        }
        if (!var_25) {
            // ndim = 2 * (condim - 1)                                                            <L 2824>
            var_29 = wp::sub(var_11, var_28);
            var_30 = wp::mul(var_27, var_29);
        }
        var_31 = wp::where(var_25, var_26, var_30);
        // worldid = worldid_in[conid]                                                            <L 2826>
        var_32 = wp::address(var_worldid_in, var_0);
        var_34 = wp::load(var_32);
        var_33 = wp::copy(var_34);
        // base_efcid = wp.atomic_add(nefc_out, worldid, ndim)                                    <L 2829>
        var_35 = wp::atomic_add(var_nefc_out, var_33, var_31);
        // for dim in range(ndim):                                                                <L 2830>
        var_36 = wp::range(var_31);
        start_for_3:;
            if (iter_cmp(var_36) == 0) goto end_for_3;
            var_37 = wp::iter_next(var_36);
            // efcid = base_efcid + dim                                                           <L 2831>
            var_38 = wp::add(var_35, var_37);
            // if efcid >= njmax_in:                                                              <L 2832>
            var_39 = (var_38 >= var_njmax_in);
            if (var_39) {
                // contact_efc_address_out[conid, dim] = -1                                       <L 2833>
                wp::array_store(var_contact_efc_address_out, var_0, var_37, var_40);
            }
            if (!var_39) {
                // contact_efc_address_out[conid, dim] = efcid                                    <L 2835>
                wp::array_store(var_contact_efc_address_out, var_0, var_37, var_38);
                // efc_id_out[worldid, efcid] = conid                                             <L 2837>
                wp::array_store(var_efc_id_out, var_33, var_38, var_0);
            }
            goto start_for_3;
        end_for_3:;
        // if wp.static(is_sparse and newton):                                                    <L 2839>
        // if base_efcid < njmax_in:                                                              <L 2840>
        var_42 = (var_35 < var_njmax_in);
        if (var_42) {
            // jgid = wp.atomic_add(efc_jtdaj_nblock_out, worldid, 1)                             <L 2841>
            var_44 = wp::atomic_add(var_efc_jtdaj_nblock_out, var_33, var_43);
            // efc_jtdaj_adr_out[worldid, jgid] = base_efcid                                      <L 2842>
            wp::array_store(var_efc_jtdaj_adr_out, var_33, var_44, var_35);
            // efc_jtdaj_nrow_out[worldid, jgid] = wp.min(ndim, njmax_in - base_efcid)            <L 2843>
            var_45 = wp::sub(var_njmax_in, var_35);
            var_46 = wp::min(var_31, var_45);
            wp::array_store(var_efc_jtdaj_nrow_out, var_33, var_44, var_46);
        }
        // if wp.static(IS_SPARSE):                                                               <L 2845>
        // geom = geom_in[conid]                                                                  <L 2846>
        var_48 = wp::address(var_geom_in, var_0);
        var_50 = wp::load(var_48);
        var_49 = wp::copy(var_50);
        // body1 = body_weldid[geom_bodyid[geom[0]]]                                              <L 2847>
        var_52 = wp::extract(var_49, var_51);
        var_53 = wp::address(var_geom_bodyid, var_52);
        var_54 = wp::load(var_53);
        var_55 = wp::address(var_body_weldid, var_54);
        var_57 = wp::load(var_55);
        var_56 = wp::copy(var_57);
        // body2 = body_weldid[geom_bodyid[geom[1]]]                                              <L 2848>
        var_59 = wp::extract(var_49, var_58);
        var_60 = wp::address(var_geom_bodyid, var_59);
        var_61 = wp::load(var_60);
        var_62 = wp::address(var_body_weldid, var_61);
        var_64 = wp::load(var_62);
        var_63 = wp::copy(var_64);
        // da1 = int(body_dofadr[body1] + body_dofnum[body1] - 1)                                 <L 2850>
        var_65 = wp::address(var_body_dofadr, var_56);
        var_66 = wp::address(var_body_dofnum, var_56);
        var_68 = wp::load(var_65);
        var_69 = wp::load(var_66);
        var_67 = wp::add(var_68, var_69);
        var_71 = wp::sub(var_67, var_70);
        var_72 = wp::int(var_71);
        // da2 = int(body_dofadr[body2] + body_dofnum[body2] - 1)                                 <L 2851>
        var_73 = wp::address(var_body_dofadr, var_63);
        var_74 = wp::address(var_body_dofnum, var_63);
        var_76 = wp::load(var_73);
        var_77 = wp::load(var_74);
        var_75 = wp::add(var_76, var_77);
        var_79 = wp::sub(var_75, var_78);
        var_80 = wp::int(var_79);
        // rownnz = int(0)                                                                        <L 2854>
        var_82 = wp::int(var_81);
        // while da1 >= 0 or da2 >= 0:                                                            <L 2855>
        start_while_5:;
        var_85 = (var_72 >= var_84);
        var_83 = var_85;
        if (!var_83) {
            var_87 = (var_80 >= var_86);
            var_83 = var_83 || var_87;
        }
        if ((var_83) == false) goto end_while_5;
            // da = wp.max(da1, da2)                                                              <L 2856>
            var_88 = wp::max(var_72, var_80);
            // if da1 == da and da2 == da:                                                        <L 2858>
            var_90 = (var_72 == var_88);
            var_89 = var_90;
            if (var_89) {
                var_91 = (var_80 == var_88);
                var_89 = var_89 && var_91;
            }
            if (var_89) {
                // break                                                                          <L 2859>
                goto end_while_5;
            }
            // if da1 == da:                                                                      <L 2860>
            var_92 = (var_72 == var_88);
            if (var_92) {
                // da1 = dof_parentid[da1]                                                        <L 2861>
                var_93 = wp::address(var_dof_parentid, var_72);
                var_95 = wp::load(var_93);
                var_94 = wp::copy(var_95);
            }
            var_96 = wp::where(var_92, var_94, var_72);
            // if da2 == da:                                                                      <L 2862>
            var_97 = (var_80 == var_88);
            if (var_97) {
                // da2 = dof_parentid[da2]                                                        <L 2863>
                var_98 = wp::address(var_dof_parentid, var_80);
                var_100 = wp::load(var_98);
                var_99 = wp::copy(var_100);
            }
            var_101 = wp::where(var_97, var_99, var_80);
            // rownnz += 1                                                                        <L 2864>
            var_103 = wp::add(var_82, var_102);
            wp::assign(var_72, var_96);
            wp::assign(var_80, var_101);
            wp::assign(var_82, var_103);
        goto start_while_5;
        end_while_5:;
        // rowadr = wp.atomic_add(efc_nnz_out, worldid, rownnz * ndim)                            <L 2866>
        var_104 = wp::mul(var_82, var_31);
        var_105 = wp::atomic_add(var_efc_nnz_out, var_33, var_104);
        // if rowadr + rownnz * ndim > njmax_nnz_in:                                              <L 2867>
        var_106 = wp::mul(var_82, var_31);
        var_107 = wp::add(var_105, var_106);
        var_108 = (var_107 > var_njmax_nnz_in);
        if (var_108) {
            // return                                                                             <L 2868>
            continue;
        }
        // for dim in range(ndim):                                                                <L 2869>
        var_109 = wp::range(var_31);
        start_for_8:;
            if (iter_cmp(var_109) == 0) goto end_for_8;
            var_110 = wp::iter_next(var_109);
            // efcid = base_efcid + dim                                                           <L 2870>
            var_111 = wp::add(var_35, var_110);
            // if efcid < njmax_in:                                                               <L 2871>
            var_112 = (var_111 < var_njmax_in);
            if (var_112) {
                // efc_J_rowadr_out[worldid, efcid] = rowadr + dim * rownnz                       <L 2872>
                var_113 = wp::mul(var_110, var_82);
                var_114 = wp::add(var_105, var_113);
                wp::array_store(var_efc_J_rowadr_out, var_33, var_111, var_114);
                // efc_J_rownnz_out[worldid, efcid] = rownnz                                      <L 2873>
                wp::array_store(var_efc_J_rownnz_out, var_33, var_111, var_82);
            }
            wp::assign(var_38, var_111);
            goto start_for_8;
        end_for_8:;
    }
}

