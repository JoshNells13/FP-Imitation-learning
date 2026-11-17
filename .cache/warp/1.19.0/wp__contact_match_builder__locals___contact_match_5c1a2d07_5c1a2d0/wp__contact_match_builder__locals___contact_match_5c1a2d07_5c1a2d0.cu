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


// C:\Users\ASUS\Documents\human-imitation-learning\.venv\Lib\site-packages\mujoco_warp\_src\util_misc.py:858
static CUDA_CALLABLE bool inside_geom_0(
    wp::vec_t<3, wp::float32> var_pos,
    wp::mat_t<3, 3, wp::float32> var_mat,
    wp::vec_t<3, wp::float32> var_size,
    wp::int32 var_geomtype,
    wp::vec_t<3, wp::float32> var_point)
{
    //---------
    // primal vars
    wp::vec_t<3, wp::float32> var_0;
    const wp::int32 var_1 = 2;
    bool var_2;
    wp::float32 var_3;
    const wp::int32 var_4 = 0;
    wp::float32 var_5;
    const wp::int32 var_6 = 0;
    wp::float32 var_7;
    wp::float32 var_8;
    bool var_9;
    wp::mat_t<3, 3, wp::float32> var_10;
    wp::vec_t<3, wp::float32> var_11;
    const wp::int32 var_12 = 3;
    bool var_13;
    const wp::int32 var_14 = 2;
    wp::float32 var_15;
    const wp::int32 var_16 = 1;
    wp::float32 var_17;
    wp::float32 var_18;
    const wp::int32 var_19 = 1;
    wp::float32 var_20;
    wp::float32 var_21;
    wp::float32 var_22;
    wp::float32 var_23;
    const wp::int32 var_24 = 0;
    wp::float32 var_25;
    const wp::int32 var_26 = 0;
    wp::float32 var_27;
    wp::float32 var_28;
    const wp::int32 var_29 = 1;
    wp::float32 var_30;
    const wp::int32 var_31 = 1;
    wp::float32 var_32;
    wp::float32 var_33;
    wp::float32 var_34;
    wp::float32 var_35;
    const wp::int32 var_36 = 0;
    wp::float32 var_37;
    const wp::int32 var_38 = 0;
    wp::float32 var_39;
    wp::float32 var_40;
    bool var_41;
    const wp::int32 var_42 = 4;
    bool var_43;
    wp::vec_t<3, wp::float32> var_44;
    wp::float32 var_45;
    const wp::float32 var_46 = 1.0;
    bool var_47;
    const wp::int32 var_48 = 5;
    bool var_49;
    bool var_50;
    const wp::int32 var_51 = 2;
    wp::float32 var_52;
    wp::float32 var_53;
    const wp::int32 var_54 = 1;
    wp::float32 var_55;
    bool var_56;
    const wp::int32 var_57 = 0;
    wp::float32 var_58;
    const wp::int32 var_59 = 0;
    wp::float32 var_60;
    wp::float32 var_61;
    const wp::int32 var_62 = 1;
    wp::float32 var_63;
    const wp::int32 var_64 = 1;
    wp::float32 var_65;
    wp::float32 var_66;
    wp::float32 var_67;
    const wp::int32 var_68 = 0;
    wp::float32 var_69;
    const wp::int32 var_70 = 0;
    wp::float32 var_71;
    wp::float32 var_72;
    bool var_73;
    const wp::int32 var_74 = 6;
    bool var_75;
    bool var_76;
    const wp::int32 var_77 = 0;
    wp::float32 var_78;
    wp::float32 var_79;
    const wp::int32 var_80 = 0;
    wp::float32 var_81;
    bool var_82;
    const wp::int32 var_83 = 1;
    wp::float32 var_84;
    wp::float32 var_85;
    const wp::int32 var_86 = 1;
    wp::float32 var_87;
    bool var_88;
    const wp::int32 var_89 = 2;
    wp::float32 var_90;
    wp::float32 var_91;
    const wp::int32 var_92 = 2;
    wp::float32 var_93;
    bool var_94;
    const wp::int32 var_95 = 0;
    bool var_96;
    const wp::int32 var_97 = 2;
    wp::float32 var_98;
    const wp::float32 var_99 = 0.0;
    bool var_100;
    const bool var_101 = false;
    //---------
    // forward
    // def inside_geom(pos: wp.vec3, mat: wp.mat33, size: wp.vec3, geomtype: int, point: wp.vec3) -> bool:       <L 859>
    // vec = point - pos                                                                      <L 862>
    var_0 = wp::sub(var_point, var_pos);
    // if geomtype == GeomType.SPHERE:                                                        <L 865>
    var_2 = (var_geomtype == var_1);
    if (var_2) {
        // return wp.dot(vec, vec) < size[0] * size[0]                                        <L 866>
        var_3 = wp::dot(var_0, var_0);
        var_5 = wp::extract(var_size, var_4);
        var_7 = wp::extract(var_size, var_6);
        var_8 = wp::mul(var_5, var_7);
        var_9 = (var_3 < var_8);
        return var_9;
    }
    // plocal = wp.transpose(mat) @ vec                                                       <L 869>
    var_10 = wp::transpose(var_mat);
    var_11 = wp::mul(var_10, var_0);
    // if geomtype == GeomType.CAPSULE:                                                       <L 872>
    var_13 = (var_geomtype == var_12);
    if (var_13) {
        // z = plocal[2]                                                                      <L 873>
        var_15 = wp::extract(var_11, var_14);
        // z_clamped = wp.clamp(z, -size[1], size[1])                                         <L 874>
        var_17 = wp::extract(var_size, var_16);
        var_18 = wp::neg(var_17);
        var_20 = wp::extract(var_size, var_19);
        var_21 = wp::clamp(var_15, var_18, var_20);
        // z_dif = z - z_clamped                                                              <L 875>
        var_22 = wp::sub(var_15, var_21);
        // z_dist_sq = z_dif * z_dif                                                          <L 876>
        var_23 = wp::mul(var_22, var_22);
        // return plocal[0] * plocal[0] + plocal[1] * plocal[1] + z_dist_sq < size[0] * size[0]       <L 877>
        var_25 = wp::extract(var_11, var_24);
        var_27 = wp::extract(var_11, var_26);
        var_28 = wp::mul(var_25, var_27);
        var_30 = wp::extract(var_11, var_29);
        var_32 = wp::extract(var_11, var_31);
        var_33 = wp::mul(var_30, var_32);
        var_34 = wp::add(var_28, var_33);
        var_35 = wp::add(var_34, var_23);
        var_37 = wp::extract(var_size, var_36);
        var_39 = wp::extract(var_size, var_38);
        var_40 = wp::mul(var_37, var_39);
        var_41 = (var_35 < var_40);
        return var_41;
    }
    if (!var_13) {
        // elif geomtype == GeomType.ELLIPSOID:                                               <L 878>
        var_43 = (var_geomtype == var_42);
        if (var_43) {
            // plocalsize = wp.cw_div(plocal, size)                                           <L 879>
            var_44 = wp::cw_div(var_11, var_size);
            // return wp.dot(plocalsize, plocalsize) < 1.0                                    <L 880>
            var_45 = wp::dot(var_44, var_44);
            var_47 = (var_45 < var_46);
            return var_47;
        }
        if (!var_43) {
            // elif geomtype == GeomType.CYLINDER:                                            <L 881>
            var_49 = (var_geomtype == var_48);
            if (var_49) {
                // return (wp.abs(plocal[2]) < size[1]) and (plocal[0] * plocal[0] + plocal[1] * plocal[1] < size[0] * size[0])       <L 882>
                var_52 = wp::extract(var_11, var_51);
                var_53 = wp::abs(var_52);
                var_55 = wp::extract(var_size, var_54);
                var_56 = (var_53 < var_55);
                var_50 = var_56;
                if (var_50) {
                    var_58 = wp::extract(var_11, var_57);
                    var_60 = wp::extract(var_11, var_59);
                    var_61 = wp::mul(var_58, var_60);
                    var_63 = wp::extract(var_11, var_62);
                    var_65 = wp::extract(var_11, var_64);
                    var_66 = wp::mul(var_63, var_65);
                    var_67 = wp::add(var_61, var_66);
                    var_69 = wp::extract(var_size, var_68);
                    var_71 = wp::extract(var_size, var_70);
                    var_72 = wp::mul(var_69, var_71);
                    var_73 = (var_67 < var_72);
                    var_50 = var_50 && var_73;
                }
                return var_50;
            }
            if (!var_49) {
                // elif geomtype == GeomType.BOX:                                             <L 883>
                var_75 = (var_geomtype == var_74);
                if (var_75) {
                    // return wp.abs(plocal[0]) < size[0] and wp.abs(plocal[1]) < size[1] and wp.abs(plocal[2]) < size[2]       <L 884>
                    var_78 = wp::extract(var_11, var_77);
                    var_79 = wp::abs(var_78);
                    var_81 = wp::extract(var_size, var_80);
                    var_82 = (var_79 < var_81);
                    var_76 = var_82;
                    if (var_76) {
                        var_84 = wp::extract(var_11, var_83);
                        var_85 = wp::abs(var_84);
                        var_87 = wp::extract(var_size, var_86);
                        var_88 = (var_85 < var_87);
                        var_76 = var_76 && var_88;
                    }
                    if (var_76) {
                        var_90 = wp::extract(var_11, var_89);
                        var_91 = wp::abs(var_90);
                        var_93 = wp::extract(var_size, var_92);
                        var_94 = (var_91 < var_93);
                        var_76 = var_76 && var_94;
                    }
                    return var_76;
                }
                if (!var_75) {
                    // elif geomtype == GeomType.PLANE:                                       <L 885>
                    var_96 = (var_geomtype == var_95);
                    if (var_96) {
                        // return plocal[2] < 0.0                                             <L 886>
                        var_98 = wp::extract(var_11, var_97);
                        var_100 = (var_98 < var_99);
                        return var_100;
                    }
                }
            }
        }
    }
    // return False                                                                           <L 888>
    return var_101;
}


// C:\Users\ASUS\Documents\human-imitation-learning\.venv\Lib\site-packages\mujoco_warp\_src\sensor.py:2227
static CUDA_CALLABLE bool _check_match_0(
    wp::array_t<wp::int32> var_body_parentid,
    wp::int32 var_body,
    wp::int32 var_geom,
    wp::int32 var_objtype,
    wp::int32 var_objid)
{
    //---------
    // primal vars
    const wp::int32 var_0 = 0;
    bool var_1;
    const bool var_2 = true;
    const wp::int32 var_3 = 6;
    bool var_4;
    const bool var_5 = true;
    const wp::int32 var_6 = 5;
    bool var_7;
    bool var_8;
    const wp::int32 var_9 = 1;
    bool var_10;
    bool var_11;
    const wp::int32 var_12 = 2;
    bool var_13;
    bool var_14;
    wp::int32* var_15;
    wp::int32 var_16;
    wp::int32 var_17;
    bool var_18;
    const bool var_19 = false;
    //---------
    // forward
    // def _check_match(body_parentid: wp.array[int], body: int, geom: int, objtype: int, objid: int) -> bool:       <L 2228>
    // if objtype == ObjType.UNKNOWN:                                                         <L 2230>
    var_1 = (var_objtype == var_0);
    if (var_1) {
        // return True                                                                        <L 2231>
        return var_2;
    }
    // if objtype == ObjType.SITE:                                                            <L 2232>
    var_4 = (var_objtype == var_3);
    if (var_4) {
        // return True  # already passed site filter test                                     <L 2233>
        return var_5;
    }
    // if objtype == ObjType.GEOM:                                                            <L 2234>
    var_7 = (var_objtype == var_6);
    if (var_7) {
        // return objid == geom                                                               <L 2235>
        var_8 = (var_objid == var_geom);
        return var_8;
    }
    // if objtype == ObjType.BODY:                                                            <L 2236>
    var_10 = (var_objtype == var_9);
    if (var_10) {
        // return objid == body                                                               <L 2237>
        var_11 = (var_objid == var_body);
        return var_11;
    }
    // if objtype == ObjType.XBODY:                                                           <L 2238>
    var_13 = (var_objtype == var_12);
    if (var_13) {
        // while body > objid:                                                                <L 2240>
    start_while_4:;
        var_14 = (var_body > var_objid);
    if ((var_14) == false) goto end_while_4;
            // body = body_parentid[body]                                                     <L 2241>
            var_15 = wp::address(var_body_parentid, var_body);
            var_17 = wp::load(var_15);
            var_16 = wp::copy(var_17);
            wp::assign(var_body, var_16);
    goto start_while_4;
    end_while_4:;
        // return body == objid                                                               <L 2242>
        var_18 = (var_body == var_objid);
        return var_18;
    }
    // return False                                                                           <L 2243>
    return var_19;
}


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


// C:\Users\ASUS\Documents\human-imitation-learning\.venv\Lib\site-packages\mujoco_warp\_src\util_misc.py:858
static CUDA_CALLABLE void adj_inside_geom_0(
    wp::vec_t<3, wp::float32> var_pos,
    wp::mat_t<3, 3, wp::float32> var_mat,
    wp::vec_t<3, wp::float32> var_size,
    wp::int32 var_geomtype,
    wp::vec_t<3, wp::float32> var_point,
    wp::vec_t<3, wp::float32> & adj_pos,
    wp::mat_t<3, 3, wp::float32> & adj_mat,
    wp::vec_t<3, wp::float32> & adj_size,
    wp::int32 & adj_geomtype,
    wp::vec_t<3, wp::float32> & adj_point,
    bool & adj_ret)
{
	// reverse mode disabled (no backward-enabled kernel depends on this function)
}


// C:\Users\ASUS\Documents\human-imitation-learning\.venv\Lib\site-packages\mujoco_warp\_src\sensor.py:2227
static CUDA_CALLABLE void adj__check_match_0(
    wp::array_t<wp::int32> var_body_parentid,
    wp::int32 var_body,
    wp::int32 var_geom,
    wp::int32 var_objtype,
    wp::int32 var_objid,
    wp::array_t<wp::int32> & adj_body_parentid,
    wp::int32 & adj_body,
    wp::int32 & adj_geom,
    wp::int32 & adj_objtype,
    wp::int32 & adj_objid,
    bool & adj_ret)
{
	// reverse mode disabled (no backward-enabled kernel depends on this function)
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



extern "C" __global__ void _contact_match_builder__locals___contact_match_8bbc0bcf_cuda_kernel_forward(
    wp::launch_bounds_t<2> dim,
    wp::int32 var_opt_cone,
    wp::int32 var_opt_contact_sensor_maxmatch,
    wp::array_t<wp::int32> var_body_parentid,
    wp::array_t<wp::int32> var_geom_bodyid,
    wp::array_t<wp::int32> var_site_type,
    wp::array_t<wp::vec_t<3, wp::float32>> var_site_size,
    wp::array_t<wp::int32> var_sensor_objtype,
    wp::array_t<wp::int32> var_sensor_objid,
    wp::array_t<wp::int32> var_sensor_reftype,
    wp::array_t<wp::int32> var_sensor_refid,
    wp::array_t<wp::int32> var_sensor_intprm,
    wp::array_t<wp::vec_t<3, wp::float32>> var_site_xpos_in,
    wp::array_t<wp::mat_t<3, 3, wp::float32>> var_site_xmat_in,
    wp::array_t<wp::float32> var_contact_dist_in,
    wp::array_t<wp::vec_t<3, wp::float32>> var_contact_pos_in,
    wp::array_t<wp::mat_t<3, 3, wp::float32>> var_contact_frame_in,
    wp::array_t<wp::vec_t<5, wp::float32>> var_contact_friction_in,
    wp::array_t<wp::int32> var_contact_dim_in,
    wp::array_t<wp::vec_t<2, wp::int32>> var_contact_geom_in,
    wp::array_t<wp::int32> var_contact_efc_address_in,
    wp::array_t<wp::int32> var_contact_worldid_in,
    wp::array_t<wp::int32> var_contact_type_in,
    wp::array_t<wp::float32> var_contact_adhesion_in,
    wp::array_t<wp::float32> var_efc_force_in,
    wp::int32 var_njmax_in,
    wp::array_t<wp::int32> var_nacon_in,
    wp::array_t<wp::int32> var_sensor_contact_adr_in,
    wp::array_t<wp::int32> var_overflow_out,
    wp::array_t<wp::int32> var_sensor_contact_nmatch_out,
    wp::array_t<wp::int32> var_sensor_contact_matchid_out,
    wp::array_t<wp::float32> var_sensor_contact_criteria_out,
    wp::array_t<wp::float32> var_sensor_contact_direction_out)
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
        wp::int32* var_2;
        wp::int32 var_3;
        wp::int32 var_4;
        const wp::int32 var_5 = 0;
        wp::int32* var_6;
        bool var_7;
        wp::int32 var_8;
        wp::int32* var_9;
        const wp::int32 var_10 = 1;
        wp::int32 var_11;
        wp::int32 var_12;
        bool var_13;
        wp::int32* var_14;
        wp::int32 var_15;
        wp::int32 var_16;
        wp::int32* var_17;
        wp::int32 var_18;
        wp::int32 var_19;
        wp::int32* var_20;
        wp::int32 var_21;
        wp::int32 var_22;
        wp::int32* var_23;
        wp::int32 var_24;
        wp::int32 var_25;
        const wp::int32 var_26 = 1;
        wp::int32* var_27;
        wp::int32 var_28;
        wp::int32 var_29;
        wp::int32* var_30;
        wp::int32 var_31;
        wp::int32 var_32;
        const wp::int32 var_33 = 6;
        bool var_34;
        wp::vec_t<3, wp::float32>* var_35;
        wp::mat_t<3, 3, wp::float32>* var_36;
        wp::vec_t<3, wp::float32>* var_37;
        wp::int32* var_38;
        wp::vec_t<3, wp::float32>* var_39;
        bool var_40;
        wp::vec_t<3, wp::float32> var_41;
        wp::mat_t<3, 3, wp::float32> var_42;
        wp::vec_t<3, wp::float32> var_43;
        wp::int32 var_44;
        wp::vec_t<3, wp::float32> var_45;
        bool var_46;
        bool var_47;
        const wp::int32 var_48 = 0;
        bool var_49;
        const wp::int32 var_50 = 0;
        bool var_51;
        const wp::float32 var_52 = 1.0;
        wp::vec_t<2, wp::int32>* var_53;
        wp::vec_t<2, wp::int32> var_54;
        wp::vec_t<2, wp::int32> var_55;
        const wp::int32 var_56 = 0;
        wp::int32 var_57;
        const wp::int32 var_58 = 1;
        wp::int32 var_59;
        wp::int32* var_60;
        wp::int32 var_61;
        wp::int32 var_62;
        wp::int32* var_63;
        wp::int32 var_64;
        wp::int32 var_65;
        bool var_66;
        bool var_67;
        bool var_68;
        bool var_69;
        bool var_70;
        bool var_71;
        bool var_72;
        bool var_73;
        bool var_74;
        bool var_75;
        const wp::float32 var_76 = 1.0;
        bool var_77;
        const wp::int32 var_78 = 0;
        bool var_79;
        const wp::int32 var_80 = 0;
        bool var_81;
        bool var_82;
        bool var_83;
        bool var_84;
        bool var_85;
        bool var_86;
        bool var_87;
        bool var_88;
        const wp::float32 var_89 = -1.0;
        wp::float32 var_90;
        const wp::int32 var_91 = 0;
        bool var_92;
        bool var_93;
        const wp::float32 var_94 = -1.0;
        wp::float32 var_95;
        const wp::int32 var_96 = 0;
        bool var_97;
        bool var_98;
        const wp::float32 var_99 = -1.0;
        wp::float32 var_100;
        wp::float32 var_101;
        wp::float32 var_102;
        wp::float32 var_103;
        wp::float32 var_104;
        wp::array_t<wp::int32> var_105;
        const wp::int32 var_106 = 1;
        wp::int32 var_107;
        bool var_108;
        const bool var_109 = true;
        const wp::str var_110 = "contact match overflow: please increase Option.contact_sensor_maxmatch beyond %u\nTo disable the print warning: m.opt.warn_overflow &= ~mjw.OverflowType.CONTACT_MATCH (or = 0 for all)\n";
        const wp::int32 var_111 = 64;
        const wp::int32 var_112 = 64;
        wp::int32 var_113;
        const wp::int32 var_114 = 1;
        bool var_115;
        wp::float32* var_116;
        wp::float32 var_117;
        wp::float32 var_118;
        const wp::int32 var_119 = 2;
        bool var_120;
        const bool var_121 = false;
        wp::vec_t<6, wp::float32> var_122;
        const wp::int32 var_123 = 0;
        wp::float32 var_124;
        const wp::int32 var_125 = 0;
        wp::float32 var_126;
        wp::float32 var_127;
        const wp::int32 var_128 = 1;
        wp::float32 var_129;
        const wp::int32 var_130 = 1;
        wp::float32 var_131;
        wp::float32 var_132;
        wp::float32 var_133;
        const wp::int32 var_134 = 2;
        wp::float32 var_135;
        const wp::int32 var_136 = 2;
        wp::float32 var_137;
        wp::float32 var_138;
        wp::float32 var_139;
        wp::float32 var_140;
        //---------
        // forward
        // def _contact_match(                                                                    <L 2249>
        // contactsensorid, contactid = wp.tid()                                                  <L 2288>
        builtin_tid2d(var_0, var_1);
        // sensorid = sensor_contact_adr_in[contactsensorid]                                      <L 2289>
        var_2 = wp::address(var_sensor_contact_adr_in, var_0);
        var_4 = wp::load(var_2);
        var_3 = wp::copy(var_4);
        // if contactid >= nacon_in[0]:                                                           <L 2291>
        var_6 = wp::address(var_nacon_in, var_5);
        var_8 = wp::load(var_6);
        var_7 = (var_1 >= var_8);
        if (var_7) {
            // return                                                                             <L 2292>
            continue;
        }
        // if not (contact_type_in[contactid] & ContactType.CONSTRAINT):                          <L 2294>
        var_9 = wp::address(var_contact_type_in, var_1);
        var_12 = wp::load(var_9);
        var_11 = wp::bit_and(var_12, var_10);
        var_13 = wp::unot(var_11);
        if (var_13) {
            // return                                                                             <L 2295>
            continue;
        }
        // objtype = sensor_objtype[sensorid]                                                     <L 2298>
        var_14 = wp::address(var_sensor_objtype, var_3);
        var_16 = wp::load(var_14);
        var_15 = wp::copy(var_16);
        // objid = sensor_objid[sensorid]                                                         <L 2299>
        var_17 = wp::address(var_sensor_objid, var_3);
        var_19 = wp::load(var_17);
        var_18 = wp::copy(var_19);
        // reftype = sensor_reftype[sensorid]                                                     <L 2300>
        var_20 = wp::address(var_sensor_reftype, var_3);
        var_22 = wp::load(var_20);
        var_21 = wp::copy(var_22);
        // refid = sensor_refid[sensorid]                                                         <L 2301>
        var_23 = wp::address(var_sensor_refid, var_3);
        var_25 = wp::load(var_23);
        var_24 = wp::copy(var_25);
        // reduce = sensor_intprm[sensorid, 1]                                                    <L 2302>
        var_27 = wp::address(var_sensor_intprm, var_3, var_26);
        var_29 = wp::load(var_27);
        var_28 = wp::copy(var_29);
        // worldid = contact_worldid_in[contactid]                                                <L 2304>
        var_30 = wp::address(var_contact_worldid_in, var_1);
        var_32 = wp::load(var_30);
        var_31 = wp::copy(var_32);
        // if objtype == ObjType.SITE:                                                            <L 2307>
        var_34 = (var_15 == var_33);
        if (var_34) {
            // if not inside_geom(                                                                <L 2308>
            // site_xpos_in[worldid, objid],                                                      <L 2309>
            var_35 = wp::address(var_site_xpos_in, var_31, var_18);
            // site_xmat_in[worldid, objid],                                                      <L 2310>
            var_36 = wp::address(var_site_xmat_in, var_31, var_18);
            // site_size[objid],                                                                  <L 2311>
            var_37 = wp::address(var_site_size, var_18);
            // site_type[objid],                                                                  <L 2312>
            var_38 = wp::address(var_site_type, var_18);
            // contact_pos_in[contactid],                                                         <L 2313>
            var_39 = wp::address(var_contact_pos_in, var_1);
            var_41 = wp::load(var_35);
            var_42 = wp::load(var_36);
            var_43 = wp::load(var_37);
            var_44 = wp::load(var_38);
            var_45 = wp::load(var_39);
            var_40 = inside_geom_0(var_41, var_42, var_43, var_44, var_45);
            var_46 = wp::unot(var_40);
            if (var_46) {
                // return                                                                         <L 2315>
                continue;
            }
        }
        // if objtype == ObjType.UNKNOWN and reftype == ObjType.UNKNOWN:                          <L 2318>
        var_49 = (var_15 == var_48);
        var_47 = var_49;
        if (var_47) {
            var_51 = (var_21 == var_50);
            var_47 = var_47 && var_51;
        }
        if (var_47) {
            // dir = 1.0                                                                          <L 2319>
        }
        if (!var_47) {
            // geom = contact_geom_in[contactid]                                                  <L 2322>
            var_53 = wp::address(var_contact_geom_in, var_1);
            var_55 = wp::load(var_53);
            var_54 = wp::copy(var_55);
            // geom1 = geom[0]                                                                    <L 2323>
            var_57 = wp::extract(var_54, var_56);
            // geom2 = geom[1]                                                                    <L 2324>
            var_59 = wp::extract(var_54, var_58);
            // body1 = geom_bodyid[geom1]                                                         <L 2325>
            var_60 = wp::address(var_geom_bodyid, var_57);
            var_62 = wp::load(var_60);
            var_61 = wp::copy(var_62);
            // body2 = geom_bodyid[geom2]                                                         <L 2326>
            var_63 = wp::address(var_geom_bodyid, var_59);
            var_65 = wp::load(var_63);
            var_64 = wp::copy(var_65);
            // match11 = _check_match(body_parentid, body1, geom1, objtype, objid)                <L 2329>
            var_66 = _check_match_0(var_body_parentid, var_61, var_57, var_15, var_18);
            // match12 = _check_match(body_parentid, body2, geom2, objtype, objid)                <L 2330>
            var_67 = _check_match_0(var_body_parentid, var_64, var_59, var_15, var_18);
            // match21 = _check_match(body_parentid, body1, geom1, reftype, refid)                <L 2331>
            var_68 = _check_match_0(var_body_parentid, var_61, var_57, var_21, var_24);
            // match22 = _check_match(body_parentid, body2, geom2, reftype, refid)                <L 2332>
            var_69 = _check_match_0(var_body_parentid, var_64, var_59, var_21, var_24);
            // if not match11 and not match12:                                                    <L 2335>
            var_71 = wp::unot(var_66);
            var_70 = var_71;
            if (var_70) {
                var_72 = wp::unot(var_67);
                var_70 = var_70 && var_72;
            }
            if (var_70) {
                // return                                                                         <L 2336>
                continue;
            }
            // if not match21 and not match22:                                                    <L 2337>
            var_74 = wp::unot(var_68);
            var_73 = var_74;
            if (var_73) {
                var_75 = wp::unot(var_69);
                var_73 = var_73 && var_75;
            }
            if (var_73) {
                // return                                                                         <L 2338>
                continue;
            }
            // dir = 1.0                                                                          <L 2341>
            // if objtype != ObjType.UNKNOWN and reftype != ObjType.UNKNOWN:                      <L 2342>
            var_79 = (var_15 != var_78);
            var_77 = var_79;
            if (var_77) {
                var_81 = (var_21 != var_80);
                var_77 = var_77 && var_81;
            }
            if (var_77) {
                // order_regular = match11 and match22                                            <L 2344>
                var_82 = var_66;
                if (var_82) {
                    var_82 = var_82 && var_69;
                }
                // order_reverse = match12 and match21                                            <L 2345>
                var_83 = var_67;
                if (var_83) {
                    var_83 = var_83 && var_68;
                }
                // if not order_regular and not order_reverse:                                    <L 2346>
                var_85 = wp::unot(var_82);
                var_84 = var_85;
                if (var_84) {
                    var_86 = wp::unot(var_83);
                    var_84 = var_84 && var_86;
                }
                if (var_84) {
                    // return                                                                     <L 2347>
                    continue;
                }
                // if order_reverse and not order_regular:                                        <L 2348>
                var_87 = var_83;
                if (var_87) {
                    var_88 = wp::unot(var_82);
                    var_87 = var_87 && var_88;
                }
                if (var_87) {
                    // dir = -1.0                                                                 <L 2349>
                }
                var_90 = wp::where(var_87, var_89, var_76);
            }
            if (!var_77) {
                // elif objtype != ObjType.UNKNOWN:                                               <L 2350>
                var_92 = (var_15 != var_91);
                if (var_92) {
                    // if not match11:                                                            <L 2351>
                    var_93 = wp::unot(var_66);
                    if (var_93) {
                        // dir = -1.0                                                             <L 2352>
                    }
                    var_95 = wp::where(var_93, var_94, var_76);
                }
                if (!var_92) {
                    // elif reftype != ObjType.UNKNOWN:                                           <L 2353>
                    var_97 = (var_21 != var_96);
                    if (var_97) {
                        // if not match22:                                                        <L 2354>
                        var_98 = wp::unot(var_69);
                        if (var_98) {
                            // dir = -1.0                                                         <L 2355>
                        }
                        var_100 = wp::where(var_98, var_99, var_76);
                    }
                    var_101 = wp::where(var_97, var_100, var_76);
                }
                var_102 = wp::where(var_92, var_95, var_101);
            }
            var_103 = wp::where(var_77, var_90, var_102);
        }
        var_104 = wp::where(var_47, var_52, var_103);
        // contactmatchid = wp.atomic_add(sensor_contact_nmatch_out[worldid], contactsensorid, 1)       <L 2357>
        var_105 = wp::view(var_sensor_contact_nmatch_out, var_31);
        var_107 = wp::atomic_add(var_105, var_0, var_106);
        // if contactmatchid >= opt_contact_sensor_maxmatch:                                      <L 2359>
        var_108 = (var_107 >= var_opt_contact_sensor_maxmatch);
        if (var_108) {
            // if wp.static(bool(warn_overflow & OverflowType.CONTACT_MATCH)):                    <L 2360>
            // wp.printf(                                                                         <L 2361>
            // "contact match overflow: please increase Option.contact_sensor_maxmatch beyond %u\n"       <L 2362>
            // opt_contact_sensor_maxmatch,                                                       <L 2364>
            printf(var_110, var_opt_contact_sensor_maxmatch);
            // wp.atomic_or(overflow_out, worldid, OverflowType.CONTACT_MATCH)                    <L 2366>
            var_113 = wp::atomic_or(var_overflow_out, var_31, var_112);
            // return                                                                             <L 2367>
            continue;
        }
        // sensor_contact_matchid_out[worldid, contactsensorid, contactmatchid] = contactid       <L 2369>
        wp::array_store(var_sensor_contact_matchid_out, var_31, var_0, var_107, var_1);
        // if reduce == 1:  # mindist                                                             <L 2371>
        var_115 = (var_28 == var_114);
        if (var_115) {
            // sensor_contact_criteria_out[worldid, contactsensorid, contactmatchid] = contact_dist_in[contactid]       <L 2372>
            var_116 = wp::address(var_contact_dist_in, var_1);
            var_118 = wp::load(var_116);
            var_117 = wp::copy(var_118);
            wp::array_store(var_sensor_contact_criteria_out, var_31, var_0, var_107, var_117);
        }
        if (!var_115) {
            // elif reduce == 2:  # maxforce                                                      <L 2373>
            var_120 = (var_28 == var_119);
            if (var_120) {
                // contact_force = support.contact_force_fn(                                      <L 2374>
                // opt_cone,                                                                      <L 2375>
                // contact_frame_in,                                                              <L 2376>
                // contact_friction_in,                                                           <L 2377>
                // contact_dim_in,                                                                <L 2378>
                // contact_efc_address_in,                                                        <L 2379>
                // contact_adhesion_in,                                                           <L 2380>
                // efc_force_in,                                                                  <L 2381>
                // njmax_in,                                                                      <L 2382>
                // nacon_in,                                                                      <L 2383>
                // worldid,                                                                       <L 2384>
                // contactid,                                                                     <L 2385>
                // False,                                                                         <L 2386>
                var_122 = contact_force_fn_0(var_opt_cone, var_contact_frame_in, var_contact_friction_in, var_contact_dim_in, var_contact_efc_address_in, var_contact_adhesion_in, var_efc_force_in, var_njmax_in, var_nacon_in, var_31, var_1, var_121);
                // force_magnitude = (                                                            <L 2388>
                // contact_force[0] * contact_force[0] + contact_force[1] * contact_force[1] + contact_force[2] * contact_force[2]       <L 2389>
                var_124 = wp::extract(var_122, var_123);
                var_126 = wp::extract(var_122, var_125);
                var_127 = wp::mul(var_124, var_126);
                var_129 = wp::extract(var_122, var_128);
                var_131 = wp::extract(var_122, var_130);
                var_132 = wp::mul(var_129, var_131);
                var_133 = wp::add(var_127, var_132);
                var_135 = wp::extract(var_122, var_134);
                var_137 = wp::extract(var_122, var_136);
                var_138 = wp::mul(var_135, var_137);
                var_139 = wp::add(var_133, var_138);
                // sensor_contact_criteria_out[worldid, contactsensorid, contactmatchid] = -force_magnitude       <L 2391>
                var_140 = wp::neg(var_139);
                wp::array_store(var_sensor_contact_criteria_out, var_31, var_0, var_107, var_140);
            }
        }
        // sensor_contact_direction_out[worldid, contactsensorid, contactmatchid] = dir           <L 2394>
        wp::array_store(var_sensor_contact_direction_out, var_31, var_0, var_107, var_104);
    }
}

