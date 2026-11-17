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


// C:\Users\ASUS\Documents\human-imitation-learning\.venv\Lib\site-packages\mujoco_warp\_src\math.py:44
static CUDA_CALLABLE wp::vec_t<3, wp::float32> rot_vec_quat_0(
    wp::vec_t<3, wp::float32> var_vec,
    wp::quat_t<wp::float32> var_quat)
{
    //---------
    // primal vars
    const wp::int32 var_0 = 0;
    wp::float32 var_1;
    const wp::int32 var_2 = 1;
    wp::float32 var_3;
    const wp::int32 var_4 = 2;
    wp::float32 var_5;
    const wp::int32 var_6 = 3;
    wp::float32 var_7;
    wp::vec_t<3, wp::float32> var_8;
    const wp::float32 var_9 = 2.0;
    wp::float32 var_10;
    wp::vec_t<3, wp::float32> var_11;
    wp::vec_t<3, wp::float32> var_12;
    wp::float32 var_13;
    wp::float32 var_14;
    wp::float32 var_15;
    wp::vec_t<3, wp::float32> var_16;
    wp::vec_t<3, wp::float32> var_17;
    const wp::float32 var_18 = 2.0;
    wp::float32 var_19;
    wp::vec_t<3, wp::float32> var_20;
    wp::vec_t<3, wp::float32> var_21;
    wp::vec_t<3, wp::float32> var_22;
    //---------
    // forward
    // def rot_vec_quat(vec: wp.vec3, quat: wp.quat) -> wp.vec3:                              <L 45>
    // s, u = quat[0], wp.vec3(quat[1], quat[2], quat[3])                                     <L 46>
    var_1 = wp::extract(var_quat, var_0);
    var_3 = wp::extract(var_quat, var_2);
    var_5 = wp::extract(var_quat, var_4);
    var_7 = wp::extract(var_quat, var_6);
    var_8 = wp::vec_t<3, wp::float32>(var_3, var_5, var_7);
    // r = 2.0 * (wp.dot(u, vec) * u) + (s * s - wp.dot(u, u)) * vec                          <L 47>
    var_10 = wp::dot(var_8, var_vec);
    var_11 = wp::mul(var_10, var_8);
    var_12 = wp::mul(var_9, var_11);
    var_13 = wp::mul(var_1, var_1);
    var_14 = wp::dot(var_8, var_8);
    var_15 = wp::sub(var_13, var_14);
    var_16 = wp::mul(var_15, var_vec);
    var_17 = wp::add(var_12, var_16);
    // r = r + 2.0 * s * wp.cross(u, vec)                                                     <L 48>
    var_19 = wp::mul(var_18, var_1);
    var_20 = wp::cross(var_8, var_vec);
    var_21 = wp::mul(var_19, var_20);
    var_22 = wp::add(var_17, var_21);
    // return r                                                                               <L 49>
    return var_22;
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


// C:\Users\ASUS\Documents\human-imitation-learning\.venv\Lib\site-packages\mujoco_warp\_src\math.py:59
static CUDA_CALLABLE wp::mat_t<3, 3, wp::float32> quat_to_mat_0(
    wp::quat_t<wp::float32> var_quat)
{
    //---------
    // primal vars
    const wp::int32 var_0 = 0;
    wp::float32 var_1;
    const wp::int32 var_2 = 0;
    wp::float32 var_3;
    wp::float32 var_4;
    const wp::int32 var_5 = 0;
    wp::float32 var_6;
    const wp::int32 var_7 = 1;
    wp::float32 var_8;
    wp::float32 var_9;
    const wp::int32 var_10 = 0;
    wp::float32 var_11;
    const wp::int32 var_12 = 2;
    wp::float32 var_13;
    wp::float32 var_14;
    const wp::int32 var_15 = 0;
    wp::float32 var_16;
    const wp::int32 var_17 = 3;
    wp::float32 var_18;
    wp::float32 var_19;
    const wp::int32 var_20 = 1;
    wp::float32 var_21;
    const wp::int32 var_22 = 1;
    wp::float32 var_23;
    wp::float32 var_24;
    const wp::int32 var_25 = 1;
    wp::float32 var_26;
    const wp::int32 var_27 = 2;
    wp::float32 var_28;
    wp::float32 var_29;
    const wp::int32 var_30 = 1;
    wp::float32 var_31;
    const wp::int32 var_32 = 3;
    wp::float32 var_33;
    wp::float32 var_34;
    const wp::int32 var_35 = 2;
    wp::float32 var_36;
    const wp::int32 var_37 = 2;
    wp::float32 var_38;
    wp::float32 var_39;
    const wp::int32 var_40 = 2;
    wp::float32 var_41;
    const wp::int32 var_42 = 3;
    wp::float32 var_43;
    wp::float32 var_44;
    const wp::int32 var_45 = 3;
    wp::float32 var_46;
    const wp::int32 var_47 = 3;
    wp::float32 var_48;
    wp::float32 var_49;
    wp::float32 var_50;
    wp::float32 var_51;
    wp::float32 var_52;
    const wp::float32 var_53 = 2.0;
    wp::float32 var_54;
    wp::float32 var_55;
    const wp::float32 var_56 = 2.0;
    wp::float32 var_57;
    wp::float32 var_58;
    const wp::float32 var_59 = 2.0;
    wp::float32 var_60;
    wp::float32 var_61;
    wp::float32 var_62;
    wp::float32 var_63;
    wp::float32 var_64;
    const wp::float32 var_65 = 2.0;
    wp::float32 var_66;
    wp::float32 var_67;
    const wp::float32 var_68 = 2.0;
    wp::float32 var_69;
    wp::float32 var_70;
    const wp::float32 var_71 = 2.0;
    wp::float32 var_72;
    wp::float32 var_73;
    wp::float32 var_74;
    wp::float32 var_75;
    wp::float32 var_76;
    wp::mat_t<3, 3, wp::float32> var_77;
    //---------
    // forward
    // def quat_to_mat(quat: wp.quat) -> wp.mat33:                                            <L 60>
    // q00 = quat[0] * quat[0]                                                                <L 62>
    var_1 = wp::extract(var_quat, var_0);
    var_3 = wp::extract(var_quat, var_2);
    var_4 = wp::mul(var_1, var_3);
    // q01 = quat[0] * quat[1]                                                                <L 63>
    var_6 = wp::extract(var_quat, var_5);
    var_8 = wp::extract(var_quat, var_7);
    var_9 = wp::mul(var_6, var_8);
    // q02 = quat[0] * quat[2]                                                                <L 64>
    var_11 = wp::extract(var_quat, var_10);
    var_13 = wp::extract(var_quat, var_12);
    var_14 = wp::mul(var_11, var_13);
    // q03 = quat[0] * quat[3]                                                                <L 65>
    var_16 = wp::extract(var_quat, var_15);
    var_18 = wp::extract(var_quat, var_17);
    var_19 = wp::mul(var_16, var_18);
    // q11 = quat[1] * quat[1]                                                                <L 66>
    var_21 = wp::extract(var_quat, var_20);
    var_23 = wp::extract(var_quat, var_22);
    var_24 = wp::mul(var_21, var_23);
    // q12 = quat[1] * quat[2]                                                                <L 67>
    var_26 = wp::extract(var_quat, var_25);
    var_28 = wp::extract(var_quat, var_27);
    var_29 = wp::mul(var_26, var_28);
    // q13 = quat[1] * quat[3]                                                                <L 68>
    var_31 = wp::extract(var_quat, var_30);
    var_33 = wp::extract(var_quat, var_32);
    var_34 = wp::mul(var_31, var_33);
    // q22 = quat[2] * quat[2]                                                                <L 69>
    var_36 = wp::extract(var_quat, var_35);
    var_38 = wp::extract(var_quat, var_37);
    var_39 = wp::mul(var_36, var_38);
    // q23 = quat[2] * quat[3]                                                                <L 70>
    var_41 = wp::extract(var_quat, var_40);
    var_43 = wp::extract(var_quat, var_42);
    var_44 = wp::mul(var_41, var_43);
    // q33 = quat[3] * quat[3]                                                                <L 71>
    var_46 = wp::extract(var_quat, var_45);
    var_48 = wp::extract(var_quat, var_47);
    var_49 = wp::mul(var_46, var_48);
    // return wp.mat33(                                                                       <L 73>
    // q00 + q11 - q22 - q33,                                                                 <L 74>
    var_50 = wp::add(var_4, var_24);
    var_51 = wp::sub(var_50, var_39);
    var_52 = wp::sub(var_51, var_49);
    // 2.0 * (q12 - q03),                                                                     <L 75>
    var_54 = wp::sub(var_29, var_19);
    var_55 = wp::mul(var_53, var_54);
    // 2.0 * (q13 + q02),                                                                     <L 76>
    var_57 = wp::add(var_34, var_14);
    var_58 = wp::mul(var_56, var_57);
    // 2.0 * (q12 + q03),                                                                     <L 77>
    var_60 = wp::add(var_29, var_19);
    var_61 = wp::mul(var_59, var_60);
    // q00 - q11 + q22 - q33,                                                                 <L 78>
    var_62 = wp::sub(var_4, var_24);
    var_63 = wp::add(var_62, var_39);
    var_64 = wp::sub(var_63, var_49);
    // 2.0 * (q23 - q01),                                                                     <L 79>
    var_66 = wp::sub(var_44, var_9);
    var_67 = wp::mul(var_65, var_66);
    // 2.0 * (q13 - q02),                                                                     <L 80>
    var_69 = wp::sub(var_34, var_14);
    var_70 = wp::mul(var_68, var_69);
    // 2.0 * (q23 + q01),                                                                     <L 81>
    var_72 = wp::add(var_44, var_9);
    var_73 = wp::mul(var_71, var_72);
    // q00 - q11 - q22 + q33,                                                                 <L 82>
    var_74 = wp::sub(var_4, var_24);
    var_75 = wp::sub(var_74, var_39);
    var_76 = wp::add(var_75, var_49);
    var_77 = wp::mat_t<3, 3, wp::float32>(var_52, var_55, var_58, var_61, var_64, var_67, var_70, var_73, var_76);
    return var_77;
}


// C:\Users\ASUS\Documents\human-imitation-learning\.venv\Lib\site-packages\mujoco_warp\_src\math.py:133
static CUDA_CALLABLE wp::vec_t<6, wp::float32> motion_cross_0(
    wp::vec_t<6, wp::float32> var_u,
    wp::vec_t<6, wp::float32> var_v)
{
    //---------
    // primal vars
    const wp::int32 var_0 = 0;
    wp::float32 var_1;
    const wp::int32 var_2 = 1;
    wp::float32 var_3;
    const wp::int32 var_4 = 2;
    wp::float32 var_5;
    wp::vec_t<3, wp::float32> var_6;
    const wp::int32 var_7 = 3;
    wp::float32 var_8;
    const wp::int32 var_9 = 4;
    wp::float32 var_10;
    const wp::int32 var_11 = 5;
    wp::float32 var_12;
    wp::vec_t<3, wp::float32> var_13;
    const wp::int32 var_14 = 0;
    wp::float32 var_15;
    const wp::int32 var_16 = 1;
    wp::float32 var_17;
    const wp::int32 var_18 = 2;
    wp::float32 var_19;
    wp::vec_t<3, wp::float32> var_20;
    const wp::int32 var_21 = 3;
    wp::float32 var_22;
    const wp::int32 var_23 = 4;
    wp::float32 var_24;
    const wp::int32 var_25 = 5;
    wp::float32 var_26;
    wp::vec_t<3, wp::float32> var_27;
    wp::vec_t<3, wp::float32> var_28;
    wp::vec_t<3, wp::float32> var_29;
    wp::vec_t<3, wp::float32> var_30;
    wp::vec_t<3, wp::float32> var_31;
    wp::vec_t<6, wp::float32> var_32;
    //---------
    // forward
    // def motion_cross(u: wp.spatial_vector, v: wp.spatial_vector) -> wp.spatial_vector:       <L 134>
    // u0 = wp.vec3(u[0], u[1], u[2])                                                         <L 136>
    var_1 = wp::extract(var_u, var_0);
    var_3 = wp::extract(var_u, var_2);
    var_5 = wp::extract(var_u, var_4);
    var_6 = wp::vec_t<3, wp::float32>(var_1, var_3, var_5);
    // u1 = wp.vec3(u[3], u[4], u[5])                                                         <L 137>
    var_8 = wp::extract(var_u, var_7);
    var_10 = wp::extract(var_u, var_9);
    var_12 = wp::extract(var_u, var_11);
    var_13 = wp::vec_t<3, wp::float32>(var_8, var_10, var_12);
    // v0 = wp.vec3(v[0], v[1], v[2])                                                         <L 138>
    var_15 = wp::extract(var_v, var_14);
    var_17 = wp::extract(var_v, var_16);
    var_19 = wp::extract(var_v, var_18);
    var_20 = wp::vec_t<3, wp::float32>(var_15, var_17, var_19);
    // v1 = wp.vec3(v[3], v[4], v[5])                                                         <L 139>
    var_22 = wp::extract(var_v, var_21);
    var_24 = wp::extract(var_v, var_23);
    var_26 = wp::extract(var_v, var_25);
    var_27 = wp::vec_t<3, wp::float32>(var_22, var_24, var_26);
    // ang = wp.cross(u0, v0)                                                                 <L 141>
    var_28 = wp::cross(var_6, var_20);
    // vel = wp.cross(u1, v0) + wp.cross(u0, v1)                                              <L 142>
    var_29 = wp::cross(var_13, var_20);
    var_30 = wp::cross(var_6, var_27);
    var_31 = wp::add(var_29, var_30);
    // return wp.spatial_vector(ang, vel)                                                     <L 144>
    var_32 = wp::vec_t<6, wp::float32>(var_28, var_31);
    return var_32;
}


// C:\Users\ASUS\Documents\human-imitation-learning\.venv\Lib\site-packages\mujoco_warp\_src\math.py:120
static CUDA_CALLABLE wp::vec_t<6, wp::float32> inert_vec_0(
    wp::vec_t<10, wp::float32> var_i,
    wp::vec_t<6, wp::float32> var_v)
{
    //---------
    // primal vars
    const wp::int32 var_0 = 0;
    wp::float32 var_1;
    const wp::int32 var_2 = 0;
    wp::float32 var_3;
    wp::float32 var_4;
    const wp::int32 var_5 = 3;
    wp::float32 var_6;
    const wp::int32 var_7 = 1;
    wp::float32 var_8;
    wp::float32 var_9;
    wp::float32 var_10;
    const wp::int32 var_11 = 4;
    wp::float32 var_12;
    const wp::int32 var_13 = 2;
    wp::float32 var_14;
    wp::float32 var_15;
    wp::float32 var_16;
    const wp::int32 var_17 = 8;
    wp::float32 var_18;
    const wp::int32 var_19 = 4;
    wp::float32 var_20;
    wp::float32 var_21;
    wp::float32 var_22;
    const wp::int32 var_23 = 7;
    wp::float32 var_24;
    const wp::int32 var_25 = 5;
    wp::float32 var_26;
    wp::float32 var_27;
    wp::float32 var_28;
    const wp::int32 var_29 = 3;
    wp::float32 var_30;
    const wp::int32 var_31 = 0;
    wp::float32 var_32;
    wp::float32 var_33;
    const wp::int32 var_34 = 1;
    wp::float32 var_35;
    const wp::int32 var_36 = 1;
    wp::float32 var_37;
    wp::float32 var_38;
    wp::float32 var_39;
    const wp::int32 var_40 = 5;
    wp::float32 var_41;
    const wp::int32 var_42 = 2;
    wp::float32 var_43;
    wp::float32 var_44;
    wp::float32 var_45;
    const wp::int32 var_46 = 8;
    wp::float32 var_47;
    const wp::int32 var_48 = 3;
    wp::float32 var_49;
    wp::float32 var_50;
    wp::float32 var_51;
    const wp::int32 var_52 = 6;
    wp::float32 var_53;
    const wp::int32 var_54 = 5;
    wp::float32 var_55;
    wp::float32 var_56;
    wp::float32 var_57;
    const wp::int32 var_58 = 4;
    wp::float32 var_59;
    const wp::int32 var_60 = 0;
    wp::float32 var_61;
    wp::float32 var_62;
    const wp::int32 var_63 = 5;
    wp::float32 var_64;
    const wp::int32 var_65 = 1;
    wp::float32 var_66;
    wp::float32 var_67;
    wp::float32 var_68;
    const wp::int32 var_69 = 2;
    wp::float32 var_70;
    const wp::int32 var_71 = 2;
    wp::float32 var_72;
    wp::float32 var_73;
    wp::float32 var_74;
    const wp::int32 var_75 = 7;
    wp::float32 var_76;
    const wp::int32 var_77 = 3;
    wp::float32 var_78;
    wp::float32 var_79;
    wp::float32 var_80;
    const wp::int32 var_81 = 6;
    wp::float32 var_82;
    const wp::int32 var_83 = 4;
    wp::float32 var_84;
    wp::float32 var_85;
    wp::float32 var_86;
    const wp::int32 var_87 = 8;
    wp::float32 var_88;
    const wp::int32 var_89 = 1;
    wp::float32 var_90;
    wp::float32 var_91;
    const wp::int32 var_92 = 7;
    wp::float32 var_93;
    const wp::int32 var_94 = 2;
    wp::float32 var_95;
    wp::float32 var_96;
    wp::float32 var_97;
    const wp::int32 var_98 = 9;
    wp::float32 var_99;
    const wp::int32 var_100 = 3;
    wp::float32 var_101;
    wp::float32 var_102;
    wp::float32 var_103;
    const wp::int32 var_104 = 6;
    wp::float32 var_105;
    const wp::int32 var_106 = 2;
    wp::float32 var_107;
    wp::float32 var_108;
    const wp::int32 var_109 = 8;
    wp::float32 var_110;
    const wp::int32 var_111 = 0;
    wp::float32 var_112;
    wp::float32 var_113;
    wp::float32 var_114;
    const wp::int32 var_115 = 9;
    wp::float32 var_116;
    const wp::int32 var_117 = 4;
    wp::float32 var_118;
    wp::float32 var_119;
    wp::float32 var_120;
    const wp::int32 var_121 = 7;
    wp::float32 var_122;
    const wp::int32 var_123 = 0;
    wp::float32 var_124;
    wp::float32 var_125;
    const wp::int32 var_126 = 6;
    wp::float32 var_127;
    const wp::int32 var_128 = 1;
    wp::float32 var_129;
    wp::float32 var_130;
    wp::float32 var_131;
    const wp::int32 var_132 = 9;
    wp::float32 var_133;
    const wp::int32 var_134 = 5;
    wp::float32 var_135;
    wp::float32 var_136;
    wp::float32 var_137;
    wp::vec_t<6, wp::float32> var_138;
    //---------
    // forward
    // def inert_vec(i: types.vec10, v: wp.spatial_vector) -> wp.spatial_vector:              <L 121>
    // return wp.spatial_vector(                                                              <L 123>
    // i[0] * v[0] + i[3] * v[1] + i[4] * v[2] - i[8] * v[4] + i[7] * v[5],                   <L 124>
    var_1 = wp::extract(var_i, var_0);
    var_3 = wp::extract(var_v, var_2);
    var_4 = wp::mul(var_1, var_3);
    var_6 = wp::extract(var_i, var_5);
    var_8 = wp::extract(var_v, var_7);
    var_9 = wp::mul(var_6, var_8);
    var_10 = wp::add(var_4, var_9);
    var_12 = wp::extract(var_i, var_11);
    var_14 = wp::extract(var_v, var_13);
    var_15 = wp::mul(var_12, var_14);
    var_16 = wp::add(var_10, var_15);
    var_18 = wp::extract(var_i, var_17);
    var_20 = wp::extract(var_v, var_19);
    var_21 = wp::mul(var_18, var_20);
    var_22 = wp::sub(var_16, var_21);
    var_24 = wp::extract(var_i, var_23);
    var_26 = wp::extract(var_v, var_25);
    var_27 = wp::mul(var_24, var_26);
    var_28 = wp::add(var_22, var_27);
    // i[3] * v[0] + i[1] * v[1] + i[5] * v[2] + i[8] * v[3] - i[6] * v[5],                   <L 125>
    var_30 = wp::extract(var_i, var_29);
    var_32 = wp::extract(var_v, var_31);
    var_33 = wp::mul(var_30, var_32);
    var_35 = wp::extract(var_i, var_34);
    var_37 = wp::extract(var_v, var_36);
    var_38 = wp::mul(var_35, var_37);
    var_39 = wp::add(var_33, var_38);
    var_41 = wp::extract(var_i, var_40);
    var_43 = wp::extract(var_v, var_42);
    var_44 = wp::mul(var_41, var_43);
    var_45 = wp::add(var_39, var_44);
    var_47 = wp::extract(var_i, var_46);
    var_49 = wp::extract(var_v, var_48);
    var_50 = wp::mul(var_47, var_49);
    var_51 = wp::add(var_45, var_50);
    var_53 = wp::extract(var_i, var_52);
    var_55 = wp::extract(var_v, var_54);
    var_56 = wp::mul(var_53, var_55);
    var_57 = wp::sub(var_51, var_56);
    // i[4] * v[0] + i[5] * v[1] + i[2] * v[2] - i[7] * v[3] + i[6] * v[4],                   <L 126>
    var_59 = wp::extract(var_i, var_58);
    var_61 = wp::extract(var_v, var_60);
    var_62 = wp::mul(var_59, var_61);
    var_64 = wp::extract(var_i, var_63);
    var_66 = wp::extract(var_v, var_65);
    var_67 = wp::mul(var_64, var_66);
    var_68 = wp::add(var_62, var_67);
    var_70 = wp::extract(var_i, var_69);
    var_72 = wp::extract(var_v, var_71);
    var_73 = wp::mul(var_70, var_72);
    var_74 = wp::add(var_68, var_73);
    var_76 = wp::extract(var_i, var_75);
    var_78 = wp::extract(var_v, var_77);
    var_79 = wp::mul(var_76, var_78);
    var_80 = wp::sub(var_74, var_79);
    var_82 = wp::extract(var_i, var_81);
    var_84 = wp::extract(var_v, var_83);
    var_85 = wp::mul(var_82, var_84);
    var_86 = wp::add(var_80, var_85);
    // i[8] * v[1] - i[7] * v[2] + i[9] * v[3],                                               <L 127>
    var_88 = wp::extract(var_i, var_87);
    var_90 = wp::extract(var_v, var_89);
    var_91 = wp::mul(var_88, var_90);
    var_93 = wp::extract(var_i, var_92);
    var_95 = wp::extract(var_v, var_94);
    var_96 = wp::mul(var_93, var_95);
    var_97 = wp::sub(var_91, var_96);
    var_99 = wp::extract(var_i, var_98);
    var_101 = wp::extract(var_v, var_100);
    var_102 = wp::mul(var_99, var_101);
    var_103 = wp::add(var_97, var_102);
    // i[6] * v[2] - i[8] * v[0] + i[9] * v[4],                                               <L 128>
    var_105 = wp::extract(var_i, var_104);
    var_107 = wp::extract(var_v, var_106);
    var_108 = wp::mul(var_105, var_107);
    var_110 = wp::extract(var_i, var_109);
    var_112 = wp::extract(var_v, var_111);
    var_113 = wp::mul(var_110, var_112);
    var_114 = wp::sub(var_108, var_113);
    var_116 = wp::extract(var_i, var_115);
    var_118 = wp::extract(var_v, var_117);
    var_119 = wp::mul(var_116, var_118);
    var_120 = wp::add(var_114, var_119);
    // i[7] * v[0] - i[6] * v[1] + i[9] * v[5],                                               <L 129>
    var_122 = wp::extract(var_i, var_121);
    var_124 = wp::extract(var_v, var_123);
    var_125 = wp::mul(var_122, var_124);
    var_127 = wp::extract(var_i, var_126);
    var_129 = wp::extract(var_v, var_128);
    var_130 = wp::mul(var_127, var_129);
    var_131 = wp::sub(var_125, var_130);
    var_133 = wp::extract(var_i, var_132);
    var_135 = wp::extract(var_v, var_134);
    var_136 = wp::mul(var_133, var_135);
    var_137 = wp::add(var_131, var_136);
    var_138 = wp::vec_t<6, wp::float32>({var_28, var_57, var_86, var_103, var_120, var_137});
    return var_138;
}


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


// C:\Users\ASUS\Documents\human-imitation-learning\.venv\Lib\site-packages\warp\_src\math.py:0
static CUDA_CALLABLE wp::float32 norm_l2_0(
    wp::vec_t<3, wp::float32> var_v)
{
    //---------
    // primal vars
    wp::float32 var_0;
    //---------
    // forward
    // def norm_l2(v: Any) -> float:                                                          <L 1>
    // return wp.length(v)                                                                    <L 12>
    var_0 = wp::length(var_v);
    return var_0;
}


// C:\Users\ASUS\Documents\human-imitation-learning\.venv\Lib\site-packages\mujoco_warp\_src\math.py:161
static CUDA_CALLABLE wp::vec_t<3, wp::float32> quat_to_vel_0(
    wp::quat_t<wp::float32> var_quat)
{
    //---------
    // primal vars
    const wp::int32 var_0 = 1;
    wp::float32 var_1;
    const wp::int32 var_2 = 2;
    wp::float32 var_3;
    const wp::int32 var_4 = 3;
    wp::float32 var_5;
    wp::vec_t<3, wp::float32> var_6;
    wp::float32 var_7;
    const wp::float32 var_8 = 0.0;
    bool var_9;
    const wp::float32 var_10 = 0.0;
    wp::vec_t<3, wp::float32> var_11;
    const wp::float32 var_12 = 2.0;
    const wp::int32 var_13 = 0;
    wp::float32 var_14;
    wp::float32 var_15;
    wp::float32 var_16;
    const wp::float32 var_17 = 3.141592653589793;
    bool var_18;
    const wp::float32 var_19 = 2.0;
    const wp::float32 var_20 = 3.141592653589793;
    wp::float32 var_21;
    wp::float32 var_22;
    wp::float32 var_23;
    wp::vec_t<3, wp::float32> var_24;
    wp::vec_t<3, wp::float32> var_25;
    //---------
    // forward
    // def quat_to_vel(quat: wp.quat) -> wp.vec3:                                             <L 162>
    // axis = wp.vec3(quat[1], quat[2], quat[3])                                              <L 163>
    var_1 = wp::extract(var_quat, var_0);
    var_3 = wp::extract(var_quat, var_2);
    var_5 = wp::extract(var_quat, var_4);
    var_6 = wp::vec_t<3, wp::float32>(var_1, var_3, var_5);
    // sin_a_2 = wp.norm_l2(axis)                                                             <L 164>
    var_7 = norm_l2_0(var_6);
    // if sin_a_2 == 0.0:                                                                     <L 166>
    var_9 = (var_7 == var_8);
    if (var_9) {
        // return wp.vec3(0.0)                                                                <L 167>
        var_11 = wp::vec_t<3, wp::float32>(var_10);
        return var_11;
    }
    // speed = 2.0 * wp.atan2(sin_a_2, quat[0])                                               <L 169>
    var_14 = wp::extract(var_quat, var_13);
    var_15 = wp::atan2(var_7, var_14);
    var_16 = wp::mul(var_12, var_15);
    // if speed > wp.pi:                                                                      <L 171>
    var_18 = (var_16 > var_17);
    if (var_18) {
        // speed -= 2.0 * wp.pi                                                               <L 172>
        var_21 = wp::mul(var_19, var_20);
        var_22 = wp::sub(var_16, var_21);
    }
    var_23 = wp::where(var_18, var_22, var_16);
    // return axis * speed / sin_a_2                                                          <L 174>
    var_24 = wp::mul(var_6, var_23);
    var_25 = wp::div(var_24, var_7);
    return var_25;
}


// C:\Users\ASUS\Documents\human-imitation-learning\.venv\Lib\site-packages\mujoco_warp\_src\math.py:0
static CUDA_CALLABLE wp::float32 safe_div_0(
    wp::float32 var_x,
    wp::float32 var_y)
{
    //---------
    // primal vars
    const wp::float32 var_0 = 0.0;
    bool var_1;
    const wp::float32 var_2 = 1e-15;
    const wp::float32 var_3 = 1e-15;
    wp::float32 var_4;
    wp::float32 var_5;
    //---------
    // forward
    // def safe_div(x: Any, y: Any) -> Any:                                                   <L 1>
    // return x / wp.where(y != 0.0, y, types.MJ_MINVAL)                                      <L 2>
    var_1 = (var_y != var_0);
    var_4 = wp::where(var_1, var_y, var_3);
    var_5 = wp::div(var_x, var_4);
    return var_5;
}


// C:\Users\ASUS\Documents\human-imitation-learning\.venv\Lib\site-packages\mujoco_warp\_src\math.py:0
static CUDA_CALLABLE wp::vec_t<3, wp::float32> safe_div_0(
    wp::vec_t<3, wp::float32> var_x,
    wp::float32 var_y)
{
    //---------
    // primal vars
    const wp::float32 var_0 = 0.0;
    bool var_1;
    const wp::float32 var_2 = 1e-15;
    const wp::float32 var_3 = 1e-15;
    wp::float32 var_4;
    wp::vec_t<3, wp::float32> var_5;
    //---------
    // forward
    // def safe_div(x: Any, y: Any) -> Any:                                                   <L 1>
    // return x / wp.where(y != 0.0, y, types.MJ_MINVAL)                                      <L 2>
    var_1 = (var_y != var_0);
    var_4 = wp::where(var_1, var_y, var_3);
    var_5 = wp::div(var_x, var_4);
    return var_5;
}


// C:\Users\ASUS\Documents\human-imitation-learning\.venv\Lib\site-packages\mujoco_warp\_src\support.py:499
static CUDA_CALLABLE void jac_dof_0(
    wp::array_t<wp::int32> var_body_parentid,
    wp::array_t<wp::int32> var_body_rootid,
    wp::array_t<wp::int32> var_dof_bodyid,
    wp::array_t<wp::int32> var_body_isdofancestor,
    wp::array_t<wp::vec_t<3, wp::float32>> var_subtree_com_in,
    wp::array_t<wp::vec_t<6, wp::float32>> var_cdof_in,
    wp::vec_t<3, wp::float32> var_point,
    wp::int32 var_bodyid,
    wp::int32 var_dofid,
    wp::int32 var_worldid,
    wp::vec_t<3, wp::float32> & ret_0,
    wp::vec_t<3, wp::float32> & ret_1)
{
    //---------
    // primal vars
    wp::int32* var_0;
    const wp::int32 var_1 = 0;
    bool var_2;
    wp::int32 var_3;
    const wp::float32 var_4 = 0.0;
    wp::vec_t<3, wp::float32> var_5;
    const wp::float32 var_6 = 0.0;
    wp::vec_t<3, wp::float32> var_7;
    wp::int32* var_8;
    wp::int32 var_9;
    wp::vec_t<3, wp::float32>* var_10;
    wp::vec_t<3, wp::float32> var_11;
    wp::vec_t<3, wp::float32> var_12;
    wp::vec_t<3, wp::float32> var_13;
    wp::vec_t<6, wp::float32>* var_14;
    wp::vec_t<6, wp::float32> var_15;
    wp::vec_t<6, wp::float32> var_16;
    wp::vec_t<3, wp::float32> var_17;
    wp::vec_t<3, wp::float32> var_18;
    wp::vec_t<3, wp::float32> var_19;
    wp::vec_t<3, wp::float32> var_20;
    wp::vec_t<3, wp::float32> var_21;
    //---------
    // forward
    // def jac_dof(                                                                           <L 500>
    // if body_isdofancestor[bodyid, dofid] == 0:                                             <L 515>
    var_0 = wp::address(var_body_isdofancestor, var_bodyid, var_dofid);
    var_3 = wp::load(var_0);
    var_2 = (var_3 == var_1);
    if (var_2) {
        // return wp.vec3(0.0), wp.vec3(0.0)                                                  <L 516>
        var_5 = wp::vec_t<3, wp::float32>(var_4);
        var_7 = wp::vec_t<3, wp::float32>(var_6);
        ret_0 = var_5;
        ret_1 = var_7;
        return;
    }
    // offset = point - wp.vec3(subtree_com_in[worldid, body_rootid[bodyid]])                 <L 518>
    var_8 = wp::address(var_body_rootid, var_bodyid);
    var_9 = wp::load(var_8);
    var_10 = wp::address(var_subtree_com_in, var_worldid, var_9);
    var_12 = wp::load(var_10);
    var_11 = wp::vec_t<3, wp::float32>(var_12);
    var_13 = wp::sub(var_point, var_11);
    // cdof = cdof_in[worldid, dofid]                                                         <L 520>
    var_14 = wp::address(var_cdof_in, var_worldid, var_dofid);
    var_16 = wp::load(var_14);
    var_15 = wp::copy(var_16);
    // cdof_ang = wp.spatial_top(cdof)                                                        <L 521>
    var_17 = wp::spatial_top(var_15);
    // cdof_lin = wp.spatial_bottom(cdof)                                                     <L 522>
    var_18 = wp::spatial_bottom(var_15);
    // jacp = cdof_lin + wp.cross(cdof_ang, offset)                                           <L 524>
    var_19 = wp::cross(var_17, var_13);
    var_20 = wp::add(var_18, var_19);
    // jacr = cdof_ang                                                                        <L 525>
    var_21 = wp::copy(var_17);
    // return jacp, jacr                                                                      <L 527>
    ret_0 = var_20;
    ret_1 = var_21;
    return;
}


// C:\Users\ASUS\Documents\human-imitation-learning\.venv\Lib\site-packages\mujoco_warp\_src\math.py:177
static CUDA_CALLABLE wp::vec_t<3, wp::float32> quat_sub_0(
    wp::quat_t<wp::float32> var_qa,
    wp::quat_t<wp::float32> var_qb)
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
    wp::quat_t<wp::float32> var_12;
    wp::vec_t<3, wp::float32> var_13;
    //---------
    // forward
    // def quat_sub(qa: wp.quat, qb: wp.quat) -> wp.vec3:                                     <L 178>
    // qneg = wp.quat(qb[0], -qb[1], -qb[2], -qb[3])                                          <L 181>
    var_1 = wp::extract(var_qb, var_0);
    var_3 = wp::extract(var_qb, var_2);
    var_4 = wp::neg(var_3);
    var_6 = wp::extract(var_qb, var_5);
    var_7 = wp::neg(var_6);
    var_9 = wp::extract(var_qb, var_8);
    var_10 = wp::neg(var_9);
    var_11 = wp::quat_t<wp::float32>(var_1, var_4, var_7, var_10);
    // qdif = mul_quat(qneg, qa)                                                              <L 182>
    var_12 = mul_quat_0(var_11, var_qa);
    // return quat_to_vel(qdif)                                                               <L 185>
    var_13 = quat_to_vel_0(var_12);
    return var_13;
}


// C:\Users\ASUS\Documents\human-imitation-learning\.venv\Lib\site-packages\mujoco_warp\_src\util_misc.py:891
static CUDA_CALLABLE wp::float32 _poly_force_0(
    wp::float32 var_linear,
    wp::vec_t<2, wp::float32> var_poly,
    wp::float32 var_x,
    wp::int32 var_flg_odd)
{
    //---------
    // primal vars
    const wp::int32 var_0 = 1;
    bool var_1;
    wp::float32 var_2;
    wp::float32 var_3;
    wp::float32 var_4;
    const wp::int32 var_5 = 0;
    wp::float32 var_6;
    wp::float32 var_7;
    wp::float32 var_8;
    const wp::int32 var_9 = 1;
    wp::float32 var_10;
    wp::float32 var_11;
    wp::float32 var_12;
    wp::float32 var_13;
    //---------
    // forward
    // def _poly_force(linear: float, poly: wp.vec2, x: float, flg_odd: int) -> float:        <L 892>
    // x_val = wp.where(flg_odd == 1, wp.abs(x), x)                                           <L 893>
    var_1 = (var_flg_odd == var_0);
    var_2 = wp::abs(var_x);
    var_3 = wp::where(var_1, var_2, var_x);
    // res = linear                                                                           <L 894>
    var_4 = wp::copy(var_linear);
    // res += poly[0] * x_val                                                                 <L 895>
    var_6 = wp::extract(var_poly, var_5);
    var_7 = wp::mul(var_6, var_3);
    var_8 = wp::add(var_4, var_7);
    // res += poly[1] * x_val * x_val                                                         <L 896>
    var_10 = wp::extract(var_poly, var_9);
    var_11 = wp::mul(var_10, var_3);
    var_12 = wp::mul(var_11, var_3);
    var_13 = wp::add(var_8, var_12);
    // return res                                                                             <L 897>
    return var_13;
}


// C:\Users\ASUS\Documents\human-imitation-learning\.venv\Lib\site-packages\mujoco_warp\_src\math.py:0
static CUDA_CALLABLE void normalize_with_norm_0(
    wp::vec_t<3, wp::float32> var_x,
    wp::vec_t<3, wp::float32> & ret_0,
    wp::float32 & ret_1)
{
    //---------
    // primal vars
    wp::float32 var_0;
    const wp::float32 var_1 = 0.0;
    bool var_2;
    const wp::float32 var_3 = 0.0;
    wp::vec_t<3, wp::float32> var_4;
    //---------
    // forward
    // def normalize_with_norm(x: Any):                                                       <L 1>
    // norm = wp.length(x)                                                                    <L 2>
    var_0 = wp::length(var_x);
    // if norm == 0.0:                                                                        <L 3>
    var_2 = (var_0 == var_1);
    if (var_2) {
        // return x, 0.0                                                                      <L 4>
        ret_0 = var_x;
        ret_1 = var_3;
        return;
    }
    // return x / norm, norm                                                                  <L 5>
    var_4 = wp::div(var_x, var_0);
    ret_0 = var_4;
    ret_1 = var_0;
    return;
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


// C:\Users\ASUS\Documents\human-imitation-learning\.venv\Lib\site-packages\mujoco_warp\_src\support.py:471
static CUDA_CALLABLE wp::vec_t<6, wp::float32> transform_force_0(
    wp::vec_t<3, wp::float32> var_force,
    wp::vec_t<3, wp::float32> var_torque,
    wp::vec_t<3, wp::float32> var_offset)
{
    //---------
    // primal vars
    wp::vec_t<3, wp::float32> var_0;
    wp::vec_t<3, wp::float32> var_1;
    wp::vec_t<6, wp::float32> var_2;
    //---------
    // forward
    // def transform_force(force: wp.vec3, torque: wp.vec3, offset: wp.vec3) -> wp.spatial_vector:       <L 472>
    // return wp.spatial_vector(torque - wp.cross(offset, force), force)                      <L 473>
    var_0 = wp::cross(var_offset, var_force);
    var_1 = wp::sub(var_torque, var_0);
    var_2 = wp::vec_t<6, wp::float32>(var_1, var_force);
    return var_2;
}


// C:\Users\ASUS\Documents\human-imitation-learning\.venv\Lib\site-packages\mujoco_warp\_src\support.py:476
static CUDA_CALLABLE wp::vec_t<6, wp::float32> transform_force_1(
    wp::vec_t<6, wp::float32> var_frc,
    wp::vec_t<3, wp::float32> var_offset)
{
    //---------
    // primal vars
    wp::vec_t<3, wp::float32> var_0;
    wp::vec_t<3, wp::float32> var_1;
    wp::vec_t<6, wp::float32> var_2;
    //---------
    // forward
    // def transform_force(frc: wp.spatial_vector, offset: wp.vec3) -> wp.spatial_vector:       <L 477>
    // force = wp.spatial_top(frc)                                                            <L 478>
    var_0 = wp::spatial_top(var_frc);
    // torque = wp.spatial_bottom(frc)                                                        <L 479>
    var_1 = wp::spatial_bottom(var_frc);
    // return transform_force(force, torque, offset)                                          <L 480>
    var_2 = transform_force_0(var_0, var_1, var_offset);
    return var_2;
}


// C:\Users\ASUS\Documents\human-imitation-learning\.venv\Lib\site-packages\mujoco_warp\_src\math.py:147
static CUDA_CALLABLE wp::vec_t<6, wp::float32> motion_cross_force_0(
    wp::vec_t<6, wp::float32> var_v,
    wp::vec_t<6, wp::float32> var_f)
{
    //---------
    // primal vars
    const wp::int32 var_0 = 0;
    wp::float32 var_1;
    const wp::int32 var_2 = 1;
    wp::float32 var_3;
    const wp::int32 var_4 = 2;
    wp::float32 var_5;
    wp::vec_t<3, wp::float32> var_6;
    const wp::int32 var_7 = 3;
    wp::float32 var_8;
    const wp::int32 var_9 = 4;
    wp::float32 var_10;
    const wp::int32 var_11 = 5;
    wp::float32 var_12;
    wp::vec_t<3, wp::float32> var_13;
    const wp::int32 var_14 = 0;
    wp::float32 var_15;
    const wp::int32 var_16 = 1;
    wp::float32 var_17;
    const wp::int32 var_18 = 2;
    wp::float32 var_19;
    wp::vec_t<3, wp::float32> var_20;
    const wp::int32 var_21 = 3;
    wp::float32 var_22;
    const wp::int32 var_23 = 4;
    wp::float32 var_24;
    const wp::int32 var_25 = 5;
    wp::float32 var_26;
    wp::vec_t<3, wp::float32> var_27;
    wp::vec_t<3, wp::float32> var_28;
    wp::vec_t<3, wp::float32> var_29;
    wp::vec_t<3, wp::float32> var_30;
    wp::vec_t<3, wp::float32> var_31;
    wp::vec_t<6, wp::float32> var_32;
    //---------
    // forward
    // def motion_cross_force(v: wp.spatial_vector, f: wp.spatial_vector) -> wp.spatial_vector:       <L 148>
    // v0 = wp.vec3(v[0], v[1], v[2])                                                         <L 150>
    var_1 = wp::extract(var_v, var_0);
    var_3 = wp::extract(var_v, var_2);
    var_5 = wp::extract(var_v, var_4);
    var_6 = wp::vec_t<3, wp::float32>(var_1, var_3, var_5);
    // v1 = wp.vec3(v[3], v[4], v[5])                                                         <L 151>
    var_8 = wp::extract(var_v, var_7);
    var_10 = wp::extract(var_v, var_9);
    var_12 = wp::extract(var_v, var_11);
    var_13 = wp::vec_t<3, wp::float32>(var_8, var_10, var_12);
    // f0 = wp.vec3(f[0], f[1], f[2])                                                         <L 152>
    var_15 = wp::extract(var_f, var_14);
    var_17 = wp::extract(var_f, var_16);
    var_19 = wp::extract(var_f, var_18);
    var_20 = wp::vec_t<3, wp::float32>(var_15, var_17, var_19);
    // f1 = wp.vec3(f[3], f[4], f[5])                                                         <L 153>
    var_22 = wp::extract(var_f, var_21);
    var_24 = wp::extract(var_f, var_23);
    var_26 = wp::extract(var_f, var_25);
    var_27 = wp::vec_t<3, wp::float32>(var_22, var_24, var_26);
    // ang = wp.cross(v0, f0) + wp.cross(v1, f1)                                              <L 155>
    var_28 = wp::cross(var_6, var_20);
    var_29 = wp::cross(var_13, var_27);
    var_30 = wp::add(var_28, var_29);
    // vel = wp.cross(v0, f1)                                                                 <L 156>
    var_31 = wp::cross(var_6, var_27);
    // return wp.spatial_vector(ang, vel)                                                     <L 158>
    var_32 = wp::vec_t<6, wp::float32>(var_30, var_31);
    return var_32;
}


// C:\Users\ASUS\Documents\human-imitation-learning\.venv\Lib\site-packages\mujoco_warp\_src\math.py:33
static CUDA_CALLABLE wp::quat_t<wp::float32> quat_mul_axis_0(
    wp::quat_t<wp::float32> var_q,
    wp::vec_t<3, wp::float32> var_axis)
{
    //---------
    // primal vars
    const wp::int32 var_0 = 1;
    wp::float32 var_1;
    wp::float32 var_2;
    const wp::int32 var_3 = 0;
    wp::float32 var_4;
    wp::float32 var_5;
    const wp::int32 var_6 = 2;
    wp::float32 var_7;
    const wp::int32 var_8 = 1;
    wp::float32 var_9;
    wp::float32 var_10;
    wp::float32 var_11;
    const wp::int32 var_12 = 3;
    wp::float32 var_13;
    const wp::int32 var_14 = 2;
    wp::float32 var_15;
    wp::float32 var_16;
    wp::float32 var_17;
    const wp::int32 var_18 = 0;
    wp::float32 var_19;
    const wp::int32 var_20 = 0;
    wp::float32 var_21;
    wp::float32 var_22;
    const wp::int32 var_23 = 2;
    wp::float32 var_24;
    const wp::int32 var_25 = 2;
    wp::float32 var_26;
    wp::float32 var_27;
    wp::float32 var_28;
    const wp::int32 var_29 = 3;
    wp::float32 var_30;
    const wp::int32 var_31 = 1;
    wp::float32 var_32;
    wp::float32 var_33;
    wp::float32 var_34;
    const wp::int32 var_35 = 0;
    wp::float32 var_36;
    const wp::int32 var_37 = 1;
    wp::float32 var_38;
    wp::float32 var_39;
    const wp::int32 var_40 = 3;
    wp::float32 var_41;
    const wp::int32 var_42 = 0;
    wp::float32 var_43;
    wp::float32 var_44;
    wp::float32 var_45;
    const wp::int32 var_46 = 1;
    wp::float32 var_47;
    const wp::int32 var_48 = 2;
    wp::float32 var_49;
    wp::float32 var_50;
    wp::float32 var_51;
    const wp::int32 var_52 = 0;
    wp::float32 var_53;
    const wp::int32 var_54 = 2;
    wp::float32 var_55;
    wp::float32 var_56;
    const wp::int32 var_57 = 1;
    wp::float32 var_58;
    const wp::int32 var_59 = 1;
    wp::float32 var_60;
    wp::float32 var_61;
    wp::float32 var_62;
    const wp::int32 var_63 = 2;
    wp::float32 var_64;
    const wp::int32 var_65 = 0;
    wp::float32 var_66;
    wp::float32 var_67;
    wp::float32 var_68;
    wp::quat_t<wp::float32> var_69;
    //---------
    // forward
    // def quat_mul_axis(q: wp.quat, axis: wp.vec3f) -> wp.quat:                              <L 34>
    // return wp.quat(                                                                        <L 36>
    // -q[1] * axis[0] - q[2] * axis[1] - q[3] * axis[2],                                     <L 37>
    var_1 = wp::extract(var_q, var_0);
    var_2 = wp::neg(var_1);
    var_4 = wp::extract(var_axis, var_3);
    var_5 = wp::mul(var_2, var_4);
    var_7 = wp::extract(var_q, var_6);
    var_9 = wp::extract(var_axis, var_8);
    var_10 = wp::mul(var_7, var_9);
    var_11 = wp::sub(var_5, var_10);
    var_13 = wp::extract(var_q, var_12);
    var_15 = wp::extract(var_axis, var_14);
    var_16 = wp::mul(var_13, var_15);
    var_17 = wp::sub(var_11, var_16);
    // q[0] * axis[0] + q[2] * axis[2] - q[3] * axis[1],                                      <L 38>
    var_19 = wp::extract(var_q, var_18);
    var_21 = wp::extract(var_axis, var_20);
    var_22 = wp::mul(var_19, var_21);
    var_24 = wp::extract(var_q, var_23);
    var_26 = wp::extract(var_axis, var_25);
    var_27 = wp::mul(var_24, var_26);
    var_28 = wp::add(var_22, var_27);
    var_30 = wp::extract(var_q, var_29);
    var_32 = wp::extract(var_axis, var_31);
    var_33 = wp::mul(var_30, var_32);
    var_34 = wp::sub(var_28, var_33);
    // q[0] * axis[1] + q[3] * axis[0] - q[1] * axis[2],                                      <L 39>
    var_36 = wp::extract(var_q, var_35);
    var_38 = wp::extract(var_axis, var_37);
    var_39 = wp::mul(var_36, var_38);
    var_41 = wp::extract(var_q, var_40);
    var_43 = wp::extract(var_axis, var_42);
    var_44 = wp::mul(var_41, var_43);
    var_45 = wp::add(var_39, var_44);
    var_47 = wp::extract(var_q, var_46);
    var_49 = wp::extract(var_axis, var_48);
    var_50 = wp::mul(var_47, var_49);
    var_51 = wp::sub(var_45, var_50);
    // q[0] * axis[2] + q[1] * axis[1] - q[2] * axis[0],                                      <L 40>
    var_53 = wp::extract(var_q, var_52);
    var_55 = wp::extract(var_axis, var_54);
    var_56 = wp::mul(var_53, var_55);
    var_58 = wp::extract(var_q, var_57);
    var_60 = wp::extract(var_axis, var_59);
    var_61 = wp::mul(var_58, var_60);
    var_62 = wp::add(var_56, var_61);
    var_64 = wp::extract(var_q, var_63);
    var_66 = wp::extract(var_axis, var_65);
    var_67 = wp::mul(var_64, var_66);
    var_68 = wp::sub(var_62, var_67);
    var_69 = wp::quat_t<wp::float32>(var_17, var_34, var_51, var_68);
    return var_69;
}


// C:\Users\ASUS\Documents\human-imitation-learning\.venv\Lib\site-packages\mujoco_warp\_src\smooth.py:2157
static CUDA_CALLABLE void _accumulate_jac_dot_chain_0(
    wp::array_t<wp::int32> var_body_parentid,
    wp::array_t<wp::int32> var_body_dofnum,
    wp::array_t<wp::int32> var_body_dofadr,
    wp::array_t<wp::int32> var_jnt_type,
    wp::array_t<wp::int32> var_jnt_dofadr,
    wp::array_t<wp::int32> var_dof_jntid,
    wp::array_t<wp::int32> var_ten_J_colind,
    wp::array_t<wp::vec_t<6, wp::float32>> var_cdof_in,
    wp::array_t<wp::vec_t<6, wp::float32>> var_cvel_in,
    wp::array_t<wp::vec_t<6, wp::float32>> var_cdof_dot_in,
    wp::vec_t<3, wp::float32> var_offset,
    wp::vec_t<3, wp::float32> var_pvel_lin,
    wp::vec_t<3, wp::float32> var_dpnt,
    wp::vec_t<3, wp::float32> var_dvel,
    wp::int32 var_bodyid,
    wp::int32 var_rowadr,
    wp::int32 var_rownnz,
    wp::float32 var_scale,
    wp::int32 var_worldid,
    wp::array_t<wp::float32> var_ten_Jdot_out)
{
    //---------
    // primal vars
    const wp::int32 var_0 = 1;
    wp::int32 var_1;
    wp::int32 var_2;
    const wp::int32 var_3 = 0;
    bool var_4;
    wp::int32* var_5;
    wp::int32 var_6;
    wp::int32 var_7;
    wp::int32* var_8;
    wp::int32 var_9;
    wp::int32 var_10;
    wp::range_t var_11;
    wp::int32 var_12;
    wp::int32 var_13;
    const wp::int32 var_14 = 1;
    wp::int32 var_15;
    wp::int32 var_16;
    const wp::int32 var_17 = 0;
    bool var_18;
    wp::int32 var_19;
    wp::int32* var_20;
    bool var_21;
    wp::int32 var_22;
    const wp::int32 var_23 = 1;
    wp::int32 var_24;
    bool var_25;
    const wp::int32 var_26 = 0;
    bool var_27;
    wp::int32* var_28;
    bool var_29;
    wp::int32 var_30;
    wp::vec_t<6, wp::float32>* var_31;
    wp::vec_t<6, wp::float32> var_32;
    wp::vec_t<6, wp::float32> var_33;
    wp::vec_t<3, wp::float32> var_34;
    wp::vec_t<3, wp::float32> var_35;
    wp::vec_t<6, wp::float32>* var_36;
    wp::vec_t<6, wp::float32> var_37;
    wp::vec_t<6, wp::float32> var_38;
    wp::int32* var_39;
    wp::int32 var_40;
    wp::int32 var_41;
    wp::int32* var_42;
    wp::int32 var_43;
    wp::int32 var_44;
    wp::int32* var_45;
    wp::int32 var_46;
    wp::int32 var_47;
    bool var_48;
    const wp::int32 var_49 = 1;
    bool var_50;
    bool var_51;
    const wp::int32 var_52 = 0;
    bool var_53;
    const wp::int32 var_54 = 3;
    wp::int32 var_55;
    bool var_56;
    wp::vec_t<6, wp::float32>* var_57;
    wp::vec_t<6, wp::float32> var_58;
    wp::vec_t<6, wp::float32> var_59;
    wp::vec_t<6, wp::float32> var_60;
    wp::vec_t<3, wp::float32> var_61;
    wp::vec_t<3, wp::float32> var_62;
    wp::vec_t<3, wp::float32> var_63;
    wp::vec_t<3, wp::float32> var_64;
    wp::vec_t<3, wp::float32> var_65;
    wp::vec_t<3, wp::float32> var_66;
    wp::vec_t<3, wp::float32> var_67;
    wp::vec_t<3, wp::float32> var_68;
    wp::float32 var_69;
    wp::float32 var_70;
    wp::float32 var_71;
    wp::float32 var_72;
    const wp::float32 var_73 = 0.0;
    bool var_74;
    wp::array_t<wp::float32> var_75;
    wp::float32 var_76;
    wp::int32* var_77;
    wp::int32 var_78;
    wp::int32 var_79;
    //---------
    // forward
    // def _accumulate_jac_dot_chain(                                                         <L 2158>
    // ptr = rownnz - 1                                                                       <L 2185>
    var_1 = wp::sub(var_rownnz, var_0);
    // bid = bodyid                                                                           <L 2186>
    var_2 = wp::copy(var_bodyid);
    // while bid > 0:                                                                         <L 2187>
    start_while_0:;
    var_4 = (var_2 > var_3);
    if ((var_4) == false) goto end_while_0;
        // bdofadr = body_dofadr[bid]                                                         <L 2188>
        var_5 = wp::address(var_body_dofadr, var_2);
        var_7 = wp::load(var_5);
        var_6 = wp::copy(var_7);
        // bdofnum = body_dofnum[bid]                                                         <L 2189>
        var_8 = wp::address(var_body_dofnum, var_2);
        var_10 = wp::load(var_8);
        var_9 = wp::copy(var_10);
        // for k_rev in range(bdofnum):                                                       <L 2191>
        var_11 = wp::range(var_9);
        start_for_2:;
            if (iter_cmp(var_11) == 0) goto end_for_2;
            var_12 = wp::iter_next(var_11);
            // dof = bdofadr + bdofnum - 1 - k_rev                                            <L 2192>
            var_13 = wp::add(var_6, var_9);
            var_15 = wp::sub(var_13, var_14);
            var_16 = wp::sub(var_15, var_12);
            // while ptr >= 0:                                                                <L 2194>
    start_while_4:;
            var_18 = (var_1 >= var_17);
    if ((var_18) == false) goto end_while_4;
                // sparseid = rowadr + ptr                                                    <L 2195>
                var_19 = wp::add(var_rowadr, var_1);
                // if ten_J_colind[sparseid] <= dof:                                          <L 2196>
                var_20 = wp::address(var_ten_J_colind, var_19);
                var_22 = wp::load(var_20);
                var_21 = (var_22 <= var_16);
                if (var_21) {
                    // break                                                                  <L 2197>
                    goto end_while_4;
                }
                // ptr -= 1                                                                   <L 2198>
                var_24 = wp::sub(var_1, var_23);
                wp::assign(var_1, var_24);
    goto start_while_4;
    end_while_4:;
            // if ptr >= 0 and ten_J_colind[sparseid] == dof:                                 <L 2199>
            var_27 = (var_1 >= var_26);
            var_25 = var_27;
            if (var_25) {
                var_28 = wp::address(var_ten_J_colind, var_19);
                var_30 = wp::load(var_28);
                var_29 = (var_30 == var_16);
                var_25 = var_25 && var_29;
            }
            if (var_25) {
                // cdof = cdof_in[worldid, dof]                                               <L 2200>
                var_31 = wp::address(var_cdof_in, var_worldid, var_16);
                var_33 = wp::load(var_31);
                var_32 = wp::copy(var_33);
                // cdof_ang = wp.spatial_top(cdof)                                            <L 2201>
                var_34 = wp::spatial_top(var_32);
                // cdof_lin = wp.spatial_bottom(cdof)                                         <L 2202>
                var_35 = wp::spatial_bottom(var_32);
                // cdof_dot = cdof_dot_in[worldid, dof]                                       <L 2203>
                var_36 = wp::address(var_cdof_dot_in, var_worldid, var_16);
                var_38 = wp::load(var_36);
                var_37 = wp::copy(var_38);
                // dofjntid = dof_jntid[dof]                                                  <L 2206>
                var_39 = wp::address(var_dof_jntid, var_16);
                var_41 = wp::load(var_39);
                var_40 = wp::copy(var_41);
                // jnttype = jnt_type[dofjntid]                                               <L 2207>
                var_42 = wp::address(var_jnt_type, var_40);
                var_44 = wp::load(var_42);
                var_43 = wp::copy(var_44);
                // jntdofadr = jnt_dofadr[dofjntid]                                           <L 2208>
                var_45 = wp::address(var_jnt_dofadr, var_40);
                var_47 = wp::load(var_45);
                var_46 = wp::copy(var_47);
                // if (jnttype == JointType.BALL) or ((jnttype == JointType.FREE) and dof >= jntdofadr + 3):       <L 2209>
                var_50 = (var_43 == var_49);
                var_48 = var_50;
                if (!var_48) {
                    var_53 = (var_43 == var_52);
                    var_51 = var_53;
                    if (var_51) {
                        var_55 = wp::add(var_46, var_54);
                        var_56 = (var_16 >= var_55);
                        var_51 = var_51 && var_56;
                    }
                    var_48 = var_48 || var_51;
                }
                if (var_48) {
                    // cdof_dot = math.motion_cross(cvel_in[worldid, bid], cdof)              <L 2210>
                    var_57 = wp::address(var_cvel_in, var_worldid, var_2);
                    var_59 = wp::load(var_57);
                    var_58 = motion_cross_0(var_59, var_32);
                }
                var_60 = wp::where(var_48, var_58, var_37);
                // cdof_dot_ang = wp.spatial_top(cdof_dot)                                    <L 2212>
                var_61 = wp::spatial_top(var_60);
                // cdof_dot_lin = wp.spatial_bottom(cdof_dot)                                 <L 2213>
                var_62 = wp::spatial_bottom(var_60);
                // jacp_dot = cdof_dot_lin + wp.cross(cdof_dot_ang, offset) + wp.cross(cdof_ang, pvel_lin)       <L 2216>
                var_63 = wp::cross(var_61, var_offset);
                var_64 = wp::add(var_62, var_63);
                var_65 = wp::cross(var_34, var_pvel_lin);
                var_66 = wp::add(var_64, var_65);
                // jacp = cdof_lin + wp.cross(cdof_ang, offset)                               <L 2219>
                var_67 = wp::cross(var_34, var_offset);
                var_68 = wp::add(var_35, var_67);
                // Jdot = (wp.dot(jacp_dot, dpnt) + wp.dot(jacp, dvel)) * scale               <L 2222>
                var_69 = wp::dot(var_66, var_dpnt);
                var_70 = wp::dot(var_68, var_dvel);
                var_71 = wp::add(var_69, var_70);
                var_72 = wp::mul(var_71, var_scale);
                // if Jdot != 0.0:                                                            <L 2223>
                var_74 = (var_72 != var_73);
                if (var_74) {
                    // wp.atomic_add(ten_Jdot_out[worldid], sparseid, Jdot)                   <L 2224>
                    var_75 = wp::view(var_ten_Jdot_out, var_worldid);
                    var_76 = wp::atomic_add(var_75, var_19, var_72);
                }
            }
            goto start_for_2;
        end_for_2:;
        // bid = body_parentid[bid]                                                           <L 2225>
        var_77 = wp::address(var_body_parentid, var_2);
        var_79 = wp::load(var_77);
        var_78 = wp::copy(var_79);
        wp::assign(var_2, var_78);
    goto start_while_0;
    end_while_0:;
}


// C:\Users\ASUS\Documents\human-imitation-learning\.venv\Lib\site-packages\mujoco_warp\_src\support.py:986
static CUDA_CALLABLE wp::float32 _phi_0(
    wp::float32 var_s,
    wp::int32 var_i)
{
    //---------
    // primal vars
    const wp::int32 var_0 = 0;
    bool var_1;
    const wp::float32 var_2 = 1.0;
    wp::float32 var_3;
    //---------
    // forward
    // def _phi(s: float, i: int) -> float:                                                   <L 987>
    // if i == 0:                                                                             <L 993>
    var_1 = (var_i == var_0);
    if (var_1) {
        // return 1.0 - s                                                                     <L 994>
        var_3 = wp::sub(var_2, var_s);
        return var_3;
    }
    // return s                                                                               <L 995>
    return var_s;
}


// C:\Users\ASUS\Documents\human-imitation-learning\.venv\Lib\site-packages\mujoco_warp\_src\support.py:998
static CUDA_CALLABLE wp::float32 eval_basis_trilinear_0(
    wp::vec_t<3, wp::float32> var_local,
    wp::int32 var_node_idx)
{
    //---------
    // primal vars
    const wp::int32 var_0 = 1;
    wp::int32 var_1;
    const wp::int32 var_2 = 1;
    wp::int32 var_3;
    const wp::int32 var_4 = 1;
    wp::int32 var_5;
    const wp::int32 var_6 = 2;
    wp::int32 var_7;
    const wp::int32 var_8 = 1;
    wp::int32 var_9;
    const wp::int32 var_10 = 0;
    wp::float32 var_11;
    wp::float32 var_12;
    const wp::int32 var_13 = 1;
    wp::float32 var_14;
    wp::float32 var_15;
    wp::float32 var_16;
    const wp::int32 var_17 = 2;
    wp::float32 var_18;
    wp::float32 var_19;
    wp::float32 var_20;
    //---------
    // forward
    // def eval_basis_trilinear(local: wp.vec3, node_idx: int) -> float:                      <L 999>
    // k = node_idx & 1                                                                       <L 1005>
    var_1 = wp::bit_and(var_node_idx, var_0);
    // j = (node_idx >> 1) & 1                                                                <L 1006>
    var_3 = wp::rshift(var_node_idx, var_2);
    var_5 = wp::bit_and(var_3, var_4);
    // i = (node_idx >> 2) & 1                                                                <L 1007>
    var_7 = wp::rshift(var_node_idx, var_6);
    var_9 = wp::bit_and(var_7, var_8);
    // return _phi(local[0], i) * _phi(local[1], j) * _phi(local[2], k)                       <L 1008>
    var_11 = wp::extract(var_local, var_10);
    var_12 = _phi_0(var_11, var_9);
    var_14 = wp::extract(var_local, var_13);
    var_15 = _phi_0(var_14, var_5);
    var_16 = wp::mul(var_12, var_15);
    var_18 = wp::extract(var_local, var_17);
    var_19 = _phi_0(var_18, var_1);
    var_20 = wp::mul(var_16, var_19);
    return var_20;
}


// C:\Users\ASUS\Documents\human-imitation-learning\.venv\Lib\site-packages\mujoco_warp\_src\math.py:0
static CUDA_CALLABLE void normalize_with_norm_0(
    wp::vec_t<2, wp::float32> var_x,
    wp::vec_t<2, wp::float32> & ret_0,
    wp::float32 & ret_1)
{
    //---------
    // primal vars
    wp::float32 var_0;
    const wp::float32 var_1 = 0.0;
    bool var_2;
    const wp::float32 var_3 = 0.0;
    wp::vec_t<2, wp::float32> var_4;
    //---------
    // forward
    // def normalize_with_norm(x: Any):                                                       <L 1>
    // norm = wp.length(x)                                                                    <L 2>
    var_0 = wp::length(var_x);
    // if norm == 0.0:                                                                        <L 3>
    var_2 = (var_0 == var_1);
    if (var_2) {
        // return x, 0.0                                                                      <L 4>
        ret_0 = var_x;
        ret_1 = var_3;
        return;
    }
    // return x / norm, norm                                                                  <L 5>
    var_4 = wp::div(var_x, var_0);
    ret_0 = var_4;
    ret_1 = var_0;
    return;
}


// C:\Users\ASUS\Documents\human-imitation-learning\.venv\Lib\site-packages\warp\_src\math.py:0
static CUDA_CALLABLE wp::float32 norm_l2_0(
    wp::vec_t<2, wp::float32> var_v)
{
    //---------
    // primal vars
    wp::float32 var_0;
    //---------
    // forward
    // def norm_l2(v: Any) -> float:                                                          <L 1>
    // return wp.length(v)                                                                    <L 12>
    var_0 = wp::length(var_v);
    return var_0;
}


// C:\Users\ASUS\Documents\human-imitation-learning\.venv\Lib\site-packages\mujoco_warp\_src\util_misc.py:205
static CUDA_CALLABLE void wrap_inside_0(
    wp::vec_t<4, wp::float32> var_end,
    wp::float32 var_radius,
    wp::float32 & ret_0,
    wp::vec_t<2, wp::float32> & ret_1,
    wp::vec_t<2, wp::float32> & ret_2)
{
    //---------
    // primal vars
    const wp::int32 var_0 = 20;
    wp::int32 var_1;
    const wp::float32 var_2 = 1e-06;
    wp::float32 var_3;
    const wp::int32 var_4 = 0;
    wp::float32 var_5;
    const wp::int32 var_6 = 1;
    wp::float32 var_7;
    wp::vec_t<2, wp::float32> var_8;
    const wp::int32 var_9 = 2;
    wp::float32 var_10;
    const wp::int32 var_11 = 3;
    wp::float32 var_12;
    wp::vec_t<2, wp::float32> var_13;
    wp::float32 var_14;
    wp::float32 var_15;
    wp::vec_t<2, wp::float32> var_16;
    wp::float32 var_17;
    bool var_18;
    bool var_19;
    bool var_20;
    const wp::float32 var_21 = 1e-15;
    bool var_22;
    bool var_23;
    bool var_24;
    const wp::float32 var_25 = -1.0;
    const wp::float32 var_26 = 10000000000.0;
    wp::vec_t<2, wp::float32> var_27;
    wp::vec_t<2, wp::float32> var_28;
    bool var_29;
    wp::float32 var_30;
    wp::float32 var_31;
    wp::float32 var_32;
    bool var_33;
    const wp::float32 var_34 = 0.0;
    bool var_35;
    const wp::float32 var_36 = 1.0;
    bool var_37;
    wp::vec_t<2, wp::float32> var_38;
    wp::vec_t<2, wp::float32> var_39;
    wp::float32 var_40;
    bool var_41;
    const wp::float32 var_42 = -1.0;
    wp::vec_t<2, wp::float32> var_43;
    wp::vec_t<2, wp::float32> var_44;
    const wp::float32 var_45 = 0.5;
    wp::vec_t<2, wp::float32> var_46;
    wp::vec_t<2, wp::float32> var_47;
    wp::vec_t<2, wp::float32> var_48;
    wp::float32 var_49;
    wp::vec_t<2, wp::float32> var_50;
    wp::float32 var_51;
    wp::float32 var_52;
    wp::float32 var_53;
    wp::float32 var_54;
    wp::float32 var_55;
    wp::float32 var_56;
    wp::float32 var_57;
    wp::float32 var_58;
    const wp::float32 var_59 = 2.0;
    wp::float32 var_60;
    wp::float32 var_61;
    wp::float32 var_62;
    const wp::float32 var_63 = -1.0;
    wp::float32 var_64;
    bool var_65;
    const wp::float32 var_66 = -1.0;
    const wp::float32 var_67 = 1.0;
    wp::float32 var_68;
    bool var_69;
    const wp::float32 var_70 = 0.0;
    wp::float32 var_71;
    const wp::float32 var_72 = 1.5707963267948966;
    wp::float32 var_73;
    const wp::float32 var_74 = 1.0;
    wp::float32 var_75;
    wp::float32 var_76;
    wp::float32 var_77;
    wp::float32 var_78;
    const wp::float32 var_79 = 3.141592653589793;
    wp::float32 var_80;
    wp::float32 var_81;
    const wp::float32 var_82 = 0.0;
    bool var_83;
    const wp::float32 var_84 = 0.0;
    const wp::int32 var_85 = 0;
    wp::int32 var_86;
    bool var_87;
    bool var_88;
    wp::float32 var_89;
    bool var_90;
    wp::float32 var_91;
    wp::float32 var_92;
    const wp::float32 var_93 = 1.0;
    wp::float32 var_94;
    wp::float32 var_95;
    wp::float32 var_96;
    wp::float32 var_97;
    wp::float32 var_98;
    const wp::float32 var_99 = 1.0;
    wp::float32 var_100;
    wp::float32 var_101;
    wp::float32 var_102;
    wp::float32 var_103;
    wp::float32 var_104;
    wp::float32 var_105;
    wp::float32 var_106;
    const wp::float32 var_107 = 2.0;
    wp::float32 var_108;
    const wp::float32 var_109 = -1e-15;
    bool var_110;
    const wp::float32 var_111 = 0.0;
    wp::float32 var_112;
    wp::float32 var_113;
    bool var_114;
    const wp::float32 var_115 = 0.0;
    wp::float32 var_116;
    wp::float32 var_117;
    wp::float32 var_118;
    wp::float32 var_119;
    wp::float32 var_120;
    wp::float32 var_121;
    wp::float32 var_122;
    const wp::float32 var_123 = 2.0;
    wp::float32 var_124;
    wp::float32 var_125;
    wp::float32 var_126;
    bool var_127;
    const wp::float32 var_128 = 0.0;
    const wp::int32 var_129 = 1;
    wp::int32 var_130;
    bool var_131;
    const wp::float32 var_132 = 0.0;
    const wp::int32 var_133 = 0;
    wp::float32 var_134;
    const wp::int32 var_135 = 3;
    wp::float32 var_136;
    wp::float32 var_137;
    const wp::int32 var_138 = 1;
    wp::float32 var_139;
    const wp::int32 var_140 = 2;
    wp::float32 var_141;
    wp::float32 var_142;
    wp::float32 var_143;
    const wp::float32 var_144 = 0.0;
    bool var_145;
    wp::vec_t<2, wp::float32> var_146;
    wp::float32 var_147;
    wp::vec_t<2, wp::float32> var_148;
    wp::float32 var_149;
    wp::vec_t<2, wp::float32> var_150;
    wp::float32 var_151;
    wp::vec_t<2, wp::float32> var_152;
    wp::float32 var_153;
    wp::float32 var_154;
    const wp::int32 var_155 = 0;
    wp::float32 var_156;
    wp::float32 var_157;
    wp::float32 var_158;
    const wp::int32 var_159 = 1;
    wp::float32 var_160;
    wp::float32 var_161;
    wp::float32 var_162;
    wp::float32 var_163;
    wp::float32 var_164;
    const wp::int32 var_165 = 0;
    wp::float32 var_166;
    wp::float32 var_167;
    wp::float32 var_168;
    const wp::int32 var_169 = 1;
    wp::float32 var_170;
    wp::float32 var_171;
    wp::float32 var_172;
    wp::float32 var_173;
    wp::vec_t<2, wp::float32> var_174;
    const wp::float32 var_175 = 0.0;
    //---------
    // forward
    // def wrap_inside(                                                                       <L 206>
    // maxiter = int(20)                                                                      <L 221>
    var_1 = wp::int(var_0);
    // tolerance = float(1.0e-6)                                                              <L 222>
    var_3 = wp::float(var_2);
    // end0 = wp.vec2(end[0], end[1])                                                         <L 224>
    var_5 = wp::extract(var_end, var_4);
    var_7 = wp::extract(var_end, var_6);
    var_8 = wp::vec_t<2, wp::float32>(var_5, var_7);
    // end1 = wp.vec2(end[2], end[3])                                                         <L 225>
    var_10 = wp::extract(var_end, var_9);
    var_12 = wp::extract(var_end, var_11);
    var_13 = wp::vec_t<2, wp::float32>(var_10, var_12);
    // len0 = wp.norm_l2(end0)                                                                <L 228>
    var_14 = norm_l2_0(var_8);
    // len1 = wp.norm_l2(end1)                                                                <L 229>
    var_15 = norm_l2_0(var_13);
    // dif = end1 - end0                                                                      <L 230>
    var_16 = wp::sub(var_13, var_8);
    // dd = wp.dot(dif, dif)                                                                  <L 231>
    var_17 = wp::dot(var_16, var_16);
    // if (len0 <= radius) or (len1 <= radius) or (radius < MJ_MINVAL) or (len0 < MJ_MINVAL) or (len1 < MJ_MINVAL):       <L 234>
    var_19 = (var_14 <= var_radius);
    var_18 = var_19;
    if (!var_18) {
        var_20 = (var_15 <= var_radius);
        var_18 = var_18 || var_20;
    }
    if (!var_18) {
        var_22 = (var_radius < var_21);
        var_18 = var_18 || var_22;
    }
    if (!var_18) {
        var_23 = (var_14 < var_21);
        var_18 = var_18 || var_23;
    }
    if (!var_18) {
        var_24 = (var_15 < var_21);
        var_18 = var_18 || var_24;
    }
    if (var_18) {
        // return -1.0, wp.vec2(MJ_MAXVAL), wp.vec2(MJ_MAXVAL)                                <L 235>
        var_27 = wp::vec_t<2, wp::float32>(var_26);
        var_28 = wp::vec_t<2, wp::float32>(var_26);
        ret_0 = var_25;
        ret_1 = var_27;
        ret_2 = var_28;
        return;
    }
    // if dd > MJ_MINVAL:                                                                     <L 238>
    var_29 = (var_17 > var_21);
    if (var_29) {
        // a = -wp.dot(dif, end0) / dd                                                        <L 240>
        var_30 = wp::dot(var_16, var_8);
        var_31 = wp::neg(var_30);
        var_32 = wp::div(var_31, var_17);
        // if (a > 0.0) and (a < 1.0):                                                        <L 243>
        var_35 = (var_32 > var_34);
        var_33 = var_35;
        if (var_33) {
            var_37 = (var_32 < var_36);
            var_33 = var_33 && var_37;
        }
        if (var_33) {
            // tmp = end0 + a * dif                                                           <L 244>
            var_38 = wp::mul(var_32, var_16);
            var_39 = wp::add(var_8, var_38);
            // if wp.norm_l2(tmp) <= radius:                                                  <L 245>
            var_40 = norm_l2_0(var_39);
            var_41 = (var_40 <= var_radius);
            if (var_41) {
                // return -1.0, wp.vec2(MJ_MAXVAL), wp.vec2(MJ_MAXVAL)                        <L 246>
                var_43 = wp::vec_t<2, wp::float32>(var_26);
                var_44 = wp::vec_t<2, wp::float32>(var_26);
                ret_0 = var_42;
                ret_1 = var_43;
                ret_2 = var_44;
                return;
            }
        }
    }
    // pnt = 0.5 * (end0 + end1)                                                              <L 249>
    var_46 = wp::add(var_8, var_13);
    var_47 = wp::mul(var_45, var_46);
    // pnt, _ = math.normalize_with_norm(pnt)                                                 <L 250>
    normalize_with_norm_0(var_47, var_48, var_49);
    // pnt *= radius                                                                          <L 251>
    var_50 = wp::mul(var_48, var_radius);
    // A = math.safe_div(radius, len0)                                                        <L 254>
    var_51 = safe_div_0(var_radius, var_14);
    // B = math.safe_div(radius, len1)                                                        <L 255>
    var_52 = safe_div_0(var_radius, var_15);
    // sq_A = A * A                                                                           <L 256>
    var_53 = wp::mul(var_51, var_51);
    // sq_B = B * B                                                                           <L 257>
    var_54 = wp::mul(var_52, var_52);
    // cosG = math.safe_div(len0 * len0 + len1 * len1 - dd, 2.0 * len0 * len1)                <L 258>
    var_55 = wp::mul(var_14, var_14);
    var_56 = wp::mul(var_15, var_15);
    var_57 = wp::add(var_55, var_56);
    var_58 = wp::sub(var_57, var_17);
    var_60 = wp::mul(var_59, var_14);
    var_61 = wp::mul(var_60, var_15);
    var_62 = safe_div_0(var_58, var_61);
    // if cosG < -1.0 + MJ_MINVAL:                                                            <L 259>
    var_64 = wp::add(var_63, var_21);
    var_65 = (var_62 < var_64);
    if (var_65) {
        // return -1.0, pnt, pnt                                                              <L 260>
        ret_0 = var_66;
        ret_1 = var_50;
        ret_2 = var_50;
        return;
    }
    if (!var_65) {
        // elif cosG > 1.0 - MJ_MINVAL:                                                       <L 261>
        var_68 = wp::sub(var_67, var_21);
        var_69 = (var_62 > var_68);
        if (var_69) {
            // return 0.0, pnt, pnt                                                           <L 262>
            ret_0 = var_70;
            ret_1 = var_50;
            ret_2 = var_50;
            return;
        }
    }
    // G = wp.acos(cosG)                                                                      <L 263>
    var_71 = wp::acos(var_62);
    // phi = float(wp.static(0.5 * wp.pi))                                                    <L 266>
    var_73 = wp::float(var_72);
    // z = float(1.0)                                                                         <L 267>
    var_75 = wp::float(var_74);
    // asin_Az = wp.asin(A)                                                                   <L 268>
    var_76 = wp::asin(var_51);
    // asin_Bz = wp.asin(B)                                                                   <L 269>
    var_77 = wp::asin(var_52);
    // f = asin_Az + asin_Bz - wp.pi + G                                                      <L 270>
    var_78 = wp::add(var_76, var_77);
    var_80 = wp::sub(var_78, var_79);
    var_81 = wp::add(var_80, var_71);
    // if f > 0.0:                                                                            <L 273>
    var_83 = (var_81 > var_82);
    if (var_83) {
        // return 0.0, pnt, pnt                                                               <L 274>
        ret_0 = var_84;
        ret_1 = var_50;
        ret_2 = var_50;
        return;
    }
    // iter = int(0)                                                                          <L 277>
    var_86 = wp::int(var_85);
    // while (iter < maxiter) and (wp.abs(f) > tolerance):                                    <L 279>
    start_while_5:;
    var_88 = (var_86 < var_1);
    var_87 = var_88;
    if (var_87) {
        var_89 = wp::abs(var_81);
        var_90 = (var_89 > var_3);
        var_87 = var_87 && var_90;
    }
    if ((var_87) == false) goto end_while_5;
        // sq_z = z * z                                                                       <L 281>
        var_91 = wp::mul(var_75, var_75);
        // df = (                                                                             <L 282>
        // wp.cos(phi) * (A / wp.max(MJ_MINVAL, wp.sqrt(1.0 - sq_z * sq_A)) + B / wp.max(MJ_MINVAL, wp.sqrt(1.0 - sq_z * sq_B)))       <L 283>
        var_92 = wp::cos(var_73);
        var_94 = wp::mul(var_91, var_53);
        var_95 = wp::sub(var_93, var_94);
        var_96 = wp::sqrt(var_95);
        var_97 = wp::max(var_21, var_96);
        var_98 = wp::div(var_51, var_97);
        var_100 = wp::mul(var_91, var_54);
        var_101 = wp::sub(var_99, var_100);
        var_102 = wp::sqrt(var_101);
        var_103 = wp::max(var_21, var_102);
        var_104 = wp::div(var_52, var_103);
        var_105 = wp::add(var_98, var_104);
        var_106 = wp::mul(var_92, var_105);
        // - 2.0                                                                              <L 284>
        var_108 = wp::sub(var_106, var_107);
        // if df > -MJ_MINVAL:                                                                <L 288>
        var_110 = (var_108 > var_109);
        if (var_110) {
            // return 0.0, pnt, pnt                                                           <L 289>
            ret_0 = var_111;
            ret_1 = var_50;
            ret_2 = var_50;
            return;
        }
        // phi1 = phi - math.safe_div(f, df)                                                  <L 292>
        var_112 = safe_div_0(var_81, var_108);
        var_113 = wp::sub(var_73, var_112);
        // if phi1 > phi:                                                                     <L 295>
        var_114 = (var_113 > var_73);
        if (var_114) {
            // return 0.0, pnt, pnt                                                           <L 296>
            ret_0 = var_115;
            ret_1 = var_50;
            ret_2 = var_50;
            return;
        }
        // phi = phi1                                                                         <L 299>
        var_116 = wp::copy(var_113);
        // z = wp.sin(phi)                                                                    <L 300>
        var_117 = wp::sin(var_116);
        // asin_Az = wp.asin(A * z)                                                           <L 301>
        var_118 = wp::mul(var_51, var_117);
        var_119 = wp::asin(var_118);
        // asin_Bz = wp.asin(B * z)                                                           <L 302>
        var_120 = wp::mul(var_52, var_117);
        var_121 = wp::asin(var_120);
        // f = asin_Az + asin_Bz - 2.0 * phi + G                                              <L 303>
        var_122 = wp::add(var_119, var_121);
        var_124 = wp::mul(var_123, var_116);
        var_125 = wp::sub(var_122, var_124);
        var_126 = wp::add(var_125, var_71);
        // if f > tolerance:                                                                  <L 306>
        var_127 = (var_126 > var_3);
        if (var_127) {
            // return 0.0, pnt, pnt                                                           <L 307>
            ret_0 = var_128;
            ret_1 = var_50;
            ret_2 = var_50;
            return;
        }
        // iter += 1                                                                          <L 309>
        var_130 = wp::add(var_86, var_129);
        wp::assign(var_73, var_116);
        wp::assign(var_75, var_117);
        wp::assign(var_76, var_119);
        wp::assign(var_77, var_121);
        wp::assign(var_81, var_126);
        wp::assign(var_86, var_130);
    goto start_while_5;
    end_while_5:;
    // if iter >= maxiter:                                                                    <L 312>
    var_131 = (var_86 >= var_1);
    if (var_131) {
        // return 0.0, pnt, pnt                                                               <L 313>
        ret_0 = var_132;
        ret_1 = var_50;
        ret_2 = var_50;
        return;
    }
    // if end[0] * end[3] - end[1] * end[2] > 0.0:                                            <L 316>
    var_134 = wp::extract(var_end, var_133);
    var_136 = wp::extract(var_end, var_135);
    var_137 = wp::mul(var_134, var_136);
    var_139 = wp::extract(var_end, var_138);
    var_141 = wp::extract(var_end, var_140);
    var_142 = wp::mul(var_139, var_141);
    var_143 = wp::sub(var_137, var_142);
    var_145 = (var_143 > var_144);
    if (var_145) {
        // vec = end0                                                                         <L 317>
        var_146 = wp::copy(var_8);
        // ang = phi - asin_Az                                                                <L 318>
        var_147 = wp::sub(var_73, var_76);
    }
    if (!var_145) {
        // vec = end1                                                                         <L 320>
        var_148 = wp::copy(var_13);
        // ang = phi - asin_Bz                                                                <L 321>
        var_149 = wp::sub(var_73, var_77);
    }
    var_150 = wp::where(var_145, var_146, var_148);
    var_151 = wp::where(var_145, var_147, var_149);
    // vec, _ = math.normalize_with_norm(vec)                                                 <L 323>
    normalize_with_norm_0(var_150, var_152, var_153);
    // pnt = wp.vec2(                                                                         <L 324>
    // radius * (wp.cos(ang) * vec[0] - wp.sin(ang) * vec[1]),                                <L 325>
    var_154 = wp::cos(var_151);
    var_156 = wp::extract(var_152, var_155);
    var_157 = wp::mul(var_154, var_156);
    var_158 = wp::sin(var_151);
    var_160 = wp::extract(var_152, var_159);
    var_161 = wp::mul(var_158, var_160);
    var_162 = wp::sub(var_157, var_161);
    var_163 = wp::mul(var_radius, var_162);
    // radius * (wp.sin(ang) * vec[0] + wp.cos(ang) * vec[1]),                                <L 326>
    var_164 = wp::sin(var_151);
    var_166 = wp::extract(var_152, var_165);
    var_167 = wp::mul(var_164, var_166);
    var_168 = wp::cos(var_151);
    var_170 = wp::extract(var_152, var_169);
    var_171 = wp::mul(var_168, var_170);
    var_172 = wp::add(var_167, var_171);
    var_173 = wp::mul(var_radius, var_172);
    var_174 = wp::vec_t<2, wp::float32>(var_163, var_173);
    // return 0.0, pnt, pnt                                                                   <L 329>
    ret_0 = var_175;
    ret_1 = var_174;
    ret_2 = var_174;
    return;
}


// C:\Users\ASUS\Documents\human-imitation-learning\.venv\Lib\site-packages\mujoco_warp\_src\util_misc.py:31
static CUDA_CALLABLE bool is_intersect_0(
    wp::vec_t<2, wp::float32> var_p1,
    wp::vec_t<2, wp::float32> var_p2,
    wp::vec_t<2, wp::float32> var_p3,
    wp::vec_t<2, wp::float32> var_p4)
{
    //---------
    // primal vars
    wp::vec_t<2, wp::float32> var_0;
    wp::vec_t<2, wp::float32> var_1;
    const wp::int32 var_2 = 1;
    wp::float32 var_3;
    const wp::int32 var_4 = 0;
    wp::float32 var_5;
    wp::float32 var_6;
    const wp::int32 var_7 = 0;
    wp::float32 var_8;
    const wp::int32 var_9 = 1;
    wp::float32 var_10;
    wp::float32 var_11;
    wp::float32 var_12;
    wp::float32 var_13;
    const wp::float32 var_14 = 1e-15;
    bool var_15;
    const bool var_16 = false;
    const wp::float32 var_17 = 0.0;
    bool var_18;
    const wp::float32 var_19 = 1.0;
    const wp::float32 var_20 = -1.0;
    wp::float32 var_21;
    wp::vec_t<2, wp::float32> var_22;
    wp::vec_t<2, wp::float32> var_23;
    const wp::int32 var_24 = 0;
    wp::float32 var_25;
    const wp::int32 var_26 = 1;
    wp::float32 var_27;
    wp::float32 var_28;
    const wp::int32 var_29 = 1;
    wp::float32 var_30;
    const wp::int32 var_31 = 0;
    wp::float32 var_32;
    wp::float32 var_33;
    wp::float32 var_34;
    wp::float32 var_35;
    const wp::int32 var_36 = 0;
    wp::float32 var_37;
    const wp::int32 var_38 = 1;
    wp::float32 var_39;
    wp::float32 var_40;
    const wp::int32 var_41 = 1;
    wp::float32 var_42;
    const wp::int32 var_43 = 0;
    wp::float32 var_44;
    wp::float32 var_45;
    wp::float32 var_46;
    wp::float32 var_47;
    const wp::int32 var_48 = 0;
    wp::float32 var_49;
    const wp::int32 var_50 = 1;
    wp::float32 var_51;
    wp::float32 var_52;
    const wp::int32 var_53 = 1;
    wp::float32 var_54;
    const wp::int32 var_55 = 0;
    wp::float32 var_56;
    wp::float32 var_57;
    wp::float32 var_58;
    wp::float32 var_59;
    const wp::int32 var_60 = 0;
    wp::float32 var_61;
    const wp::int32 var_62 = 1;
    wp::float32 var_63;
    wp::float32 var_64;
    const wp::int32 var_65 = 1;
    wp::float32 var_66;
    const wp::int32 var_67 = 0;
    wp::float32 var_68;
    wp::float32 var_69;
    wp::float32 var_70;
    wp::float32 var_71;
    bool var_72;
    const wp::float32 var_73 = 0.0;
    bool var_74;
    const wp::float32 var_75 = 0.0;
    bool var_76;
    const wp::float32 var_77 = 0.0;
    bool var_78;
    const wp::float32 var_79 = 0.0;
    bool var_80;
    //---------
    // forward
    // def is_intersect(p1: wp.vec2, p2: wp.vec2, p3: wp.vec2, p4: wp.vec2) -> bool:          <L 32>
    // u = p2 - p1                                                                            <L 44>
    var_0 = wp::sub(var_p2, var_p1);
    // v = p4 - p3                                                                            <L 45>
    var_1 = wp::sub(var_p4, var_p3);
    // det = v[1] * u[0] - v[0] * u[1]                                                        <L 48>
    var_3 = wp::extract(var_1, var_2);
    var_5 = wp::extract(var_0, var_4);
    var_6 = wp::mul(var_3, var_5);
    var_8 = wp::extract(var_1, var_7);
    var_10 = wp::extract(var_0, var_9);
    var_11 = wp::mul(var_8, var_10);
    var_12 = wp::sub(var_6, var_11);
    // if wp.abs(det) < MJ_MINVAL:                                                            <L 49>
    var_13 = wp::abs(var_12);
    var_15 = (var_13 < var_14);
    if (var_15) {
        // return False                                                                       <L 50>
        return var_16;
    }
    // s = wp.where(det > 0.0, 1.0, -1.0)                                                     <L 53>
    var_18 = (var_12 > var_17);
    var_21 = wp::where(var_18, var_19, var_20);
    // w = p1 - p3                                                                            <L 54>
    var_22 = wp::sub(var_p1, var_p3);
    // d = p2 - p4                                                                            <L 55>
    var_23 = wp::sub(var_p2, var_p4);
    // a0 = (v[0] * w[1] - v[1] * w[0]) * s                                                   <L 56>
    var_25 = wp::extract(var_1, var_24);
    var_27 = wp::extract(var_22, var_26);
    var_28 = wp::mul(var_25, var_27);
    var_30 = wp::extract(var_1, var_29);
    var_32 = wp::extract(var_22, var_31);
    var_33 = wp::mul(var_30, var_32);
    var_34 = wp::sub(var_28, var_33);
    var_35 = wp::mul(var_34, var_21);
    // a1 = (v[0] * d[1] - v[1] * d[0]) * s                                                   <L 57>
    var_37 = wp::extract(var_1, var_36);
    var_39 = wp::extract(var_23, var_38);
    var_40 = wp::mul(var_37, var_39);
    var_42 = wp::extract(var_1, var_41);
    var_44 = wp::extract(var_23, var_43);
    var_45 = wp::mul(var_42, var_44);
    var_46 = wp::sub(var_40, var_45);
    var_47 = wp::mul(var_46, var_21);
    // b0 = (u[0] * w[1] - u[1] * w[0]) * s                                                   <L 58>
    var_49 = wp::extract(var_0, var_48);
    var_51 = wp::extract(var_22, var_50);
    var_52 = wp::mul(var_49, var_51);
    var_54 = wp::extract(var_0, var_53);
    var_56 = wp::extract(var_22, var_55);
    var_57 = wp::mul(var_54, var_56);
    var_58 = wp::sub(var_52, var_57);
    var_59 = wp::mul(var_58, var_21);
    // b1 = (u[0] * d[1] - u[1] * d[0]) * s                                                   <L 59>
    var_61 = wp::extract(var_0, var_60);
    var_63 = wp::extract(var_23, var_62);
    var_64 = wp::mul(var_61, var_63);
    var_66 = wp::extract(var_0, var_65);
    var_68 = wp::extract(var_23, var_67);
    var_69 = wp::mul(var_66, var_68);
    var_70 = wp::sub(var_64, var_69);
    var_71 = wp::mul(var_70, var_21);
    // return a0 >= 0.0 and a1 <= 0.0 and b0 >= 0.0 and b1 <= 0.0                             <L 61>
    var_74 = (var_35 >= var_73);
    var_72 = var_74;
    if (var_72) {
        var_76 = (var_47 <= var_75);
        var_72 = var_72 && var_76;
    }
    if (var_72) {
        var_78 = (var_59 >= var_77);
        var_72 = var_72 && var_78;
    }
    if (var_72) {
        var_80 = (var_71 <= var_79);
        var_72 = var_72 && var_80;
    }
    return var_72;
}


// C:\Users\ASUS\Documents\human-imitation-learning\.venv\Lib\site-packages\mujoco_warp\_src\util_misc.py:81
static CUDA_CALLABLE wp::float32 length_circle_0(
    wp::vec_t<2, wp::float32> var_p0,
    wp::vec_t<2, wp::float32> var_p1,
    wp::int32 var_ind,
    wp::float32 var_radius)
{
    //---------
    // primal vars
    const wp::int32 var_0 = 1;
    wp::float32 var_1;
    const wp::int32 var_2 = 0;
    wp::float32 var_3;
    wp::float32 var_4;
    const wp::int32 var_5 = 0;
    wp::float32 var_6;
    const wp::int32 var_7 = 1;
    wp::float32 var_8;
    wp::float32 var_9;
    wp::float32 var_10;
    wp::float32 var_11;
    wp::float32 var_12;
    wp::float32 var_13;
    bool var_14;
    bool var_15;
    const wp::float32 var_16 = 0.0;
    bool var_17;
    const wp::int32 var_18 = 0;
    bool var_19;
    bool var_20;
    const wp::float32 var_21 = 0.0;
    bool var_22;
    const wp::int32 var_23 = 0;
    bool var_24;
    const wp::float32 var_25 = 2.0;
    const wp::float32 var_26 = 3.141592653589793;
    wp::float32 var_27;
    wp::float32 var_28;
    wp::float32 var_29;
    wp::float32 var_30;
    //---------
    // forward
    // def length_circle(p0: wp.vec2, p1: wp.vec2, ind: int, radius: float) -> float:         <L 82>
    // cross = p0[1] * p1[0] - p0[0] * p1[1]                                                  <L 95>
    var_1 = wp::extract(var_p0, var_0);
    var_3 = wp::extract(var_p1, var_2);
    var_4 = wp::mul(var_1, var_3);
    var_6 = wp::extract(var_p0, var_5);
    var_8 = wp::extract(var_p1, var_7);
    var_9 = wp::mul(var_6, var_8);
    var_10 = wp::sub(var_4, var_9);
    // angle = wp.atan2(wp.abs(cross), wp.dot(p0, p1))                                        <L 96>
    var_11 = wp::abs(var_10);
    var_12 = wp::dot(var_p0, var_p1);
    var_13 = wp::atan2(var_11, var_12);
    // if (cross > 0.0 and ind != 0) or (cross < 0.0 and ind == 0):                           <L 99>
    var_17 = (var_10 > var_16);
    var_15 = var_17;
    if (var_15) {
        var_19 = (var_ind != var_18);
        var_15 = var_15 && var_19;
    }
    var_14 = var_15;
    if (!var_14) {
        var_22 = (var_10 < var_21);
        var_20 = var_22;
        if (var_20) {
            var_24 = (var_ind == var_23);
            var_20 = var_20 && var_24;
        }
        var_14 = var_14 || var_20;
    }
    if (var_14) {
        // angle = 2.0 * wp.pi - angle                                                        <L 100>
        var_27 = wp::mul(var_25, var_26);
        var_28 = wp::sub(var_27, var_13);
    }
    var_29 = wp::where(var_14, var_28, var_13);
    // return radius * angle                                                                  <L 102>
    var_30 = wp::mul(var_radius, var_29);
    return var_30;
}


// C:\Users\ASUS\Documents\human-imitation-learning\.venv\Lib\site-packages\mujoco_warp\_src\util_misc.py:105
static CUDA_CALLABLE void wrap_circle_0(
    wp::vec_t<4, wp::float32> var_end,
    wp::vec_t<2, wp::float32> var_side,
    wp::float32 var_radius,
    wp::float32 & ret_0,
    wp::vec_t<2, wp::float32> & ret_1,
    wp::vec_t<2, wp::float32> & ret_2)
{
    //---------
    // primal vars
    wp::float32 var_0;
    const wp::float32 var_1 = 10000000000.0;
    bool var_2;
    const wp::int32 var_3 = 0;
    wp::float32 var_4;
    const wp::int32 var_5 = 1;
    wp::float32 var_6;
    wp::vec_t<2, wp::float32> var_7;
    const wp::int32 var_8 = 2;
    wp::float32 var_9;
    const wp::int32 var_10 = 3;
    wp::float32 var_11;
    wp::vec_t<2, wp::float32> var_12;
    wp::float32 var_13;
    wp::float32 var_14;
    wp::float32 var_15;
    bool var_16;
    bool var_17;
    bool var_18;
    const wp::float32 var_19 = 1e-15;
    bool var_20;
    const wp::float32 var_21 = -1.0;
    wp::vec_t<2, wp::float32> var_22;
    wp::vec_t<2, wp::float32> var_23;
    wp::vec_t<2, wp::float32> var_24;
    wp::float32 var_25;
    bool var_26;
    const wp::float32 var_27 = -1.0;
    wp::vec_t<2, wp::float32> var_28;
    wp::vec_t<2, wp::float32> var_29;
    wp::float32 var_30;
    wp::float32 var_31;
    wp::float32 var_32;
    const wp::float32 var_33 = 0.0;
    const wp::float32 var_34 = 1.0;
    wp::float32 var_35;
    wp::vec_t<2, wp::float32> var_36;
    wp::vec_t<2, wp::float32> var_37;
    bool var_38;
    wp::float32 var_39;
    bool var_40;
    bool var_41;
    bool var_42;
    wp::float32 var_43;
    const wp::float32 var_44 = 0.0;
    bool var_45;
    const wp::float32 var_46 = -1.0;
    wp::vec_t<2, wp::float32> var_47;
    wp::vec_t<2, wp::float32> var_48;
    wp::float32 var_49;
    wp::float32 var_50;
    wp::float32 var_51;
    wp::float32 var_52;
    const wp::int32 var_53 = 0;
    wp::float32 var_54;
    wp::float32 var_55;
    const wp::int32 var_56 = 1;
    wp::float32 var_57;
    wp::float32 var_58;
    wp::float32 var_59;
    wp::float32 var_60;
    wp::float32 var_61;
    const wp::int32 var_62 = 1;
    wp::float32 var_63;
    wp::float32 var_64;
    const wp::int32 var_65 = 0;
    wp::float32 var_66;
    wp::float32 var_67;
    wp::float32 var_68;
    wp::float32 var_69;
    wp::float32 var_70;
    wp::vec_t<2, wp::float32> var_71;
    const wp::int32 var_72 = 2;
    wp::float32 var_73;
    wp::float32 var_74;
    const wp::int32 var_75 = 3;
    wp::float32 var_76;
    wp::float32 var_77;
    wp::float32 var_78;
    wp::float32 var_79;
    wp::float32 var_80;
    const wp::int32 var_81 = 3;
    wp::float32 var_82;
    wp::float32 var_83;
    const wp::int32 var_84 = 2;
    wp::float32 var_85;
    wp::float32 var_86;
    wp::float32 var_87;
    wp::float32 var_88;
    wp::float32 var_89;
    wp::vec_t<2, wp::float32> var_90;
    const wp::int32 var_91 = 0;
    wp::float32 var_92;
    wp::float32 var_93;
    const wp::int32 var_94 = 1;
    wp::float32 var_95;
    wp::float32 var_96;
    wp::float32 var_97;
    wp::float32 var_98;
    wp::float32 var_99;
    const wp::int32 var_100 = 1;
    wp::float32 var_101;
    wp::float32 var_102;
    const wp::int32 var_103 = 0;
    wp::float32 var_104;
    wp::float32 var_105;
    wp::float32 var_106;
    wp::float32 var_107;
    wp::float32 var_108;
    wp::vec_t<2, wp::float32> var_109;
    const wp::int32 var_110 = 2;
    wp::float32 var_111;
    wp::float32 var_112;
    const wp::int32 var_113 = 3;
    wp::float32 var_114;
    wp::float32 var_115;
    wp::float32 var_116;
    wp::float32 var_117;
    wp::float32 var_118;
    const wp::int32 var_119 = 3;
    wp::float32 var_120;
    wp::float32 var_121;
    const wp::int32 var_122 = 2;
    wp::float32 var_123;
    wp::float32 var_124;
    wp::float32 var_125;
    wp::float32 var_126;
    wp::float32 var_127;
    wp::vec_t<2, wp::float32> var_128;
    wp::vec_t<2, wp::float32> var_129;
    wp::vec_t<2, wp::float32> var_130;
    wp::float32 var_131;
    wp::float32 var_132;
    wp::vec_t<2, wp::float32> var_133;
    wp::vec_t<2, wp::float32> var_134;
    wp::float32 var_135;
    wp::float32 var_136;
    wp::vec_t<2, wp::float32> var_137;
    wp::float32 var_138;
    wp::float32 var_139;
    wp::vec_t<2, wp::float32> var_140;
    wp::float32 var_141;
    wp::float32 var_142;
    wp::vec_t<2, wp::float32> var_143;
    wp::float32 var_144;
    wp::vec_t<2, wp::float32> var_145;
    wp::float32 var_146;
    bool var_147;
    const wp::float32 var_148 = -10000.0;
    wp::float32 var_149;
    bool var_150;
    const wp::float32 var_151 = -10000.0;
    wp::float32 var_152;
    bool var_153;
    const wp::float32 var_154 = -1.0;
    wp::vec_t<2, wp::float32> var_155;
    wp::vec_t<2, wp::float32> var_156;
    wp::vec_t<2, wp::float32> var_157;
    wp::vec_t<2, wp::float32> var_158;
    const wp::int32 var_159 = 0;
    const wp::float32 var_160 = -1.0;
    wp::vec_t<2, wp::float32> var_161;
    wp::vec_t<2, wp::float32> var_162;
    wp::vec_t<2, wp::float32> var_163;
    wp::vec_t<2, wp::float32> var_164;
    const wp::int32 var_165 = 1;
    wp::vec_t<2, wp::float32> var_166;
    wp::vec_t<2, wp::float32> var_167;
    wp::int32 var_168;
    wp::float32 var_169;
    //---------
    // forward
    // def wrap_circle(end: wp.vec4, side: wp.vec2, radius: float) -> Tuple[float, wp.vec2, wp.vec2]:       <L 106>
    // valid_side = wp.norm_l2(side) < MJ_MAXVAL                                              <L 117>
    var_0 = norm_l2_0(var_side);
    var_2 = (var_0 < var_1);
    // end0 = wp.vec2(end[0], end[1])                                                         <L 119>
    var_4 = wp::extract(var_end, var_3);
    var_6 = wp::extract(var_end, var_5);
    var_7 = wp::vec_t<2, wp::float32>(var_4, var_6);
    // end1 = wp.vec2(end[2], end[3])                                                         <L 120>
    var_9 = wp::extract(var_end, var_8);
    var_11 = wp::extract(var_end, var_10);
    var_12 = wp::vec_t<2, wp::float32>(var_9, var_11);
    // sqlen0 = wp.dot(end0, end0)                                                            <L 122>
    var_13 = wp::dot(var_7, var_7);
    // sqlen1 = wp.dot(end1, end1)                                                            <L 123>
    var_14 = wp::dot(var_12, var_12);
    // sqrad = radius * radius                                                                <L 124>
    var_15 = wp::mul(var_radius, var_radius);
    // if (sqlen0 < sqrad) or (sqlen1 < sqrad) or (radius < MJ_MINVAL):                       <L 127>
    var_17 = (var_13 < var_15);
    var_16 = var_17;
    if (!var_16) {
        var_18 = (var_14 < var_15);
        var_16 = var_16 || var_18;
    }
    if (!var_16) {
        var_20 = (var_radius < var_19);
        var_16 = var_16 || var_20;
    }
    if (var_16) {
        // return -1.0, wp.vec2(MJ_MAXVAL), wp.vec2(MJ_MAXVAL)                                <L 128>
        var_22 = wp::vec_t<2, wp::float32>(var_1);
        var_23 = wp::vec_t<2, wp::float32>(var_1);
        ret_0 = var_21;
        ret_1 = var_22;
        ret_2 = var_23;
        return;
    }
    // dif = end1 - end0                                                                      <L 131>
    var_24 = wp::sub(var_12, var_7);
    // dd = wp.dot(dif, dif)                                                                  <L 132>
    var_25 = wp::dot(var_24, var_24);
    // if dd < MJ_MINVAL:                                                                     <L 133>
    var_26 = (var_25 < var_19);
    if (var_26) {
        // return -1.0, wp.vec2(MJ_MAXVAL), wp.vec2(MJ_MAXVAL)                                <L 134>
        var_28 = wp::vec_t<2, wp::float32>(var_1);
        var_29 = wp::vec_t<2, wp::float32>(var_1);
        ret_0 = var_27;
        ret_1 = var_28;
        ret_2 = var_29;
        return;
    }
    // a = -wp.dot(dif, end0) / dd                                                            <L 137>
    var_30 = wp::dot(var_24, var_7);
    var_31 = wp::neg(var_30);
    var_32 = wp::div(var_31, var_25);
    // a = wp.clamp(a, 0.0, 1.0)                                                              <L 138>
    var_35 = wp::clamp(var_32, var_33, var_34);
    // tmp = a * dif + end0                                                                   <L 141>
    var_36 = wp::mul(var_35, var_24);
    var_37 = wp::add(var_36, var_7);
    // if (wp.dot(tmp, tmp) > sqrad) and (not valid_side or wp.dot(side, tmp) >= 0.0):        <L 142>
    var_39 = wp::dot(var_37, var_37);
    var_40 = (var_39 > var_15);
    var_38 = var_40;
    if (var_38) {
        var_42 = wp::unot(var_2);
        var_41 = var_42;
        if (!var_41) {
            var_43 = wp::dot(var_side, var_37);
            var_45 = (var_43 >= var_44);
            var_41 = var_41 || var_45;
        }
        var_38 = var_38 && var_41;
    }
    if (var_38) {
        // return -1.0, wp.vec2(MJ_MAXVAL), wp.vec2(MJ_MAXVAL)                                <L 143>
        var_47 = wp::vec_t<2, wp::float32>(var_1);
        var_48 = wp::vec_t<2, wp::float32>(var_1);
        ret_0 = var_46;
        ret_1 = var_47;
        ret_2 = var_48;
        return;
    }
    // sqrt0 = wp.sqrt(sqlen0 - sqrad)                                                        <L 145>
    var_49 = wp::sub(var_13, var_15);
    var_50 = wp::sqrt(var_49);
    // sqrt1 = wp.sqrt(sqlen1 - sqrad)                                                        <L 146>
    var_51 = wp::sub(var_14, var_15);
    var_52 = wp::sqrt(var_51);
    // sol00 = wp.vec2(                                                                       <L 149>
    // math.safe_div(end[0] * sqrad + radius * end[1] * sqrt0, sqlen0),                       <L 150>
    var_54 = wp::extract(var_end, var_53);
    var_55 = wp::mul(var_54, var_15);
    var_57 = wp::extract(var_end, var_56);
    var_58 = wp::mul(var_radius, var_57);
    var_59 = wp::mul(var_58, var_50);
    var_60 = wp::add(var_55, var_59);
    var_61 = safe_div_0(var_60, var_13);
    // math.safe_div(end[1] * sqrad - radius * end[0] * sqrt0, sqlen0),                       <L 151>
    var_63 = wp::extract(var_end, var_62);
    var_64 = wp::mul(var_63, var_15);
    var_66 = wp::extract(var_end, var_65);
    var_67 = wp::mul(var_radius, var_66);
    var_68 = wp::mul(var_67, var_50);
    var_69 = wp::sub(var_64, var_68);
    var_70 = safe_div_0(var_69, var_13);
    var_71 = wp::vec_t<2, wp::float32>(var_61, var_70);
    // sol01 = wp.vec2(                                                                       <L 153>
    // math.safe_div(end[2] * sqrad - radius * end[3] * sqrt1, sqlen1),                       <L 154>
    var_73 = wp::extract(var_end, var_72);
    var_74 = wp::mul(var_73, var_15);
    var_76 = wp::extract(var_end, var_75);
    var_77 = wp::mul(var_radius, var_76);
    var_78 = wp::mul(var_77, var_52);
    var_79 = wp::sub(var_74, var_78);
    var_80 = safe_div_0(var_79, var_14);
    // math.safe_div(end[3] * sqrad + radius * end[2] * sqrt1, sqlen1),                       <L 155>
    var_82 = wp::extract(var_end, var_81);
    var_83 = wp::mul(var_82, var_15);
    var_85 = wp::extract(var_end, var_84);
    var_86 = wp::mul(var_radius, var_85);
    var_87 = wp::mul(var_86, var_52);
    var_88 = wp::add(var_83, var_87);
    var_89 = safe_div_0(var_88, var_14);
    var_90 = wp::vec_t<2, wp::float32>(var_80, var_89);
    // sol10 = wp.vec2(                                                                       <L 158>
    // math.safe_div(end[0] * sqrad - radius * end[1] * sqrt0, sqlen0),                       <L 159>
    var_92 = wp::extract(var_end, var_91);
    var_93 = wp::mul(var_92, var_15);
    var_95 = wp::extract(var_end, var_94);
    var_96 = wp::mul(var_radius, var_95);
    var_97 = wp::mul(var_96, var_50);
    var_98 = wp::sub(var_93, var_97);
    var_99 = safe_div_0(var_98, var_13);
    // math.safe_div(end[1] * sqrad + radius * end[0] * sqrt0, sqlen0),                       <L 160>
    var_101 = wp::extract(var_end, var_100);
    var_102 = wp::mul(var_101, var_15);
    var_104 = wp::extract(var_end, var_103);
    var_105 = wp::mul(var_radius, var_104);
    var_106 = wp::mul(var_105, var_50);
    var_107 = wp::add(var_102, var_106);
    var_108 = safe_div_0(var_107, var_13);
    var_109 = wp::vec_t<2, wp::float32>(var_99, var_108);
    // sol11 = wp.vec2(                                                                       <L 162>
    // math.safe_div(end[2] * sqrad + radius * end[3] * sqrt1, sqlen1),                       <L 163>
    var_111 = wp::extract(var_end, var_110);
    var_112 = wp::mul(var_111, var_15);
    var_114 = wp::extract(var_end, var_113);
    var_115 = wp::mul(var_radius, var_114);
    var_116 = wp::mul(var_115, var_52);
    var_117 = wp::add(var_112, var_116);
    var_118 = safe_div_0(var_117, var_14);
    // math.safe_div(end[3] * sqrad - radius * end[2] * sqrt1, sqlen1),                       <L 164>
    var_120 = wp::extract(var_end, var_119);
    var_121 = wp::mul(var_120, var_15);
    var_123 = wp::extract(var_end, var_122);
    var_124 = wp::mul(var_radius, var_123);
    var_125 = wp::mul(var_124, var_52);
    var_126 = wp::sub(var_121, var_125);
    var_127 = safe_div_0(var_126, var_14);
    var_128 = wp::vec_t<2, wp::float32>(var_118, var_127);
    // if valid_side:                                                                         <L 168>
    if (var_2) {
        // tmp0, _ = math.normalize_with_norm(sol00 + sol01)                                  <L 169>
        var_129 = wp::add(var_71, var_90);
        normalize_with_norm_0(var_129, var_130, var_131);
        // good0 = wp.dot(tmp0, side)                                                         <L 170>
        var_132 = wp::dot(var_130, var_side);
        // tmp1, _ = math.normalize_with_norm(sol10 + sol11)                                  <L 171>
        var_133 = wp::add(var_109, var_128);
        normalize_with_norm_0(var_133, var_134, var_135);
        // good1 = wp.dot(tmp1, side)                                                         <L 172>
        var_136 = wp::dot(var_134, var_side);
    }
    if (!var_2) {
        // tmp0 = sol00 - sol01                                                               <L 174>
        var_137 = wp::sub(var_71, var_90);
        // good0 = -wp.dot(tmp0, tmp0)                                                        <L 175>
        var_138 = wp::dot(var_137, var_137);
        var_139 = wp::neg(var_138);
        // tmp1 = sol10 - sol11                                                               <L 176>
        var_140 = wp::sub(var_109, var_128);
        // good1 = -wp.dot(tmp1, tmp1)                                                        <L 177>
        var_141 = wp::dot(var_140, var_140);
        var_142 = wp::neg(var_141);
    }
    var_143 = wp::where(var_2, var_130, var_137);
    var_144 = wp::where(var_2, var_132, var_139);
    var_145 = wp::where(var_2, var_134, var_140);
    var_146 = wp::where(var_2, var_136, var_142);
    // intersect0 = is_intersect(end0, sol00, end1, sol01)                                    <L 180>
    var_147 = is_intersect_0(var_7, var_71, var_12, var_90);
    // if intersect0:                                                                         <L 181>
    if (var_147) {
        // good0 = -10000.0                                                                   <L 182>
    }
    var_149 = wp::where(var_147, var_148, var_144);
    // intersect1 = is_intersect(end0, sol10, end1, sol11)                                    <L 183>
    var_150 = is_intersect_0(var_7, var_109, var_12, var_128);
    // if intersect1:                                                                         <L 184>
    if (var_150) {
        // good1 = -10000.0                                                                   <L 185>
    }
    var_152 = wp::where(var_150, var_151, var_146);
    // if good0 > good1:                                                                      <L 188>
    var_153 = (var_149 > var_152);
    if (var_153) {
        // if intersect0:                                                                     <L 189>
        if (var_147) {
            // return -1.0, wp.vec2(MJ_MAXVAL), wp.vec2(MJ_MAXVAL)                            <L 190>
            var_155 = wp::vec_t<2, wp::float32>(var_1);
            var_156 = wp::vec_t<2, wp::float32>(var_1);
            ret_0 = var_154;
            ret_1 = var_155;
            ret_2 = var_156;
            return;
        }
        // pnt0 = sol00                                                                       <L 191>
        var_157 = wp::copy(var_71);
        // pnt1 = sol01                                                                       <L 192>
        var_158 = wp::copy(var_90);
        // ind = 0                                                                            <L 193>
    }
    if (!var_153) {
        // if intersect1:                                                                     <L 195>
        if (var_150) {
            // return -1.0, wp.vec2(MJ_MAXVAL), wp.vec2(MJ_MAXVAL)                            <L 196>
            var_161 = wp::vec_t<2, wp::float32>(var_1);
            var_162 = wp::vec_t<2, wp::float32>(var_1);
            ret_0 = var_160;
            ret_1 = var_161;
            ret_2 = var_162;
            return;
        }
        // pnt0 = sol10                                                                       <L 197>
        var_163 = wp::copy(var_109);
        // pnt1 = sol11                                                                       <L 198>
        var_164 = wp::copy(var_128);
        // ind = 1                                                                            <L 199>
    }
    var_166 = wp::where(var_153, var_157, var_163);
    var_167 = wp::where(var_153, var_158, var_164);
    var_168 = wp::where(var_153, var_159, var_165);
    // return length_circle(pnt0, pnt1, ind, radius), pnt0, pnt1                              <L 202>
    var_169 = length_circle_0(var_166, var_167, var_168, var_radius);
    ret_0 = var_169;
    ret_1 = var_166;
    ret_2 = var_167;
    return;
}


// C:\Users\ASUS\Documents\human-imitation-learning\.venv\Lib\site-packages\mujoco_warp\_src\util_misc.py:332
static CUDA_CALLABLE void wrap_0(
    wp::vec_t<3, wp::float32> var_x0,
    wp::vec_t<3, wp::float32> var_x1,
    wp::vec_t<3, wp::float32> var_pos,
    wp::mat_t<3, 3, wp::float32> var_mat,
    wp::float32 var_radius,
    wp::int32 var_geomtype,
    wp::vec_t<3, wp::float32> var_side,
    wp::float32 & ret_0,
    wp::vec_t<3, wp::float32> & ret_1,
    wp::vec_t<3, wp::float32> & ret_2)
{
    //---------
    // primal vars
    bool var_0;
    const wp::int32 var_1 = 4;
    bool var_2;
    const wp::int32 var_3 = 5;
    bool var_4;
    const wp::float32 var_5 = 10000000000.0;
    wp::vec_t<3, wp::float32> var_6;
    wp::vec_t<3, wp::float32> var_7;
    wp::mat_t<3, 3, wp::float32> var_8;
    wp::vec_t<3, wp::float32> var_9;
    wp::vec_t<3, wp::float32> var_10;
    wp::vec_t<3, wp::float32> var_11;
    wp::vec_t<3, wp::float32> var_12;
    bool var_13;
    wp::float32 var_14;
    const wp::float32 var_15 = 1e-15;
    bool var_16;
    wp::float32 var_17;
    bool var_18;
    const wp::float32 var_19 = -1.0;
    wp::vec_t<3, wp::float32> var_20;
    wp::vec_t<3, wp::float32> var_21;
    const wp::int32 var_22 = 4;
    bool var_23;
    wp::vec_t<3, wp::float32> var_24;
    wp::float32 var_25;
    wp::vec_t<3, wp::float32> var_26;
    wp::vec_t<3, wp::float32> var_27;
    wp::float32 var_28;
    bool var_29;
    wp::vec_t<3, wp::float32> var_30;
    const wp::int32 var_31 = 0;
    wp::int32 var_32;
    bool var_33;
    const wp::int32 var_34 = 1;
    wp::float32 var_35;
    const wp::int32 var_36 = 0;
    wp::float32 var_37;
    bool var_38;
    const wp::int32 var_39 = 1;
    wp::float32 var_40;
    const wp::int32 var_41 = 2;
    wp::float32 var_42;
    bool var_43;
    const wp::int32 var_44 = 1;
    wp::int32 var_45;
    bool var_46;
    const wp::int32 var_47 = 2;
    wp::float32 var_48;
    const wp::int32 var_49 = 0;
    wp::float32 var_50;
    bool var_51;
    const wp::int32 var_52 = 2;
    wp::float32 var_53;
    const wp::int32 var_54 = 1;
    wp::float32 var_55;
    bool var_56;
    const wp::int32 var_57 = 2;
    wp::int32 var_58;
    const wp::float32 var_59 = 1.0;
    wp::vec_t<3, wp::float32> var_60;
    const wp::float32 var_61 = 0.0;
    wp::vec_t<3, wp::float32> var_62;
    wp::vec_t<3, wp::float32> var_63;
    wp::float32 var_64;
    wp::float32 var_65;
    wp::vec_t<3, wp::float32> var_66;
    wp::vec_t<3, wp::float32> var_67;
    wp::vec_t<3, wp::float32> var_68;
    wp::float32 var_69;
    const wp::float32 var_70 = 1.0;
    const wp::float32 var_71 = 0.0;
    const wp::float32 var_72 = 0.0;
    wp::vec_t<3, wp::float32> var_73;
    const wp::float32 var_74 = 0.0;
    const wp::float32 var_75 = 1.0;
    const wp::float32 var_76 = 0.0;
    wp::vec_t<3, wp::float32> var_77;
    wp::vec_t<3, wp::float32> var_78;
    wp::vec_t<3, wp::float32> var_79;
    wp::float32 var_80;
    wp::float32 var_81;
    wp::float32 var_82;
    wp::float32 var_83;
    wp::vec_t<4, wp::float32> var_84;
    wp::float32 var_85;
    bool var_86;
    wp::vec_t<3, wp::float32> var_87;
    wp::vec_t<3, wp::float32> var_88;
    wp::float32 var_89;
    wp::float32 var_90;
    wp::vec_t<2, wp::float32> var_91;
    wp::vec_t<2, wp::float32> var_92;
    wp::float32 var_93;
    wp::vec_t<2, wp::float32> var_94;
    wp::vec_t<2, wp::float32> var_95;
    wp::float32 var_96;
    wp::vec_t<2, wp::float32> var_97;
    bool var_98;
    wp::float32 var_99;
    bool var_100;
    wp::float32 var_101;
    wp::vec_t<2, wp::float32> var_102;
    wp::vec_t<2, wp::float32> var_103;
    wp::float32 var_104;
    wp::vec_t<2, wp::float32> var_105;
    wp::vec_t<2, wp::float32> var_106;
    wp::float32 var_107;
    wp::vec_t<2, wp::float32> var_108;
    wp::vec_t<2, wp::float32> var_109;
    const wp::float32 var_110 = 0.0;
    bool var_111;
    const wp::float32 var_112 = -1.0;
    wp::vec_t<3, wp::float32> var_113;
    wp::vec_t<3, wp::float32> var_114;
    const wp::int32 var_115 = 0;
    wp::float32 var_116;
    wp::vec_t<3, wp::float32> var_117;
    const wp::int32 var_118 = 1;
    wp::float32 var_119;
    wp::vec_t<3, wp::float32> var_120;
    wp::vec_t<3, wp::float32> var_121;
    const wp::int32 var_122 = 0;
    wp::float32 var_123;
    wp::vec_t<3, wp::float32> var_124;
    const wp::int32 var_125 = 1;
    wp::float32 var_126;
    wp::vec_t<3, wp::float32> var_127;
    wp::vec_t<3, wp::float32> var_128;
    const wp::int32 var_129 = 5;
    bool var_130;
    const wp::int32 var_131 = 0;
    wp::float32 var_132;
    const wp::int32 var_133 = 0;
    wp::float32 var_134;
    wp::float32 var_135;
    const wp::int32 var_136 = 0;
    wp::float32 var_137;
    const wp::int32 var_138 = 0;
    wp::float32 var_139;
    wp::float32 var_140;
    wp::float32 var_141;
    const wp::int32 var_142 = 1;
    wp::float32 var_143;
    const wp::int32 var_144 = 1;
    wp::float32 var_145;
    wp::float32 var_146;
    const wp::int32 var_147 = 1;
    wp::float32 var_148;
    const wp::int32 var_149 = 1;
    wp::float32 var_150;
    wp::float32 var_151;
    wp::float32 var_152;
    wp::float32 var_153;
    wp::float32 var_154;
    const wp::int32 var_155 = 0;
    wp::float32 var_156;
    const wp::int32 var_157 = 0;
    wp::float32 var_158;
    wp::float32 var_159;
    const wp::int32 var_160 = 0;
    wp::float32 var_161;
    const wp::int32 var_162 = 0;
    wp::float32 var_163;
    wp::float32 var_164;
    wp::float32 var_165;
    const wp::int32 var_166 = 1;
    wp::float32 var_167;
    const wp::int32 var_168 = 1;
    wp::float32 var_169;
    wp::float32 var_170;
    const wp::int32 var_171 = 1;
    wp::float32 var_172;
    const wp::int32 var_173 = 1;
    wp::float32 var_174;
    wp::float32 var_175;
    wp::float32 var_176;
    wp::float32 var_177;
    wp::float32 var_178;
    const wp::int32 var_179 = 2;
    wp::float32 var_180;
    const wp::int32 var_181 = 2;
    wp::float32 var_182;
    const wp::int32 var_183 = 2;
    wp::float32 var_184;
    wp::float32 var_185;
    wp::float32 var_186;
    wp::float32 var_187;
    wp::float32 var_188;
    wp::float32 var_189;
    wp::float32 var_190;
    const wp::int32 var_191 = 2;
    const wp::int32 var_192 = 2;
    wp::float32 var_193;
    const wp::int32 var_194 = 2;
    wp::float32 var_195;
    const wp::int32 var_196 = 2;
    wp::float32 var_197;
    wp::float32 var_198;
    wp::float32 var_199;
    wp::float32 var_200;
    wp::float32 var_201;
    wp::float32 var_202;
    wp::float32 var_203;
    wp::float32 var_204;
    const wp::int32 var_205 = 2;
    const wp::int32 var_206 = 2;
    wp::float32 var_207;
    const wp::int32 var_208 = 2;
    wp::float32 var_209;
    wp::float32 var_210;
    wp::float32 var_211;
    wp::float32 var_212;
    wp::float32 var_213;
    wp::float32 var_214;
    wp::float32 var_215;
    wp::float32 var_216;
    wp::vec_t<3, wp::float32> var_217;
    wp::vec_t<3, wp::float32> var_218;
    wp::vec_t<3, wp::float32> var_219;
    wp::vec_t<3, wp::float32> var_220;
    //---------
    // forward
    // def wrap(                                                                              <L 333>
    // if geomtype != WrapType.SPHERE and geomtype != WrapType.CYLINDER:                      <L 351>
    var_2 = (var_geomtype != var_1);
    var_0 = var_2;
    if (var_0) {
        var_4 = (var_geomtype != var_3);
        var_0 = var_0 && var_4;
    }
    if (var_0) {
        // return MJ_MAXVAL, wp.vec3(MJ_MAXVAL), wp.vec3(MJ_MAXVAL)                           <L 352>
        var_6 = wp::vec_t<3, wp::float32>(var_5);
        var_7 = wp::vec_t<3, wp::float32>(var_5);
        ret_0 = var_5;
        ret_1 = var_6;
        ret_2 = var_7;
        return;
    }
    // matT = wp.transpose(mat)                                                               <L 355>
    var_8 = wp::transpose(var_mat);
    // p0 = matT @ (x0 - pos)                                                                 <L 356>
    var_9 = wp::sub(var_x0, var_pos);
    var_10 = wp::mul(var_8, var_9);
    // p1 = matT @ (x1 - pos)                                                                 <L 357>
    var_11 = wp::sub(var_x1, var_pos);
    var_12 = wp::mul(var_8, var_11);
    // if (wp.norm_l2(p0) < MJ_MINVAL) or (wp.norm_l2(p1) < MJ_MINVAL):                       <L 360>
    var_14 = norm_l2_0(var_10);
    var_16 = (var_14 < var_15);
    var_13 = var_16;
    if (!var_13) {
        var_17 = norm_l2_0(var_12);
        var_18 = (var_17 < var_15);
        var_13 = var_13 || var_18;
    }
    if (var_13) {
        // return -1.0, wp.vec3(MJ_MAXVAL), wp.vec3(MJ_MAXVAL)                                <L 361>
        var_20 = wp::vec_t<3, wp::float32>(var_5);
        var_21 = wp::vec_t<3, wp::float32>(var_5);
        ret_0 = var_19;
        ret_1 = var_20;
        ret_2 = var_21;
        return;
    }
    // if geomtype == WrapType.SPHERE:                                                        <L 364>
    var_23 = (var_geomtype == var_22);
    if (var_23) {
        // axis0, _ = math.normalize_with_norm(p0)                                            <L 366>
        normalize_with_norm_0(var_10, var_24, var_25);
        // normal = wp.cross(p0, p1)                                                          <L 369>
        var_26 = wp::cross(var_10, var_12);
        // normal, nrm = math.normalize_with_norm(normal)                                     <L 370>
        normalize_with_norm_0(var_26, var_27, var_28);
        // if nrm < MJ_MINVAL:                                                                <L 373>
        var_29 = (var_28 < var_15);
        if (var_29) {
            // axis0_abs = wp.abs(axis0)                                                      <L 375>
            var_30 = wp::abs(var_24);
            // i = int(0)                                                                     <L 376>
            var_32 = wp::int(var_31);
            // if (axis0_abs[1] > axis0_abs[0]) and (axis0_abs[1] > axis0_abs[2]):            <L 377>
            var_35 = wp::extract(var_30, var_34);
            var_37 = wp::extract(var_30, var_36);
            var_38 = (var_35 > var_37);
            var_33 = var_38;
            if (var_33) {
                var_40 = wp::extract(var_30, var_39);
                var_42 = wp::extract(var_30, var_41);
                var_43 = (var_40 > var_42);
                var_33 = var_33 && var_43;
            }
            if (var_33) {
                // i = 1                                                                      <L 378>
            }
            var_45 = wp::where(var_33, var_44, var_32);
            // if (axis0_abs[2] > axis0_abs[0]) and (axis0_abs[2] > axis0_abs[1]):            <L 379>
            var_48 = wp::extract(var_30, var_47);
            var_50 = wp::extract(var_30, var_49);
            var_51 = (var_48 > var_50);
            var_46 = var_51;
            if (var_46) {
                var_53 = wp::extract(var_30, var_52);
                var_55 = wp::extract(var_30, var_54);
                var_56 = (var_53 > var_55);
                var_46 = var_46 && var_56;
            }
            if (var_46) {
                // i = 2                                                                      <L 380>
            }
            var_58 = wp::where(var_46, var_57, var_45);
            // axis1 = wp.vec3(1.0)                                                           <L 383>
            var_60 = wp::vec_t<3, wp::float32>(var_59);
            // axis1[i] = 0.0                                                                 <L 384>
            wp::assign_inplace(var_60, var_58, var_61);
            // normal = wp.cross(axis0, axis1)                                                <L 387>
            var_62 = wp::cross(var_24, var_60);
            // normal, _ = math.normalize_with_norm(normal)                                   <L 388>
            normalize_with_norm_0(var_62, var_63, var_64);
        }
        var_65 = wp::where(var_29, var_64, var_25);
        var_66 = wp::where(var_29, var_63, var_27);
        // axis1 = wp.cross(normal, axis0)                                                    <L 391>
        var_67 = wp::cross(var_66, var_24);
        // axis1, _ = math.normalize_with_norm(axis1)                                         <L 392>
        normalize_with_norm_0(var_67, var_68, var_69);
    }
    if (!var_23) {
        // axis0 = wp.vec3(1.0, 0.0, 0.0)                                                     <L 395>
        var_73 = wp::vec_t<3, wp::float32>(var_70, var_71, var_72);
        // axis1 = wp.vec3(0.0, 1.0, 0.0)                                                     <L 398>
        var_77 = wp::vec_t<3, wp::float32>(var_74, var_75, var_76);
    }
    var_78 = wp::where(var_23, var_24, var_73);
    var_79 = wp::where(var_23, var_68, var_77);
    // end = wp.vec4(                                                                         <L 401>
    // wp.dot(p0, axis0),                                                                     <L 402>
    var_80 = wp::dot(var_10, var_78);
    // wp.dot(p0, axis1),                                                                     <L 403>
    var_81 = wp::dot(var_10, var_79);
    // wp.dot(p1, axis0),                                                                     <L 404>
    var_82 = wp::dot(var_12, var_78);
    // wp.dot(p1, axis1),                                                                     <L 405>
    var_83 = wp::dot(var_12, var_79);
    var_84 = wp::vec_t<4, wp::float32>(var_80, var_81, var_82, var_83);
    // valid_side = wp.norm_l2(side) < MJ_MAXVAL                                              <L 409>
    var_85 = norm_l2_0(var_side);
    var_86 = (var_85 < var_5);
    // if valid_side:                                                                         <L 411>
    if (var_86) {
        // sidepnt = matT @ (side - pos)                                                      <L 413>
        var_87 = wp::sub(var_side, var_pos);
        var_88 = wp::mul(var_8, var_87);
        // sidepnt_proj = wp.vec2(                                                            <L 416>
        // wp.dot(sidepnt, axis0),                                                            <L 417>
        var_89 = wp::dot(var_88, var_78);
        // wp.dot(sidepnt, axis1),                                                            <L 418>
        var_90 = wp::dot(var_88, var_79);
        var_91 = wp::vec_t<2, wp::float32>(var_89, var_90);
        // sidepnt_proj, _ = math.normalize_with_norm(sidepnt_proj)                           <L 421>
        normalize_with_norm_0(var_91, var_92, var_93);
        // sidepnt_proj *= radius                                                             <L 422>
        var_94 = wp::mul(var_92, var_radius);
    }
    if (!var_86) {
        // sidepnt_proj = wp.vec2(MJ_MAXVAL)                                                  <L 424>
        var_95 = wp::vec_t<2, wp::float32>(var_5);
    }
    var_96 = wp::where(var_86, var_93, var_69);
    var_97 = wp::where(var_86, var_94, var_95);
    // if valid_side and wp.norm_l2(sidepnt) < radius:                                        <L 427>
    var_98 = var_86;
    if (var_98) {
        var_99 = norm_l2_0(var_88);
        var_100 = (var_99 < var_radius);
        var_98 = var_98 && var_100;
    }
    if (var_98) {
        // wlen, pnt0, pnt1 = wrap_inside(end, radius)                                        <L 428>
        wrap_inside_0(var_84, var_radius, var_101, var_102, var_103);
    }
    if (!var_98) {
        // wlen, pnt0, pnt1 = wrap_circle(end, sidepnt_proj, radius)                          <L 430>
        wrap_circle_0(var_84, var_97, var_radius, var_104, var_105, var_106);
    }
    var_107 = wp::where(var_98, var_101, var_104);
    var_108 = wp::where(var_98, var_102, var_105);
    var_109 = wp::where(var_98, var_103, var_106);
    // if wlen < 0.0:                                                                         <L 433>
    var_111 = (var_107 < var_110);
    if (var_111) {
        // return -1.0, wp.vec3(MJ_MAXVAL), wp.vec3(MJ_MAXVAL)                                <L 434>
        var_113 = wp::vec_t<3, wp::float32>(var_5);
        var_114 = wp::vec_t<3, wp::float32>(var_5);
        ret_0 = var_112;
        ret_1 = var_113;
        ret_2 = var_114;
        return;
    }
    // res0 = axis0 * pnt0[0] + axis1 * pnt0[1]                                               <L 437>
    var_116 = wp::extract(var_108, var_115);
    var_117 = wp::mul(var_78, var_116);
    var_119 = wp::extract(var_108, var_118);
    var_120 = wp::mul(var_79, var_119);
    var_121 = wp::add(var_117, var_120);
    // res1 = axis0 * pnt1[0] + axis1 * pnt1[1]                                               <L 438>
    var_123 = wp::extract(var_109, var_122);
    var_124 = wp::mul(var_78, var_123);
    var_126 = wp::extract(var_109, var_125);
    var_127 = wp::mul(var_79, var_126);
    var_128 = wp::add(var_124, var_127);
    // if geomtype == WrapType.CYLINDER:                                                      <L 441>
    var_130 = (var_geomtype == var_129);
    if (var_130) {
        // L0 = wp.sqrt((p0[0] - res0[0]) * (p0[0] - res0[0]) + (p0[1] - res0[1]) * (p0[1] - res0[1]))       <L 443>
        var_132 = wp::extract(var_10, var_131);
        var_134 = wp::extract(var_121, var_133);
        var_135 = wp::sub(var_132, var_134);
        var_137 = wp::extract(var_10, var_136);
        var_139 = wp::extract(var_121, var_138);
        var_140 = wp::sub(var_137, var_139);
        var_141 = wp::mul(var_135, var_140);
        var_143 = wp::extract(var_10, var_142);
        var_145 = wp::extract(var_121, var_144);
        var_146 = wp::sub(var_143, var_145);
        var_148 = wp::extract(var_10, var_147);
        var_150 = wp::extract(var_121, var_149);
        var_151 = wp::sub(var_148, var_150);
        var_152 = wp::mul(var_146, var_151);
        var_153 = wp::add(var_141, var_152);
        var_154 = wp::sqrt(var_153);
        // L1 = wp.sqrt((p1[0] - res1[0]) * (p1[0] - res1[0]) + (p1[1] - res1[1]) * (p1[1] - res1[1]))       <L 444>
        var_156 = wp::extract(var_12, var_155);
        var_158 = wp::extract(var_128, var_157);
        var_159 = wp::sub(var_156, var_158);
        var_161 = wp::extract(var_12, var_160);
        var_163 = wp::extract(var_128, var_162);
        var_164 = wp::sub(var_161, var_163);
        var_165 = wp::mul(var_159, var_164);
        var_167 = wp::extract(var_12, var_166);
        var_169 = wp::extract(var_128, var_168);
        var_170 = wp::sub(var_167, var_169);
        var_172 = wp::extract(var_12, var_171);
        var_174 = wp::extract(var_128, var_173);
        var_175 = wp::sub(var_172, var_174);
        var_176 = wp::mul(var_170, var_175);
        var_177 = wp::add(var_165, var_176);
        var_178 = wp::sqrt(var_177);
        // res0[2] = p0[2] + (p1[2] - p0[2]) * math.safe_div(L0, L0 + wlen + L1)              <L 445>
        var_180 = wp::extract(var_10, var_179);
        var_182 = wp::extract(var_12, var_181);
        var_184 = wp::extract(var_10, var_183);
        var_185 = wp::sub(var_182, var_184);
        var_186 = wp::add(var_154, var_107);
        var_187 = wp::add(var_186, var_178);
        var_188 = safe_div_0(var_154, var_187);
        var_189 = wp::mul(var_185, var_188);
        var_190 = wp::add(var_180, var_189);
        wp::assign_inplace(var_121, var_191, var_190);
        // res1[2] = p0[2] + (p1[2] - p0[2]) * math.safe_div(L0 + wlen, L0 + wlen + L1)       <L 446>
        var_193 = wp::extract(var_10, var_192);
        var_195 = wp::extract(var_12, var_194);
        var_197 = wp::extract(var_10, var_196);
        var_198 = wp::sub(var_195, var_197);
        var_199 = wp::add(var_154, var_107);
        var_200 = wp::add(var_154, var_107);
        var_201 = wp::add(var_200, var_178);
        var_202 = safe_div_0(var_199, var_201);
        var_203 = wp::mul(var_198, var_202);
        var_204 = wp::add(var_193, var_203);
        wp::assign_inplace(var_128, var_205, var_204);
        // height = wp.abs(res1[2] - res0[2])                                                 <L 449>
        var_207 = wp::extract(var_128, var_206);
        var_209 = wp::extract(var_121, var_208);
        var_210 = wp::sub(var_207, var_209);
        var_211 = wp::abs(var_210);
        // wlen = wp.sqrt(wlen * wlen + height * height)                                      <L 450>
        var_212 = wp::mul(var_107, var_107);
        var_213 = wp::mul(var_211, var_211);
        var_214 = wp::add(var_212, var_213);
        var_215 = wp::sqrt(var_214);
    }
    var_216 = wp::where(var_130, var_215, var_107);
    // wpnt0 = mat @ res0 + pos                                                               <L 453>
    var_217 = wp::mul(var_mat, var_121);
    var_218 = wp::add(var_217, var_pos);
    // wpnt1 = mat @ res1 + pos                                                               <L 454>
    var_219 = wp::mul(var_mat, var_128);
    var_220 = wp::add(var_219, var_pos);
    // return wlen, wpnt0, wpnt1                                                              <L 456>
    ret_0 = var_216;
    ret_1 = var_218;
    ret_2 = var_220;
    return;
}


// C:\Users\ASUS\Documents\human-imitation-learning\.venv\Lib\site-packages\mujoco_warp\_src\smooth.py:4025
static CUDA_CALLABLE void _accumulate_jac_chain_0(
    wp::array_t<wp::int32> var_body_parentid,
    wp::array_t<wp::int32> var_body_dofnum,
    wp::array_t<wp::int32> var_body_dofadr,
    wp::array_t<wp::int32> var_ten_J_colind,
    wp::array_t<wp::vec_t<6, wp::float32>> var_cdof_in,
    wp::vec_t<3, wp::float32> var_offset,
    wp::vec_t<3, wp::float32> var_vec,
    wp::int32 var_bodyid,
    wp::int32 var_rowadr,
    wp::int32 var_rownnz,
    wp::float32 var_scale,
    wp::int32 var_worldid,
    wp::array_t<wp::float32> var_ten_J_out)
{
    //---------
    // primal vars
    const wp::int32 var_0 = 1;
    wp::int32 var_1;
    wp::int32 var_2;
    const wp::int32 var_3 = 0;
    bool var_4;
    wp::int32* var_5;
    wp::int32 var_6;
    wp::int32 var_7;
    wp::int32* var_8;
    wp::int32 var_9;
    wp::int32 var_10;
    wp::range_t var_11;
    wp::int32 var_12;
    wp::int32 var_13;
    const wp::int32 var_14 = 1;
    wp::int32 var_15;
    wp::int32 var_16;
    const wp::int32 var_17 = 0;
    bool var_18;
    wp::int32 var_19;
    wp::int32* var_20;
    bool var_21;
    wp::int32 var_22;
    const wp::int32 var_23 = 1;
    wp::int32 var_24;
    bool var_25;
    const wp::int32 var_26 = 0;
    bool var_27;
    wp::int32* var_28;
    bool var_29;
    wp::int32 var_30;
    wp::vec_t<6, wp::float32>* var_31;
    wp::vec_t<6, wp::float32> var_32;
    wp::vec_t<6, wp::float32> var_33;
    wp::vec_t<3, wp::float32> var_34;
    wp::vec_t<3, wp::float32> var_35;
    wp::vec_t<3, wp::float32> var_36;
    wp::vec_t<3, wp::float32> var_37;
    wp::float32 var_38;
    wp::float32 var_39;
    const wp::float32 var_40 = 0.0;
    bool var_41;
    wp::array_t<wp::float32> var_42;
    wp::float32 var_43;
    wp::int32* var_44;
    wp::int32 var_45;
    wp::int32 var_46;
    //---------
    // forward
    // def _accumulate_jac_chain(                                                             <L 4026>
    // ptr = rownnz - 1                                                                       <L 4046>
    var_1 = wp::sub(var_rownnz, var_0);
    // bid = bodyid                                                                           <L 4047>
    var_2 = wp::copy(var_bodyid);
    // while bid > 0:                                                                         <L 4048>
    start_while_0:;
    var_4 = (var_2 > var_3);
    if ((var_4) == false) goto end_while_0;
        // bdofadr = body_dofadr[bid]                                                         <L 4049>
        var_5 = wp::address(var_body_dofadr, var_2);
        var_7 = wp::load(var_5);
        var_6 = wp::copy(var_7);
        // bdofnum = body_dofnum[bid]                                                         <L 4050>
        var_8 = wp::address(var_body_dofnum, var_2);
        var_10 = wp::load(var_8);
        var_9 = wp::copy(var_10);
        // for k_rev in range(bdofnum):                                                       <L 4052>
        var_11 = wp::range(var_9);
        start_for_2:;
            if (iter_cmp(var_11) == 0) goto end_for_2;
            var_12 = wp::iter_next(var_11);
            // dof = bdofadr + bdofnum - 1 - k_rev                                            <L 4053>
            var_13 = wp::add(var_6, var_9);
            var_15 = wp::sub(var_13, var_14);
            var_16 = wp::sub(var_15, var_12);
            // while ptr >= 0:                                                                <L 4055>
    start_while_4:;
            var_18 = (var_1 >= var_17);
    if ((var_18) == false) goto end_while_4;
                // sparseid = rowadr + ptr                                                    <L 4056>
                var_19 = wp::add(var_rowadr, var_1);
                // if ten_J_colind[sparseid] <= dof:                                          <L 4057>
                var_20 = wp::address(var_ten_J_colind, var_19);
                var_22 = wp::load(var_20);
                var_21 = (var_22 <= var_16);
                if (var_21) {
                    // break                                                                  <L 4058>
                    goto end_while_4;
                }
                // ptr -= 1                                                                   <L 4059>
                var_24 = wp::sub(var_1, var_23);
                wp::assign(var_1, var_24);
    goto start_while_4;
    end_while_4:;
            // if ptr >= 0 and ten_J_colind[sparseid] == dof:                                 <L 4060>
            var_27 = (var_1 >= var_26);
            var_25 = var_27;
            if (var_25) {
                var_28 = wp::address(var_ten_J_colind, var_19);
                var_30 = wp::load(var_28);
                var_29 = (var_30 == var_16);
                var_25 = var_25 && var_29;
            }
            if (var_25) {
                // cdof = cdof_in[worldid, dof]                                               <L 4061>
                var_31 = wp::address(var_cdof_in, var_worldid, var_16);
                var_33 = wp::load(var_31);
                var_32 = wp::copy(var_33);
                // cdof_ang = wp.spatial_top(cdof)                                            <L 4062>
                var_34 = wp::spatial_top(var_32);
                // cdof_lin = wp.spatial_bottom(cdof)                                         <L 4063>
                var_35 = wp::spatial_bottom(var_32);
                // jacp = cdof_lin + wp.cross(cdof_ang, offset)                               <L 4064>
                var_36 = wp::cross(var_34, var_offset);
                var_37 = wp::add(var_35, var_36);
                // J = wp.dot(jacp, vec) * scale                                              <L 4065>
                var_38 = wp::dot(var_37, var_vec);
                var_39 = wp::mul(var_38, var_scale);
                // if J != 0.0:                                                               <L 4066>
                var_41 = (var_39 != var_40);
                if (var_41) {
                    // wp.atomic_add(ten_J_out[worldid], sparseid, J)                         <L 4067>
                    var_42 = wp::view(var_ten_J_out, var_worldid);
                    var_43 = wp::atomic_add(var_42, var_19, var_39);
                }
            }
            goto start_for_2;
        end_for_2:;
        // bid = body_parentid[bid]                                                           <L 4068>
        var_44 = wp::address(var_body_parentid, var_2);
        var_46 = wp::load(var_44);
        var_45 = wp::copy(var_46);
        wp::assign(var_2, var_45);
    goto start_while_0;
    end_while_0:;
}


// C:\Users\ASUS\Documents\human-imitation-learning\.venv\Lib\site-packages\mujoco_warp\_src\math.py:52
static CUDA_CALLABLE wp::quat_t<wp::float32> axis_angle_to_quat_0(
    wp::vec_t<3, wp::float32> var_axis,
    wp::float32 var_angle)
{
    //---------
    // primal vars
    const wp::float32 var_0 = 0.5;
    wp::float32 var_1;
    wp::float32 var_2;
    const wp::float32 var_3 = 0.5;
    wp::float32 var_4;
    wp::float32 var_5;
    wp::vec_t<3, wp::float32> var_6;
    const wp::int32 var_7 = 0;
    wp::float32 var_8;
    const wp::int32 var_9 = 1;
    wp::float32 var_10;
    const wp::int32 var_11 = 2;
    wp::float32 var_12;
    wp::quat_t<wp::float32> var_13;
    //---------
    // forward
    // def axis_angle_to_quat(axis: wp.vec3, angle: float) -> wp.quat:                        <L 53>
    // s, c = wp.sin(angle * 0.5), wp.cos(angle * 0.5)                                        <L 54>
    var_1 = wp::mul(var_angle, var_0);
    var_2 = wp::sin(var_1);
    var_4 = wp::mul(var_angle, var_3);
    var_5 = wp::cos(var_4);
    // axis = axis * s                                                                        <L 55>
    var_6 = wp::mul(var_axis, var_2);
    // return wp.quat(c, axis[0], axis[1], axis[2])                                           <L 56>
    var_8 = wp::extract(var_6, var_7);
    var_10 = wp::extract(var_6, var_9);
    var_12 = wp::extract(var_6, var_11);
    var_13 = wp::quat_t<wp::float32>(var_5, var_8, var_10, var_12);
    return var_13;
}


// C:\Users\ASUS\Documents\human-imitation-learning\.venv\Lib\site-packages\mujoco_warp\_src\support.py:1067
static CUDA_CALLABLE void get_face_metadata_0(
    wp::int32 var_cellnum_x,
    wp::int32 var_cellnum_y,
    wp::int32 var_cellnum_z,
    wp::int32 var_face_elem_idx,
    wp::int32 var_order_abs,
    wp::int32 & ret_0,
    wp::int32 & ret_1,
    wp::int32 & ret_2,
    wp::int32 & ret_3,
    wp::int32 & ret_4,
    wp::int32 & ret_5)
{
    //---------
    // primal vars
    wp::int32 var_0;
    wp::int32 var_1;
    wp::int32 var_2;
    const wp::int32 var_3 = 0;
    const wp::int32 var_4 = 0;
    bool var_5;
    const wp::int32 var_6 = 0;
    wp::int32 var_7;
    const wp::int32 var_8 = 2;
    wp::int32 var_9;
    bool var_10;
    const wp::int32 var_11 = 1;
    wp::int32 var_12;
    const wp::int32 var_13 = 2;
    wp::int32 var_14;
    wp::int32 var_15;
    bool var_16;
    const wp::int32 var_17 = 2;
    const wp::int32 var_18 = 2;
    wp::int32 var_19;
    wp::int32 var_20;
    const wp::int32 var_21 = 2;
    wp::int32 var_22;
    const wp::int32 var_23 = 2;
    wp::int32 var_24;
    wp::int32 var_25;
    bool var_26;
    const wp::int32 var_27 = 3;
    const wp::int32 var_28 = 2;
    wp::int32 var_29;
    wp::int32 var_30;
    wp::int32 var_31;
    const wp::int32 var_32 = 2;
    wp::int32 var_33;
    const wp::int32 var_34 = 2;
    wp::int32 var_35;
    wp::int32 var_36;
    wp::int32 var_37;
    bool var_38;
    const wp::int32 var_39 = 4;
    const wp::int32 var_40 = 2;
    wp::int32 var_41;
    wp::int32 var_42;
    const wp::int32 var_43 = 2;
    wp::int32 var_44;
    wp::int32 var_45;
    const wp::int32 var_46 = 5;
    const wp::int32 var_47 = 2;
    wp::int32 var_48;
    wp::int32 var_49;
    const wp::int32 var_50 = 2;
    wp::int32 var_51;
    wp::int32 var_52;
    wp::int32 var_53;
    wp::int32 var_54;
    wp::int32 var_55;
    wp::int32 var_56;
    wp::int32 var_57;
    wp::int32 var_58;
    wp::int32 var_59;
    wp::int32 var_60;
    wp::int32 var_61;
    wp::int32 var_62;
    wp::int32 var_63;
    const wp::int32 var_64 = 2;
    wp::int32 var_65;
    const wp::int32 var_66 = 0;
    bool var_67;
    const wp::int32 var_68 = 0;
    bool var_69;
    const wp::int32 var_70 = 1;
    bool var_71;
    wp::int32 var_72;
    bool var_73;
    const wp::int32 var_74 = 2;
    bool var_75;
    const wp::int32 var_76 = 3;
    bool var_77;
    wp::int32 var_78;
    wp::int32 var_79;
    wp::int32 var_80;
    wp::int32 var_81;
    const wp::int32 var_82 = 0;
    const wp::int32 var_83 = 0;
    bool var_84;
    wp::int32 var_85;
    const wp::int32 var_86 = 1;
    bool var_87;
    wp::int32 var_88;
    wp::int32 var_89;
    wp::int32 var_90;
    wp::int32 var_91;
    const wp::int32 var_92 = 2;
    wp::int32 var_93;
    wp::int32 var_94;
    wp::int32 var_95;
    wp::int32 var_96;
    wp::int32 var_97;
    wp::int32 var_98;
    const wp::int32 var_99 = 1;
    wp::int32 var_100;
    wp::int32 var_101;
    const wp::int32 var_102 = 1;
    wp::int32 var_103;
    //---------
    // forward
    // def get_face_metadata(                                                                 <L 1068>
    // size01 = cellnum_y * cellnum_z                                                         <L 1076>
    var_0 = wp::mul(var_cellnum_y, var_cellnum_z);
    // size23 = cellnum_x * cellnum_z                                                         <L 1077>
    var_1 = wp::mul(var_cellnum_x, var_cellnum_z);
    // size45 = cellnum_x * cellnum_y                                                         <L 1078>
    var_2 = wp::mul(var_cellnum_x, var_cellnum_y);
    // face_id = 0                                                                            <L 1080>
    // within_face = 0                                                                        <L 1081>
    // if face_elem_idx < size01:                                                             <L 1083>
    var_5 = (var_face_elem_idx < var_0);
    if (var_5) {
        // face_id = 0                                                                        <L 1084>
        // within_face = face_elem_idx                                                        <L 1085>
        var_7 = wp::copy(var_face_elem_idx);
    }
    if (!var_5) {
        // elif face_elem_idx < 2 * size01:                                                   <L 1086>
        var_9 = wp::mul(var_8, var_0);
        var_10 = (var_face_elem_idx < var_9);
        if (var_10) {
            // face_id = 1                                                                    <L 1087>
            // within_face = face_elem_idx - size01                                           <L 1088>
            var_12 = wp::sub(var_face_elem_idx, var_0);
        }
        if (!var_10) {
            // elif face_elem_idx < 2 * size01 + size23:                                      <L 1089>
            var_14 = wp::mul(var_13, var_0);
            var_15 = wp::add(var_14, var_1);
            var_16 = (var_face_elem_idx < var_15);
            if (var_16) {
                // face_id = 2                                                                <L 1090>
                // within_face = face_elem_idx - 2 * size01                                   <L 1091>
                var_19 = wp::mul(var_18, var_0);
                var_20 = wp::sub(var_face_elem_idx, var_19);
            }
            if (!var_16) {
                // elif face_elem_idx < 2 * size01 + 2 * size23:                              <L 1092>
                var_22 = wp::mul(var_21, var_0);
                var_24 = wp::mul(var_23, var_1);
                var_25 = wp::add(var_22, var_24);
                var_26 = (var_face_elem_idx < var_25);
                if (var_26) {
                    // face_id = 3                                                            <L 1093>
                    // within_face = face_elem_idx - 2 * size01 - size23                      <L 1094>
                    var_29 = wp::mul(var_28, var_0);
                    var_30 = wp::sub(var_face_elem_idx, var_29);
                    var_31 = wp::sub(var_30, var_1);
                }
                if (!var_26) {
                    // elif face_elem_idx < 2 * size01 + 2 * size23 + size45:                 <L 1095>
                    var_33 = wp::mul(var_32, var_0);
                    var_35 = wp::mul(var_34, var_1);
                    var_36 = wp::add(var_33, var_35);
                    var_37 = wp::add(var_36, var_2);
                    var_38 = (var_face_elem_idx < var_37);
                    if (var_38) {
                        // face_id = 4                                                        <L 1096>
                        // within_face = face_elem_idx - 2 * size01 - 2 * size23              <L 1097>
                        var_41 = wp::mul(var_40, var_0);
                        var_42 = wp::sub(var_face_elem_idx, var_41);
                        var_44 = wp::mul(var_43, var_1);
                        var_45 = wp::sub(var_42, var_44);
                    }
                    if (!var_38) {
                        // face_id = 5                                                        <L 1099>
                        // within_face = face_elem_idx - 2 * size01 - 2 * size23 - size45       <L 1100>
                        var_48 = wp::mul(var_47, var_0);
                        var_49 = wp::sub(var_face_elem_idx, var_48);
                        var_51 = wp::mul(var_50, var_1);
                        var_52 = wp::sub(var_49, var_51);
                        var_53 = wp::sub(var_52, var_2);
                    }
                    var_54 = wp::where(var_38, var_39, var_46);
                    var_55 = wp::where(var_38, var_45, var_53);
                }
                var_56 = wp::where(var_26, var_27, var_54);
                var_57 = wp::where(var_26, var_31, var_55);
            }
            var_58 = wp::where(var_16, var_17, var_56);
            var_59 = wp::where(var_16, var_20, var_57);
        }
        var_60 = wp::where(var_10, var_11, var_58);
        var_61 = wp::where(var_10, var_12, var_59);
    }
    var_62 = wp::where(var_5, var_6, var_60);
    var_63 = wp::where(var_5, var_7, var_61);
    // normal_axis = face_id // 2                                                             <L 1102>
    var_65 = wp::floordiv(var_62, var_64);
    // c1 = 0                                                                                 <L 1104>
    // if face_id == 0 or face_id == 1:                                                       <L 1105>
    var_69 = (var_62 == var_68);
    var_67 = var_69;
    if (!var_67) {
        var_71 = (var_62 == var_70);
        var_67 = var_67 || var_71;
    }
    if (var_67) {
        // c1 = cellnum_z                                                                     <L 1106>
        var_72 = wp::copy(var_cellnum_z);
    }
    if (!var_67) {
        // elif face_id == 2 or face_id == 3:                                                 <L 1107>
        var_75 = (var_62 == var_74);
        var_73 = var_75;
        if (!var_73) {
            var_77 = (var_62 == var_76);
            var_73 = var_73 || var_77;
        }
        if (var_73) {
            // c1 = cellnum_x                                                                 <L 1108>
            var_78 = wp::copy(var_cellnum_x);
        }
        if (!var_73) {
            // c1 = cellnum_y                                                                 <L 1110>
            var_79 = wp::copy(var_cellnum_y);
        }
        var_80 = wp::where(var_73, var_78, var_79);
    }
    var_81 = wp::where(var_67, var_72, var_80);
    // fixed_dim = 0                                                                          <L 1112>
    // if normal_axis == 0:                                                                   <L 1113>
    var_84 = (var_65 == var_83);
    if (var_84) {
        // fixed_dim = cellnum_x                                                              <L 1114>
        var_85 = wp::copy(var_cellnum_x);
    }
    if (!var_84) {
        // elif normal_axis == 1:                                                             <L 1115>
        var_87 = (var_65 == var_86);
        if (var_87) {
            // fixed_dim = cellnum_y                                                          <L 1116>
            var_88 = wp::copy(var_cellnum_y);
        }
        if (!var_87) {
            // fixed_dim = cellnum_z                                                          <L 1118>
            var_89 = wp::copy(var_cellnum_z);
        }
        var_90 = wp::where(var_87, var_88, var_89);
    }
    var_91 = wp::where(var_84, var_85, var_90);
    // g_fixed = (face_id % 2) * fixed_dim * order_abs                                        <L 1119>
    var_93 = wp::mod(var_62, var_92);
    var_94 = wp::mul(var_93, var_91);
    var_95 = wp::mul(var_94, var_order_abs);
    // q0 = within_face // c1                                                                 <L 1121>
    var_96 = wp::floordiv(var_63, var_81);
    // q1 = within_face % c1                                                                  <L 1122>
    var_97 = wp::mod(var_63, var_81);
    // ny_g = cellnum_y * order_abs + 1                                                       <L 1124>
    var_98 = wp::mul(var_cellnum_y, var_order_abs);
    var_100 = wp::add(var_98, var_99);
    // nz_g = cellnum_z * order_abs + 1                                                       <L 1125>
    var_101 = wp::mul(var_cellnum_z, var_order_abs);
    var_103 = wp::add(var_101, var_102);
    // return normal_axis, g_fixed, q0, q1, ny_g, nz_g                                        <L 1127>
    ret_0 = var_65;
    ret_1 = var_95;
    ret_2 = var_96;
    ret_3 = var_97;
    ret_4 = var_100;
    ret_5 = var_103;
    return;
}


// C:\Users\ASUS\Documents\human-imitation-learning\.venv\Lib\site-packages\mujoco_warp\_src\support.py:67
static CUDA_CALLABLE wp::quat_t<wp::float32> mat33_to_quat_polar_0(
    wp::mat_t<3, 3, wp::float32> var_F)
{
    //---------
    // primal vars
    const wp::float32 var_0 = 0.0;
    const wp::float32 var_1 = 0.0;
    const wp::float32 var_2 = 0.0;
    const wp::float32 var_3 = 1.0;
    wp::quat_t<wp::float32> var_4;
    const wp::int32 var_5 = 50;
    wp::range_t var_6;
    wp::int32 var_7;
    wp::mat_t<3, 3, wp::float32> var_8;
    wp::mat_t<3, 3, wp::float32> var_9;
    const wp::int32 var_10 = 0;
    wp::vec_t<3, wp::float32> var_11;
    const wp::int32 var_12 = 1;
    wp::vec_t<3, wp::float32> var_13;
    const wp::int32 var_14 = 2;
    wp::vec_t<3, wp::float32> var_15;
    wp::mat_t<3, 3, wp::float32> var_16;
    const wp::int32 var_17 = 0;
    wp::vec_t<3, wp::float32> var_18;
    const wp::int32 var_19 = 1;
    wp::vec_t<3, wp::float32> var_20;
    const wp::int32 var_21 = 2;
    wp::vec_t<3, wp::float32> var_22;
    wp::vec_t<3, wp::float32> var_23;
    wp::vec_t<3, wp::float32> var_24;
    wp::vec_t<3, wp::float32> var_25;
    wp::vec_t<3, wp::float32> var_26;
    wp::vec_t<3, wp::float32> var_27;
    wp::float32 var_28;
    wp::float32 var_29;
    wp::float32 var_30;
    wp::float32 var_31;
    wp::float32 var_32;
    wp::float32 var_33;
    const wp::float32 var_34 = 1e-10;
    wp::float32 var_35;
    wp::vec_t<3, wp::float32> var_36;
    wp::float32 var_37;
    const wp::float32 var_38 = 1e-06;
    bool var_39;
    wp::vec_t<3, wp::float32> var_40;
    const wp::float32 var_41 = 0.5;
    wp::float32 var_42;
    const wp::int32 var_43 = 0;
    wp::float32 var_44;
    wp::float32 var_45;
    wp::float32 var_46;
    const wp::int32 var_47 = 1;
    wp::float32 var_48;
    wp::float32 var_49;
    wp::float32 var_50;
    const wp::int32 var_51 = 2;
    wp::float32 var_52;
    wp::float32 var_53;
    wp::float32 var_54;
    wp::float32 var_55;
    wp::quat_t<wp::float32> var_56;
    wp::quat_t<wp::float32> var_57;
    wp::quat_t<wp::float32> var_58;
    //---------
    // forward
    // def mat33_to_quat_polar(F: wp.mat33) -> wp.quat:                                       <L 68>
    // cell_quat = wp.quat(0.0, 0.0, 0.0, 1.0)                                                <L 69>
    var_4 = wp::quat_t<wp::float32>(var_0, var_1, var_2, var_3);
    // for _iter in range(50):                                                                <L 70>
    var_6 = wp::range(var_5);
    start_for_0:;
        if (iter_cmp(var_6) == 0) goto end_for_0;
        var_7 = wp::iter_next(var_6);
        // rot = wp.quat_to_matrix(cell_quat)                                                 <L 71>
        var_8 = wp::quat_to_matrix(var_4);
        // rot_t = wp.transpose(rot)                                                          <L 72>
        var_9 = wp::transpose(var_8);
        // col1_rot = rot_t[0]                                                                <L 73>
        var_11 = wp::extract(var_9, var_10);
        // col2_rot = rot_t[1]                                                                <L 74>
        var_13 = wp::extract(var_9, var_12);
        // col3_rot = rot_t[2]                                                                <L 75>
        var_15 = wp::extract(var_9, var_14);
        // F_t = wp.transpose(F)                                                              <L 76>
        var_16 = wp::transpose(var_F);
        // col1_mat = F_t[0]                                                                  <L 77>
        var_18 = wp::extract(var_16, var_17);
        // col2_mat = F_t[1]                                                                  <L 78>
        var_20 = wp::extract(var_16, var_19);
        // col3_mat = F_t[2]                                                                  <L 79>
        var_22 = wp::extract(var_16, var_21);
        // omega = wp.cross(col1_rot, col1_mat) + wp.cross(col2_rot, col2_mat) + wp.cross(col3_rot, col3_mat)       <L 81>
        var_23 = wp::cross(var_11, var_18);
        var_24 = wp::cross(var_13, var_20);
        var_25 = wp::add(var_23, var_24);
        var_26 = wp::cross(var_15, var_22);
        var_27 = wp::add(var_25, var_26);
        // denom = wp.abs(wp.dot(col1_rot, col1_mat) + wp.dot(col2_rot, col2_mat) + wp.dot(col3_rot, col3_mat)) + 1.0e-10       <L 82>
        var_28 = wp::dot(var_11, var_18);
        var_29 = wp::dot(var_13, var_20);
        var_30 = wp::add(var_28, var_29);
        var_31 = wp::dot(var_15, var_22);
        var_32 = wp::add(var_30, var_31);
        var_33 = wp::abs(var_32);
        var_35 = wp::add(var_33, var_34);
        // omega = omega / denom                                                              <L 83>
        var_36 = wp::div(var_27, var_35);
        // w = wp.length(omega)                                                               <L 85>
        var_37 = wp::length(var_36);
        // if w < 1.0e-6:                                                                     <L 86>
        var_39 = (var_37 < var_38);
        if (var_39) {
            // break                                                                          <L 87>
            goto end_for_0;
        }
        // axis = omega / w                                                                   <L 89>
        var_40 = wp::div(var_36, var_37);
        // half_w = 0.5 * w                                                                   <L 90>
        var_42 = wp::mul(var_41, var_37);
        // qrot = wp.quat(                                                                    <L 91>
        // axis[0] * wp.sin(half_w),                                                          <L 92>
        var_44 = wp::extract(var_40, var_43);
        var_45 = wp::sin(var_42);
        var_46 = wp::mul(var_44, var_45);
        // axis[1] * wp.sin(half_w),                                                          <L 93>
        var_48 = wp::extract(var_40, var_47);
        var_49 = wp::sin(var_42);
        var_50 = wp::mul(var_48, var_49);
        // axis[2] * wp.sin(half_w),                                                          <L 94>
        var_52 = wp::extract(var_40, var_51);
        var_53 = wp::sin(var_42);
        var_54 = wp::mul(var_52, var_53);
        // wp.cos(half_w),                                                                    <L 95>
        var_55 = wp::cos(var_42);
        var_56 = wp::quat_t<wp::float32>(var_46, var_50, var_54, var_55);
        // cell_quat = wp.normalize(qrot * cell_quat)                                         <L 97>
        var_57 = wp::mul(var_56, var_4);
        var_58 = wp::normalize(var_57);
        wp::assign(var_4, var_58);
        goto start_for_0;
    end_for_0:;
    // return cell_quat                                                                       <L 98>
    return var_4;
}


// C:\Users\ASUS\Documents\human-imitation-learning\.venv\Lib\site-packages\mujoco_warp\_src\math.py:44
static CUDA_CALLABLE void adj_rot_vec_quat_0(
    wp::vec_t<3, wp::float32> var_vec,
    wp::quat_t<wp::float32> var_quat,
    wp::vec_t<3, wp::float32> & adj_vec,
    wp::quat_t<wp::float32> & adj_quat,
    wp::vec_t<3, wp::float32> & adj_ret)
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


// C:\Users\ASUS\Documents\human-imitation-learning\.venv\Lib\site-packages\mujoco_warp\_src\math.py:59
static CUDA_CALLABLE void adj_quat_to_mat_0(
    wp::quat_t<wp::float32> var_quat,
    wp::quat_t<wp::float32> & adj_quat,
    wp::mat_t<3, 3, wp::float32> & adj_ret)
{
	// reverse mode disabled (no backward-enabled kernel depends on this function)
}


// C:\Users\ASUS\Documents\human-imitation-learning\.venv\Lib\site-packages\mujoco_warp\_src\math.py:133
static CUDA_CALLABLE void adj_motion_cross_0(
    wp::vec_t<6, wp::float32> var_u,
    wp::vec_t<6, wp::float32> var_v,
    wp::vec_t<6, wp::float32> & adj_u,
    wp::vec_t<6, wp::float32> & adj_v,
    wp::vec_t<6, wp::float32> & adj_ret)
{
	// reverse mode disabled (no backward-enabled kernel depends on this function)
}


// C:\Users\ASUS\Documents\human-imitation-learning\.venv\Lib\site-packages\mujoco_warp\_src\math.py:120
static CUDA_CALLABLE void adj_inert_vec_0(
    wp::vec_t<10, wp::float32> var_i,
    wp::vec_t<6, wp::float32> var_v,
    wp::vec_t<10, wp::float32> & adj_i,
    wp::vec_t<6, wp::float32> & adj_v,
    wp::vec_t<6, wp::float32> & adj_ret)
{
	// reverse mode disabled (no backward-enabled kernel depends on this function)
}


// C:\Users\ASUS\Documents\human-imitation-learning\.venv\Lib\site-packages\mujoco_warp\_src\math.py:115
static CUDA_CALLABLE void adj_quat_inv_0(
    wp::quat_t<wp::float32> var_quat,
    wp::quat_t<wp::float32> & adj_quat,
    wp::quat_t<wp::float32> & adj_ret)
{
	// reverse mode disabled (no backward-enabled kernel depends on this function)
}


// C:\Users\ASUS\Documents\human-imitation-learning\.venv\Lib\site-packages\warp\_src\math.py:0
static CUDA_CALLABLE void adj_norm_l2_0(
    wp::vec_t<3, wp::float32> var_v,
    wp::vec_t<3, wp::float32> & adj_v,
    wp::float32 & adj_ret)
{
	// reverse mode disabled (no backward-enabled kernel depends on this function)
}


// C:\Users\ASUS\Documents\human-imitation-learning\.venv\Lib\site-packages\mujoco_warp\_src\math.py:161
static CUDA_CALLABLE void adj_quat_to_vel_0(
    wp::quat_t<wp::float32> var_quat,
    wp::quat_t<wp::float32> & adj_quat,
    wp::vec_t<3, wp::float32> & adj_ret)
{
	// reverse mode disabled (no backward-enabled kernel depends on this function)
}


// C:\Users\ASUS\Documents\human-imitation-learning\.venv\Lib\site-packages\mujoco_warp\_src\math.py:0
static CUDA_CALLABLE void adj_safe_div_0(
    wp::float32 var_x,
    wp::float32 var_y,
    wp::float32 & adj_x,
    wp::float32 & adj_y,
    wp::float32 & adj_ret)
{
	// reverse mode disabled (no backward-enabled kernel depends on this function)
}


// C:\Users\ASUS\Documents\human-imitation-learning\.venv\Lib\site-packages\mujoco_warp\_src\math.py:0
static CUDA_CALLABLE void adj_safe_div_0(
    wp::vec_t<3, wp::float32> var_x,
    wp::float32 var_y,
    wp::vec_t<3, wp::float32> & adj_x,
    wp::float32 & adj_y,
    wp::vec_t<3, wp::float32> & adj_ret)
{
	// reverse mode disabled (no backward-enabled kernel depends on this function)
}


// C:\Users\ASUS\Documents\human-imitation-learning\.venv\Lib\site-packages\mujoco_warp\_src\support.py:499
static CUDA_CALLABLE void adj_jac_dof_0(
    wp::array_t<wp::int32> var_body_parentid,
    wp::array_t<wp::int32> var_body_rootid,
    wp::array_t<wp::int32> var_dof_bodyid,
    wp::array_t<wp::int32> var_body_isdofancestor,
    wp::array_t<wp::vec_t<3, wp::float32>> var_subtree_com_in,
    wp::array_t<wp::vec_t<6, wp::float32>> var_cdof_in,
    wp::vec_t<3, wp::float32> var_point,
    wp::int32 var_bodyid,
    wp::int32 var_dofid,
    wp::int32 var_worldid,
    wp::vec_t<3, wp::float32> & ret_0,
    wp::vec_t<3, wp::float32> & ret_1,
    wp::array_t<wp::int32> & adj_body_parentid,
    wp::array_t<wp::int32> & adj_body_rootid,
    wp::array_t<wp::int32> & adj_dof_bodyid,
    wp::array_t<wp::int32> & adj_body_isdofancestor,
    wp::array_t<wp::vec_t<3, wp::float32>> & adj_subtree_com_in,
    wp::array_t<wp::vec_t<6, wp::float32>> & adj_cdof_in,
    wp::vec_t<3, wp::float32> & adj_point,
    wp::int32 & adj_bodyid,
    wp::int32 & adj_dofid,
    wp::int32 & adj_worldid,
    wp::vec_t<3, wp::float32> & adj_ret_0,
    wp::vec_t<3, wp::float32> & adj_ret_1)
{
	// reverse mode disabled (no backward-enabled kernel depends on this function)
}


// C:\Users\ASUS\Documents\human-imitation-learning\.venv\Lib\site-packages\mujoco_warp\_src\math.py:177
static CUDA_CALLABLE void adj_quat_sub_0(
    wp::quat_t<wp::float32> var_qa,
    wp::quat_t<wp::float32> var_qb,
    wp::quat_t<wp::float32> & adj_qa,
    wp::quat_t<wp::float32> & adj_qb,
    wp::vec_t<3, wp::float32> & adj_ret)
{
	// reverse mode disabled (no backward-enabled kernel depends on this function)
}


// C:\Users\ASUS\Documents\human-imitation-learning\.venv\Lib\site-packages\mujoco_warp\_src\util_misc.py:891
static CUDA_CALLABLE void adj__poly_force_0(
    wp::float32 var_linear,
    wp::vec_t<2, wp::float32> var_poly,
    wp::float32 var_x,
    wp::int32 var_flg_odd,
    wp::float32 & adj_linear,
    wp::vec_t<2, wp::float32> & adj_poly,
    wp::float32 & adj_x,
    wp::int32 & adj_flg_odd,
    wp::float32 & adj_ret)
{
	// reverse mode disabled (no backward-enabled kernel depends on this function)
}


// C:\Users\ASUS\Documents\human-imitation-learning\.venv\Lib\site-packages\mujoco_warp\_src\math.py:0
static CUDA_CALLABLE void adj_normalize_with_norm_0(
    wp::vec_t<3, wp::float32> var_x,
    wp::vec_t<3, wp::float32> & ret_0,
    wp::float32 & ret_1,
    wp::vec_t<3, wp::float32> & adj_x,
    wp::vec_t<3, wp::float32> & adj_ret_0,
    wp::float32 & adj_ret_1)
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


// C:\Users\ASUS\Documents\human-imitation-learning\.venv\Lib\site-packages\mujoco_warp\_src\support.py:471
static CUDA_CALLABLE void adj_transform_force_0(
    wp::vec_t<3, wp::float32> var_force,
    wp::vec_t<3, wp::float32> var_torque,
    wp::vec_t<3, wp::float32> var_offset,
    wp::vec_t<3, wp::float32> & adj_force,
    wp::vec_t<3, wp::float32> & adj_torque,
    wp::vec_t<3, wp::float32> & adj_offset,
    wp::vec_t<6, wp::float32> & adj_ret)
{
	// reverse mode disabled (no backward-enabled kernel depends on this function)
}


// C:\Users\ASUS\Documents\human-imitation-learning\.venv\Lib\site-packages\mujoco_warp\_src\support.py:476
static CUDA_CALLABLE void adj_transform_force_1(
    wp::vec_t<6, wp::float32> var_frc,
    wp::vec_t<3, wp::float32> var_offset,
    wp::vec_t<6, wp::float32> & adj_frc,
    wp::vec_t<3, wp::float32> & adj_offset,
    wp::vec_t<6, wp::float32> & adj_ret)
{
	// reverse mode disabled (no backward-enabled kernel depends on this function)
}


// C:\Users\ASUS\Documents\human-imitation-learning\.venv\Lib\site-packages\mujoco_warp\_src\math.py:147
static CUDA_CALLABLE void adj_motion_cross_force_0(
    wp::vec_t<6, wp::float32> var_v,
    wp::vec_t<6, wp::float32> var_f,
    wp::vec_t<6, wp::float32> & adj_v,
    wp::vec_t<6, wp::float32> & adj_f,
    wp::vec_t<6, wp::float32> & adj_ret)
{
	// reverse mode disabled (no backward-enabled kernel depends on this function)
}


// C:\Users\ASUS\Documents\human-imitation-learning\.venv\Lib\site-packages\mujoco_warp\_src\math.py:33
static CUDA_CALLABLE void adj_quat_mul_axis_0(
    wp::quat_t<wp::float32> var_q,
    wp::vec_t<3, wp::float32> var_axis,
    wp::quat_t<wp::float32> & adj_q,
    wp::vec_t<3, wp::float32> & adj_axis,
    wp::quat_t<wp::float32> & adj_ret)
{
	// reverse mode disabled (no backward-enabled kernel depends on this function)
}


// C:\Users\ASUS\Documents\human-imitation-learning\.venv\Lib\site-packages\mujoco_warp\_src\smooth.py:2157
static CUDA_CALLABLE void adj__accumulate_jac_dot_chain_0(
    wp::array_t<wp::int32> var_body_parentid,
    wp::array_t<wp::int32> var_body_dofnum,
    wp::array_t<wp::int32> var_body_dofadr,
    wp::array_t<wp::int32> var_jnt_type,
    wp::array_t<wp::int32> var_jnt_dofadr,
    wp::array_t<wp::int32> var_dof_jntid,
    wp::array_t<wp::int32> var_ten_J_colind,
    wp::array_t<wp::vec_t<6, wp::float32>> var_cdof_in,
    wp::array_t<wp::vec_t<6, wp::float32>> var_cvel_in,
    wp::array_t<wp::vec_t<6, wp::float32>> var_cdof_dot_in,
    wp::vec_t<3, wp::float32> var_offset,
    wp::vec_t<3, wp::float32> var_pvel_lin,
    wp::vec_t<3, wp::float32> var_dpnt,
    wp::vec_t<3, wp::float32> var_dvel,
    wp::int32 var_bodyid,
    wp::int32 var_rowadr,
    wp::int32 var_rownnz,
    wp::float32 var_scale,
    wp::int32 var_worldid,
    wp::array_t<wp::float32> var_ten_Jdot_out,
    wp::array_t<wp::int32> & adj_body_parentid,
    wp::array_t<wp::int32> & adj_body_dofnum,
    wp::array_t<wp::int32> & adj_body_dofadr,
    wp::array_t<wp::int32> & adj_jnt_type,
    wp::array_t<wp::int32> & adj_jnt_dofadr,
    wp::array_t<wp::int32> & adj_dof_jntid,
    wp::array_t<wp::int32> & adj_ten_J_colind,
    wp::array_t<wp::vec_t<6, wp::float32>> & adj_cdof_in,
    wp::array_t<wp::vec_t<6, wp::float32>> & adj_cvel_in,
    wp::array_t<wp::vec_t<6, wp::float32>> & adj_cdof_dot_in,
    wp::vec_t<3, wp::float32> & adj_offset,
    wp::vec_t<3, wp::float32> & adj_pvel_lin,
    wp::vec_t<3, wp::float32> & adj_dpnt,
    wp::vec_t<3, wp::float32> & adj_dvel,
    wp::int32 & adj_bodyid,
    wp::int32 & adj_rowadr,
    wp::int32 & adj_rownnz,
    wp::float32 & adj_scale,
    wp::int32 & adj_worldid,
    wp::array_t<wp::float32> & adj_ten_Jdot_out)
{
	// reverse mode disabled (no backward-enabled kernel depends on this function)
}


// C:\Users\ASUS\Documents\human-imitation-learning\.venv\Lib\site-packages\mujoco_warp\_src\support.py:986
static CUDA_CALLABLE void adj__phi_0(
    wp::float32 var_s,
    wp::int32 var_i,
    wp::float32 & adj_s,
    wp::int32 & adj_i,
    wp::float32 & adj_ret)
{
	// reverse mode disabled (no backward-enabled kernel depends on this function)
}


// C:\Users\ASUS\Documents\human-imitation-learning\.venv\Lib\site-packages\mujoco_warp\_src\support.py:998
static CUDA_CALLABLE void adj_eval_basis_trilinear_0(
    wp::vec_t<3, wp::float32> var_local,
    wp::int32 var_node_idx,
    wp::vec_t<3, wp::float32> & adj_local,
    wp::int32 & adj_node_idx,
    wp::float32 & adj_ret)
{
	// reverse mode disabled (no backward-enabled kernel depends on this function)
}


// C:\Users\ASUS\Documents\human-imitation-learning\.venv\Lib\site-packages\mujoco_warp\_src\math.py:0
static CUDA_CALLABLE void adj_normalize_with_norm_0(
    wp::vec_t<2, wp::float32> var_x,
    wp::vec_t<2, wp::float32> & ret_0,
    wp::float32 & ret_1,
    wp::vec_t<2, wp::float32> & adj_x,
    wp::vec_t<2, wp::float32> & adj_ret_0,
    wp::float32 & adj_ret_1)
{
	// reverse mode disabled (no backward-enabled kernel depends on this function)
}


// C:\Users\ASUS\Documents\human-imitation-learning\.venv\Lib\site-packages\warp\_src\math.py:0
static CUDA_CALLABLE void adj_norm_l2_0(
    wp::vec_t<2, wp::float32> var_v,
    wp::vec_t<2, wp::float32> & adj_v,
    wp::float32 & adj_ret)
{
	// reverse mode disabled (no backward-enabled kernel depends on this function)
}


// C:\Users\ASUS\Documents\human-imitation-learning\.venv\Lib\site-packages\mujoco_warp\_src\util_misc.py:205
static CUDA_CALLABLE void adj_wrap_inside_0(
    wp::vec_t<4, wp::float32> var_end,
    wp::float32 var_radius,
    wp::float32 & ret_0,
    wp::vec_t<2, wp::float32> & ret_1,
    wp::vec_t<2, wp::float32> & ret_2,
    wp::vec_t<4, wp::float32> & adj_end,
    wp::float32 & adj_radius,
    wp::float32 & adj_ret_0,
    wp::vec_t<2, wp::float32> & adj_ret_1,
    wp::vec_t<2, wp::float32> & adj_ret_2)
{
	// reverse mode disabled (no backward-enabled kernel depends on this function)
}


// C:\Users\ASUS\Documents\human-imitation-learning\.venv\Lib\site-packages\mujoco_warp\_src\util_misc.py:31
static CUDA_CALLABLE void adj_is_intersect_0(
    wp::vec_t<2, wp::float32> var_p1,
    wp::vec_t<2, wp::float32> var_p2,
    wp::vec_t<2, wp::float32> var_p3,
    wp::vec_t<2, wp::float32> var_p4,
    wp::vec_t<2, wp::float32> & adj_p1,
    wp::vec_t<2, wp::float32> & adj_p2,
    wp::vec_t<2, wp::float32> & adj_p3,
    wp::vec_t<2, wp::float32> & adj_p4,
    bool & adj_ret)
{
	// reverse mode disabled (no backward-enabled kernel depends on this function)
}


// C:\Users\ASUS\Documents\human-imitation-learning\.venv\Lib\site-packages\mujoco_warp\_src\util_misc.py:81
static CUDA_CALLABLE void adj_length_circle_0(
    wp::vec_t<2, wp::float32> var_p0,
    wp::vec_t<2, wp::float32> var_p1,
    wp::int32 var_ind,
    wp::float32 var_radius,
    wp::vec_t<2, wp::float32> & adj_p0,
    wp::vec_t<2, wp::float32> & adj_p1,
    wp::int32 & adj_ind,
    wp::float32 & adj_radius,
    wp::float32 & adj_ret)
{
	// reverse mode disabled (no backward-enabled kernel depends on this function)
}


// C:\Users\ASUS\Documents\human-imitation-learning\.venv\Lib\site-packages\mujoco_warp\_src\util_misc.py:105
static CUDA_CALLABLE void adj_wrap_circle_0(
    wp::vec_t<4, wp::float32> var_end,
    wp::vec_t<2, wp::float32> var_side,
    wp::float32 var_radius,
    wp::float32 & ret_0,
    wp::vec_t<2, wp::float32> & ret_1,
    wp::vec_t<2, wp::float32> & ret_2,
    wp::vec_t<4, wp::float32> & adj_end,
    wp::vec_t<2, wp::float32> & adj_side,
    wp::float32 & adj_radius,
    wp::float32 & adj_ret_0,
    wp::vec_t<2, wp::float32> & adj_ret_1,
    wp::vec_t<2, wp::float32> & adj_ret_2)
{
	// reverse mode disabled (no backward-enabled kernel depends on this function)
}


// C:\Users\ASUS\Documents\human-imitation-learning\.venv\Lib\site-packages\mujoco_warp\_src\util_misc.py:332
static CUDA_CALLABLE void adj_wrap_0(
    wp::vec_t<3, wp::float32> var_x0,
    wp::vec_t<3, wp::float32> var_x1,
    wp::vec_t<3, wp::float32> var_pos,
    wp::mat_t<3, 3, wp::float32> var_mat,
    wp::float32 var_radius,
    wp::int32 var_geomtype,
    wp::vec_t<3, wp::float32> var_side,
    wp::float32 & ret_0,
    wp::vec_t<3, wp::float32> & ret_1,
    wp::vec_t<3, wp::float32> & ret_2,
    wp::vec_t<3, wp::float32> & adj_x0,
    wp::vec_t<3, wp::float32> & adj_x1,
    wp::vec_t<3, wp::float32> & adj_pos,
    wp::mat_t<3, 3, wp::float32> & adj_mat,
    wp::float32 & adj_radius,
    wp::int32 & adj_geomtype,
    wp::vec_t<3, wp::float32> & adj_side,
    wp::float32 & adj_ret_0,
    wp::vec_t<3, wp::float32> & adj_ret_1,
    wp::vec_t<3, wp::float32> & adj_ret_2)
{
	// reverse mode disabled (no backward-enabled kernel depends on this function)
}


// C:\Users\ASUS\Documents\human-imitation-learning\.venv\Lib\site-packages\mujoco_warp\_src\smooth.py:4025
static CUDA_CALLABLE void adj__accumulate_jac_chain_0(
    wp::array_t<wp::int32> var_body_parentid,
    wp::array_t<wp::int32> var_body_dofnum,
    wp::array_t<wp::int32> var_body_dofadr,
    wp::array_t<wp::int32> var_ten_J_colind,
    wp::array_t<wp::vec_t<6, wp::float32>> var_cdof_in,
    wp::vec_t<3, wp::float32> var_offset,
    wp::vec_t<3, wp::float32> var_vec,
    wp::int32 var_bodyid,
    wp::int32 var_rowadr,
    wp::int32 var_rownnz,
    wp::float32 var_scale,
    wp::int32 var_worldid,
    wp::array_t<wp::float32> var_ten_J_out,
    wp::array_t<wp::int32> & adj_body_parentid,
    wp::array_t<wp::int32> & adj_body_dofnum,
    wp::array_t<wp::int32> & adj_body_dofadr,
    wp::array_t<wp::int32> & adj_ten_J_colind,
    wp::array_t<wp::vec_t<6, wp::float32>> & adj_cdof_in,
    wp::vec_t<3, wp::float32> & adj_offset,
    wp::vec_t<3, wp::float32> & adj_vec,
    wp::int32 & adj_bodyid,
    wp::int32 & adj_rowadr,
    wp::int32 & adj_rownnz,
    wp::float32 & adj_scale,
    wp::int32 & adj_worldid,
    wp::array_t<wp::float32> & adj_ten_J_out)
{
	// reverse mode disabled (no backward-enabled kernel depends on this function)
}


// C:\Users\ASUS\Documents\human-imitation-learning\.venv\Lib\site-packages\mujoco_warp\_src\math.py:52
static CUDA_CALLABLE void adj_axis_angle_to_quat_0(
    wp::vec_t<3, wp::float32> var_axis,
    wp::float32 var_angle,
    wp::vec_t<3, wp::float32> & adj_axis,
    wp::float32 & adj_angle,
    wp::quat_t<wp::float32> & adj_ret)
{
	// reverse mode disabled (no backward-enabled kernel depends on this function)
}


// C:\Users\ASUS\Documents\human-imitation-learning\.venv\Lib\site-packages\mujoco_warp\_src\support.py:1067
static CUDA_CALLABLE void adj_get_face_metadata_0(
    wp::int32 var_cellnum_x,
    wp::int32 var_cellnum_y,
    wp::int32 var_cellnum_z,
    wp::int32 var_face_elem_idx,
    wp::int32 var_order_abs,
    wp::int32 & ret_0,
    wp::int32 & ret_1,
    wp::int32 & ret_2,
    wp::int32 & ret_3,
    wp::int32 & ret_4,
    wp::int32 & ret_5,
    wp::int32 & adj_cellnum_x,
    wp::int32 & adj_cellnum_y,
    wp::int32 & adj_cellnum_z,
    wp::int32 & adj_face_elem_idx,
    wp::int32 & adj_order_abs,
    wp::int32 & adj_ret_0,
    wp::int32 & adj_ret_1,
    wp::int32 & adj_ret_2,
    wp::int32 & adj_ret_3,
    wp::int32 & adj_ret_4,
    wp::int32 & adj_ret_5)
{
	// reverse mode disabled (no backward-enabled kernel depends on this function)
}


// C:\Users\ASUS\Documents\human-imitation-learning\.venv\Lib\site-packages\mujoco_warp\_src\support.py:67
static CUDA_CALLABLE void adj_mat33_to_quat_polar_0(
    wp::mat_t<3, 3, wp::float32> var_F,
    wp::mat_t<3, 3, wp::float32> & adj_F,
    wp::quat_t<wp::float32> & adj_ret)
{
	// reverse mode disabled (no backward-enabled kernel depends on this function)
}



extern "C" __global__ void _subtree_com_init_002b02fd_cuda_kernel_forward(
    wp::launch_bounds_t<2> dim,
    wp::array_t<wp::float32> var_body_mass,
    wp::array_t<wp::vec_t<3, wp::float32>> var_xipos_in,
    wp::array_t<wp::vec_t<3, wp::float32>> var_subtree_com_out)
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
        wp::vec_t<3, wp::float32>* var_2;
        wp::shape_t* var_3;
        const wp::int32 var_4 = 0;
        wp::int32 var_5;
        wp::shape_t var_6;
        wp::int32 var_7;
        wp::float32* var_8;
        wp::vec_t<3, wp::float32> var_9;
        wp::vec_t<3, wp::float32> var_10;
        wp::float32 var_11;
        //---------
        // forward
        // def _subtree_com_init(                                                                 <L 657>
        // worldid, bodyid = wp.tid()                                                             <L 665>
        builtin_tid2d(var_0, var_1);
        // subtree_com_out[worldid, bodyid] = xipos_in[worldid, bodyid] * body_mass[worldid % body_mass.shape[0], bodyid]       <L 666>
        var_2 = wp::address(var_xipos_in, var_0, var_1);
        var_3 = &(var_body_mass.shape);
        var_6 = wp::load(var_3);
        var_5 = wp::extract(var_6, var_4);
        var_7 = wp::mod(var_0, var_5);
        var_8 = wp::address(var_body_mass, var_7, var_1);
        var_10 = wp::load(var_2);
        var_11 = wp::load(var_8);
        var_9 = wp::mul(var_10, var_11);
        wp::array_store(var_subtree_com_out, var_0, var_1, var_9);
    }
}



extern "C" __global__ void _cfrc_ext_tendon_constraint_06fedf68_cuda_kernel_forward(
    wp::launch_bounds_t<2> dim,
    wp::array_t<wp::int32> var_eq_type,
    wp::array_t<wp::int32> var_eq_obj1id,
    wp::array_t<wp::int32> var_eq_obj2id,
    wp::array_t<wp::vec_t<11, wp::float32>> var_eq_data,
    wp::array_t<wp::vec_t<2, wp::float32>> var_tendon_range,
    wp::array_t<wp::float32> var_tendon_length0,
    wp::array_t<wp::int32> var_nefc_in,
    wp::array_t<wp::float32> var_ten_length_in,
    wp::array_t<wp::int32> var_efc_type_in,
    wp::array_t<wp::int32> var_efc_id_in,
    wp::array_t<wp::float32> var_efc_force_in,
    wp::array_t<wp::float32> var_ten_frc_out)
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
        bool var_3;
        wp::int32 var_4;
        wp::float32* var_5;
        wp::float32 var_6;
        wp::float32 var_7;
        const wp::float32 var_8 = 0.0;
        bool var_9;
        wp::int32* var_10;
        wp::int32 var_11;
        wp::int32 var_12;
        const wp::int32 var_13 = 0;
        bool var_14;
        wp::int32* var_15;
        wp::int32 var_16;
        wp::int32 var_17;
        wp::int32* var_18;
        const wp::int32 var_19 = 3;
        bool var_20;
        wp::int32 var_21;
        wp::int32* var_22;
        wp::int32 var_23;
        wp::int32 var_24;
        wp::int32* var_25;
        wp::int32 var_26;
        wp::int32 var_27;
        wp::array_t<wp::float32> var_28;
        wp::float32 var_29;
        const wp::int32 var_30 = 0;
        bool var_31;
        wp::float32* var_32;
        wp::shape_t* var_33;
        const wp::int32 var_34 = 0;
        wp::int32 var_35;
        wp::shape_t var_36;
        wp::int32 var_37;
        wp::float32* var_38;
        wp::float32 var_39;
        wp::float32 var_40;
        wp::float32 var_41;
        wp::shape_t* var_42;
        const wp::int32 var_43 = 0;
        wp::int32 var_44;
        wp::shape_t var_45;
        wp::int32 var_46;
        wp::vec_t<11, wp::float32>* var_47;
        wp::vec_t<11, wp::float32> var_48;
        wp::vec_t<11, wp::float32> var_49;
        const wp::int32 var_50 = 1;
        wp::float32 var_51;
        const wp::float32 var_52 = 2.0;
        const wp::int32 var_53 = 2;
        wp::float32 var_54;
        wp::float32 var_55;
        wp::float32 var_56;
        wp::float32 var_57;
        const wp::float32 var_58 = 3.0;
        const wp::int32 var_59 = 3;
        wp::float32 var_60;
        wp::float32 var_61;
        wp::float32 var_62;
        wp::float32 var_63;
        wp::float32 var_64;
        const wp::float32 var_65 = 4.0;
        const wp::int32 var_66 = 4;
        wp::float32 var_67;
        wp::float32 var_68;
        wp::float32 var_69;
        wp::float32 var_70;
        wp::float32 var_71;
        wp::float32 var_72;
        wp::array_t<wp::float32> var_73;
        wp::float32 var_74;
        wp::float32 var_75;
        const wp::int32 var_76 = 2;
        bool var_77;
        wp::int32* var_78;
        wp::int32 var_79;
        wp::int32 var_80;
        wp::array_t<wp::float32> var_81;
        wp::float32 var_82;
        const wp::int32 var_83 = 4;
        bool var_84;
        wp::int32* var_85;
        wp::int32 var_86;
        wp::int32 var_87;
        wp::shape_t* var_88;
        const wp::int32 var_89 = 0;
        wp::int32 var_90;
        wp::shape_t var_91;
        wp::int32 var_92;
        wp::vec_t<2, wp::float32>* var_93;
        wp::vec_t<2, wp::float32> var_94;
        wp::vec_t<2, wp::float32> var_95;
        wp::float32* var_96;
        wp::float32 var_97;
        wp::float32 var_98;
        const wp::int32 var_99 = 0;
        wp::float32 var_100;
        wp::float32 var_101;
        const wp::int32 var_102 = 1;
        wp::float32 var_103;
        wp::float32 var_104;
        bool var_105;
        const wp::float32 var_106 = 1.0;
        const wp::float32 var_107 = -1.0;
        wp::float32 var_108;
        wp::array_t<wp::float32> var_109;
        wp::float32 var_110;
        wp::float32 var_111;
        wp::int32 var_112;
        //---------
        // forward
        // def _cfrc_ext_tendon_constraint(                                                       <L 1734>
        // worldid, efcid = wp.tid()                                                              <L 1751>
        builtin_tid2d(var_0, var_1);
        // if efcid >= nefc_in[worldid]:                                                          <L 1752>
        var_2 = wp::address(var_nefc_in, var_0);
        var_4 = wp::load(var_2);
        var_3 = (var_1 >= var_4);
        if (var_3) {
            // return                                                                             <L 1753>
            continue;
        }
        // efc_force = efc_force_in[worldid, efcid]                                               <L 1755>
        var_5 = wp::address(var_efc_force_in, var_0, var_1);
        var_7 = wp::load(var_5);
        var_6 = wp::copy(var_7);
        // if efc_force == 0.0:                                                                   <L 1756>
        var_9 = (var_6 == var_8);
        if (var_9) {
            // return                                                                             <L 1757>
            continue;
        }
        // efc_type = efc_type_in[worldid, efcid]                                                 <L 1759>
        var_10 = wp::address(var_efc_type_in, var_0, var_1);
        var_12 = wp::load(var_10);
        var_11 = wp::copy(var_12);
        // if efc_type == ConstraintType.EQUALITY:                                                <L 1761>
        var_14 = (var_11 == var_13);
        if (var_14) {
            // eqid = efc_id_in[worldid, efcid]                                                   <L 1762>
            var_15 = wp::address(var_efc_id_in, var_0, var_1);
            var_17 = wp::load(var_15);
            var_16 = wp::copy(var_17);
            // if eq_type[eqid] != EqType.TENDON:                                                 <L 1763>
            var_18 = wp::address(var_eq_type, var_16);
            var_21 = wp::load(var_18);
            var_20 = (var_21 != var_19);
            if (var_20) {
                // return                                                                         <L 1764>
                continue;
            }
            // obj1 = eq_obj1id[eqid]                                                             <L 1765>
            var_22 = wp::address(var_eq_obj1id, var_16);
            var_24 = wp::load(var_22);
            var_23 = wp::copy(var_24);
            // obj2 = eq_obj2id[eqid]                                                             <L 1766>
            var_25 = wp::address(var_eq_obj2id, var_16);
            var_27 = wp::load(var_25);
            var_26 = wp::copy(var_27);
            // wp.atomic_add(ten_frc_out[worldid], obj1, efc_force)                               <L 1767>
            var_28 = wp::view(var_ten_frc_out, var_0);
            var_29 = wp::atomic_add(var_28, var_23, var_6);
            // if obj2 >= 0:                                                                      <L 1768>
            var_31 = (var_26 >= var_30);
            if (var_31) {
                // dif = ten_length_in[worldid, obj2] - tendon_length0[worldid % tendon_length0.shape[0], obj2]       <L 1769>
                var_32 = wp::address(var_ten_length_in, var_0, var_26);
                var_33 = &(var_tendon_length0.shape);
                var_36 = wp::load(var_33);
                var_35 = wp::extract(var_36, var_34);
                var_37 = wp::mod(var_0, var_35);
                var_38 = wp::address(var_tendon_length0, var_37, var_26);
                var_40 = wp::load(var_32);
                var_41 = wp::load(var_38);
                var_39 = wp::sub(var_40, var_41);
                // eq_data_ = eq_data[worldid % eq_data.shape[0], eqid]                           <L 1770>
                var_42 = &(var_eq_data.shape);
                var_45 = wp::load(var_42);
                var_44 = wp::extract(var_45, var_43);
                var_46 = wp::mod(var_0, var_44);
                var_47 = wp::address(var_eq_data, var_46, var_16);
                var_49 = wp::load(var_47);
                var_48 = wp::copy(var_49);
                // deriv = eq_data_[1] + 2.0 * eq_data_[2] * dif + 3.0 * eq_data_[3] * dif * dif + 4.0 * eq_data_[4] * dif * dif * dif       <L 1771>
                var_51 = wp::extract(var_48, var_50);
                var_54 = wp::extract(var_48, var_53);
                var_55 = wp::mul(var_52, var_54);
                var_56 = wp::mul(var_55, var_39);
                var_57 = wp::add(var_51, var_56);
                var_60 = wp::extract(var_48, var_59);
                var_61 = wp::mul(var_58, var_60);
                var_62 = wp::mul(var_61, var_39);
                var_63 = wp::mul(var_62, var_39);
                var_64 = wp::add(var_57, var_63);
                var_67 = wp::extract(var_48, var_66);
                var_68 = wp::mul(var_65, var_67);
                var_69 = wp::mul(var_68, var_39);
                var_70 = wp::mul(var_69, var_39);
                var_71 = wp::mul(var_70, var_39);
                var_72 = wp::add(var_64, var_71);
                // wp.atomic_sub(ten_frc_out[worldid], obj2, deriv * efc_force)                   <L 1772>
                var_73 = wp::view(var_ten_frc_out, var_0);
                var_74 = wp::mul(var_72, var_6);
                var_75 = wp::atomic_sub(var_73, var_26, var_74);
            }
            // return                                                                             <L 1773>
            continue;
        }
        // if efc_type == ConstraintType.FRICTION_TENDON:                                         <L 1775>
        var_77 = (var_11 == var_76);
        if (var_77) {
            // tenid = efc_id_in[worldid, efcid]                                                  <L 1776>
            var_78 = wp::address(var_efc_id_in, var_0, var_1);
            var_80 = wp::load(var_78);
            var_79 = wp::copy(var_80);
            // wp.atomic_add(ten_frc_out[worldid], tenid, efc_force)                              <L 1777>
            var_81 = wp::view(var_ten_frc_out, var_0);
            var_82 = wp::atomic_add(var_81, var_79, var_6);
            // return                                                                             <L 1778>
            continue;
        }
        // if efc_type == ConstraintType.LIMIT_TENDON:                                            <L 1780>
        var_84 = (var_11 == var_83);
        if (var_84) {
            // tenid = efc_id_in[worldid, efcid]                                                  <L 1781>
            var_85 = wp::address(var_efc_id_in, var_0, var_1);
            var_87 = wp::load(var_85);
            var_86 = wp::copy(var_87);
            // tenrange = tendon_range[worldid % tendon_range.shape[0], tenid]                    <L 1782>
            var_88 = &(var_tendon_range.shape);
            var_91 = wp::load(var_88);
            var_90 = wp::extract(var_91, var_89);
            var_92 = wp::mod(var_0, var_90);
            var_93 = wp::address(var_tendon_range, var_92, var_86);
            var_95 = wp::load(var_93);
            var_94 = wp::copy(var_95);
            // length = ten_length_in[worldid, tenid]                                             <L 1783>
            var_96 = wp::address(var_ten_length_in, var_0, var_86);
            var_98 = wp::load(var_96);
            var_97 = wp::copy(var_98);
            // dist_min = length - tenrange[0]                                                    <L 1784>
            var_100 = wp::extract(var_94, var_99);
            var_101 = wp::sub(var_97, var_100);
            // dist_max = tenrange[1] - length                                                    <L 1785>
            var_103 = wp::extract(var_94, var_102);
            var_104 = wp::sub(var_103, var_97);
            // scl = wp.where(dist_min < dist_max, 1.0, -1.0)                                     <L 1786>
            var_105 = (var_101 < var_104);
            var_108 = wp::where(var_105, var_106, var_107);
            // wp.atomic_add(ten_frc_out[worldid], tenid, scl * efc_force)                        <L 1787>
            var_109 = wp::view(var_ten_frc_out, var_0);
            var_110 = wp::mul(var_108, var_6);
            var_111 = wp::atomic_add(var_109, var_86, var_110);
        }
        var_112 = wp::where(var_84, var_86, var_79);
    }
}



extern "C" __global__ void _crb_accumulate_073a11b8_cuda_kernel_forward(
    wp::launch_bounds_t<2> dim,
    wp::array_t<wp::int32> var_body_parentid,
    wp::array_t<wp::vec_t<10, wp::float32>> var_crb_in,
    wp::array_t<wp::int32> var_body_tree_,
    wp::array_t<wp::vec_t<10, wp::float32>> var_crb_out)
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
        wp::int32* var_5;
        wp::int32 var_6;
        wp::int32 var_7;
        const wp::int32 var_8 = 0;
        bool var_9;
        wp::vec_t<10, wp::float32>* var_10;
        wp::vec_t<10, wp::float32> var_11;
        wp::vec_t<10, wp::float32> var_12;
        //---------
        // forward
        // def _crb_accumulate(                                                                   <L 1000>
        // worldid, nodeid = wp.tid()                                                             <L 1010>
        builtin_tid2d(var_0, var_1);
        // bodyid = body_tree_[nodeid]                                                            <L 1011>
        var_2 = wp::address(var_body_tree_, var_1);
        var_4 = wp::load(var_2);
        var_3 = wp::copy(var_4);
        // pid = body_parentid[bodyid]                                                            <L 1012>
        var_5 = wp::address(var_body_parentid, var_3);
        var_7 = wp::load(var_5);
        var_6 = wp::copy(var_7);
        // if pid == 0:                                                                           <L 1013>
        var_9 = (var_6 == var_8);
        if (var_9) {
            // return                                                                             <L 1014>
            continue;
        }
        // wp.atomic_add(crb_out, worldid, pid, crb_in[worldid, bodyid])                          <L 1015>
        var_10 = wp::address(var_crb_in, var_0, var_3);
        var_12 = wp::load(var_10);
        var_11 = wp::atomic_add(var_crb_out, var_0, var_6, var_12);
    }
}



extern "C" __global__ void _site_local_to_global_084f92a4_cuda_kernel_forward(
    wp::launch_bounds_t<2> dim,
    wp::array_t<wp::int32> var_site_bodyid,
    wp::array_t<wp::vec_t<3, wp::float32>> var_site_pos,
    wp::array_t<wp::quat_t<wp::float32>> var_site_quat,
    wp::array_t<wp::vec_t<3, wp::float32>> var_xpos_in,
    wp::array_t<wp::quat_t<wp::float32>> var_xquat_in,
    wp::array_t<wp::vec_t<3, wp::float32>> var_site_xpos_out,
    wp::array_t<wp::mat_t<3, 3, wp::float32>> var_site_xmat_out)
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
        wp::vec_t<3, wp::float32>* var_5;
        wp::vec_t<3, wp::float32> var_6;
        wp::vec_t<3, wp::float32> var_7;
        wp::quat_t<wp::float32>* var_8;
        wp::quat_t<wp::float32> var_9;
        wp::quat_t<wp::float32> var_10;
        wp::shape_t* var_11;
        const wp::int32 var_12 = 0;
        wp::int32 var_13;
        wp::shape_t var_14;
        wp::int32 var_15;
        wp::vec_t<3, wp::float32>* var_16;
        wp::vec_t<3, wp::float32> var_17;
        wp::vec_t<3, wp::float32> var_18;
        wp::vec_t<3, wp::float32> var_19;
        wp::shape_t* var_20;
        const wp::int32 var_21 = 0;
        wp::int32 var_22;
        wp::shape_t var_23;
        wp::int32 var_24;
        wp::quat_t<wp::float32>* var_25;
        wp::quat_t<wp::float32> var_26;
        wp::quat_t<wp::float32> var_27;
        wp::mat_t<3, 3, wp::float32> var_28;
        //---------
        // forward
        // def _site_local_to_global(                                                             <L 210>
        // worldid, siteid = wp.tid()                                                             <L 222>
        builtin_tid2d(var_0, var_1);
        // bodyid = site_bodyid[siteid]                                                           <L 223>
        var_2 = wp::address(var_site_bodyid, var_1);
        var_4 = wp::load(var_2);
        var_3 = wp::copy(var_4);
        // xpos = xpos_in[worldid, bodyid]                                                        <L 224>
        var_5 = wp::address(var_xpos_in, var_0, var_3);
        var_7 = wp::load(var_5);
        var_6 = wp::copy(var_7);
        // xquat = xquat_in[worldid, bodyid]                                                      <L 225>
        var_8 = wp::address(var_xquat_in, var_0, var_3);
        var_10 = wp::load(var_8);
        var_9 = wp::copy(var_10);
        // site_xpos_out[worldid, siteid] = xpos + math.rot_vec_quat(site_pos[worldid % site_pos.shape[0], siteid], xquat)       <L 226>
        var_11 = &(var_site_pos.shape);
        var_14 = wp::load(var_11);
        var_13 = wp::extract(var_14, var_12);
        var_15 = wp::mod(var_0, var_13);
        var_16 = wp::address(var_site_pos, var_15, var_1);
        var_18 = wp::load(var_16);
        var_17 = rot_vec_quat_0(var_18, var_9);
        var_19 = wp::add(var_6, var_17);
        wp::array_store(var_site_xpos_out, var_0, var_1, var_19);
        // site_xmat_out[worldid, siteid] = math.quat_to_mat(math.mul_quat(xquat, site_quat[worldid % site_quat.shape[0], siteid]))       <L 227>
        var_20 = &(var_site_quat.shape);
        var_23 = wp::load(var_20);
        var_22 = wp::extract(var_23, var_21);
        var_24 = wp::mod(var_0, var_22);
        var_25 = wp::address(var_site_quat, var_24, var_1);
        var_27 = wp::load(var_25);
        var_26 = mul_quat_0(var_9, var_27);
        var_28 = quat_to_mat_0(var_26);
        wp::array_store(var_site_xmat_out, var_0, var_1, var_28);
    }
}



extern "C" __global__ void _tendon_armature_0d777ffa_cuda_kernel_forward(
    wp::launch_bounds_t<3> dim,
    wp::array_t<wp::int32> var_dof_parentid,
    wp::array_t<wp::int32> var_ten_J_rownnz,
    wp::array_t<wp::int32> var_ten_J_rowadr,
    wp::array_t<wp::int32> var_ten_J_colind,
    wp::array_t<wp::float32> var_tendon_armature,
    wp::array_t<wp::int32> var_M_rownnz,
    wp::array_t<wp::int32> var_M_rowadr,
    wp::array_t<wp::float32> var_ten_J_in,
    wp::array_t<wp::float32> var_M_out)
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
        wp::shape_t* var_3;
        const wp::int32 var_4 = 0;
        wp::int32 var_5;
        wp::shape_t var_6;
        wp::int32 var_7;
        wp::float32* var_8;
        wp::float32 var_9;
        wp::float32 var_10;
        const wp::float32 var_11 = 0.0;
        bool var_12;
        wp::int32* var_13;
        wp::int32 var_14;
        wp::int32 var_15;
        bool var_16;
        wp::int32* var_17;
        wp::int32 var_18;
        wp::int32 var_19;
        wp::int32 var_20;
        wp::int32 var_21;
        wp::int32* var_22;
        wp::int32 var_23;
        wp::int32 var_24;
        wp::float32* var_25;
        wp::float32 var_26;
        wp::float32 var_27;
        const wp::float32 var_28 = 0.0;
        bool var_29;
        wp::int32* var_30;
        wp::int32* var_31;
        wp::int32 var_32;
        wp::int32 var_33;
        wp::int32 var_34;
        const wp::int32 var_35 = 1;
        wp::int32 var_36;
        wp::int32 var_37;
        wp::int32 var_38;
        const wp::int32 var_39 = 0;
        bool var_40;
        bool var_41;
        wp::float32 var_42;
        const wp::int32 var_43 = 0;
        bool var_44;
        wp::int32 var_45;
        wp::int32* var_46;
        bool var_47;
        wp::int32 var_48;
        wp::int32 var_49;
        const wp::int32 var_50 = 1;
        wp::int32 var_51;
        bool var_52;
        const wp::int32 var_53 = 0;
        bool var_54;
        wp::int32* var_55;
        bool var_56;
        wp::int32 var_57;
        wp::float32* var_58;
        wp::float32 var_59;
        wp::float32 var_60;
        const wp::float32 var_61 = 0.0;
        wp::float32 var_62;
        wp::float32 var_63;
        wp::float32 var_64;
        wp::float32 var_65;
        wp::float32 var_66;
        wp::array_t<wp::float32> var_67;
        wp::float32 var_68;
        const wp::int32 var_69 = 1;
        wp::int32 var_70;
        wp::int32* var_71;
        wp::int32 var_72;
        wp::int32 var_73;
        //---------
        // forward
        // def _tendon_armature(                                                                  <L 1072>
        // worldid, tenid, dofid = wp.tid()                                                       <L 1086>
        builtin_tid3d(var_0, var_1, var_2);
        // armature = tendon_armature[worldid % tendon_armature.shape[0], tenid]                  <L 1088>
        var_3 = &(var_tendon_armature.shape);
        var_6 = wp::load(var_3);
        var_5 = wp::extract(var_6, var_4);
        var_7 = wp::mod(var_0, var_5);
        var_8 = wp::address(var_tendon_armature, var_7, var_1);
        var_10 = wp::load(var_8);
        var_9 = wp::copy(var_10);
        // if armature == 0.0:                                                                    <L 1090>
        var_12 = (var_9 == var_11);
        if (var_12) {
            // return                                                                             <L 1091>
            continue;
        }
        // rownnz = ten_J_rownnz[tenid]                                                           <L 1093>
        var_13 = wp::address(var_ten_J_rownnz, var_1);
        var_15 = wp::load(var_13);
        var_14 = wp::copy(var_15);
        // if dofid >= rownnz:                                                                    <L 1094>
        var_16 = (var_2 >= var_14);
        if (var_16) {
            // return                                                                             <L 1095>
            continue;
        }
        // rowadr = ten_J_rowadr[tenid]                                                           <L 1096>
        var_17 = wp::address(var_ten_J_rowadr, var_1);
        var_19 = wp::load(var_17);
        var_18 = wp::copy(var_19);
        // dofid_sparse = dofid                                                                   <L 1097>
        var_20 = wp::copy(var_2);
        // sparseid = rowadr + dofid_sparse                                                       <L 1098>
        var_21 = wp::add(var_18, var_20);
        // dofid = ten_J_colind[sparseid]                                                         <L 1099>
        var_22 = wp::address(var_ten_J_colind, var_21);
        var_24 = wp::load(var_22);
        var_23 = wp::copy(var_24);
        // ten_Ji = ten_J_in[worldid, sparseid]                                                   <L 1100>
        var_25 = wp::address(var_ten_J_in, var_0, var_21);
        var_27 = wp::load(var_25);
        var_26 = wp::copy(var_27);
        // if ten_Ji == 0.0:                                                                      <L 1102>
        var_29 = (var_26 == var_28);
        if (var_29) {
            // return                                                                             <L 1103>
            continue;
        }
        // madr_ij = M_rowadr[dofid] + M_rownnz[dofid] - 1                                        <L 1106>
        var_30 = wp::address(var_M_rowadr, var_23);
        var_31 = wp::address(var_M_rownnz, var_23);
        var_33 = wp::load(var_30);
        var_34 = wp::load(var_31);
        var_32 = wp::add(var_33, var_34);
        var_36 = wp::sub(var_32, var_35);
        // dofidi = dofid                                                                         <L 1109>
        var_37 = wp::copy(var_23);
        // ptr = dofid_sparse                                                                     <L 1110>
        var_38 = wp::copy(var_20);
        // while dofid >= 0:                                                                      <L 1111>
        start_while_3:;
        var_40 = (var_23 >= var_39);
        if ((var_40) == false) goto end_while_3;
            // if dofid == dofidi:                                                                <L 1112>
            var_41 = (var_23 == var_37);
            if (var_41) {
                // ten_Jj = ten_Ji                                                                <L 1113>
                var_42 = wp::copy(var_26);
            }
            if (!var_41) {
                // while ptr >= 0:                                                                <L 1116>
        start_while_5:;
                var_44 = (var_38 >= var_43);
        if ((var_44) == false) goto end_while_5;
                    // sparseid = rowadr + ptr                                                    <L 1117>
                    var_45 = wp::add(var_18, var_38);
                    // if ten_J_colind[sparseid] <= dofid:                                        <L 1118>
                    var_46 = wp::address(var_ten_J_colind, var_45);
                    var_48 = wp::load(var_46);
                    var_47 = (var_48 <= var_23);
                    if (var_47) {
                        // break                                                                  <L 1119>
                        wp::assign(var_21, var_45);
                        goto end_while_5;
                    }
                    var_49 = wp::where(var_47, var_21, var_45);
                    // ptr -= 1                                                                   <L 1120>
                    var_51 = wp::sub(var_38, var_50);
                    wp::assign(var_21, var_49);
                    wp::assign(var_38, var_51);
        goto start_while_5;
        end_while_5:;
                // if ptr >= 0 and ten_J_colind[sparseid] == dofid:                               <L 1121>
                var_54 = (var_38 >= var_53);
                var_52 = var_54;
                if (var_52) {
                    var_55 = wp::address(var_ten_J_colind, var_21);
                    var_57 = wp::load(var_55);
                    var_56 = (var_57 == var_23);
                    var_52 = var_52 && var_56;
                }
                if (var_52) {
                    // ten_Jj = ten_J_in[worldid, sparseid]                                       <L 1122>
                    var_58 = wp::address(var_ten_J_in, var_0, var_21);
                    var_60 = wp::load(var_58);
                    var_59 = wp::copy(var_60);
                }
                if (!var_52) {
                    // ten_Jj = float(0.0)                                                        <L 1124>
                    var_62 = wp::float(var_61);
                }
                var_63 = wp::where(var_52, var_59, var_62);
            }
            var_64 = wp::where(var_41, var_42, var_63);
            // Mij = armature * ten_Jj * ten_Ji                                                   <L 1126>
            var_65 = wp::mul(var_9, var_64);
            var_66 = wp::mul(var_65, var_26);
            // wp.atomic_add(M_out[worldid], madr_ij, Mij)                                        <L 1128>
            var_67 = wp::view(var_M_out, var_0);
            var_68 = wp::atomic_add(var_67, var_36, var_66);
            // madr_ij -= 1                                                                       <L 1129>
            var_70 = wp::sub(var_36, var_69);
            // dofid = dof_parentid[dofid]                                                        <L 1131>
            var_71 = wp::address(var_dof_parentid, var_23);
            var_73 = wp::load(var_71);
            var_72 = wp::copy(var_73);
            wp::assign(var_23, var_72);
            wp::assign(var_36, var_70);
        goto start_while_3;
        end_while_3:;
    }
}



extern "C" __global__ void _subtree_div_0ef57c26_cuda_kernel_forward(
    wp::launch_bounds_t<2> dim,
    wp::array_t<wp::float32> var_body_subtreemass,
    wp::array_t<wp::vec_t<3, wp::float32>> var_subtree_com_in,
    wp::array_t<wp::vec_t<3, wp::float32>> var_subtree_com_out)
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
        wp::vec_t<3, wp::float32>* var_2;
        wp::vec_t<3, wp::float32> var_3;
        wp::vec_t<3, wp::float32> var_4;
        wp::shape_t* var_5;
        const wp::int32 var_6 = 0;
        wp::int32 var_7;
        wp::shape_t var_8;
        wp::int32 var_9;
        wp::float32* var_10;
        wp::float32 var_11;
        wp::float32 var_12;
        const wp::float32 var_13 = 0.0;
        bool var_14;
        wp::vec_t<3, wp::float32> var_15;
        //---------
        // forward
        // def _subtree_div(                                                                      <L 688>
        // worldid, bodyid = wp.tid()                                                             <L 696>
        builtin_tid2d(var_0, var_1);
        // com = subtree_com_in[worldid, bodyid]                                                  <L 697>
        var_2 = wp::address(var_subtree_com_in, var_0, var_1);
        var_4 = wp::load(var_2);
        var_3 = wp::copy(var_4);
        // mass = body_subtreemass[worldid % body_subtreemass.shape[0], bodyid]                   <L 698>
        var_5 = &(var_body_subtreemass.shape);
        var_8 = wp::load(var_5);
        var_7 = wp::extract(var_8, var_6);
        var_9 = wp::mod(var_0, var_7);
        var_10 = wp::address(var_body_subtreemass, var_9, var_1);
        var_12 = wp::load(var_10);
        var_11 = wp::copy(var_12);
        // if mass != 0.0:                                                                        <L 699>
        var_14 = (var_11 != var_13);
        if (var_14) {
            // subtree_com_out[worldid, bodyid] = com / mass                                      <L 700>
            var_15 = wp::div(var_3, var_11);
            wp::array_store(var_subtree_com_out, var_0, var_1, var_15);
        }
    }
}



extern "C" __global__ void _comvel_branch_0f933c58_cuda_kernel_forward(
    wp::launch_bounds_t<2> dim,
    wp::array_t<wp::int32> var_body_parentid,
    wp::array_t<wp::int32> var_body_jntnum,
    wp::array_t<wp::int32> var_body_jntadr,
    wp::array_t<wp::int32> var_body_dofadr,
    wp::array_t<wp::int32> var_jnt_type,
    wp::array_t<wp::int32> var_body_branches,
    wp::array_t<wp::int32> var_body_branch_start,
    wp::array_t<wp::float32> var_qvel_in,
    wp::array_t<wp::vec_t<6, wp::float32>> var_cdof_in,
    wp::array_t<wp::vec_t<6, wp::float32>> var_cvel_out,
    wp::array_t<wp::vec_t<6, wp::float32>> var_cdof_dot_out)
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
        const wp::int32 var_5 = 1;
        wp::int32 var_6;
        wp::int32* var_7;
        wp::int32 var_8;
        wp::int32 var_9;
        wp::array_t<wp::float32> var_10;
        wp::array_t<wp::vec_t<6, wp::float32>> var_11;
        wp::range_t var_12;
        wp::int32 var_13;
        wp::int32* var_14;
        wp::int32 var_15;
        wp::int32 var_16;
        wp::int32* var_17;
        wp::int32 var_18;
        wp::int32 var_19;
        wp::vec_t<6, wp::float32>* var_20;
        wp::vec_t<6, wp::float32> var_21;
        wp::vec_t<6, wp::float32> var_22;
        wp::int32* var_23;
        wp::int32 var_24;
        wp::int32 var_25;
        wp::int32* var_26;
        wp::int32 var_27;
        wp::int32 var_28;
        wp::int32* var_29;
        wp::int32 var_30;
        wp::int32 var_31;
        const wp::int32 var_32 = 0;
        bool var_33;
        wp::int32 var_34;
        wp::range_t var_35;
        wp::int32 var_36;
        wp::int32* var_37;
        wp::int32 var_38;
        wp::int32 var_39;
        const wp::int32 var_40 = 0;
        bool var_41;
        const wp::int32 var_42 = 0;
        wp::int32 var_43;
        wp::vec_t<6, wp::float32>* var_44;
        const wp::int32 var_45 = 0;
        wp::int32 var_46;
        wp::float32* var_47;
        wp::vec_t<6, wp::float32> var_48;
        wp::vec_t<6, wp::float32> var_49;
        wp::float32 var_50;
        wp::vec_t<6, wp::float32> var_51;
        const wp::int32 var_52 = 1;
        wp::int32 var_53;
        wp::vec_t<6, wp::float32>* var_54;
        const wp::int32 var_55 = 1;
        wp::int32 var_56;
        wp::float32* var_57;
        wp::vec_t<6, wp::float32> var_58;
        wp::vec_t<6, wp::float32> var_59;
        wp::float32 var_60;
        wp::vec_t<6, wp::float32> var_61;
        const wp::int32 var_62 = 2;
        wp::int32 var_63;
        wp::vec_t<6, wp::float32>* var_64;
        const wp::int32 var_65 = 2;
        wp::int32 var_66;
        wp::float32* var_67;
        wp::vec_t<6, wp::float32> var_68;
        wp::vec_t<6, wp::float32> var_69;
        wp::float32 var_70;
        wp::vec_t<6, wp::float32> var_71;
        const wp::float32 var_72 = 0.0;
        const wp::float32 var_73 = 0.0;
        const wp::float32 var_74 = 0.0;
        const wp::float32 var_75 = 0.0;
        const wp::float32 var_76 = 0.0;
        const wp::float32 var_77 = 0.0;
        wp::vec_t<6, wp::float32> var_78;
        const wp::int32 var_79 = 0;
        wp::int32 var_80;
        const wp::float32 var_81 = 0.0;
        const wp::float32 var_82 = 0.0;
        const wp::float32 var_83 = 0.0;
        const wp::float32 var_84 = 0.0;
        const wp::float32 var_85 = 0.0;
        const wp::float32 var_86 = 0.0;
        wp::vec_t<6, wp::float32> var_87;
        const wp::int32 var_88 = 1;
        wp::int32 var_89;
        const wp::float32 var_90 = 0.0;
        const wp::float32 var_91 = 0.0;
        const wp::float32 var_92 = 0.0;
        const wp::float32 var_93 = 0.0;
        const wp::float32 var_94 = 0.0;
        const wp::float32 var_95 = 0.0;
        wp::vec_t<6, wp::float32> var_96;
        const wp::int32 var_97 = 2;
        wp::int32 var_98;
        const wp::int32 var_99 = 3;
        wp::int32 var_100;
        wp::vec_t<6, wp::float32>* var_101;
        wp::vec_t<6, wp::float32> var_102;
        wp::vec_t<6, wp::float32> var_103;
        const wp::int32 var_104 = 3;
        wp::int32 var_105;
        const wp::int32 var_106 = 4;
        wp::int32 var_107;
        wp::vec_t<6, wp::float32>* var_108;
        wp::vec_t<6, wp::float32> var_109;
        wp::vec_t<6, wp::float32> var_110;
        const wp::int32 var_111 = 4;
        wp::int32 var_112;
        const wp::int32 var_113 = 5;
        wp::int32 var_114;
        wp::vec_t<6, wp::float32>* var_115;
        wp::vec_t<6, wp::float32> var_116;
        wp::vec_t<6, wp::float32> var_117;
        const wp::int32 var_118 = 5;
        wp::int32 var_119;
        const wp::int32 var_120 = 3;
        wp::int32 var_121;
        wp::vec_t<6, wp::float32>* var_122;
        const wp::int32 var_123 = 3;
        wp::int32 var_124;
        wp::float32* var_125;
        wp::vec_t<6, wp::float32> var_126;
        wp::vec_t<6, wp::float32> var_127;
        wp::float32 var_128;
        wp::vec_t<6, wp::float32> var_129;
        const wp::int32 var_130 = 4;
        wp::int32 var_131;
        wp::vec_t<6, wp::float32>* var_132;
        const wp::int32 var_133 = 4;
        wp::int32 var_134;
        wp::float32* var_135;
        wp::vec_t<6, wp::float32> var_136;
        wp::vec_t<6, wp::float32> var_137;
        wp::float32 var_138;
        wp::vec_t<6, wp::float32> var_139;
        const wp::int32 var_140 = 5;
        wp::int32 var_141;
        wp::vec_t<6, wp::float32>* var_142;
        const wp::int32 var_143 = 5;
        wp::int32 var_144;
        wp::float32* var_145;
        wp::vec_t<6, wp::float32> var_146;
        wp::vec_t<6, wp::float32> var_147;
        wp::float32 var_148;
        wp::vec_t<6, wp::float32> var_149;
        const wp::int32 var_150 = 6;
        wp::int32 var_151;
        const wp::int32 var_152 = 1;
        bool var_153;
        const wp::int32 var_154 = 0;
        wp::int32 var_155;
        wp::vec_t<6, wp::float32>* var_156;
        wp::vec_t<6, wp::float32> var_157;
        wp::vec_t<6, wp::float32> var_158;
        const wp::int32 var_159 = 0;
        wp::int32 var_160;
        const wp::int32 var_161 = 1;
        wp::int32 var_162;
        wp::vec_t<6, wp::float32>* var_163;
        wp::vec_t<6, wp::float32> var_164;
        wp::vec_t<6, wp::float32> var_165;
        const wp::int32 var_166 = 1;
        wp::int32 var_167;
        const wp::int32 var_168 = 2;
        wp::int32 var_169;
        wp::vec_t<6, wp::float32>* var_170;
        wp::vec_t<6, wp::float32> var_171;
        wp::vec_t<6, wp::float32> var_172;
        const wp::int32 var_173 = 2;
        wp::int32 var_174;
        const wp::int32 var_175 = 0;
        wp::int32 var_176;
        wp::vec_t<6, wp::float32>* var_177;
        const wp::int32 var_178 = 0;
        wp::int32 var_179;
        wp::float32* var_180;
        wp::vec_t<6, wp::float32> var_181;
        wp::vec_t<6, wp::float32> var_182;
        wp::float32 var_183;
        wp::vec_t<6, wp::float32> var_184;
        const wp::int32 var_185 = 1;
        wp::int32 var_186;
        wp::vec_t<6, wp::float32>* var_187;
        const wp::int32 var_188 = 1;
        wp::int32 var_189;
        wp::float32* var_190;
        wp::vec_t<6, wp::float32> var_191;
        wp::vec_t<6, wp::float32> var_192;
        wp::float32 var_193;
        wp::vec_t<6, wp::float32> var_194;
        const wp::int32 var_195 = 2;
        wp::int32 var_196;
        wp::vec_t<6, wp::float32>* var_197;
        const wp::int32 var_198 = 2;
        wp::int32 var_199;
        wp::float32* var_200;
        wp::vec_t<6, wp::float32> var_201;
        wp::vec_t<6, wp::float32> var_202;
        wp::float32 var_203;
        wp::vec_t<6, wp::float32> var_204;
        const wp::int32 var_205 = 3;
        wp::int32 var_206;
        wp::vec_t<6, wp::float32>* var_207;
        wp::vec_t<6, wp::float32> var_208;
        wp::vec_t<6, wp::float32> var_209;
        wp::vec_t<6, wp::float32>* var_210;
        wp::float32* var_211;
        wp::vec_t<6, wp::float32> var_212;
        wp::vec_t<6, wp::float32> var_213;
        wp::float32 var_214;
        wp::vec_t<6, wp::float32> var_215;
        const wp::int32 var_216 = 1;
        wp::int32 var_217;
        wp::vec_t<6, wp::float32> var_218;
        wp::int32 var_219;
        wp::vec_t<6, wp::float32> var_220;
        wp::int32 var_221;
        //---------
        // forward
        // def _comvel_branch(                                                                    <L 2515>
        // worldid, branchid = wp.tid()                                                           <L 2531>
        builtin_tid2d(var_0, var_1);
        // start = body_branch_start[branchid]                                                    <L 2533>
        var_2 = wp::address(var_body_branch_start, var_1);
        var_4 = wp::load(var_2);
        var_3 = wp::copy(var_4);
        // end = body_branch_start[branchid + 1]                                                  <L 2534>
        var_6 = wp::add(var_1, var_5);
        var_7 = wp::address(var_body_branch_start, var_6);
        var_9 = wp::load(var_7);
        var_8 = wp::copy(var_9);
        // qvel = qvel_in[worldid]                                                                <L 2536>
        var_10 = wp::view(var_qvel_in, var_0);
        // cdof = cdof_in[worldid]                                                                <L 2537>
        var_11 = wp::view(var_cdof_in, var_0);
        // for i in range(start, end):                                                            <L 2539>
        var_12 = wp::range(var_3, var_8);
        start_for_0:;
            if (iter_cmp(var_12) == 0) goto end_for_0;
            var_13 = wp::iter_next(var_12);
            // bodyid = body_branches[i]                                                          <L 2540>
            var_14 = wp::address(var_body_branches, var_13);
            var_16 = wp::load(var_14);
            var_15 = wp::copy(var_16);
            // pid = body_parentid[bodyid]                                                        <L 2541>
            var_17 = wp::address(var_body_parentid, var_15);
            var_19 = wp::load(var_17);
            var_18 = wp::copy(var_19);
            // cvel = cvel_out[worldid, pid]                                                      <L 2542>
            var_20 = wp::address(var_cvel_out, var_0, var_18);
            var_22 = wp::load(var_20);
            var_21 = wp::copy(var_22);
            // dofid = body_dofadr[bodyid]                                                        <L 2543>
            var_23 = wp::address(var_body_dofadr, var_15);
            var_25 = wp::load(var_23);
            var_24 = wp::copy(var_25);
            // jntid = body_jntadr[bodyid]                                                        <L 2544>
            var_26 = wp::address(var_body_jntadr, var_15);
            var_28 = wp::load(var_26);
            var_27 = wp::copy(var_28);
            // jntnum = body_jntnum[bodyid]                                                       <L 2545>
            var_29 = wp::address(var_body_jntnum, var_15);
            var_31 = wp::load(var_29);
            var_30 = wp::copy(var_31);
            // if jntnum == 0:                                                                    <L 2547>
            var_33 = (var_30 == var_32);
            if (var_33) {
                // cvel_out[worldid, bodyid] = cvel                                               <L 2548>
                wp::array_store(var_cvel_out, var_0, var_15, var_21);
                // continue                                                                       <L 2549>
                goto start_for_0;
            }
            // for j in range(jntid, jntid + jntnum):                                             <L 2551>
            var_34 = wp::add(var_27, var_30);
            var_35 = wp::range(var_27, var_34);
            start_for_2:;
                if (iter_cmp(var_35) == 0) goto end_for_2;
                var_36 = wp::iter_next(var_35);
                // jnttype = jnt_type[j]                                                          <L 2552>
                var_37 = wp::address(var_jnt_type, var_36);
                var_39 = wp::load(var_37);
                var_38 = wp::copy(var_39);
                // if jnttype == JointType.FREE:                                                  <L 2554>
                var_41 = (var_38 == var_40);
                if (var_41) {
                    // cvel += cdof[dofid + 0] * qvel[dofid + 0]                                  <L 2555>
                    var_43 = wp::add(var_24, var_42);
                    var_44 = wp::address(var_11, var_43);
                    var_46 = wp::add(var_24, var_45);
                    var_47 = wp::address(var_10, var_46);
                    var_49 = wp::load(var_44);
                    var_50 = wp::load(var_47);
                    var_48 = wp::mul(var_49, var_50);
                    var_51 = wp::add(var_21, var_48);
                    // cvel += cdof[dofid + 1] * qvel[dofid + 1]                                  <L 2556>
                    var_53 = wp::add(var_24, var_52);
                    var_54 = wp::address(var_11, var_53);
                    var_56 = wp::add(var_24, var_55);
                    var_57 = wp::address(var_10, var_56);
                    var_59 = wp::load(var_54);
                    var_60 = wp::load(var_57);
                    var_58 = wp::mul(var_59, var_60);
                    var_61 = wp::add(var_51, var_58);
                    // cvel += cdof[dofid + 2] * qvel[dofid + 2]                                  <L 2557>
                    var_63 = wp::add(var_24, var_62);
                    var_64 = wp::address(var_11, var_63);
                    var_66 = wp::add(var_24, var_65);
                    var_67 = wp::address(var_10, var_66);
                    var_69 = wp::load(var_64);
                    var_70 = wp::load(var_67);
                    var_68 = wp::mul(var_69, var_70);
                    var_71 = wp::add(var_61, var_68);
                    // cdof_dot_out[worldid, dofid + 0] = wp.spatial_vector(0.0, 0.0, 0.0, 0.0, 0.0, 0.0)       <L 2559>
                    var_78 = wp::vec_t<6, wp::float32>({var_72, var_73, var_74, var_75, var_76, var_77});
                    var_80 = wp::add(var_24, var_79);
                    wp::array_store(var_cdof_dot_out, var_0, var_80, var_78);
                    // cdof_dot_out[worldid, dofid + 1] = wp.spatial_vector(0.0, 0.0, 0.0, 0.0, 0.0, 0.0)       <L 2560>
                    var_87 = wp::vec_t<6, wp::float32>({var_81, var_82, var_83, var_84, var_85, var_86});
                    var_89 = wp::add(var_24, var_88);
                    wp::array_store(var_cdof_dot_out, var_0, var_89, var_87);
                    // cdof_dot_out[worldid, dofid + 2] = wp.spatial_vector(0.0, 0.0, 0.0, 0.0, 0.0, 0.0)       <L 2561>
                    var_96 = wp::vec_t<6, wp::float32>({var_90, var_91, var_92, var_93, var_94, var_95});
                    var_98 = wp::add(var_24, var_97);
                    wp::array_store(var_cdof_dot_out, var_0, var_98, var_96);
                    // cdof_dot_out[worldid, dofid + 3] = math.motion_cross(cvel, cdof[dofid + 3])       <L 2562>
                    var_100 = wp::add(var_24, var_99);
                    var_101 = wp::address(var_11, var_100);
                    var_103 = wp::load(var_101);
                    var_102 = motion_cross_0(var_71, var_103);
                    var_105 = wp::add(var_24, var_104);
                    wp::array_store(var_cdof_dot_out, var_0, var_105, var_102);
                    // cdof_dot_out[worldid, dofid + 4] = math.motion_cross(cvel, cdof[dofid + 4])       <L 2563>
                    var_107 = wp::add(var_24, var_106);
                    var_108 = wp::address(var_11, var_107);
                    var_110 = wp::load(var_108);
                    var_109 = motion_cross_0(var_71, var_110);
                    var_112 = wp::add(var_24, var_111);
                    wp::array_store(var_cdof_dot_out, var_0, var_112, var_109);
                    // cdof_dot_out[worldid, dofid + 5] = math.motion_cross(cvel, cdof[dofid + 5])       <L 2564>
                    var_114 = wp::add(var_24, var_113);
                    var_115 = wp::address(var_11, var_114);
                    var_117 = wp::load(var_115);
                    var_116 = motion_cross_0(var_71, var_117);
                    var_119 = wp::add(var_24, var_118);
                    wp::array_store(var_cdof_dot_out, var_0, var_119, var_116);
                    // cvel += cdof[dofid + 3] * qvel[dofid + 3]                                  <L 2566>
                    var_121 = wp::add(var_24, var_120);
                    var_122 = wp::address(var_11, var_121);
                    var_124 = wp::add(var_24, var_123);
                    var_125 = wp::address(var_10, var_124);
                    var_127 = wp::load(var_122);
                    var_128 = wp::load(var_125);
                    var_126 = wp::mul(var_127, var_128);
                    var_129 = wp::add(var_71, var_126);
                    // cvel += cdof[dofid + 4] * qvel[dofid + 4]                                  <L 2567>
                    var_131 = wp::add(var_24, var_130);
                    var_132 = wp::address(var_11, var_131);
                    var_134 = wp::add(var_24, var_133);
                    var_135 = wp::address(var_10, var_134);
                    var_137 = wp::load(var_132);
                    var_138 = wp::load(var_135);
                    var_136 = wp::mul(var_137, var_138);
                    var_139 = wp::add(var_129, var_136);
                    // cvel += cdof[dofid + 5] * qvel[dofid + 5]                                  <L 2568>
                    var_141 = wp::add(var_24, var_140);
                    var_142 = wp::address(var_11, var_141);
                    var_144 = wp::add(var_24, var_143);
                    var_145 = wp::address(var_10, var_144);
                    var_147 = wp::load(var_142);
                    var_148 = wp::load(var_145);
                    var_146 = wp::mul(var_147, var_148);
                    var_149 = wp::add(var_139, var_146);
                    // dofid += 6                                                                 <L 2570>
                    var_151 = wp::add(var_24, var_150);
                }
                if (!var_41) {
                    // elif jnttype == JointType.BALL:                                            <L 2571>
                    var_153 = (var_38 == var_152);
                    if (var_153) {
                        // cdof_dot_out[worldid, dofid + 0] = math.motion_cross(cvel, cdof[dofid + 0])       <L 2572>
                        var_155 = wp::add(var_24, var_154);
                        var_156 = wp::address(var_11, var_155);
                        var_158 = wp::load(var_156);
                        var_157 = motion_cross_0(var_21, var_158);
                        var_160 = wp::add(var_24, var_159);
                        wp::array_store(var_cdof_dot_out, var_0, var_160, var_157);
                        // cdof_dot_out[worldid, dofid + 1] = math.motion_cross(cvel, cdof[dofid + 1])       <L 2573>
                        var_162 = wp::add(var_24, var_161);
                        var_163 = wp::address(var_11, var_162);
                        var_165 = wp::load(var_163);
                        var_164 = motion_cross_0(var_21, var_165);
                        var_167 = wp::add(var_24, var_166);
                        wp::array_store(var_cdof_dot_out, var_0, var_167, var_164);
                        // cdof_dot_out[worldid, dofid + 2] = math.motion_cross(cvel, cdof[dofid + 2])       <L 2574>
                        var_169 = wp::add(var_24, var_168);
                        var_170 = wp::address(var_11, var_169);
                        var_172 = wp::load(var_170);
                        var_171 = motion_cross_0(var_21, var_172);
                        var_174 = wp::add(var_24, var_173);
                        wp::array_store(var_cdof_dot_out, var_0, var_174, var_171);
                        // cvel += cdof[dofid + 0] * qvel[dofid + 0]                              <L 2576>
                        var_176 = wp::add(var_24, var_175);
                        var_177 = wp::address(var_11, var_176);
                        var_179 = wp::add(var_24, var_178);
                        var_180 = wp::address(var_10, var_179);
                        var_182 = wp::load(var_177);
                        var_183 = wp::load(var_180);
                        var_181 = wp::mul(var_182, var_183);
                        var_184 = wp::add(var_21, var_181);
                        // cvel += cdof[dofid + 1] * qvel[dofid + 1]                              <L 2577>
                        var_186 = wp::add(var_24, var_185);
                        var_187 = wp::address(var_11, var_186);
                        var_189 = wp::add(var_24, var_188);
                        var_190 = wp::address(var_10, var_189);
                        var_192 = wp::load(var_187);
                        var_193 = wp::load(var_190);
                        var_191 = wp::mul(var_192, var_193);
                        var_194 = wp::add(var_184, var_191);
                        // cvel += cdof[dofid + 2] * qvel[dofid + 2]                              <L 2578>
                        var_196 = wp::add(var_24, var_195);
                        var_197 = wp::address(var_11, var_196);
                        var_199 = wp::add(var_24, var_198);
                        var_200 = wp::address(var_10, var_199);
                        var_202 = wp::load(var_197);
                        var_203 = wp::load(var_200);
                        var_201 = wp::mul(var_202, var_203);
                        var_204 = wp::add(var_194, var_201);
                        // dofid += 3                                                             <L 2580>
                        var_206 = wp::add(var_24, var_205);
                    }
                    if (!var_153) {
                        // cdof_dot_out[worldid, dofid] = math.motion_cross(cvel, cdof[dofid])       <L 2582>
                        var_207 = wp::address(var_11, var_24);
                        var_209 = wp::load(var_207);
                        var_208 = motion_cross_0(var_21, var_209);
                        wp::array_store(var_cdof_dot_out, var_0, var_24, var_208);
                        // cvel += cdof[dofid] * qvel[dofid]                                      <L 2583>
                        var_210 = wp::address(var_11, var_24);
                        var_211 = wp::address(var_10, var_24);
                        var_213 = wp::load(var_210);
                        var_214 = wp::load(var_211);
                        var_212 = wp::mul(var_213, var_214);
                        var_215 = wp::add(var_21, var_212);
                        // dofid += 1                                                             <L 2585>
                        var_217 = wp::add(var_24, var_216);
                    }
                    var_218 = wp::where(var_153, var_204, var_215);
                    var_219 = wp::where(var_153, var_206, var_217);
                }
                var_220 = wp::where(var_41, var_149, var_218);
                var_221 = wp::where(var_41, var_151, var_219);
                wp::assign(var_21, var_220);
                wp::assign(var_24, var_221);
                goto start_for_2;
            end_for_2:;
            // cvel_out[worldid, bodyid] = cvel                                                   <L 2587>
            wp::array_store(var_cvel_out, var_0, var_15, var_21);
            goto start_for_0;
        end_for_0:;
    }
}



extern "C" __global__ void _qLD_acc_120d68f2_cuda_kernel_forward(
    wp::launch_bounds_t<2> dim,
    wp::array_t<wp::int32> var_M_rownnz,
    wp::array_t<wp::int32> var_M_rowadr,
    wp::array_t<wp::vec_t<3, wp::int32>> var_qLD_updates_,
    wp::array_t<wp::float32> var_L_in,
    wp::array_t<wp::float32> var_L_out)
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
        wp::vec_t<3, wp::int32>* var_2;
        wp::vec_t<3, wp::int32> var_3;
        wp::vec_t<3, wp::int32> var_4;
        const wp::int32 var_5 = 0;
        wp::int32 var_6;
        const wp::int32 var_7 = 1;
        wp::int32 var_8;
        const wp::int32 var_9 = 2;
        wp::int32 var_10;
        wp::int32* var_11;
        wp::int32 var_12;
        wp::int32 var_13;
        wp::int32* var_14;
        wp::int32* var_15;
        wp::int32 var_16;
        wp::int32 var_17;
        wp::int32 var_18;
        const wp::int32 var_19 = 1;
        wp::int32 var_20;
        wp::float32* var_21;
        wp::float32* var_22;
        wp::float32 var_23;
        wp::float32 var_24;
        wp::float32 var_25;
        wp::int32* var_26;
        wp::range_t var_27;
        wp::int32 var_28;
        wp::int32 var_29;
        wp::array_t<wp::float32> var_30;
        wp::int32 var_31;
        wp::int32* var_32;
        wp::int32 var_33;
        wp::int32 var_34;
        wp::float32* var_35;
        wp::float32 var_36;
        wp::float32 var_37;
        wp::float32 var_38;
        //---------
        // forward
        // def _qLD_acc(                                                                          <L 1155>
        // worldid, nodeid = wp.tid()                                                             <L 1165>
        builtin_tid2d(var_0, var_1);
        // update = qLD_updates_[nodeid]                                                          <L 1166>
        var_2 = wp::address(var_qLD_updates_, var_1);
        var_4 = wp::load(var_2);
        var_3 = wp::copy(var_4);
        // i, k, Madr_ki = update[0], update[1], update[2]                                        <L 1167>
        var_6 = wp::extract(var_3, var_5);
        var_8 = wp::extract(var_3, var_7);
        var_10 = wp::extract(var_3, var_9);
        // Madr_i = M_rowadr[i]  # Address of row being updated                                   <L 1168>
        var_11 = wp::address(var_M_rowadr, var_6);
        var_13 = wp::load(var_11);
        var_12 = wp::copy(var_13);
        // diag_k = M_rowadr[k] + M_rownnz[k] - 1  # Address of diagonal element of k             <L 1169>
        var_14 = wp::address(var_M_rowadr, var_8);
        var_15 = wp::address(var_M_rownnz, var_8);
        var_17 = wp::load(var_14);
        var_18 = wp::load(var_15);
        var_16 = wp::add(var_17, var_18);
        var_20 = wp::sub(var_16, var_19);
        // tmp = L_out[worldid, Madr_ki] / L_out[worldid, diag_k]                                 <L 1171>
        var_21 = wp::address(var_L_out, var_0, var_10);
        var_22 = wp::address(var_L_out, var_0, var_20);
        var_24 = wp::load(var_21);
        var_25 = wp::load(var_22);
        var_23 = wp::div(var_24, var_25);
        // for j in range(M_rownnz[i]):                                                           <L 1172>
        var_26 = wp::address(var_M_rownnz, var_6);
        var_28 = wp::load(var_26);
        var_27 = wp::range(var_28);
        start_for_0:;
            if (iter_cmp(var_27) == 0) goto end_for_0;
            var_29 = wp::iter_next(var_27);
            // wp.atomic_sub(L_out[worldid], Madr_i + j, L_in[worldid, M_rowadr[k] + j] * tmp)       <L 1174>
            var_30 = wp::view(var_L_out, var_0);
            var_31 = wp::add(var_12, var_29);
            var_32 = wp::address(var_M_rowadr, var_8);
            var_34 = wp::load(var_32);
            var_33 = wp::add(var_34, var_29);
            var_35 = wp::address(var_L_in, var_0, var_33);
            var_37 = wp::load(var_35);
            var_36 = wp::mul(var_37, var_23);
            var_38 = wp::atomic_sub(var_30, var_31, var_36);
            goto start_for_0;
        end_for_0:;
        // L_out[worldid, Madr_ki] = tmp                                                          <L 1176>
        wp::array_store(var_L_out, var_0, var_10, var_23);
    }
}



extern "C" __global__ void _cinert_1261260c_cuda_kernel_forward(
    wp::launch_bounds_t<2> dim,
    wp::array_t<wp::int32> var_body_rootid,
    wp::array_t<wp::float32> var_body_mass,
    wp::array_t<wp::vec_t<3, wp::float32>> var_body_inertia,
    wp::array_t<wp::vec_t<3, wp::float32>> var_xipos_in,
    wp::array_t<wp::mat_t<3, 3, wp::float32>> var_ximat_in,
    wp::array_t<wp::vec_t<3, wp::float32>> var_subtree_com_in,
    wp::array_t<wp::vec_t<10, wp::float32>> var_cinert_out)
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
        wp::mat_t<3, 3, wp::float32>* var_2;
        wp::mat_t<3, 3, wp::float32> var_3;
        wp::mat_t<3, 3, wp::float32> var_4;
        wp::shape_t* var_5;
        const wp::int32 var_6 = 0;
        wp::int32 var_7;
        wp::shape_t var_8;
        wp::int32 var_9;
        wp::vec_t<3, wp::float32>* var_10;
        wp::vec_t<3, wp::float32> var_11;
        wp::vec_t<3, wp::float32> var_12;
        wp::shape_t* var_13;
        const wp::int32 var_14 = 0;
        wp::int32 var_15;
        wp::shape_t var_16;
        wp::int32 var_17;
        wp::float32* var_18;
        wp::float32 var_19;
        wp::float32 var_20;
        wp::vec_t<3, wp::float32>* var_21;
        wp::int32* var_22;
        wp::int32 var_23;
        wp::vec_t<3, wp::float32>* var_24;
        wp::vec_t<3, wp::float32> var_25;
        wp::vec_t<3, wp::float32> var_26;
        wp::vec_t<3, wp::float32> var_27;
        wp::vec_t<10, wp::float32> var_28;
        wp::mat_t<3, 3, wp::float32> var_29;
        wp::mat_t<3, 3, wp::float32> var_30;
        wp::mat_t<3, 3, wp::float32> var_31;
        wp::mat_t<3, 3, wp::float32> var_32;
        const wp::int32 var_33 = 0;
        const wp::int32 var_34 = 0;
        wp::float32 var_35;
        const wp::int32 var_36 = 0;
        const wp::int32 var_37 = 1;
        const wp::int32 var_38 = 1;
        wp::float32 var_39;
        const wp::int32 var_40 = 1;
        const wp::int32 var_41 = 2;
        const wp::int32 var_42 = 2;
        wp::float32 var_43;
        const wp::int32 var_44 = 2;
        const wp::int32 var_45 = 0;
        const wp::int32 var_46 = 1;
        wp::float32 var_47;
        const wp::int32 var_48 = 3;
        const wp::int32 var_49 = 0;
        const wp::int32 var_50 = 2;
        wp::float32 var_51;
        const wp::int32 var_52 = 4;
        const wp::int32 var_53 = 1;
        const wp::int32 var_54 = 2;
        wp::float32 var_55;
        const wp::int32 var_56 = 5;
        const wp::int32 var_57 = 0;
        const wp::int32 var_58 = 1;
        wp::float32 var_59;
        const wp::int32 var_60 = 1;
        wp::float32 var_61;
        wp::float32 var_62;
        const wp::int32 var_63 = 2;
        wp::float32 var_64;
        const wp::int32 var_65 = 2;
        wp::float32 var_66;
        wp::float32 var_67;
        wp::float32 var_68;
        wp::float32 var_69;
        const wp::int32 var_70 = 1;
        const wp::int32 var_71 = 0;
        wp::float32 var_72;
        const wp::int32 var_73 = 0;
        wp::float32 var_74;
        wp::float32 var_75;
        const wp::int32 var_76 = 2;
        wp::float32 var_77;
        const wp::int32 var_78 = 2;
        wp::float32 var_79;
        wp::float32 var_80;
        wp::float32 var_81;
        wp::float32 var_82;
        const wp::int32 var_83 = 2;
        const wp::int32 var_84 = 0;
        wp::float32 var_85;
        const wp::int32 var_86 = 0;
        wp::float32 var_87;
        wp::float32 var_88;
        const wp::int32 var_89 = 1;
        wp::float32 var_90;
        const wp::int32 var_91 = 1;
        wp::float32 var_92;
        wp::float32 var_93;
        wp::float32 var_94;
        wp::float32 var_95;
        const wp::int32 var_96 = 3;
        const wp::int32 var_97 = 0;
        wp::float32 var_98;
        wp::float32 var_99;
        const wp::int32 var_100 = 1;
        wp::float32 var_101;
        wp::float32 var_102;
        const wp::int32 var_103 = 4;
        const wp::int32 var_104 = 0;
        wp::float32 var_105;
        wp::float32 var_106;
        const wp::int32 var_107 = 2;
        wp::float32 var_108;
        wp::float32 var_109;
        const wp::int32 var_110 = 5;
        const wp::int32 var_111 = 1;
        wp::float32 var_112;
        wp::float32 var_113;
        const wp::int32 var_114 = 2;
        wp::float32 var_115;
        wp::float32 var_116;
        const wp::int32 var_117 = 0;
        wp::float32 var_118;
        wp::float32 var_119;
        const wp::int32 var_120 = 6;
        const wp::int32 var_121 = 1;
        wp::float32 var_122;
        wp::float32 var_123;
        const wp::int32 var_124 = 7;
        const wp::int32 var_125 = 2;
        wp::float32 var_126;
        wp::float32 var_127;
        const wp::int32 var_128 = 8;
        const wp::int32 var_129 = 9;
        //---------
        // forward
        // def _cinert(                                                                           <L 704>
        // worldid, bodyid = wp.tid()                                                             <L 716>
        builtin_tid2d(var_0, var_1);
        // mat = ximat_in[worldid, bodyid]                                                        <L 717>
        var_2 = wp::address(var_ximat_in, var_0, var_1);
        var_4 = wp::load(var_2);
        var_3 = wp::copy(var_4);
        // inert = body_inertia[worldid % body_inertia.shape[0], bodyid]                          <L 718>
        var_5 = &(var_body_inertia.shape);
        var_8 = wp::load(var_5);
        var_7 = wp::extract(var_8, var_6);
        var_9 = wp::mod(var_0, var_7);
        var_10 = wp::address(var_body_inertia, var_9, var_1);
        var_12 = wp::load(var_10);
        var_11 = wp::copy(var_12);
        // mass = body_mass[worldid % body_mass.shape[0], bodyid]                                 <L 719>
        var_13 = &(var_body_mass.shape);
        var_16 = wp::load(var_13);
        var_15 = wp::extract(var_16, var_14);
        var_17 = wp::mod(var_0, var_15);
        var_18 = wp::address(var_body_mass, var_17, var_1);
        var_20 = wp::load(var_18);
        var_19 = wp::copy(var_20);
        // dif = xipos_in[worldid, bodyid] - subtree_com_in[worldid, body_rootid[bodyid]]         <L 720>
        var_21 = wp::address(var_xipos_in, var_0, var_1);
        var_22 = wp::address(var_body_rootid, var_1);
        var_23 = wp::load(var_22);
        var_24 = wp::address(var_subtree_com_in, var_0, var_23);
        var_26 = wp::load(var_21);
        var_27 = wp::load(var_24);
        var_25 = wp::sub(var_26, var_27);
        // res = vec10()                                                                          <L 723>
        var_28 = wp::vec_t<10, wp::float32>();
        // tmp = mat @ wp.diag(inert) @ wp.transpose(mat)                                         <L 725>
        var_29 = wp::diag(var_11);
        var_30 = wp::mul(var_3, var_29);
        var_31 = wp::transpose(var_3);
        var_32 = wp::mul(var_30, var_31);
        // res[0] = tmp[0, 0]                                                                     <L 726>
        var_35 = wp::extract(var_32, var_33, var_34);
        wp::assign_inplace(var_28, var_36, var_35);
        // res[1] = tmp[1, 1]                                                                     <L 727>
        var_39 = wp::extract(var_32, var_37, var_38);
        wp::assign_inplace(var_28, var_40, var_39);
        // res[2] = tmp[2, 2]                                                                     <L 728>
        var_43 = wp::extract(var_32, var_41, var_42);
        wp::assign_inplace(var_28, var_44, var_43);
        // res[3] = tmp[0, 1]                                                                     <L 729>
        var_47 = wp::extract(var_32, var_45, var_46);
        wp::assign_inplace(var_28, var_48, var_47);
        // res[4] = tmp[0, 2]                                                                     <L 730>
        var_51 = wp::extract(var_32, var_49, var_50);
        wp::assign_inplace(var_28, var_52, var_51);
        // res[5] = tmp[1, 2]                                                                     <L 731>
        var_55 = wp::extract(var_32, var_53, var_54);
        wp::assign_inplace(var_28, var_56, var_55);
        // res[0] += mass * (dif[1] * dif[1] + dif[2] * dif[2])                                   <L 733>
        var_59 = wp::extract(var_25, var_58);
        var_61 = wp::extract(var_25, var_60);
        var_62 = wp::mul(var_59, var_61);
        var_64 = wp::extract(var_25, var_63);
        var_66 = wp::extract(var_25, var_65);
        var_67 = wp::mul(var_64, var_66);
        var_68 = wp::add(var_62, var_67);
        var_69 = wp::mul(var_19, var_68);
        wp::add_inplace(var_28, var_57, var_69);
        // res[1] += mass * (dif[0] * dif[0] + dif[2] * dif[2])                                   <L 734>
        var_72 = wp::extract(var_25, var_71);
        var_74 = wp::extract(var_25, var_73);
        var_75 = wp::mul(var_72, var_74);
        var_77 = wp::extract(var_25, var_76);
        var_79 = wp::extract(var_25, var_78);
        var_80 = wp::mul(var_77, var_79);
        var_81 = wp::add(var_75, var_80);
        var_82 = wp::mul(var_19, var_81);
        wp::add_inplace(var_28, var_70, var_82);
        // res[2] += mass * (dif[0] * dif[0] + dif[1] * dif[1])                                   <L 735>
        var_85 = wp::extract(var_25, var_84);
        var_87 = wp::extract(var_25, var_86);
        var_88 = wp::mul(var_85, var_87);
        var_90 = wp::extract(var_25, var_89);
        var_92 = wp::extract(var_25, var_91);
        var_93 = wp::mul(var_90, var_92);
        var_94 = wp::add(var_88, var_93);
        var_95 = wp::mul(var_19, var_94);
        wp::add_inplace(var_28, var_83, var_95);
        // res[3] -= mass * dif[0] * dif[1]                                                       <L 736>
        var_98 = wp::extract(var_25, var_97);
        var_99 = wp::mul(var_19, var_98);
        var_101 = wp::extract(var_25, var_100);
        var_102 = wp::mul(var_99, var_101);
        wp::sub_inplace(var_28, var_96, var_102);
        // res[4] -= mass * dif[0] * dif[2]                                                       <L 737>
        var_105 = wp::extract(var_25, var_104);
        var_106 = wp::mul(var_19, var_105);
        var_108 = wp::extract(var_25, var_107);
        var_109 = wp::mul(var_106, var_108);
        wp::sub_inplace(var_28, var_103, var_109);
        // res[5] -= mass * dif[1] * dif[2]                                                       <L 738>
        var_112 = wp::extract(var_25, var_111);
        var_113 = wp::mul(var_19, var_112);
        var_115 = wp::extract(var_25, var_114);
        var_116 = wp::mul(var_113, var_115);
        wp::sub_inplace(var_28, var_110, var_116);
        // res[6] = mass * dif[0]                                                                 <L 740>
        var_118 = wp::extract(var_25, var_117);
        var_119 = wp::mul(var_19, var_118);
        wp::assign_inplace(var_28, var_120, var_119);
        // res[7] = mass * dif[1]                                                                 <L 741>
        var_122 = wp::extract(var_25, var_121);
        var_123 = wp::mul(var_19, var_122);
        wp::assign_inplace(var_28, var_124, var_123);
        // res[8] = mass * dif[2]                                                                 <L 742>
        var_126 = wp::extract(var_25, var_125);
        var_127 = wp::mul(var_19, var_126);
        wp::assign_inplace(var_28, var_128, var_127);
        // res[9] = mass                                                                          <L 744>
        wp::assign_inplace(var_28, var_129, var_19);
        // cinert_out[worldid, bodyid] = res                                                      <L 746>
        wp::array_store(var_cinert_out, var_0, var_1, var_28);
    }
}



extern "C" __global__ void _M_13671fca_cuda_kernel_forward(
    wp::launch_bounds_t<2> dim,
    wp::array_t<wp::int32> var_dof_bodyid,
    wp::array_t<wp::int32> var_dof_parentid,
    wp::array_t<wp::float32> var_dof_armature,
    wp::array_t<wp::int32> var_M_rownnz,
    wp::array_t<wp::int32> var_M_rowadr,
    wp::array_t<wp::vec_t<6, wp::float32>> var_cdof_in,
    wp::array_t<wp::vec_t<10, wp::float32>> var_crb_in,
    wp::array_t<wp::float32> var_M_out)
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
        wp::int32* var_5;
        wp::int32* var_6;
        wp::int32 var_7;
        wp::int32 var_8;
        wp::int32 var_9;
        const wp::int32 var_10 = 1;
        wp::int32 var_11;
        wp::shape_t* var_12;
        const wp::int32 var_13 = 0;
        wp::int32 var_14;
        wp::shape_t var_15;
        wp::int32 var_16;
        wp::float32* var_17;
        wp::float32 var_18;
        wp::float32 var_19;
        wp::vec_t<10, wp::float32>* var_20;
        wp::vec_t<6, wp::float32>* var_21;
        wp::vec_t<6, wp::float32> var_22;
        wp::vec_t<10, wp::float32> var_23;
        wp::vec_t<6, wp::float32> var_24;
        const wp::int32 var_25 = 0;
        bool var_26;
        wp::vec_t<6, wp::float32>* var_27;
        wp::float32 var_28;
        wp::vec_t<6, wp::float32> var_29;
        wp::float32 var_30;
        const wp::int32 var_31 = 1;
        wp::int32 var_32;
        wp::int32* var_33;
        wp::int32 var_34;
        wp::int32 var_35;
        //---------
        // forward
        // def _M(                                                                                <L 1019>
        // worldid, dofid = wp.tid()                                                              <L 1032>
        builtin_tid2d(var_0, var_1);
        // bodyid = dof_bodyid[dofid]                                                             <L 1033>
        var_2 = wp::address(var_dof_bodyid, var_1);
        var_4 = wp::load(var_2);
        var_3 = wp::copy(var_4);
        // madr_ij = M_rowadr[dofid] + M_rownnz[dofid] - 1                                        <L 1034>
        var_5 = wp::address(var_M_rowadr, var_1);
        var_6 = wp::address(var_M_rownnz, var_1);
        var_8 = wp::load(var_5);
        var_9 = wp::load(var_6);
        var_7 = wp::add(var_8, var_9);
        var_11 = wp::sub(var_7, var_10);
        // M_out[worldid, madr_ij] = dof_armature[worldid % dof_armature.shape[0], dofid]         <L 1037>
        var_12 = &(var_dof_armature.shape);
        var_15 = wp::load(var_12);
        var_14 = wp::extract(var_15, var_13);
        var_16 = wp::mod(var_0, var_14);
        var_17 = wp::address(var_dof_armature, var_16, var_1);
        var_19 = wp::load(var_17);
        var_18 = wp::copy(var_19);
        wp::array_store(var_M_out, var_0, var_11, var_18);
        // buf = math.inert_vec(crb_in[worldid, bodyid], cdof_in[worldid, dofid])                 <L 1040>
        var_20 = wp::address(var_crb_in, var_0, var_3);
        var_21 = wp::address(var_cdof_in, var_0, var_1);
        var_23 = wp::load(var_20);
        var_24 = wp::load(var_21);
        var_22 = inert_vec_0(var_23, var_24);
        // while dofid >= 0:                                                                      <L 1043>
        start_while_0:;
        var_26 = (var_1 >= var_25);
        if ((var_26) == false) goto end_while_0;
            // M_out[worldid, madr_ij] += wp.dot(cdof_in[worldid, dofid], buf)                    <L 1044>
            var_27 = wp::address(var_cdof_in, var_0, var_1);
            var_29 = wp::load(var_27);
            var_28 = wp::dot(var_29, var_22);
            var_30 = wp::atomic_add(var_M_out, var_0, var_11, var_28);
            // madr_ij -= 1                                                                       <L 1045>
            var_32 = wp::sub(var_11, var_31);
            // dofid = dof_parentid[dofid]                                                        <L 1046>
            var_33 = wp::address(var_dof_parentid, var_1);
            var_35 = wp::load(var_33);
            var_34 = wp::copy(var_35);
            wp::assign(var_1, var_34);
            wp::assign(var_11, var_32);
        goto start_while_0;
        end_while_0:;
    }
}



extern "C" __global__ void _transmission_13b1d1f2_cuda_kernel_forward(
    wp::launch_bounds_t<2> dim,
    wp::int32 var_nv,
    wp::array_t<wp::int32> var_body_parentid,
    wp::array_t<wp::int32> var_body_rootid,
    wp::array_t<wp::int32> var_body_weldid,
    wp::array_t<wp::int32> var_body_dofnum,
    wp::array_t<wp::int32> var_body_dofadr,
    wp::array_t<wp::int32> var_jnt_type,
    wp::array_t<wp::int32> var_jnt_qposadr,
    wp::array_t<wp::int32> var_jnt_dofadr,
    wp::array_t<wp::int32> var_dof_bodyid,
    wp::array_t<wp::int32> var_dof_parentid,
    wp::array_t<wp::int32> var_site_bodyid,
    wp::array_t<wp::quat_t<wp::float32>> var_site_quat,
    wp::array_t<wp::int32> var_ten_J_rownnz,
    wp::array_t<wp::int32> var_ten_J_rowadr,
    wp::array_t<wp::int32> var_ten_J_colind,
    wp::array_t<wp::int32> var_actuator_trntype,
    wp::array_t<wp::vec_t<2, wp::int32>> var_actuator_trnid,
    wp::array_t<wp::float32> var_actuator_cranklength,
    wp::array_t<wp::vec_t<6, wp::float32>> var_actuator_gear,
    wp::array_t<wp::int32> var_body_isdofancestor,
    wp::array_t<wp::float32> var_qpos_in,
    wp::array_t<wp::quat_t<wp::float32>> var_xquat_in,
    wp::array_t<wp::vec_t<3, wp::float32>> var_site_xpos_in,
    wp::array_t<wp::mat_t<3, 3, wp::float32>> var_site_xmat_in,
    wp::array_t<wp::vec_t<3, wp::float32>> var_subtree_com_in,
    wp::array_t<wp::vec_t<6, wp::float32>> var_cdof_in,
    wp::array_t<wp::float32> var_ten_J_in,
    wp::array_t<wp::float32> var_ten_length_in,
    wp::array_t<wp::int32> var_moment_nnz,
    wp::array_t<wp::float32> var_actuator_length_out,
    wp::array_t<wp::int32> var_moment_rownnz_out,
    wp::array_t<wp::int32> var_moment_rowadr_out,
    wp::array_t<wp::int32> var_moment_colind_out,
    wp::array_t<wp::float32> var_actuator_moment_out)
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
        wp::shape_t* var_5;
        const wp::int32 var_6 = 0;
        wp::int32 var_7;
        wp::shape_t var_8;
        wp::int32 var_9;
        wp::vec_t<6, wp::float32>* var_10;
        wp::vec_t<6, wp::float32> var_11;
        wp::vec_t<6, wp::float32> var_12;
        bool var_13;
        const wp::int32 var_14 = 0;
        bool var_15;
        const wp::int32 var_16 = 1;
        bool var_17;
        wp::array_t<wp::float32> var_18;
        wp::vec_t<2, wp::int32>* var_19;
        const wp::int32 var_20 = 0;
        wp::int32 var_21;
        wp::vec_t<2, wp::int32> var_22;
        wp::int32* var_23;
        wp::int32 var_24;
        wp::int32 var_25;
        wp::int32* var_26;
        wp::int32 var_27;
        wp::int32 var_28;
        wp::int32* var_29;
        wp::int32 var_30;
        wp::int32 var_31;
        const wp::int32 var_32 = 0;
        bool var_33;
        const wp::int32 var_34 = 6;
        const wp::int32 var_35 = 6;
        wp::int32 var_36;
        const wp::int32 var_37 = 0;
        wp::int32 var_38;
        const wp::int32 var_39 = 0;
        wp::int32 var_40;
        const wp::int32 var_41 = 1;
        wp::int32 var_42;
        const wp::int32 var_43 = 1;
        wp::int32 var_44;
        const wp::int32 var_45 = 2;
        wp::int32 var_46;
        const wp::int32 var_47 = 2;
        wp::int32 var_48;
        const wp::int32 var_49 = 3;
        wp::int32 var_50;
        const wp::int32 var_51 = 3;
        wp::int32 var_52;
        const wp::int32 var_53 = 4;
        wp::int32 var_54;
        const wp::int32 var_55 = 4;
        wp::int32 var_56;
        const wp::int32 var_57 = 5;
        wp::int32 var_58;
        const wp::int32 var_59 = 5;
        wp::int32 var_60;
        const wp::float32 var_61 = 0.0;
        const wp::int32 var_62 = 1;
        bool var_63;
        const wp::int32 var_64 = 3;
        wp::int32 var_65;
        wp::float32* var_66;
        const wp::int32 var_67 = 4;
        wp::int32 var_68;
        wp::float32* var_69;
        const wp::int32 var_70 = 5;
        wp::int32 var_71;
        wp::float32* var_72;
        const wp::int32 var_73 = 6;
        wp::int32 var_74;
        wp::float32* var_75;
        wp::quat_t<wp::float32> var_76;
        wp::float32 var_77;
        wp::float32 var_78;
        wp::float32 var_79;
        wp::float32 var_80;
        wp::quat_t<wp::float32> var_81;
        wp::quat_t<wp::float32> var_82;
        wp::vec_t<3, wp::float32> var_83;
        wp::vec_t<3, wp::float32> var_84;
        const wp::int32 var_85 = 0;
        wp::float32 var_86;
        const wp::int32 var_87 = 0;
        wp::int32 var_88;
        const wp::int32 var_89 = 1;
        wp::float32 var_90;
        const wp::int32 var_91 = 1;
        wp::int32 var_92;
        const wp::int32 var_93 = 2;
        wp::float32 var_94;
        const wp::int32 var_95 = 2;
        wp::int32 var_96;
        const wp::int32 var_97 = 0;
        wp::float32 var_98;
        const wp::int32 var_99 = 3;
        wp::int32 var_100;
        const wp::int32 var_101 = 1;
        wp::float32 var_102;
        const wp::int32 var_103 = 4;
        wp::int32 var_104;
        const wp::int32 var_105 = 2;
        wp::float32 var_106;
        const wp::int32 var_107 = 5;
        wp::int32 var_108;
        const wp::int32 var_109 = 0;
        wp::float32 var_110;
        const wp::int32 var_111 = 0;
        wp::int32 var_112;
        const wp::int32 var_113 = 1;
        wp::float32 var_114;
        const wp::int32 var_115 = 1;
        wp::int32 var_116;
        const wp::int32 var_117 = 2;
        wp::float32 var_118;
        const wp::int32 var_119 = 2;
        wp::int32 var_120;
        const wp::int32 var_121 = 3;
        wp::float32 var_122;
        const wp::int32 var_123 = 3;
        wp::int32 var_124;
        const wp::int32 var_125 = 4;
        wp::float32 var_126;
        const wp::int32 var_127 = 4;
        wp::int32 var_128;
        const wp::int32 var_129 = 5;
        wp::float32 var_130;
        const wp::int32 var_131 = 5;
        wp::int32 var_132;
        const wp::int32 var_133 = 1;
        bool var_134;
        const wp::int32 var_135 = 0;
        wp::int32 var_136;
        wp::float32* var_137;
        const wp::int32 var_138 = 1;
        wp::int32 var_139;
        wp::float32* var_140;
        const wp::int32 var_141 = 2;
        wp::int32 var_142;
        wp::float32* var_143;
        const wp::int32 var_144 = 3;
        wp::int32 var_145;
        wp::float32* var_146;
        wp::quat_t<wp::float32> var_147;
        wp::float32 var_148;
        wp::float32 var_149;
        wp::float32 var_150;
        wp::float32 var_151;
        wp::quat_t<wp::float32> var_152;
        wp::vec_t<3, wp::float32> var_153;
        wp::vec_t<3, wp::float32> var_154;
        const wp::int32 var_155 = 1;
        bool var_156;
        wp::quat_t<wp::float32> var_157;
        wp::vec_t<3, wp::float32> var_158;
        wp::vec_t<3, wp::float32> var_159;
        wp::float32 var_160;
        const wp::int32 var_161 = 3;
        wp::int32 var_162;
        const wp::int32 var_163 = 0;
        wp::int32 var_164;
        wp::int32 var_165;
        wp::float32 var_166;
        const wp::int32 var_167 = 1;
        wp::int32 var_168;
        wp::int32 var_169;
        wp::float32 var_170;
        const wp::int32 var_171 = 2;
        wp::int32 var_172;
        wp::int32 var_173;
        wp::float32 var_174;
        bool var_175;
        const wp::int32 var_176 = 2;
        bool var_177;
        const wp::int32 var_178 = 3;
        bool var_179;
        wp::float32* var_180;
        const wp::int32 var_181 = 0;
        wp::float32 var_182;
        wp::float32 var_183;
        wp::float32 var_184;
        const wp::int32 var_185 = 1;
        wp::int32 var_186;
        const wp::int32 var_187 = 0;
        wp::float32 var_188;
        const wp::str var_189 = "unrecognized joint type";
        wp::int32 var_190;
        wp::int32 var_191;
        wp::int32 var_192;
        wp::quat_t<wp::float32> var_193;
        wp::vec_t<3, wp::float32> var_194;
        const wp::int32 var_195 = 2;
        bool var_196;
        wp::vec_t<2, wp::int32>* var_197;
        wp::vec_t<2, wp::int32> var_198;
        wp::vec_t<2, wp::int32> var_199;
        const wp::int32 var_200 = 0;
        wp::int32 var_201;
        const wp::int32 var_202 = 1;
        wp::int32 var_203;
        const wp::int32 var_204 = 0;
        wp::float32 var_205;
        wp::shape_t* var_206;
        const wp::int32 var_207 = 0;
        wp::int32 var_208;
        wp::shape_t var_209;
        wp::int32 var_210;
        wp::float32* var_211;
        wp::float32 var_212;
        wp::float32 var_213;
        wp::mat_t<3, 3, wp::float32>* var_214;
        wp::mat_t<3, 3, wp::float32> var_215;
        wp::mat_t<3, 3, wp::float32> var_216;
        const wp::int32 var_217 = 0;
        const wp::int32 var_218 = 2;
        wp::float32 var_219;
        const wp::int32 var_220 = 1;
        const wp::int32 var_221 = 2;
        wp::float32 var_222;
        const wp::int32 var_223 = 2;
        const wp::int32 var_224 = 2;
        wp::float32 var_225;
        wp::vec_t<3, wp::float32> var_226;
        wp::vec_t<3, wp::float32>* var_227;
        wp::vec_t<3, wp::float32> var_228;
        wp::vec_t<3, wp::float32> var_229;
        wp::vec_t<3, wp::float32>* var_230;
        wp::vec_t<3, wp::float32> var_231;
        wp::vec_t<3, wp::float32> var_232;
        wp::vec_t<3, wp::float32> var_233;
        wp::float32 var_234;
        wp::float32 var_235;
        wp::float32 var_236;
        wp::float32 var_237;
        wp::float32 var_238;
        wp::float32 var_239;
        const wp::int32 var_240 = 1;
        const wp::float32 var_241 = 0.0;
        bool var_242;
        const wp::int32 var_243 = 0;
        const wp::float32 var_244 = 0.0;
        wp::float32 var_245;
        wp::float32 var_246;
        wp::float32 var_247;
        wp::int32 var_248;
        wp::float32 var_249;
        wp::float32 var_250;
        wp::float32 var_251;
        const wp::int32 var_252 = 1;
        bool var_253;
        const wp::float32 var_254 = 1.0;
        wp::float32 var_255;
        wp::float32 var_256;
        wp::vec_t<3, wp::float32> var_257;
        wp::vec_t<3, wp::float32> var_258;
        wp::vec_t<3, wp::float32> var_259;
        wp::vec_t<3, wp::float32> var_260;
        wp::vec_t<3, wp::float32> var_261;
        wp::vec_t<3, wp::float32> var_262;
        wp::vec_t<3, wp::float32> var_263;
        wp::vec_t<3, wp::float32> var_264;
        wp::int32* var_265;
        wp::int32 var_266;
        wp::int32* var_267;
        wp::int32 var_268;
        wp::int32 var_269;
        wp::int32* var_270;
        wp::int32 var_271;
        wp::int32* var_272;
        wp::int32 var_273;
        wp::int32 var_274;
        const wp::int32 var_275 = -1;
        wp::int32 var_276;
        const wp::int32 var_277 = -1;
        wp::int32 var_278;
        const wp::int32 var_279 = 0;
        bool var_280;
        wp::int32* var_281;
        wp::int32* var_282;
        wp::int32 var_283;
        wp::int32 var_284;
        wp::int32 var_285;
        const wp::int32 var_286 = 1;
        wp::int32 var_287;
        wp::int32 var_288;
        const wp::int32 var_289 = 0;
        bool var_290;
        wp::int32* var_291;
        wp::int32* var_292;
        wp::int32 var_293;
        wp::int32 var_294;
        wp::int32 var_295;
        const wp::int32 var_296 = 1;
        wp::int32 var_297;
        wp::int32 var_298;
        wp::int32 var_299;
        wp::int32 var_300;
        const wp::int32 var_301 = 0;
        wp::int32 var_302;
        bool var_303;
        const wp::int32 var_304 = 0;
        bool var_305;
        const wp::int32 var_306 = 0;
        bool var_307;
        wp::int32 var_308;
        const wp::int32 var_309 = 1;
        wp::int32 var_310;
        bool var_311;
        wp::int32* var_312;
        wp::int32 var_313;
        wp::int32 var_314;
        wp::int32 var_315;
        bool var_316;
        wp::int32* var_317;
        wp::int32 var_318;
        wp::int32 var_319;
        wp::int32 var_320;
        wp::int32 var_321;
        wp::int32 var_322;
        wp::int32 var_323;
        const wp::int32 var_324 = 1;
        wp::int32 var_325;
        bool var_326;
        const wp::int32 var_327 = 0;
        bool var_328;
        const wp::int32 var_329 = 0;
        bool var_330;
        wp::int32 var_331;
        wp::int32* var_332;
        wp::vec_t<3, wp::float32> var_333;
        wp::vec_t<3, wp::float32> var_334;
        wp::int32 var_335;
        wp::vec_t<3, wp::float32> var_336;
        wp::vec_t<3, wp::float32> var_337;
        wp::int32* var_338;
        wp::vec_t<3, wp::float32> var_339;
        wp::vec_t<3, wp::float32> var_340;
        wp::int32 var_341;
        wp::vec_t<3, wp::float32> var_342;
        wp::float32 var_343;
        wp::float32 var_344;
        wp::float32 var_345;
        wp::int32 var_346;
        wp::float32 var_347;
        const wp::int32 var_348 = 1;
        wp::int32 var_349;
        bool var_350;
        wp::int32* var_351;
        wp::int32 var_352;
        wp::int32 var_353;
        wp::int32 var_354;
        bool var_355;
        wp::int32* var_356;
        wp::int32 var_357;
        wp::int32 var_358;
        wp::int32 var_359;
        const wp::int32 var_360 = 3;
        bool var_361;
        wp::vec_t<2, wp::int32>* var_362;
        const wp::int32 var_363 = 0;
        wp::int32 var_364;
        wp::vec_t<2, wp::int32> var_365;
        const wp::int32 var_366 = 0;
        wp::float32 var_367;
        wp::float32* var_368;
        wp::float32 var_369;
        wp::float32 var_370;
        wp::int32* var_371;
        wp::int32 var_372;
        wp::int32 var_373;
        wp::int32* var_374;
        wp::int32 var_375;
        wp::int32 var_376;
        wp::int32 var_377;
        wp::range_t var_378;
        wp::int32 var_379;
        wp::int32 var_380;
        wp::int32 var_381;
        wp::int32* var_382;
        wp::int32 var_383;
        wp::int32 var_384;
        wp::float32* var_385;
        wp::float32 var_386;
        wp::float32 var_387;
        const wp::int32 var_388 = 5;
        bool var_389;
        const wp::float32 var_390 = 0.0;
        wp::int32 var_391;
        wp::range_t var_392;
        wp::int32 var_393;
        wp::int32 var_394;
        const wp::float32 var_395 = 0.0;
        const wp::int32 var_396 = 4;
        bool var_397;
        wp::vec_t<2, wp::int32>* var_398;
        wp::vec_t<2, wp::int32> var_399;
        wp::vec_t<2, wp::int32> var_400;
        const wp::int32 var_401 = 0;
        wp::int32 var_402;
        const wp::int32 var_403 = 1;
        wp::int32 var_404;
        wp::vec_t<6, wp::float32>* var_405;
        wp::vec_t<6, wp::float32> var_406;
        wp::vec_t<6, wp::float32> var_407;
        wp::shape_t* var_408;
        const wp::int32 var_409 = 0;
        wp::int32 var_410;
        wp::shape_t var_411;
        wp::int32 var_412;
        wp::vec_t<3, wp::float32> var_413;
        wp::vec_t<3, wp::float32> var_414;
        const wp::int32 var_415 = -1;
        bool var_416;
        wp::mat_t<3, 3, wp::float32>* var_417;
        wp::mat_t<3, 3, wp::float32> var_418;
        wp::mat_t<3, 3, wp::float32> var_419;
        wp::vec_t<3, wp::float32> var_420;
        wp::vec_t<3, wp::float32> var_421;
        wp::int32* var_422;
        wp::int32 var_423;
        wp::int32* var_424;
        wp::int32 var_425;
        wp::int32 var_426;
        const wp::int32 var_427 = -1;
        wp::int32 var_428;
        const wp::int32 var_429 = 0;
        bool var_430;
        wp::int32* var_431;
        wp::int32* var_432;
        wp::int32 var_433;
        wp::int32 var_434;
        wp::int32 var_435;
        const wp::int32 var_436 = 1;
        wp::int32 var_437;
        wp::int32 var_438;
        wp::int32 var_439;
        const wp::int32 var_440 = 0;
        wp::int32 var_441;
        const wp::int32 var_442 = 0;
        bool var_443;
        const wp::int32 var_444 = 1;
        wp::int32 var_445;
        wp::int32* var_446;
        wp::int32 var_447;
        wp::int32 var_448;
        wp::int32 var_449;
        const wp::float32 var_450 = 0.0;
        wp::int32 var_451;
        const wp::int32 var_452 = 1;
        wp::int32 var_453;
        const wp::int32 var_454 = 0;
        bool var_455;
        wp::vec_t<3, wp::float32>* var_456;
        wp::int32* var_457;
        wp::vec_t<3, wp::float32> var_458;
        wp::vec_t<3, wp::float32> var_459;
        wp::vec_t<3, wp::float32> var_460;
        wp::int32 var_461;
        wp::float32 var_462;
        wp::float32 var_463;
        wp::float32 var_464;
        wp::int32 var_465;
        const wp::int32 var_466 = 1;
        wp::int32 var_467;
        wp::int32* var_468;
        wp::int32 var_469;
        wp::int32 var_470;
        wp::int32* var_471;
        wp::int32 var_472;
        wp::int32 var_473;
        wp::int32* var_474;
        wp::int32 var_475;
        wp::int32 var_476;
        wp::int32* var_477;
        wp::int32 var_478;
        wp::int32 var_479;
        wp::int32* var_480;
        wp::int32 var_481;
        wp::int32 var_482;
        wp::int32* var_483;
        wp::int32* var_484;
        wp::int32 var_485;
        wp::int32 var_486;
        wp::int32 var_487;
        const wp::int32 var_488 = 1;
        wp::int32 var_489;
        wp::int32* var_490;
        wp::int32* var_491;
        wp::int32 var_492;
        wp::int32 var_493;
        wp::int32 var_494;
        const wp::int32 var_495 = 1;
        wp::int32 var_496;
        const wp::int32 var_497 = -1;
        bool var_498;
        const wp::int32 var_499 = 0;
        bool var_500;
        const wp::int32 var_501 = 0;
        bool var_502;
        bool var_503;
        bool var_504;
        wp::int32* var_505;
        wp::int32 var_506;
        wp::int32 var_507;
        wp::int32* var_508;
        wp::int32 var_509;
        wp::int32 var_510;
        wp::int32 var_511;
        wp::int32 var_512;
        bool var_513;
        const wp::int32 var_514 = -1;
        bool var_515;
        const wp::int32 var_516 = -1;
        bool var_517;
        wp::int32 var_518;
        wp::int32 var_519;
        bool var_520;
        wp::int32 var_521;
        wp::int32 var_522;
        wp::int32 var_523;
        bool var_524;
        const wp::int32 var_525 = 0;
        wp::float32 var_526;
        const wp::float32 var_527 = 0.0;
        bool var_528;
        const wp::int32 var_529 = 1;
        wp::float32 var_530;
        const wp::float32 var_531 = 0.0;
        bool var_532;
        const wp::int32 var_533 = 2;
        wp::float32 var_534;
        const wp::float32 var_535 = 0.0;
        bool var_536;
        bool var_537;
        bool var_538;
        const wp::int32 var_539 = 3;
        wp::float32 var_540;
        const wp::float32 var_541 = 0.0;
        bool var_542;
        const wp::int32 var_543 = 4;
        wp::float32 var_544;
        const wp::float32 var_545 = 0.0;
        bool var_546;
        const wp::int32 var_547 = 5;
        wp::float32 var_548;
        const wp::float32 var_549 = 0.0;
        bool var_550;
        bool var_551;
        wp::vec_t<3, wp::float32>* var_552;
        wp::vec_t<3, wp::float32> var_553;
        wp::vec_t<3, wp::float32> var_554;
        wp::vec_t<3, wp::float32>* var_555;
        wp::vec_t<3, wp::float32> var_556;
        wp::vec_t<3, wp::float32> var_557;
        wp::mat_t<3, 3, wp::float32>* var_558;
        wp::mat_t<3, 3, wp::float32> var_559;
        wp::mat_t<3, 3, wp::float32> var_560;
        const wp::float32 var_561 = 0.0;
        wp::float32 var_562;
        wp::mat_t<3, 3, wp::float32> var_563;
        wp::vec_t<3, wp::float32> var_564;
        wp::vec_t<3, wp::float32> var_565;
        wp::float32 var_566;
        wp::float32 var_567;
        wp::vec_t<3, wp::float32> var_568;
        wp::float32 var_569;
        wp::quat_t<wp::float32>* var_570;
        wp::quat_t<wp::float32>* var_571;
        wp::quat_t<wp::float32> var_572;
        wp::quat_t<wp::float32> var_573;
        wp::quat_t<wp::float32> var_574;
        wp::quat_t<wp::float32>* var_575;
        wp::quat_t<wp::float32>* var_576;
        wp::quat_t<wp::float32> var_577;
        wp::quat_t<wp::float32> var_578;
        wp::quat_t<wp::float32> var_579;
        wp::vec_t<3, wp::float32> var_580;
        wp::float32 var_581;
        wp::float32 var_582;
        wp::vec_t<3, wp::float32> var_583;
        wp::float32 var_584;
        wp::vec_t<3, wp::float32> var_585;
        const wp::int32 var_586 = -1;
        wp::int32 var_587;
        const wp::int32 var_588 = -1;
        wp::int32 var_589;
        const wp::int32 var_590 = 0;
        bool var_591;
        wp::int32* var_592;
        wp::int32* var_593;
        wp::int32 var_594;
        wp::int32 var_595;
        wp::int32 var_596;
        const wp::int32 var_597 = 1;
        wp::int32 var_598;
        wp::int32 var_599;
        const wp::int32 var_600 = 0;
        bool var_601;
        wp::int32* var_602;
        wp::int32* var_603;
        wp::int32 var_604;
        wp::int32 var_605;
        wp::int32 var_606;
        const wp::int32 var_607 = 1;
        wp::int32 var_608;
        wp::int32 var_609;
        wp::int32 var_610;
        wp::int32 var_611;
        const wp::int32 var_612 = 0;
        wp::int32 var_613;
        bool var_614;
        const wp::int32 var_615 = 0;
        bool var_616;
        const wp::int32 var_617 = 0;
        bool var_618;
        wp::int32 var_619;
        bool var_620;
        bool var_621;
        bool var_622;
        const wp::int32 var_623 = 1;
        wp::int32 var_624;
        bool var_625;
        wp::int32* var_626;
        wp::int32 var_627;
        wp::int32 var_628;
        wp::int32 var_629;
        bool var_630;
        wp::int32* var_631;
        wp::int32 var_632;
        wp::int32 var_633;
        wp::int32 var_634;
        wp::int32 var_635;
        wp::int32 var_636;
        wp::int32 var_637;
        const wp::int32 var_638 = 1;
        wp::int32 var_639;
        bool var_640;
        const wp::int32 var_641 = 0;
        bool var_642;
        const wp::int32 var_643 = 0;
        bool var_644;
        wp::int32 var_645;
        bool var_646;
        bool var_647;
        bool var_648;
        wp::int32 var_649;
        wp::int32* var_650;
        wp::vec_t<3, wp::float32> var_651;
        wp::vec_t<3, wp::float32> var_652;
        wp::int32 var_653;
        wp::int32* var_654;
        wp::vec_t<3, wp::float32> var_655;
        wp::vec_t<3, wp::float32> var_656;
        wp::int32 var_657;
        const wp::float32 var_658 = 0.0;
        wp::float32 var_659;
        wp::vec_t<3, wp::float32> var_660;
        wp::float32 var_661;
        wp::float32 var_662;
        wp::float32 var_663;
        wp::vec_t<3, wp::float32> var_664;
        wp::float32 var_665;
        wp::float32 var_666;
        wp::float32 var_667;
        wp::int32 var_668;
        const wp::int32 var_669 = 1;
        wp::int32 var_670;
        bool var_671;
        wp::int32* var_672;
        wp::int32 var_673;
        wp::int32 var_674;
        wp::int32 var_675;
        bool var_676;
        wp::int32* var_677;
        wp::int32 var_678;
        wp::int32 var_679;
        wp::int32 var_680;
        wp::vec_t<3, wp::float32> var_681;
        wp::vec_t<3, wp::float32> var_682;
        wp::int32 var_683;
        wp::int32 var_684;
        wp::int32 var_685;
        wp::int32 var_686;
        wp::int32 var_687;
        wp::vec_t<3, wp::float32> var_688;
        wp::vec_t<3, wp::float32> var_689;
        wp::float32 var_690;
        wp::int32 var_691;
        const wp::str var_692 = "unhandled transmission type %d\n";
        wp::vec_t<6, wp::float32> var_693;
        wp::vec_t<6, wp::float32> var_694;
        wp::int32 var_695;
        wp::int32 var_696;
        wp::vec_t<6, wp::float32> var_697;
        wp::vec_t<6, wp::float32> var_698;
        wp::vec_t<2, wp::int32> var_699;
        wp::float32 var_700;
        wp::mat_t<3, 3, wp::float32> var_701;
        wp::vec_t<3, wp::float32> var_702;
        wp::float32 var_703;
        wp::int32 var_704;
        wp::int32 var_705;
        wp::int32 var_706;
        wp::int32 var_707;
        wp::int32 var_708;
        wp::int32 var_709;
        wp::int32 var_710;
        wp::int32 var_711;
        wp::int32 var_712;
        wp::vec_t<3, wp::float32> var_713;
        wp::vec_t<3, wp::float32> var_714;
        wp::float32 var_715;
        wp::int32 var_716;
        wp::vec_t<6, wp::float32> var_717;
        wp::int32 var_718;
        wp::quat_t<wp::float32> var_719;
        wp::int32 var_720;
        wp::int32 var_721;
        //---------
        // forward
        // def _transmission(                                                                     <L 2618>
        // worldid, actid = wp.tid()                                                              <L 2659>
        builtin_tid2d(var_0, var_1);
        // trntype = actuator_trntype[actid]                                                      <L 2660>
        var_2 = wp::address(var_actuator_trntype, var_1);
        var_4 = wp::load(var_2);
        var_3 = wp::copy(var_4);
        // actuator_gear_id = worldid % actuator_gear.shape[0]                                    <L 2661>
        var_5 = &(var_actuator_gear.shape);
        var_8 = wp::load(var_5);
        var_7 = wp::extract(var_8, var_6);
        var_9 = wp::mod(var_0, var_7);
        // gear = actuator_gear[actuator_gear_id, actid]                                          <L 2662>
        var_10 = wp::address(var_actuator_gear, var_9, var_1);
        var_12 = wp::load(var_10);
        var_11 = wp::copy(var_12);
        // if trntype == TrnType.JOINT or trntype == TrnType.JOINTINPARENT:                       <L 2663>
        var_15 = (var_3 == var_14);
        var_13 = var_15;
        if (!var_13) {
            var_17 = (var_3 == var_16);
            var_13 = var_13 || var_17;
        }
        if (var_13) {
            // qpos = qpos_in[worldid]                                                            <L 2664>
            var_18 = wp::view(var_qpos_in, var_0);
            // jntid = actuator_trnid[actid][0]                                                   <L 2665>
            var_19 = wp::address(var_actuator_trnid, var_1);
            var_22 = wp::load(var_19);
            var_21 = wp::extract(var_22, var_20);
            // jnt_typ = jnt_type[jntid]                                                          <L 2666>
            var_23 = wp::address(var_jnt_type, var_21);
            var_25 = wp::load(var_23);
            var_24 = wp::copy(var_25);
            // qadr = jnt_qposadr[jntid]                                                          <L 2667>
            var_26 = wp::address(var_jnt_qposadr, var_21);
            var_28 = wp::load(var_26);
            var_27 = wp::copy(var_28);
            // vadr = jnt_dofadr[jntid]                                                           <L 2668>
            var_29 = wp::address(var_jnt_dofadr, var_21);
            var_31 = wp::load(var_29);
            var_30 = wp::copy(var_31);
            // if jnt_typ == JointType.FREE:                                                      <L 2669>
            var_33 = (var_24 == var_32);
            if (var_33) {
                // moment_rownnz_out[worldid, actid] = 6                                          <L 2670>
                wp::array_store(var_moment_rownnz_out, var_0, var_1, var_34);
                // rowadr = wp.atomic_add(moment_nnz, worldid, 6)                                 <L 2671>
                var_36 = wp::atomic_add(var_moment_nnz, var_0, var_35);
                // moment_rowadr_out[worldid, actid] = rowadr                                     <L 2672>
                wp::array_store(var_moment_rowadr_out, var_0, var_1, var_36);
                // moment_colind_out[worldid, rowadr + 0] = vadr + 0                              <L 2673>
                var_38 = wp::add(var_30, var_37);
                var_40 = wp::add(var_36, var_39);
                wp::array_store(var_moment_colind_out, var_0, var_40, var_38);
                // moment_colind_out[worldid, rowadr + 1] = vadr + 1                              <L 2674>
                var_42 = wp::add(var_30, var_41);
                var_44 = wp::add(var_36, var_43);
                wp::array_store(var_moment_colind_out, var_0, var_44, var_42);
                // moment_colind_out[worldid, rowadr + 2] = vadr + 2                              <L 2675>
                var_46 = wp::add(var_30, var_45);
                var_48 = wp::add(var_36, var_47);
                wp::array_store(var_moment_colind_out, var_0, var_48, var_46);
                // moment_colind_out[worldid, rowadr + 3] = vadr + 3                              <L 2676>
                var_50 = wp::add(var_30, var_49);
                var_52 = wp::add(var_36, var_51);
                wp::array_store(var_moment_colind_out, var_0, var_52, var_50);
                // moment_colind_out[worldid, rowadr + 4] = vadr + 4                              <L 2677>
                var_54 = wp::add(var_30, var_53);
                var_56 = wp::add(var_36, var_55);
                wp::array_store(var_moment_colind_out, var_0, var_56, var_54);
                // moment_colind_out[worldid, rowadr + 5] = vadr + 5                              <L 2678>
                var_58 = wp::add(var_30, var_57);
                var_60 = wp::add(var_36, var_59);
                wp::array_store(var_moment_colind_out, var_0, var_60, var_58);
                // actuator_length_out[worldid, actid] = 0.0                                      <L 2679>
                wp::array_store(var_actuator_length_out, var_0, var_1, var_61);
                // if trntype == TrnType.JOINTINPARENT:                                           <L 2680>
                var_63 = (var_3 == var_62);
                if (var_63) {
                    // quat = wp.normalize(wp.quat(qpos[qadr + 3], qpos[qadr + 4], qpos[qadr + 5], qpos[qadr + 6]))       <L 2681>
                    var_65 = wp::add(var_27, var_64);
                    var_66 = wp::address(var_18, var_65);
                    var_68 = wp::add(var_27, var_67);
                    var_69 = wp::address(var_18, var_68);
                    var_71 = wp::add(var_27, var_70);
                    var_72 = wp::address(var_18, var_71);
                    var_74 = wp::add(var_27, var_73);
                    var_75 = wp::address(var_18, var_74);
                    var_77 = wp::load(var_66);
                    var_78 = wp::load(var_69);
                    var_79 = wp::load(var_72);
                    var_80 = wp::load(var_75);
                    var_76 = wp::quat_t<wp::float32>(var_77, var_78, var_79, var_80);
                    var_81 = wp::normalize(var_76);
                    // quat_neg = math.quat_inv(quat)                                             <L 2682>
                    var_82 = quat_inv_0(var_81);
                    // gearaxis = math.rot_vec_quat(wp.spatial_bottom(gear), quat_neg)            <L 2683>
                    var_83 = wp::spatial_bottom(var_11);
                    var_84 = rot_vec_quat_0(var_83, var_82);
                    // actuator_moment_out[worldid, rowadr + 0] = gear[0]                         <L 2684>
                    var_86 = wp::extract(var_11, var_85);
                    var_88 = wp::add(var_36, var_87);
                    wp::array_store(var_actuator_moment_out, var_0, var_88, var_86);
                    // actuator_moment_out[worldid, rowadr + 1] = gear[1]                         <L 2685>
                    var_90 = wp::extract(var_11, var_89);
                    var_92 = wp::add(var_36, var_91);
                    wp::array_store(var_actuator_moment_out, var_0, var_92, var_90);
                    // actuator_moment_out[worldid, rowadr + 2] = gear[2]                         <L 2686>
                    var_94 = wp::extract(var_11, var_93);
                    var_96 = wp::add(var_36, var_95);
                    wp::array_store(var_actuator_moment_out, var_0, var_96, var_94);
                    // actuator_moment_out[worldid, rowadr + 3] = gearaxis[0]                     <L 2687>
                    var_98 = wp::extract(var_84, var_97);
                    var_100 = wp::add(var_36, var_99);
                    wp::array_store(var_actuator_moment_out, var_0, var_100, var_98);
                    // actuator_moment_out[worldid, rowadr + 4] = gearaxis[1]                     <L 2688>
                    var_102 = wp::extract(var_84, var_101);
                    var_104 = wp::add(var_36, var_103);
                    wp::array_store(var_actuator_moment_out, var_0, var_104, var_102);
                    // actuator_moment_out[worldid, rowadr + 5] = gearaxis[2]                     <L 2689>
                    var_106 = wp::extract(var_84, var_105);
                    var_108 = wp::add(var_36, var_107);
                    wp::array_store(var_actuator_moment_out, var_0, var_108, var_106);
                }
                if (!var_63) {
                    // actuator_moment_out[worldid, rowadr + 0] = gear[0]                         <L 2691>
                    var_110 = wp::extract(var_11, var_109);
                    var_112 = wp::add(var_36, var_111);
                    wp::array_store(var_actuator_moment_out, var_0, var_112, var_110);
                    // actuator_moment_out[worldid, rowadr + 1] = gear[1]                         <L 2692>
                    var_114 = wp::extract(var_11, var_113);
                    var_116 = wp::add(var_36, var_115);
                    wp::array_store(var_actuator_moment_out, var_0, var_116, var_114);
                    // actuator_moment_out[worldid, rowadr + 2] = gear[2]                         <L 2693>
                    var_118 = wp::extract(var_11, var_117);
                    var_120 = wp::add(var_36, var_119);
                    wp::array_store(var_actuator_moment_out, var_0, var_120, var_118);
                    // actuator_moment_out[worldid, rowadr + 3] = gear[3]                         <L 2694>
                    var_122 = wp::extract(var_11, var_121);
                    var_124 = wp::add(var_36, var_123);
                    wp::array_store(var_actuator_moment_out, var_0, var_124, var_122);
                    // actuator_moment_out[worldid, rowadr + 4] = gear[4]                         <L 2695>
                    var_126 = wp::extract(var_11, var_125);
                    var_128 = wp::add(var_36, var_127);
                    wp::array_store(var_actuator_moment_out, var_0, var_128, var_126);
                    // actuator_moment_out[worldid, rowadr + 5] = gear[5]                         <L 2696>
                    var_130 = wp::extract(var_11, var_129);
                    var_132 = wp::add(var_36, var_131);
                    wp::array_store(var_actuator_moment_out, var_0, var_132, var_130);
                }
            }
            if (!var_33) {
                // elif jnt_typ == JointType.BALL:                                                <L 2697>
                var_134 = (var_24 == var_133);
                if (var_134) {
                    // q = wp.quat(qpos[qadr + 0], qpos[qadr + 1], qpos[qadr + 2], qpos[qadr + 3])       <L 2698>
                    var_136 = wp::add(var_27, var_135);
                    var_137 = wp::address(var_18, var_136);
                    var_139 = wp::add(var_27, var_138);
                    var_140 = wp::address(var_18, var_139);
                    var_142 = wp::add(var_27, var_141);
                    var_143 = wp::address(var_18, var_142);
                    var_145 = wp::add(var_27, var_144);
                    var_146 = wp::address(var_18, var_145);
                    var_148 = wp::load(var_137);
                    var_149 = wp::load(var_140);
                    var_150 = wp::load(var_143);
                    var_151 = wp::load(var_146);
                    var_147 = wp::quat_t<wp::float32>(var_148, var_149, var_150, var_151);
                    // q = wp.normalize(q)                                                        <L 2699>
                    var_152 = wp::normalize(var_147);
                    // axis_angle = math.quat_to_vel(q)                                           <L 2700>
                    var_153 = quat_to_vel_0(var_152);
                    // gearaxis = wp.spatial_top(gear)  # [:3]                                    <L 2701>
                    var_154 = wp::spatial_top(var_11);
                    // if trntype == TrnType.JOINTINPARENT:                                       <L 2702>
                    var_156 = (var_3 == var_155);
                    if (var_156) {
                        // quat_neg = math.quat_inv(q)                                            <L 2703>
                        var_157 = quat_inv_0(var_152);
                        // gearaxis = math.rot_vec_quat(gearaxis, quat_neg)                       <L 2704>
                        var_158 = rot_vec_quat_0(var_154, var_157);
                    }
                    var_159 = wp::where(var_156, var_158, var_154);
                    // actuator_length_out[worldid, actid] = wp.dot(axis_angle, gearaxis)         <L 2705>
                    var_160 = wp::dot(var_153, var_159);
                    wp::array_store(var_actuator_length_out, var_0, var_1, var_160);
                    // nnz = 3                                                                    <L 2707>
                    // moment_rownnz_out[worldid, actid] = nnz                                    <L 2708>
                    wp::array_store(var_moment_rownnz_out, var_0, var_1, var_161);
                    // rowadr = wp.atomic_add(moment_nnz, worldid, nnz)                           <L 2709>
                    var_162 = wp::atomic_add(var_moment_nnz, var_0, var_161);
                    // moment_rowadr_out[worldid, actid] = rowadr                                 <L 2710>
                    wp::array_store(var_moment_rowadr_out, var_0, var_1, var_162);
                    // for i in range(3):                                                         <L 2712>
                    // sparseid = rowadr + i                                                      <L 2713>
                    var_164 = wp::add(var_162, var_163);
                    // moment_colind_out[worldid, sparseid] = vadr + i                            <L 2714>
                    var_165 = wp::add(var_30, var_163);
                    wp::array_store(var_moment_colind_out, var_0, var_164, var_165);
                    // actuator_moment_out[worldid, sparseid] = gearaxis[i]                       <L 2715>
                    var_166 = wp::extract(var_159, var_163);
                    wp::array_store(var_actuator_moment_out, var_0, var_164, var_166);
                    // sparseid = rowadr + i                                                      <L 2713>
                    var_168 = wp::add(var_162, var_167);
                    // moment_colind_out[worldid, sparseid] = vadr + i                            <L 2714>
                    var_169 = wp::add(var_30, var_167);
                    wp::array_store(var_moment_colind_out, var_0, var_168, var_169);
                    // actuator_moment_out[worldid, sparseid] = gearaxis[i]                       <L 2715>
                    var_170 = wp::extract(var_159, var_167);
                    wp::array_store(var_actuator_moment_out, var_0, var_168, var_170);
                    // sparseid = rowadr + i                                                      <L 2713>
                    var_172 = wp::add(var_162, var_171);
                    // moment_colind_out[worldid, sparseid] = vadr + i                            <L 2714>
                    var_173 = wp::add(var_30, var_171);
                    wp::array_store(var_moment_colind_out, var_0, var_172, var_173);
                    // actuator_moment_out[worldid, sparseid] = gearaxis[i]                       <L 2715>
                    var_174 = wp::extract(var_159, var_171);
                    wp::array_store(var_actuator_moment_out, var_0, var_172, var_174);
                }
                if (!var_134) {
                    // elif jnt_typ == JointType.SLIDE or jnt_typ == JointType.HINGE:             <L 2716>
                    var_177 = (var_24 == var_176);
                    var_175 = var_177;
                    if (!var_175) {
                        var_179 = (var_24 == var_178);
                        var_175 = var_175 || var_179;
                    }
                    if (var_175) {
                        // actuator_length_out[worldid, actid] = qpos[qadr] * gear[0]             <L 2717>
                        var_180 = wp::address(var_18, var_27);
                        var_182 = wp::extract(var_11, var_181);
                        var_184 = wp::load(var_180);
                        var_183 = wp::mul(var_184, var_182);
                        wp::array_store(var_actuator_length_out, var_0, var_1, var_183);
                        // nnz = 1                                                                <L 2719>
                        // moment_rownnz_out[worldid, actid] = nnz                                <L 2720>
                        wp::array_store(var_moment_rownnz_out, var_0, var_1, var_185);
                        // rowadr = wp.atomic_add(moment_nnz, worldid, nnz)                       <L 2721>
                        var_186 = wp::atomic_add(var_moment_nnz, var_0, var_185);
                        // moment_rowadr_out[worldid, actid] = rowadr                             <L 2722>
                        wp::array_store(var_moment_rowadr_out, var_0, var_1, var_186);
                        // moment_colind_out[worldid, rowadr] = vadr                              <L 2723>
                        wp::array_store(var_moment_colind_out, var_0, var_186, var_30);
                        // actuator_moment_out[worldid, rowadr] = gear[0]                         <L 2724>
                        var_188 = wp::extract(var_11, var_187);
                        wp::array_store(var_actuator_moment_out, var_0, var_186, var_188);
                    }
                    if (!var_175) {
                        // wp.printf("unrecognized joint type")                                   <L 2726>
                        printf(var_189);
                    }
                }
                var_190 = wp::where(var_134, var_161, var_185);
                var_191 = wp::where(var_134, var_162, var_186);
            }
            var_192 = wp::where(var_33, var_36, var_191);
            var_193 = wp::where(var_33, var_82, var_157);
            var_194 = wp::where(var_33, var_84, var_159);
        }
        if (!var_13) {
            // elif trntype == TrnType.SLIDERCRANK:                                               <L 2727>
            var_196 = (var_3 == var_195);
            if (var_196) {
                // trnid = actuator_trnid[actid]                                                  <L 2729>
                var_197 = wp::address(var_actuator_trnid, var_1);
                var_199 = wp::load(var_197);
                var_198 = wp::copy(var_199);
                // id = trnid[0]                                                                  <L 2730>
                var_201 = wp::extract(var_198, var_200);
                // idslider = trnid[1]                                                            <L 2731>
                var_203 = wp::extract(var_198, var_202);
                // gear0 = gear[0]                                                                <L 2732>
                var_205 = wp::extract(var_11, var_204);
                // rod = actuator_cranklength[worldid % actuator_cranklength.shape[0], actid]       <L 2733>
                var_206 = &(var_actuator_cranklength.shape);
                var_209 = wp::load(var_206);
                var_208 = wp::extract(var_209, var_207);
                var_210 = wp::mod(var_0, var_208);
                var_211 = wp::address(var_actuator_cranklength, var_210, var_1);
                var_213 = wp::load(var_211);
                var_212 = wp::copy(var_213);
                // site_xmat = site_xmat_in[worldid, idslider]                                    <L 2734>
                var_214 = wp::address(var_site_xmat_in, var_0, var_203);
                var_216 = wp::load(var_214);
                var_215 = wp::copy(var_216);
                // axis = wp.vec3(site_xmat[0, 2], site_xmat[1, 2], site_xmat[2, 2])              <L 2735>
                var_219 = wp::extract(var_215, var_217, var_218);
                var_222 = wp::extract(var_215, var_220, var_221);
                var_225 = wp::extract(var_215, var_223, var_224);
                var_226 = wp::vec_t<3, wp::float32>(var_219, var_222, var_225);
                // site_xpos_id = site_xpos_in[worldid, id]                                       <L 2736>
                var_227 = wp::address(var_site_xpos_in, var_0, var_201);
                var_229 = wp::load(var_227);
                var_228 = wp::copy(var_229);
                // site_xpos_idslider = site_xpos_in[worldid, idslider]                           <L 2737>
                var_230 = wp::address(var_site_xpos_in, var_0, var_203);
                var_232 = wp::load(var_230);
                var_231 = wp::copy(var_232);
                // vec = site_xpos_id - site_xpos_idslider                                        <L 2738>
                var_233 = wp::sub(var_228, var_231);
                // av = wp.dot(vec, axis)                                                         <L 2742>
                var_234 = wp::dot(var_233, var_226);
                // det = av * av + rod * rod - wp.dot(vec, vec)                                   <L 2743>
                var_235 = wp::mul(var_234, var_234);
                var_236 = wp::mul(var_212, var_212);
                var_237 = wp::add(var_235, var_236);
                var_238 = wp::dot(var_233, var_233);
                var_239 = wp::sub(var_237, var_238);
                // ok = 1                                                                         <L 2744>
                // if det <= 0.0:                                                                 <L 2745>
                var_242 = (var_239 <= var_241);
                if (var_242) {
                    // ok = 0                                                                     <L 2746>
                    // sdet = 0.0                                                                 <L 2747>
                    // length = av                                                                <L 2748>
                    var_245 = wp::copy(var_234);
                }
                if (!var_242) {
                    // sdet = wp.sqrt(det)                                                        <L 2750>
                    var_246 = wp::sqrt(var_239);
                    // length = av - sdet                                                         <L 2751>
                    var_247 = wp::sub(var_234, var_246);
                }
                var_248 = wp::where(var_242, var_243, var_240);
                var_249 = wp::where(var_242, var_244, var_246);
                var_250 = wp::where(var_242, var_245, var_247);
                // actuator_length_out[worldid, actid] = length * gear0                           <L 2753>
                var_251 = wp::mul(var_250, var_205);
                wp::array_store(var_actuator_length_out, var_0, var_1, var_251);
                // if ok == 1:                                                                    <L 2756>
                var_253 = (var_248 == var_252);
                if (var_253) {
                    // scale = 1.0 - math.safe_div(av, sdet)                                      <L 2757>
                    var_255 = safe_div_0(var_234, var_249);
                    var_256 = wp::sub(var_254, var_255);
                    // dldv = axis * scale + math.safe_div(vec, sdet)                             <L 2758>
                    var_257 = wp::mul(var_226, var_256);
                    var_258 = safe_div_0(var_233, var_249);
                    var_259 = wp::add(var_257, var_258);
                    // dlda = vec * scale                                                         <L 2759>
                    var_260 = wp::mul(var_233, var_256);
                }
                if (!var_253) {
                    // dldv = axis                                                                <L 2761>
                    var_261 = wp::copy(var_226);
                    // dlda = vec                                                                 <L 2762>
                    var_262 = wp::copy(var_233);
                }
                var_263 = wp::where(var_253, var_259, var_261);
                var_264 = wp::where(var_253, var_260, var_262);
                // b1 = body_weldid[site_bodyid[id]]                                              <L 2765>
                var_265 = wp::address(var_site_bodyid, var_201);
                var_266 = wp::load(var_265);
                var_267 = wp::address(var_body_weldid, var_266);
                var_269 = wp::load(var_267);
                var_268 = wp::copy(var_269);
                // b2 = body_weldid[site_bodyid[idslider]]                                        <L 2766>
                var_270 = wp::address(var_site_bodyid, var_203);
                var_271 = wp::load(var_270);
                var_272 = wp::address(var_body_weldid, var_271);
                var_274 = wp::load(var_272);
                var_273 = wp::copy(var_274);
                // da1_init = int(-1)                                                             <L 2767>
                var_276 = wp::int(var_275);
                // da2_init = int(-1)                                                             <L 2768>
                var_278 = wp::int(var_277);
                // if b1 > 0:                                                                     <L 2769>
                var_280 = (var_268 > var_279);
                if (var_280) {
                    // da1_init = body_dofadr[b1] + body_dofnum[b1] - 1                           <L 2770>
                    var_281 = wp::address(var_body_dofadr, var_268);
                    var_282 = wp::address(var_body_dofnum, var_268);
                    var_284 = wp::load(var_281);
                    var_285 = wp::load(var_282);
                    var_283 = wp::add(var_284, var_285);
                    var_287 = wp::sub(var_283, var_286);
                }
                var_288 = wp::where(var_280, var_287, var_276);
                // if b2 > 0:                                                                     <L 2771>
                var_290 = (var_273 > var_289);
                if (var_290) {
                    // da2_init = body_dofadr[b2] + body_dofnum[b2] - 1                           <L 2772>
                    var_291 = wp::address(var_body_dofadr, var_273);
                    var_292 = wp::address(var_body_dofnum, var_273);
                    var_294 = wp::load(var_291);
                    var_295 = wp::load(var_292);
                    var_293 = wp::add(var_294, var_295);
                    var_297 = wp::sub(var_293, var_296);
                }
                var_298 = wp::where(var_290, var_297, var_278);
                // da1 = da1_init                                                                 <L 2774>
                var_299 = wp::copy(var_288);
                // da2 = da2_init                                                                 <L 2775>
                var_300 = wp::copy(var_298);
                // ndof = int(0)                                                                  <L 2776>
                var_302 = wp::int(var_301);
                // while da1 >= 0 or da2 >= 0:                                                    <L 2777>
        start_while_0:;
                var_305 = (var_299 >= var_304);
                var_303 = var_305;
                if (!var_303) {
                    var_307 = (var_300 >= var_306);
                    var_303 = var_303 || var_307;
                }
        if ((var_303) == false) goto end_while_0;
                    // da = wp.max(da1, da2)                                                      <L 2778>
                    var_308 = wp::max(var_299, var_300);
                    // ndof += 1                                                                  <L 2779>
                    var_310 = wp::add(var_302, var_309);
                    // if da1 == da:                                                              <L 2780>
                    var_311 = (var_299 == var_308);
                    if (var_311) {
                        // da1 = dof_parentid[da1]                                                <L 2781>
                        var_312 = wp::address(var_dof_parentid, var_299);
                        var_314 = wp::load(var_312);
                        var_313 = wp::copy(var_314);
                    }
                    var_315 = wp::where(var_311, var_313, var_299);
                    // if da2 == da:                                                              <L 2782>
                    var_316 = (var_300 == var_308);
                    if (var_316) {
                        // da2 = dof_parentid[da2]                                                <L 2783>
                        var_317 = wp::address(var_dof_parentid, var_300);
                        var_319 = wp::load(var_317);
                        var_318 = wp::copy(var_319);
                    }
                    var_320 = wp::where(var_316, var_318, var_300);
                    wp::assign(var_299, var_315);
                    wp::assign(var_300, var_320);
                    wp::assign(var_302, var_310);
        goto start_while_0;
        end_while_0:;
                // moment_rownnz_out[worldid, actid] = ndof                                       <L 2785>
                wp::array_store(var_moment_rownnz_out, var_0, var_1, var_302);
                // rowadr = wp.atomic_add(moment_nnz, worldid, ndof)                              <L 2786>
                var_321 = wp::atomic_add(var_moment_nnz, var_0, var_302);
                // moment_rowadr_out[worldid, actid] = rowadr                                     <L 2787>
                wp::array_store(var_moment_rowadr_out, var_0, var_1, var_321);
                // da1 = da1_init                                                                 <L 2790>
                var_322 = wp::copy(var_288);
                // da2 = da2_init                                                                 <L 2791>
                var_323 = wp::copy(var_298);
                // ptr = ndof - 1                                                                 <L 2793>
                var_325 = wp::sub(var_302, var_324);
                // while da1 >= 0 or da2 >= 0:                                                    <L 2794>
        start_while_2:;
                var_328 = (var_322 >= var_327);
                var_326 = var_328;
                if (!var_326) {
                    var_330 = (var_323 >= var_329);
                    var_326 = var_326 || var_330;
                }
        if ((var_326) == false) goto end_while_2;
                    // da = wp.max(da1, da2)                                                      <L 2795>
                    var_331 = wp::max(var_322, var_323);
                    // jacp, jacr = support.jac_dof(                                              <L 2798>
                    // body_parentid,                                                             <L 2799>
                    // body_rootid,                                                               <L 2800>
                    // dof_bodyid,                                                                <L 2801>
                    // body_isdofancestor,                                                        <L 2802>
                    // subtree_com_in,                                                            <L 2803>
                    // cdof_in,                                                                   <L 2804>
                    // site_xpos_idslider,                                                        <L 2805>
                    // site_bodyid[idslider],                                                     <L 2806>
                    var_332 = wp::address(var_site_bodyid, var_203);
                    // da,                                                                        <L 2807>
                    // worldid,                                                                   <L 2808>
                    var_335 = wp::load(var_332);
                    jac_dof_0(var_body_parentid, var_body_rootid, var_dof_bodyid, var_body_isdofancestor, var_subtree_com_in, var_cdof_in, var_231, var_335, var_331, var_0, var_333, var_334);
                    // jacS = jacp                                                                <L 2810>
                    var_336 = wp::copy(var_333);
                    // jacA = wp.cross(jacr, axis)                                                <L 2811>
                    var_337 = wp::cross(var_334, var_226);
                    // jac, _ = support.jac_dof(                                                  <L 2812>
                    // body_parentid,                                                             <L 2813>
                    // body_rootid,                                                               <L 2814>
                    // dof_bodyid,                                                                <L 2815>
                    // body_isdofancestor,                                                        <L 2816>
                    // subtree_com_in,                                                            <L 2817>
                    // cdof_in,                                                                   <L 2818>
                    // site_xpos_id,                                                              <L 2819>
                    // site_bodyid[id],                                                           <L 2820>
                    var_338 = wp::address(var_site_bodyid, var_201);
                    // da,                                                                        <L 2821>
                    // worldid,                                                                   <L 2822>
                    var_341 = wp::load(var_338);
                    jac_dof_0(var_body_parentid, var_body_rootid, var_dof_bodyid, var_body_isdofancestor, var_subtree_com_in, var_cdof_in, var_228, var_341, var_331, var_0, var_339, var_340);
                    // jac -= jacS                                                                <L 2824>
                    var_342 = wp::sub(var_339, var_336);
                    // moment = wp.dot(dlda, jacA) + wp.dot(dldv, jac)                            <L 2827>
                    var_343 = wp::dot(var_264, var_337);
                    var_344 = wp::dot(var_263, var_342);
                    var_345 = wp::add(var_343, var_344);
                    // sparseid = rowadr + ptr                                                    <L 2828>
                    var_346 = wp::add(var_321, var_325);
                    // moment_colind_out[worldid, sparseid] = da                                  <L 2829>
                    wp::array_store(var_moment_colind_out, var_0, var_346, var_331);
                    // actuator_moment_out[worldid, sparseid] = moment * gear0                    <L 2830>
                    var_347 = wp::mul(var_345, var_205);
                    wp::array_store(var_actuator_moment_out, var_0, var_346, var_347);
                    // ptr -= 1                                                                   <L 2831>
                    var_349 = wp::sub(var_325, var_348);
                    // if da1 == da:                                                              <L 2833>
                    var_350 = (var_322 == var_331);
                    if (var_350) {
                        // da1 = dof_parentid[da1]                                                <L 2834>
                        var_351 = wp::address(var_dof_parentid, var_322);
                        var_353 = wp::load(var_351);
                        var_352 = wp::copy(var_353);
                    }
                    var_354 = wp::where(var_350, var_352, var_322);
                    // if da2 == da:                                                              <L 2835>
                    var_355 = (var_323 == var_331);
                    if (var_355) {
                        // da2 = dof_parentid[da2]                                                <L 2836>
                        var_356 = wp::address(var_dof_parentid, var_323);
                        var_358 = wp::load(var_356);
                        var_357 = wp::copy(var_358);
                    }
                    var_359 = wp::where(var_355, var_357, var_323);
                    wp::assign(var_322, var_354);
                    wp::assign(var_323, var_359);
                    wp::assign(var_308, var_331);
                    wp::assign(var_325, var_349);
        goto start_while_2;
        end_while_2:;
            }
            if (!var_196) {
                // elif trntype == TrnType.TENDON:                                                <L 2837>
                var_361 = (var_3 == var_360);
                if (var_361) {
                    // tenid = actuator_trnid[actid][0]                                           <L 2838>
                    var_362 = wp::address(var_actuator_trnid, var_1);
                    var_365 = wp::load(var_362);
                    var_364 = wp::extract(var_365, var_363);
                    // gear0 = gear[0]                                                            <L 2840>
                    var_367 = wp::extract(var_11, var_366);
                    // actuator_length_out[worldid, actid] = ten_length_in[worldid, tenid] * gear0       <L 2841>
                    var_368 = wp::address(var_ten_length_in, var_0, var_364);
                    var_370 = wp::load(var_368);
                    var_369 = wp::mul(var_370, var_367);
                    wp::array_store(var_actuator_length_out, var_0, var_1, var_369);
                    // rownnz_ten = ten_J_rownnz[tenid]                                           <L 2843>
                    var_371 = wp::address(var_ten_J_rownnz, var_364);
                    var_373 = wp::load(var_371);
                    var_372 = wp::copy(var_373);
                    // rowadr_ten = ten_J_rowadr[tenid]                                           <L 2844>
                    var_374 = wp::address(var_ten_J_rowadr, var_364);
                    var_376 = wp::load(var_374);
                    var_375 = wp::copy(var_376);
                    // rowadr_mom = wp.atomic_add(moment_nnz, worldid, rownnz_ten)                <L 2846>
                    var_377 = wp::atomic_add(var_moment_nnz, var_0, var_372);
                    // moment_rownnz_out[worldid, actid] = rownnz_ten                             <L 2847>
                    wp::array_store(var_moment_rownnz_out, var_0, var_1, var_372);
                    // moment_rowadr_out[worldid, actid] = rowadr_mom                             <L 2848>
                    wp::array_store(var_moment_rowadr_out, var_0, var_1, var_377);
                    // for k in range(rownnz_ten):                                                <L 2850>
                    var_378 = wp::range(var_372);
                    start_for_4:;
                        if (iter_cmp(var_378) == 0) goto end_for_4;
                        var_379 = wp::iter_next(var_378);
                        // sparseid_ten = rowadr_ten + k                                          <L 2851>
                        var_380 = wp::add(var_375, var_379);
                        // sparseid_mom = rowadr_mom + k                                          <L 2852>
                        var_381 = wp::add(var_377, var_379);
                        // moment_colind_out[worldid, sparseid_mom] = ten_J_colind[sparseid_ten]       <L 2853>
                        var_382 = wp::address(var_ten_J_colind, var_380);
                        var_384 = wp::load(var_382);
                        var_383 = wp::copy(var_384);
                        wp::array_store(var_moment_colind_out, var_0, var_381, var_383);
                        // actuator_moment_out[worldid, sparseid_mom] = ten_J_in[worldid, sparseid_ten] * gear0       <L 2854>
                        var_385 = wp::address(var_ten_J_in, var_0, var_380);
                        var_387 = wp::load(var_385);
                        var_386 = wp::mul(var_387, var_367);
                        wp::array_store(var_actuator_moment_out, var_0, var_381, var_386);
                        goto start_for_4;
                    end_for_4:;
                }
                if (!var_361) {
                    // elif trntype == TrnType.BODY:                                              <L 2855>
                    var_389 = (var_3 == var_388);
                    if (var_389) {
                        // actuator_length_out[worldid, actid] = 0.0                              <L 2857>
                        wp::array_store(var_actuator_length_out, var_0, var_1, var_390);
                        // rowadr = wp.atomic_add(moment_nnz, worldid, nv)                        <L 2860>
                        var_391 = wp::atomic_add(var_moment_nnz, var_0, var_nv);
                        // moment_rownnz_out[worldid, actid] = nv                                 <L 2861>
                        wp::array_store(var_moment_rownnz_out, var_0, var_1, var_nv);
                        // moment_rowadr_out[worldid, actid] = rowadr                             <L 2862>
                        wp::array_store(var_moment_rowadr_out, var_0, var_1, var_391);
                        // for i in range(nv):                                                    <L 2863>
                        var_392 = wp::range(var_nv);
                        start_for_6:;
                            if (iter_cmp(var_392) == 0) goto end_for_6;
                            var_393 = wp::iter_next(var_392);
                            // sparseid = rowadr + i                                              <L 2864>
                            var_394 = wp::add(var_391, var_393);
                            // moment_colind_out[worldid, sparseid] = i                           <L 2865>
                            wp::array_store(var_moment_colind_out, var_0, var_394, var_393);
                            // actuator_moment_out[worldid, sparseid] = 0.0                       <L 2866>
                            wp::array_store(var_actuator_moment_out, var_0, var_394, var_395);
                            goto start_for_6;
                        end_for_6:;
                    }
                    if (!var_389) {
                        // elif trntype == TrnType.SITE:                                          <L 2869>
                        var_397 = (var_3 == var_396);
                        if (var_397) {
                            // trnid = actuator_trnid[actid]                                      <L 2870>
                            var_398 = wp::address(var_actuator_trnid, var_1);
                            var_400 = wp::load(var_398);
                            var_399 = wp::copy(var_400);
                            // siteid = trnid[0]                                                  <L 2871>
                            var_402 = wp::extract(var_399, var_401);
                            // refid = trnid[1]                                                   <L 2872>
                            var_404 = wp::extract(var_399, var_403);
                            // gear = actuator_gear[actuator_gear_id, actid]                      <L 2874>
                            var_405 = wp::address(var_actuator_gear, var_9, var_1);
                            var_407 = wp::load(var_405);
                            var_406 = wp::copy(var_407);
                            // site_quat_id = worldid % site_quat.shape[0]                        <L 2875>
                            var_408 = &(var_site_quat.shape);
                            var_411 = wp::load(var_408);
                            var_410 = wp::extract(var_411, var_409);
                            var_412 = wp::mod(var_0, var_410);
                            // gear_translation = wp.spatial_top(gear)                            <L 2876>
                            var_413 = wp::spatial_top(var_406);
                            // gear_rotational = wp.spatial_bottom(gear)                          <L 2877>
                            var_414 = wp::spatial_bottom(var_406);
                            // if refid == -1:                                                    <L 2880>
                            var_416 = (var_404 == var_415);
                            if (var_416) {
                                // site_xmat = site_xmat_in[worldid, siteid]                      <L 2882>
                                var_417 = wp::address(var_site_xmat_in, var_0, var_402);
                                var_419 = wp::load(var_417);
                                var_418 = wp::copy(var_419);
                                // wrench_translation = site_xmat @ gear_translation              <L 2883>
                                var_420 = wp::mul(var_418, var_413);
                                // wrench_rotation = site_xmat @ gear_rotational                  <L 2884>
                                var_421 = wp::mul(var_418, var_414);
                                // b1 = body_weldid[site_bodyid[siteid]]                          <L 2887>
                                var_422 = wp::address(var_site_bodyid, var_402);
                                var_423 = wp::load(var_422);
                                var_424 = wp::address(var_body_weldid, var_423);
                                var_426 = wp::load(var_424);
                                var_425 = wp::copy(var_426);
                                // da_init = int(-1)                                              <L 2888>
                                var_428 = wp::int(var_427);
                                // if b1 > 0:                                                     <L 2889>
                                var_430 = (var_425 > var_429);
                                if (var_430) {
                                    // da_init = body_dofadr[b1] + body_dofnum[b1] - 1            <L 2890>
                                    var_431 = wp::address(var_body_dofadr, var_425);
                                    var_432 = wp::address(var_body_dofnum, var_425);
                                    var_434 = wp::load(var_431);
                                    var_435 = wp::load(var_432);
                                    var_433 = wp::add(var_434, var_435);
                                    var_437 = wp::sub(var_433, var_436);
                                }
                                var_438 = wp::where(var_430, var_437, var_428);
                                // da = da_init                                                   <L 2892>
                                var_439 = wp::copy(var_438);
                                // ndof = int(0)                                                  <L 2893>
                                var_441 = wp::int(var_440);
                                // while da >= 0:                                                 <L 2894>
        start_while_8:;
                                var_443 = (var_439 >= var_442);
        if ((var_443) == false) goto end_while_8;
                                    // ndof += 1                                                  <L 2895>
                                    var_445 = wp::add(var_441, var_444);
                                    // da = dof_parentid[da]                                      <L 2896>
                                    var_446 = wp::address(var_dof_parentid, var_439);
                                    var_448 = wp::load(var_446);
                                    var_447 = wp::copy(var_448);
                                    wp::assign(var_439, var_447);
                                    wp::assign(var_441, var_445);
        goto start_while_8;
        end_while_8:;
                                // moment_rownnz_out[worldid, actid] = ndof                       <L 2898>
                                wp::array_store(var_moment_rownnz_out, var_0, var_1, var_441);
                                // rowadr = wp.atomic_add(moment_nnz, worldid, ndof)              <L 2899>
                                var_449 = wp::atomic_add(var_moment_nnz, var_0, var_441);
                                // moment_rowadr_out[worldid, actid] = rowadr                     <L 2900>
                                wp::array_store(var_moment_rowadr_out, var_0, var_1, var_449);
                                // actuator_length_out[worldid, actid] = 0.0                      <L 2901>
                                wp::array_store(var_actuator_length_out, var_0, var_1, var_450);
                                // da = da_init                                                   <L 2904>
                                var_451 = wp::copy(var_438);
                                // ptr = ndof - 1                                                 <L 2905>
                                var_453 = wp::sub(var_441, var_452);
                                // while da >= 0:                                                 <L 2906>
        start_while_10:;
                                var_455 = (var_451 >= var_454);
        if ((var_455) == false) goto end_while_10;
                                    // jacp, jacr = support.jac_dof(                              <L 2907>
                                    // body_parentid,                                             <L 2908>
                                    // body_rootid,                                               <L 2909>
                                    // dof_bodyid,                                                <L 2910>
                                    // body_isdofancestor,                                        <L 2911>
                                    // subtree_com_in,                                            <L 2912>
                                    // cdof_in,                                                   <L 2913>
                                    // site_xpos_in[worldid, siteid],                             <L 2914>
                                    var_456 = wp::address(var_site_xpos_in, var_0, var_402);
                                    // site_bodyid[siteid],                                       <L 2915>
                                    var_457 = wp::address(var_site_bodyid, var_402);
                                    // da,                                                        <L 2916>
                                    // worldid,                                                   <L 2917>
                                    var_460 = wp::load(var_456);
                                    var_461 = wp::load(var_457);
                                    jac_dof_0(var_body_parentid, var_body_rootid, var_dof_bodyid, var_body_isdofancestor, var_subtree_com_in, var_cdof_in, var_460, var_461, var_451, var_0, var_458, var_459);
                                    // moment = wp.dot(jacp, wrench_translation) + wp.dot(jacr, wrench_rotation)       <L 2919>
                                    var_462 = wp::dot(var_458, var_420);
                                    var_463 = wp::dot(var_459, var_421);
                                    var_464 = wp::add(var_462, var_463);
                                    // sparseid = rowadr + ptr                                    <L 2920>
                                    var_465 = wp::add(var_449, var_453);
                                    // moment_colind_out[worldid, sparseid] = da                  <L 2921>
                                    wp::array_store(var_moment_colind_out, var_0, var_465, var_451);
                                    // actuator_moment_out[worldid, sparseid] = moment            <L 2922>
                                    wp::array_store(var_actuator_moment_out, var_0, var_465, var_464);
                                    // ptr -= 1                                                   <L 2923>
                                    var_467 = wp::sub(var_453, var_466);
                                    // da = dof_parentid[da]                                      <L 2924>
                                    var_468 = wp::address(var_dof_parentid, var_451);
                                    var_470 = wp::load(var_468);
                                    var_469 = wp::copy(var_470);
                                    wp::assign(var_451, var_469);
                                    wp::assign(var_453, var_467);
        goto start_while_10;
        end_while_10:;
                            }
                            if (!var_416) {
                                // bodyid = site_bodyid[siteid]                                   <L 2928>
                                var_471 = wp::address(var_site_bodyid, var_402);
                                var_473 = wp::load(var_471);
                                var_472 = wp::copy(var_473);
                                // bodyrefid = site_bodyid[refid]                                 <L 2929>
                                var_474 = wp::address(var_site_bodyid, var_404);
                                var_476 = wp::load(var_474);
                                var_475 = wp::copy(var_476);
                                // b0 = body_weldid[bodyid]                                       <L 2930>
                                var_477 = wp::address(var_body_weldid, var_472);
                                var_479 = wp::load(var_477);
                                var_478 = wp::copy(var_479);
                                // b1 = body_weldid[bodyrefid]                                    <L 2931>
                                var_480 = wp::address(var_body_weldid, var_475);
                                var_482 = wp::load(var_480);
                                var_481 = wp::copy(var_482);
                                // dofadr0 = body_dofadr[b0] + body_dofnum[b0] - 1                <L 2932>
                                var_483 = wp::address(var_body_dofadr, var_478);
                                var_484 = wp::address(var_body_dofnum, var_478);
                                var_486 = wp::load(var_483);
                                var_487 = wp::load(var_484);
                                var_485 = wp::add(var_486, var_487);
                                var_489 = wp::sub(var_485, var_488);
                                // dofadr1 = body_dofadr[b1] + body_dofnum[b1] - 1                <L 2933>
                                var_490 = wp::address(var_body_dofadr, var_481);
                                var_491 = wp::address(var_body_dofnum, var_481);
                                var_493 = wp::load(var_490);
                                var_494 = wp::load(var_491);
                                var_492 = wp::add(var_493, var_494);
                                var_496 = wp::sub(var_492, var_495);
                                // dofadr_common = -1                                             <L 2936>
                                // if dofadr0 >= 0 and dofadr1 >= 0:                              <L 2937>
                                var_500 = (var_489 >= var_499);
                                var_498 = var_500;
                                if (var_498) {
                                    var_502 = (var_496 >= var_501);
                                    var_498 = var_498 && var_502;
                                }
                                if (var_498) {
                                    // while dofadr0 != dofadr1:                                  <L 2939>
        start_while_12:;
                                    var_503 = (var_489 != var_496);
        if ((var_503) == false) goto end_while_12;
                                        // if dofadr0 < dofadr1:                                  <L 2940>
                                        var_504 = (var_489 < var_496);
                                        if (var_504) {
                                            // dofadr1 = dof_parentid[dofadr1]                    <L 2941>
                                            var_505 = wp::address(var_dof_parentid, var_496);
                                            var_507 = wp::load(var_505);
                                            var_506 = wp::copy(var_507);
                                        }
                                        if (!var_504) {
                                            // dofadr0 = dof_parentid[dofadr0]                    <L 2943>
                                            var_508 = wp::address(var_dof_parentid, var_489);
                                            var_510 = wp::load(var_508);
                                            var_509 = wp::copy(var_510);
                                        }
                                        var_511 = wp::where(var_504, var_489, var_509);
                                        var_512 = wp::where(var_504, var_506, var_496);
                                        // if dofadr0 == -1 or dofadr1 == -1:                     <L 2945>
                                        var_515 = (var_511 == var_514);
                                        var_513 = var_515;
                                        if (!var_513) {
                                            var_517 = (var_512 == var_516);
                                            var_513 = var_513 || var_517;
                                        }
                                        if (var_513) {
                                            // break                                              <L 2947>
                                            wp::assign(var_489, var_511);
                                            wp::assign(var_496, var_512);
                                            goto end_while_12;
                                        }
                                        var_518 = wp::where(var_513, var_489, var_511);
                                        var_519 = wp::where(var_513, var_496, var_512);
                                        wp::assign(var_489, var_518);
                                        wp::assign(var_496, var_519);
        goto start_while_12;
        end_while_12:;
                                    // if dofadr0 == dofadr1:                                     <L 2950>
                                    var_520 = (var_489 == var_496);
                                    if (var_520) {
                                        // dofadr_common = dofadr0                                <L 2951>
                                        var_521 = wp::copy(var_489);
                                    }
                                    var_522 = wp::where(var_520, var_521, var_497);
                                }
                                var_523 = wp::where(var_498, var_522, var_497);
                                // translational_transmission = not (gear[0] == 0.0 and gear[1] == 0.0 and gear[2] == 0.0)       <L 2953>
                                var_526 = wp::extract(var_406, var_525);
                                var_528 = (var_526 == var_527);
                                var_524 = var_528;
                                if (var_524) {
                                    var_530 = wp::extract(var_406, var_529);
                                    var_532 = (var_530 == var_531);
                                    var_524 = var_524 && var_532;
                                }
                                if (var_524) {
                                    var_534 = wp::extract(var_406, var_533);
                                    var_536 = (var_534 == var_535);
                                    var_524 = var_524 && var_536;
                                }
                                var_537 = wp::unot(var_524);
                                // rotational_transmission = not (gear[3] == 0.0 and gear[4] == 0.0 and gear[5] == 0.0)       <L 2954>
                                var_540 = wp::extract(var_406, var_539);
                                var_542 = (var_540 == var_541);
                                var_538 = var_542;
                                if (var_538) {
                                    var_544 = wp::extract(var_406, var_543);
                                    var_546 = (var_544 == var_545);
                                    var_538 = var_538 && var_546;
                                }
                                if (var_538) {
                                    var_548 = wp::extract(var_406, var_547);
                                    var_550 = (var_548 == var_549);
                                    var_538 = var_538 && var_550;
                                }
                                var_551 = wp::unot(var_538);
                                // site_xpos = site_xpos_in[worldid, siteid]                      <L 2956>
                                var_552 = wp::address(var_site_xpos_in, var_0, var_402);
                                var_554 = wp::load(var_552);
                                var_553 = wp::copy(var_554);
                                // ref_xpos = site_xpos_in[worldid, refid]                        <L 2957>
                                var_555 = wp::address(var_site_xpos_in, var_0, var_404);
                                var_557 = wp::load(var_555);
                                var_556 = wp::copy(var_557);
                                // ref_xmat = site_xmat_in[worldid, refid]                        <L 2958>
                                var_558 = wp::address(var_site_xmat_in, var_0, var_404);
                                var_560 = wp::load(var_558);
                                var_559 = wp::copy(var_560);
                                // length = float(0.0)                                            <L 2960>
                                var_562 = wp::float(var_561);
                                // if translational_transmission:                                 <L 2962>
                                if (var_537) {
                                    // vec = wp.transpose(ref_xmat) @ (site_xpos - ref_xpos)       <L 2964>
                                    var_563 = wp::transpose(var_559);
                                    var_564 = wp::sub(var_553, var_556);
                                    var_565 = wp::mul(var_563, var_564);
                                    // length += wp.dot(vec, gear_translation)                    <L 2965>
                                    var_566 = wp::dot(var_565, var_413);
                                    var_567 = wp::add(var_562, var_566);
                                    // wrench_translation = ref_xmat @ gear_translation           <L 2967>
                                    var_568 = wp::mul(var_559, var_413);
                                }
                                var_569 = wp::where(var_537, var_567, var_562);
                                // if rotational_transmission:                                    <L 2969>
                                if (var_551) {
                                    // quat = math.mul_quat(xquat_in[worldid, bodyid], site_quat[site_quat_id, siteid])       <L 2971>
                                    var_570 = wp::address(var_xquat_in, var_0, var_472);
                                    var_571 = wp::address(var_site_quat, var_412, var_402);
                                    var_573 = wp::load(var_570);
                                    var_574 = wp::load(var_571);
                                    var_572 = mul_quat_0(var_573, var_574);
                                    // refquat = math.mul_quat(xquat_in[worldid, bodyrefid], site_quat[site_quat_id, refid])       <L 2972>
                                    var_575 = wp::address(var_xquat_in, var_0, var_475);
                                    var_576 = wp::address(var_site_quat, var_412, var_404);
                                    var_578 = wp::load(var_575);
                                    var_579 = wp::load(var_576);
                                    var_577 = mul_quat_0(var_578, var_579);
                                    // vec = math.quat_sub(quat, refquat)                         <L 2975>
                                    var_580 = quat_sub_0(var_572, var_577);
                                    // length += wp.dot(vec, gear_rotational)                     <L 2976>
                                    var_581 = wp::dot(var_580, var_414);
                                    var_582 = wp::add(var_569, var_581);
                                    // wrench_rotation = ref_xmat @ gear_rotational               <L 2978>
                                    var_583 = wp::mul(var_559, var_414);
                                }
                                var_584 = wp::where(var_551, var_582, var_569);
                                var_585 = wp::where(var_551, var_580, var_565);
                                // actuator_length_out[worldid, actid] = length                   <L 2980>
                                wp::array_store(var_actuator_length_out, var_0, var_1, var_584);
                                // da1_init = int(-1)                                             <L 2983>
                                var_587 = wp::int(var_586);
                                // da2_init = int(-1)                                             <L 2984>
                                var_589 = wp::int(var_588);
                                // if b0 > 0:                                                     <L 2985>
                                var_591 = (var_478 > var_590);
                                if (var_591) {
                                    // da1_init = body_dofadr[b0] + body_dofnum[b0] - 1           <L 2986>
                                    var_592 = wp::address(var_body_dofadr, var_478);
                                    var_593 = wp::address(var_body_dofnum, var_478);
                                    var_595 = wp::load(var_592);
                                    var_596 = wp::load(var_593);
                                    var_594 = wp::add(var_595, var_596);
                                    var_598 = wp::sub(var_594, var_597);
                                }
                                var_599 = wp::where(var_591, var_598, var_587);
                                // if b1 > 0:                                                     <L 2987>
                                var_601 = (var_481 > var_600);
                                if (var_601) {
                                    // da2_init = body_dofadr[b1] + body_dofnum[b1] - 1           <L 2988>
                                    var_602 = wp::address(var_body_dofadr, var_481);
                                    var_603 = wp::address(var_body_dofnum, var_481);
                                    var_605 = wp::load(var_602);
                                    var_606 = wp::load(var_603);
                                    var_604 = wp::add(var_605, var_606);
                                    var_608 = wp::sub(var_604, var_607);
                                }
                                var_609 = wp::where(var_601, var_608, var_589);
                                // da1 = da1_init                                                 <L 2990>
                                var_610 = wp::copy(var_599);
                                // da2 = da2_init                                                 <L 2991>
                                var_611 = wp::copy(var_609);
                                // ndof = int(0)                                                  <L 2992>
                                var_613 = wp::int(var_612);
                                // while da1 >= 0 or da2 >= 0:                                    <L 2993>
        start_while_14:;
                                var_616 = (var_610 >= var_615);
                                var_614 = var_616;
                                if (!var_614) {
                                    var_618 = (var_611 >= var_617);
                                    var_614 = var_614 || var_618;
                                }
        if ((var_614) == false) goto end_while_14;
                                    // da = wp.max(da1, da2)                                      <L 2994>
                                    var_619 = wp::max(var_610, var_611);
                                    // if da1 == da and da2 == da:                                <L 2995>
                                    var_621 = (var_610 == var_619);
                                    var_620 = var_621;
                                    if (var_620) {
                                        var_622 = (var_611 == var_619);
                                        var_620 = var_620 && var_622;
                                    }
                                    if (var_620) {
                                        // break                                                  <L 2996>
                                        goto end_while_14;
                                    }
                                    // ndof += 1                                                  <L 2997>
                                    var_624 = wp::add(var_613, var_623);
                                    // if da1 == da:                                              <L 2998>
                                    var_625 = (var_610 == var_619);
                                    if (var_625) {
                                        // da1 = dof_parentid[da1]                                <L 2999>
                                        var_626 = wp::address(var_dof_parentid, var_610);
                                        var_628 = wp::load(var_626);
                                        var_627 = wp::copy(var_628);
                                    }
                                    var_629 = wp::where(var_625, var_627, var_610);
                                    // if da2 == da:                                              <L 3000>
                                    var_630 = (var_611 == var_619);
                                    if (var_630) {
                                        // da2 = dof_parentid[da2]                                <L 3001>
                                        var_631 = wp::address(var_dof_parentid, var_611);
                                        var_633 = wp::load(var_631);
                                        var_632 = wp::copy(var_633);
                                    }
                                    var_634 = wp::where(var_630, var_632, var_611);
                                    wp::assign(var_610, var_629);
                                    wp::assign(var_611, var_634);
                                    wp::assign(var_613, var_624);
        goto start_while_14;
        end_while_14:;
                                // moment_rownnz_out[worldid, actid] = ndof                       <L 3003>
                                wp::array_store(var_moment_rownnz_out, var_0, var_1, var_613);
                                // rowadr = wp.atomic_add(moment_nnz, worldid, ndof)              <L 3004>
                                var_635 = wp::atomic_add(var_moment_nnz, var_0, var_613);
                                // moment_rowadr_out[worldid, actid] = rowadr                     <L 3005>
                                wp::array_store(var_moment_rowadr_out, var_0, var_1, var_635);
                                // da1 = da1_init                                                 <L 3008>
                                var_636 = wp::copy(var_599);
                                // da2 = da2_init                                                 <L 3009>
                                var_637 = wp::copy(var_609);
                                // ptr = ndof - 1                                                 <L 3011>
                                var_639 = wp::sub(var_613, var_638);
                                // while da1 >= 0 or da2 >= 0:                                    <L 3012>
        start_while_16:;
                                var_642 = (var_636 >= var_641);
                                var_640 = var_642;
                                if (!var_640) {
                                    var_644 = (var_637 >= var_643);
                                    var_640 = var_640 || var_644;
                                }
        if ((var_640) == false) goto end_while_16;
                                    // da = wp.max(da1, da2)                                      <L 3013>
                                    var_645 = wp::max(var_636, var_637);
                                    // if da1 == da and da2 == da:                                <L 3014>
                                    var_647 = (var_636 == var_645);
                                    var_646 = var_647;
                                    if (var_646) {
                                        var_648 = (var_637 == var_645);
                                        var_646 = var_646 && var_648;
                                    }
                                    if (var_646) {
                                        // break                                                  <L 3015>
                                        wp::assign(var_619, var_645);
                                        goto end_while_16;
                                    }
                                    var_649 = wp::where(var_646, var_619, var_645);
                                    // jacp, jacr = support.jac_dof(                              <L 3017>
                                    // body_parentid,                                             <L 3018>
                                    // body_rootid,                                               <L 3019>
                                    // dof_bodyid,                                                <L 3020>
                                    // body_isdofancestor,                                        <L 3021>
                                    // subtree_com_in,                                            <L 3022>
                                    // cdof_in,                                                   <L 3023>
                                    // site_xpos,                                                 <L 3024>
                                    // site_bodyid[siteid],                                       <L 3025>
                                    var_650 = wp::address(var_site_bodyid, var_402);
                                    // da,                                                        <L 3026>
                                    // worldid,                                                   <L 3027>
                                    var_653 = wp::load(var_650);
                                    jac_dof_0(var_body_parentid, var_body_rootid, var_dof_bodyid, var_body_isdofancestor, var_subtree_com_in, var_cdof_in, var_553, var_653, var_649, var_0, var_651, var_652);
                                    // jacpref, jacrref = support.jac_dof(                        <L 3029>
                                    // body_parentid,                                             <L 3030>
                                    // body_rootid,                                               <L 3031>
                                    // dof_bodyid,                                                <L 3032>
                                    // body_isdofancestor,                                        <L 3033>
                                    // subtree_com_in,                                            <L 3034>
                                    // cdof_in,                                                   <L 3035>
                                    // ref_xpos,                                                  <L 3036>
                                    // site_bodyid[refid],                                        <L 3037>
                                    var_654 = wp::address(var_site_bodyid, var_404);
                                    // da,                                                        <L 3038>
                                    // worldid,                                                   <L 3039>
                                    var_657 = wp::load(var_654);
                                    jac_dof_0(var_body_parentid, var_body_rootid, var_dof_bodyid, var_body_isdofancestor, var_subtree_com_in, var_cdof_in, var_556, var_657, var_649, var_0, var_655, var_656);
                                    // moment = float(0.0)                                        <L 3042>
                                    var_659 = wp::float(var_658);
                                    // if translational_transmission:                             <L 3043>
                                    if (var_537) {
                                        // moment += wp.dot(jacp - jacpref, wrench_translation)       <L 3044>
                                        var_660 = wp::sub(var_651, var_655);
                                        var_661 = wp::dot(var_660, var_568);
                                        var_662 = wp::add(var_659, var_661);
                                    }
                                    var_663 = wp::where(var_537, var_662, var_659);
                                    // if rotational_transmission:                                <L 3045>
                                    if (var_551) {
                                        // moment += wp.dot(jacr - jacrref, wrench_rotation)       <L 3046>
                                        var_664 = wp::sub(var_652, var_656);
                                        var_665 = wp::dot(var_664, var_583);
                                        var_666 = wp::add(var_663, var_665);
                                    }
                                    var_667 = wp::where(var_551, var_666, var_663);
                                    // sparseid = rowadr + ptr                                    <L 3048>
                                    var_668 = wp::add(var_635, var_639);
                                    // moment_colind_out[worldid, sparseid] = da                  <L 3049>
                                    wp::array_store(var_moment_colind_out, var_0, var_668, var_649);
                                    // actuator_moment_out[worldid, sparseid] = moment            <L 3050>
                                    wp::array_store(var_actuator_moment_out, var_0, var_668, var_667);
                                    // ptr -= 1                                                   <L 3051>
                                    var_670 = wp::sub(var_639, var_669);
                                    // if da1 == da:                                              <L 3053>
                                    var_671 = (var_636 == var_649);
                                    if (var_671) {
                                        // da1 = dof_parentid[da1]                                <L 3054>
                                        var_672 = wp::address(var_dof_parentid, var_636);
                                        var_674 = wp::load(var_672);
                                        var_673 = wp::copy(var_674);
                                    }
                                    var_675 = wp::where(var_671, var_673, var_636);
                                    // if da2 == da:                                              <L 3055>
                                    var_676 = (var_637 == var_649);
                                    if (var_676) {
                                        // da2 = dof_parentid[da2]                                <L 3056>
                                        var_677 = wp::address(var_dof_parentid, var_637);
                                        var_679 = wp::load(var_677);
                                        var_678 = wp::copy(var_679);
                                    }
                                    var_680 = wp::where(var_676, var_678, var_637);
                                    wp::assign(var_636, var_675);
                                    wp::assign(var_637, var_680);
                                    wp::assign(var_619, var_649);
                                    wp::assign(var_639, var_670);
        goto start_while_16;
        end_while_16:;
                            }
                            var_681 = wp::where(var_416, var_420, var_568);
                            var_682 = wp::where(var_416, var_421, var_583);
                            var_683 = wp::where(var_416, var_425, var_481);
                            var_684 = wp::where(var_416, var_451, var_619);
                            var_685 = wp::where(var_416, var_441, var_613);
                            var_686 = wp::where(var_416, var_449, var_635);
                            var_687 = wp::where(var_416, var_453, var_639);
                            var_688 = wp::where(var_416, var_458, var_651);
                            var_689 = wp::where(var_416, var_459, var_652);
                            var_690 = wp::where(var_416, var_464, var_667);
                            var_691 = wp::where(var_416, var_465, var_668);
                        }
                        if (!var_397) {
                            // wp.printf("unhandled transmission type %d\n", trntype)             <L 3058>
                            printf(var_692, var_3);
                        }
                        var_693 = wp::where(var_397, var_406, var_11);
                    }
                    var_694 = wp::where(var_389, var_11, var_693);
                    var_695 = wp::where(var_389, var_391, var_686);
                    var_696 = wp::where(var_389, var_394, var_691);
                }
                var_697 = wp::where(var_361, var_11, var_694);
            }
            var_698 = wp::where(var_196, var_11, var_697);
            var_699 = wp::where(var_196, var_198, var_399);
            var_700 = wp::where(var_196, var_205, var_367);
            var_701 = wp::where(var_196, var_215, var_418);
            var_702 = wp::where(var_196, var_233, var_585);
            var_703 = wp::where(var_196, var_250, var_584);
            var_704 = wp::where(var_196, var_268, var_683);
            var_705 = wp::where(var_196, var_288, var_599);
            var_706 = wp::where(var_196, var_298, var_609);
            var_707 = wp::where(var_196, var_322, var_636);
            var_708 = wp::where(var_196, var_323, var_637);
            var_709 = wp::where(var_196, var_302, var_685);
            var_710 = wp::where(var_196, var_308, var_684);
            var_711 = wp::where(var_196, var_321, var_695);
            var_712 = wp::where(var_196, var_325, var_687);
            var_713 = wp::where(var_196, var_333, var_688);
            var_714 = wp::where(var_196, var_334, var_689);
            var_715 = wp::where(var_196, var_345, var_690);
            var_716 = wp::where(var_196, var_346, var_696);
        }
        var_717 = wp::where(var_13, var_11, var_698);
        var_718 = wp::where(var_13, var_192, var_711);
        var_719 = wp::where(var_13, var_81, var_572);
        var_720 = wp::where(var_13, var_171, var_393);
        var_721 = wp::where(var_13, var_172, var_716);
    }
}



extern "C" __global__ void _comvel_root_15820f35_cuda_kernel_forward(
    wp::launch_bounds_t<2> dim,
    wp::array_t<wp::vec_t<6, wp::float32>> var_cvel_out)
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
        const wp::float32 var_2 = 0.0;
        const wp::int32 var_3 = 0;
        wp::vec_t<6, wp::float32>* var_4;
        //---------
        // forward
        // def _comvel_root(cvel_out: wp.array2d[wp.spatial_vector]):                             <L 2509>
        // worldid, elementid = wp.tid()                                                          <L 2510>
        builtin_tid2d(var_0, var_1);
        // cvel_out[worldid, 0][elementid] = 0.0                                                  <L 2511>
        var_4 = &(wp::index(var_cvel_out, var_0, var_3));
        wp::store(&(wp::index(var_cvel_out, var_0, var_3).component_ref(var_1)), var_2);
    }
}



extern "C" __global__ void _light_local_to_global_1966b84b_cuda_kernel_forward(
    wp::launch_bounds_t<2> dim,
    wp::array_t<wp::int32> var_light_mode,
    wp::array_t<wp::int32> var_light_bodyid,
    wp::array_t<wp::int32> var_light_targetbodyid,
    wp::array_t<wp::vec_t<3, wp::float32>> var_light_pos,
    wp::array_t<wp::vec_t<3, wp::float32>> var_light_dir,
    wp::array_t<wp::vec_t<3, wp::float32>> var_light_poscom0,
    wp::array_t<wp::vec_t<3, wp::float32>> var_light_pos0,
    wp::array_t<wp::vec_t<3, wp::float32>> var_light_dir0,
    wp::array_t<wp::vec_t<3, wp::float32>> var_xpos_in,
    wp::array_t<wp::quat_t<wp::float32>> var_xquat_in,
    wp::array_t<wp::vec_t<3, wp::float32>> var_subtree_com_in,
    wp::array_t<wp::vec_t<3, wp::float32>> var_light_xpos_out,
    wp::array_t<wp::vec_t<3, wp::float32>> var_light_xdir_out)
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
        wp::int32* var_13;
        const wp::int32 var_14 = 3;
        bool var_15;
        wp::int32 var_16;
        wp::int32* var_17;
        const wp::int32 var_18 = 4;
        bool var_19;
        wp::int32 var_20;
        bool var_21;
        wp::int32* var_22;
        const wp::int32 var_23 = 0;
        bool var_24;
        wp::int32 var_25;
        wp::int32* var_26;
        wp::int32 var_27;
        wp::int32 var_28;
        wp::vec_t<3, wp::float32>* var_29;
        wp::vec_t<3, wp::float32> var_30;
        wp::vec_t<3, wp::float32> var_31;
        wp::quat_t<wp::float32>* var_32;
        wp::quat_t<wp::float32> var_33;
        wp::quat_t<wp::float32> var_34;
        wp::vec_t<3, wp::float32>* var_35;
        wp::vec_t<3, wp::float32> var_36;
        wp::vec_t<3, wp::float32> var_37;
        wp::vec_t<3, wp::float32> var_38;
        wp::vec_t<3, wp::float32>* var_39;
        wp::vec_t<3, wp::float32> var_40;
        wp::vec_t<3, wp::float32> var_41;
        wp::int32* var_42;
        const wp::int32 var_43 = 1;
        bool var_44;
        wp::int32 var_45;
        wp::shape_t* var_46;
        const wp::int32 var_47 = 0;
        wp::int32 var_48;
        wp::shape_t var_49;
        wp::int32 var_50;
        wp::vec_t<3, wp::float32>* var_51;
        wp::vec_t<3, wp::float32> var_52;
        wp::vec_t<3, wp::float32> var_53;
        wp::int32* var_54;
        wp::int32 var_55;
        wp::vec_t<3, wp::float32>* var_56;
        wp::vec_t<3, wp::float32> var_57;
        wp::vec_t<3, wp::float32> var_58;
        wp::shape_t* var_59;
        const wp::int32 var_60 = 0;
        wp::int32 var_61;
        wp::shape_t var_62;
        wp::int32 var_63;
        wp::vec_t<3, wp::float32>* var_64;
        wp::vec_t<3, wp::float32> var_65;
        wp::vec_t<3, wp::float32> var_66;
        wp::int32* var_67;
        const wp::int32 var_68 = 2;
        bool var_69;
        wp::int32 var_70;
        wp::shape_t* var_71;
        const wp::int32 var_72 = 0;
        wp::int32 var_73;
        wp::shape_t var_74;
        wp::int32 var_75;
        wp::vec_t<3, wp::float32>* var_76;
        wp::vec_t<3, wp::float32> var_77;
        wp::vec_t<3, wp::float32> var_78;
        wp::int32* var_79;
        wp::int32 var_80;
        wp::vec_t<3, wp::float32>* var_81;
        wp::shape_t* var_82;
        const wp::int32 var_83 = 0;
        wp::int32 var_84;
        wp::shape_t var_85;
        wp::int32 var_86;
        wp::vec_t<3, wp::float32>* var_87;
        wp::vec_t<3, wp::float32> var_88;
        wp::vec_t<3, wp::float32> var_89;
        wp::vec_t<3, wp::float32> var_90;
        bool var_91;
        wp::int32* var_92;
        const wp::int32 var_93 = 3;
        bool var_94;
        wp::int32 var_95;
        wp::int32* var_96;
        const wp::int32 var_97 = 4;
        bool var_98;
        wp::int32 var_99;
        wp::int32* var_100;
        wp::int32 var_101;
        wp::int32 var_102;
        wp::vec_t<3, wp::float32>* var_103;
        wp::vec_t<3, wp::float32> var_104;
        wp::vec_t<3, wp::float32> var_105;
        wp::quat_t<wp::float32>* var_106;
        wp::quat_t<wp::float32> var_107;
        wp::quat_t<wp::float32> var_108;
        wp::vec_t<3, wp::float32>* var_109;
        wp::vec_t<3, wp::float32> var_110;
        wp::vec_t<3, wp::float32> var_111;
        wp::vec_t<3, wp::float32> var_112;
        wp::int32* var_113;
        wp::int32 var_114;
        wp::vec_t<3, wp::float32>* var_115;
        wp::vec_t<3, wp::float32> var_116;
        wp::vec_t<3, wp::float32> var_117;
        wp::int32* var_118;
        const wp::int32 var_119 = 4;
        bool var_120;
        wp::int32 var_121;
        wp::int32* var_122;
        wp::int32 var_123;
        wp::vec_t<3, wp::float32>* var_124;
        wp::vec_t<3, wp::float32> var_125;
        wp::vec_t<3, wp::float32> var_126;
        wp::vec_t<3, wp::float32> var_127;
        wp::vec_t<3, wp::float32>* var_128;
        wp::vec_t<3, wp::float32> var_129;
        wp::vec_t<3, wp::float32> var_130;
        wp::int32* var_131;
        wp::int32 var_132;
        wp::int32 var_133;
        wp::vec_t<3, wp::float32>* var_134;
        wp::vec_t<3, wp::float32> var_135;
        wp::vec_t<3, wp::float32> var_136;
        wp::quat_t<wp::float32>* var_137;
        wp::quat_t<wp::float32> var_138;
        wp::quat_t<wp::float32> var_139;
        wp::vec_t<3, wp::float32>* var_140;
        wp::vec_t<3, wp::float32> var_141;
        wp::vec_t<3, wp::float32> var_142;
        wp::vec_t<3, wp::float32> var_143;
        wp::vec_t<3, wp::float32>* var_144;
        wp::vec_t<3, wp::float32> var_145;
        wp::vec_t<3, wp::float32> var_146;
        wp::int32 var_147;
        wp::vec_t<3, wp::float32> var_148;
        wp::quat_t<wp::float32> var_149;
        wp::int32 var_150;
        wp::vec_t<3, wp::float32> var_151;
        wp::quat_t<wp::float32> var_152;
        wp::vec_t<3, wp::float32>* var_153;
        wp::vec_t<3, wp::float32> var_154;
        wp::vec_t<3, wp::float32> var_155;
        //---------
        // forward
        // def _light_local_to_global(                                                            <L 896>
        // worldid, lightid = wp.tid()                                                            <L 914>
        builtin_tid2d(var_0, var_1);
        // light_pos_id = worldid % light_pos.shape[0]                                            <L 915>
        var_2 = &(var_light_pos.shape);
        var_5 = wp::load(var_2);
        var_4 = wp::extract(var_5, var_3);
        var_6 = wp::mod(var_0, var_4);
        // light_dir_id = worldid % light_dir.shape[0]                                            <L 916>
        var_7 = &(var_light_dir.shape);
        var_10 = wp::load(var_7);
        var_9 = wp::extract(var_10, var_8);
        var_11 = wp::mod(var_0, var_9);
        // is_target_light = (light_mode[lightid] == CamLightType.TARGETBODY) or (light_mode[lightid] == CamLightType.TARGETBODYCOM)       <L 917>
        var_13 = wp::address(var_light_mode, var_1);
        var_16 = wp::load(var_13);
        var_15 = (var_16 == var_14);
        var_12 = var_15;
        if (!var_12) {
            var_17 = wp::address(var_light_mode, var_1);
            var_20 = wp::load(var_17);
            var_19 = (var_20 == var_18);
            var_12 = var_12 || var_19;
        }
        // invalid_target = is_target_light and (light_targetbodyid[lightid] < 0)                 <L 918>
        var_21 = var_12;
        if (var_21) {
            var_22 = wp::address(var_light_targetbodyid, var_1);
            var_25 = wp::load(var_22);
            var_24 = (var_25 < var_23);
            var_21 = var_21 && var_24;
        }
        // if invalid_target:                                                                     <L 919>
        if (var_21) {
            // bodyid = light_bodyid[lightid]                                                     <L 920>
            var_26 = wp::address(var_light_bodyid, var_1);
            var_28 = wp::load(var_26);
            var_27 = wp::copy(var_28);
            // xpos = xpos_in[worldid, bodyid]                                                    <L 921>
            var_29 = wp::address(var_xpos_in, var_0, var_27);
            var_31 = wp::load(var_29);
            var_30 = wp::copy(var_31);
            // xquat = xquat_in[worldid, bodyid]                                                  <L 922>
            var_32 = wp::address(var_xquat_in, var_0, var_27);
            var_34 = wp::load(var_32);
            var_33 = wp::copy(var_34);
            // light_xpos_out[worldid, lightid] = xpos + math.rot_vec_quat(light_pos[light_pos_id, lightid], xquat)       <L 923>
            var_35 = wp::address(var_light_pos, var_6, var_1);
            var_37 = wp::load(var_35);
            var_36 = rot_vec_quat_0(var_37, var_33);
            var_38 = wp::add(var_30, var_36);
            wp::array_store(var_light_xpos_out, var_0, var_1, var_38);
            // light_xdir_out[worldid, lightid] = math.rot_vec_quat(light_dir[light_dir_id, lightid], xquat)       <L 924>
            var_39 = wp::address(var_light_dir, var_11, var_1);
            var_41 = wp::load(var_39);
            var_40 = rot_vec_quat_0(var_41, var_33);
            wp::array_store(var_light_xdir_out, var_0, var_1, var_40);
            // return                                                                             <L 925>
            continue;
        }
        if (!var_21) {
            // elif light_mode[lightid] == CamLightType.TRACK:                                    <L 926>
            var_42 = wp::address(var_light_mode, var_1);
            var_45 = wp::load(var_42);
            var_44 = (var_45 == var_43);
            if (var_44) {
                // light_xdir_out[worldid, lightid] = light_dir0[worldid % light_dir0.shape[0], lightid]       <L 927>
                var_46 = &(var_light_dir0.shape);
                var_49 = wp::load(var_46);
                var_48 = wp::extract(var_49, var_47);
                var_50 = wp::mod(var_0, var_48);
                var_51 = wp::address(var_light_dir0, var_50, var_1);
                var_53 = wp::load(var_51);
                var_52 = wp::copy(var_53);
                wp::array_store(var_light_xdir_out, var_0, var_1, var_52);
                // body_xpos = xpos_in[worldid, light_bodyid[lightid]]                            <L 928>
                var_54 = wp::address(var_light_bodyid, var_1);
                var_55 = wp::load(var_54);
                var_56 = wp::address(var_xpos_in, var_0, var_55);
                var_58 = wp::load(var_56);
                var_57 = wp::copy(var_58);
                // light_xpos_out[worldid, lightid] = body_xpos + light_pos0[worldid % light_pos0.shape[0], lightid]       <L 929>
                var_59 = &(var_light_pos0.shape);
                var_62 = wp::load(var_59);
                var_61 = wp::extract(var_62, var_60);
                var_63 = wp::mod(var_0, var_61);
                var_64 = wp::address(var_light_pos0, var_63, var_1);
                var_66 = wp::load(var_64);
                var_65 = wp::add(var_57, var_66);
                wp::array_store(var_light_xpos_out, var_0, var_1, var_65);
            }
            if (!var_44) {
                // elif light_mode[lightid] == CamLightType.TRACKCOM:                             <L 930>
                var_67 = wp::address(var_light_mode, var_1);
                var_70 = wp::load(var_67);
                var_69 = (var_70 == var_68);
                if (var_69) {
                    // light_xdir_out[worldid, lightid] = light_dir0[worldid % light_dir0.shape[0], lightid]       <L 931>
                    var_71 = &(var_light_dir0.shape);
                    var_74 = wp::load(var_71);
                    var_73 = wp::extract(var_74, var_72);
                    var_75 = wp::mod(var_0, var_73);
                    var_76 = wp::address(var_light_dir0, var_75, var_1);
                    var_78 = wp::load(var_76);
                    var_77 = wp::copy(var_78);
                    wp::array_store(var_light_xdir_out, var_0, var_1, var_77);
                    // light_xpos_out[worldid, lightid] = (                                       <L 932>
                    // subtree_com_in[worldid, light_bodyid[lightid]] + light_poscom0[worldid % light_poscom0.shape[0], lightid]       <L 933>
                    var_79 = wp::address(var_light_bodyid, var_1);
                    var_80 = wp::load(var_79);
                    var_81 = wp::address(var_subtree_com_in, var_0, var_80);
                    var_82 = &(var_light_poscom0.shape);
                    var_85 = wp::load(var_82);
                    var_84 = wp::extract(var_85, var_83);
                    var_86 = wp::mod(var_0, var_84);
                    var_87 = wp::address(var_light_poscom0, var_86, var_1);
                    var_89 = wp::load(var_81);
                    var_90 = wp::load(var_87);
                    var_88 = wp::add(var_89, var_90);
                    // light_xpos_out[worldid, lightid] = (                                       <L 932>
                    wp::array_store(var_light_xpos_out, var_0, var_1, var_88);
                }
                if (!var_69) {
                    // elif light_mode[lightid] == CamLightType.TARGETBODY or light_mode[lightid] == CamLightType.TARGETBODYCOM:       <L 935>
                    var_92 = wp::address(var_light_mode, var_1);
                    var_95 = wp::load(var_92);
                    var_94 = (var_95 == var_93);
                    var_91 = var_94;
                    if (!var_91) {
                        var_96 = wp::address(var_light_mode, var_1);
                        var_99 = wp::load(var_96);
                        var_98 = (var_99 == var_97);
                        var_91 = var_91 || var_98;
                    }
                    if (var_91) {
                        // bodyid = light_bodyid[lightid]                                         <L 936>
                        var_100 = wp::address(var_light_bodyid, var_1);
                        var_102 = wp::load(var_100);
                        var_101 = wp::copy(var_102);
                        // xpos = xpos_in[worldid, bodyid]                                        <L 937>
                        var_103 = wp::address(var_xpos_in, var_0, var_101);
                        var_105 = wp::load(var_103);
                        var_104 = wp::copy(var_105);
                        // xquat = xquat_in[worldid, bodyid]                                      <L 938>
                        var_106 = wp::address(var_xquat_in, var_0, var_101);
                        var_108 = wp::load(var_106);
                        var_107 = wp::copy(var_108);
                        // light_xpos_out[worldid, lightid] = xpos + math.rot_vec_quat(light_pos[light_pos_id, lightid], xquat)       <L 939>
                        var_109 = wp::address(var_light_pos, var_6, var_1);
                        var_111 = wp::load(var_109);
                        var_110 = rot_vec_quat_0(var_111, var_107);
                        var_112 = wp::add(var_104, var_110);
                        wp::array_store(var_light_xpos_out, var_0, var_1, var_112);
                        // pos = xpos_in[worldid, light_targetbodyid[lightid]]                    <L 940>
                        var_113 = wp::address(var_light_targetbodyid, var_1);
                        var_114 = wp::load(var_113);
                        var_115 = wp::address(var_xpos_in, var_0, var_114);
                        var_117 = wp::load(var_115);
                        var_116 = wp::copy(var_117);
                        // if light_mode[lightid] == CamLightType.TARGETBODYCOM:                  <L 941>
                        var_118 = wp::address(var_light_mode, var_1);
                        var_121 = wp::load(var_118);
                        var_120 = (var_121 == var_119);
                        if (var_120) {
                            // pos = subtree_com_in[worldid, light_targetbodyid[lightid]]         <L 942>
                            var_122 = wp::address(var_light_targetbodyid, var_1);
                            var_123 = wp::load(var_122);
                            var_124 = wp::address(var_subtree_com_in, var_0, var_123);
                            var_126 = wp::load(var_124);
                            var_125 = wp::copy(var_126);
                        }
                        var_127 = wp::where(var_120, var_125, var_116);
                        // light_xdir_out[worldid, lightid] = pos - light_xpos_out[worldid, lightid]       <L 943>
                        var_128 = wp::address(var_light_xpos_out, var_0, var_1);
                        var_130 = wp::load(var_128);
                        var_129 = wp::sub(var_127, var_130);
                        wp::array_store(var_light_xdir_out, var_0, var_1, var_129);
                    }
                    if (!var_91) {
                        // bodyid = light_bodyid[lightid]                                         <L 945>
                        var_131 = wp::address(var_light_bodyid, var_1);
                        var_133 = wp::load(var_131);
                        var_132 = wp::copy(var_133);
                        // xpos = xpos_in[worldid, bodyid]                                        <L 946>
                        var_134 = wp::address(var_xpos_in, var_0, var_132);
                        var_136 = wp::load(var_134);
                        var_135 = wp::copy(var_136);
                        // xquat = xquat_in[worldid, bodyid]                                      <L 947>
                        var_137 = wp::address(var_xquat_in, var_0, var_132);
                        var_139 = wp::load(var_137);
                        var_138 = wp::copy(var_139);
                        // light_xpos_out[worldid, lightid] = xpos + math.rot_vec_quat(light_pos[light_pos_id, lightid], xquat)       <L 948>
                        var_140 = wp::address(var_light_pos, var_6, var_1);
                        var_142 = wp::load(var_140);
                        var_141 = rot_vec_quat_0(var_142, var_138);
                        var_143 = wp::add(var_135, var_141);
                        wp::array_store(var_light_xpos_out, var_0, var_1, var_143);
                        // light_xdir_out[worldid, lightid] = math.rot_vec_quat(light_dir[light_dir_id, lightid], xquat)       <L 949>
                        var_144 = wp::address(var_light_dir, var_11, var_1);
                        var_146 = wp::load(var_144);
                        var_145 = rot_vec_quat_0(var_146, var_138);
                        wp::array_store(var_light_xdir_out, var_0, var_1, var_145);
                    }
                    var_147 = wp::where(var_91, var_101, var_132);
                    var_148 = wp::where(var_91, var_104, var_135);
                    var_149 = wp::where(var_91, var_107, var_138);
                }
            }
        }
        var_150 = wp::where(var_21, var_27, var_147);
        var_151 = wp::where(var_21, var_30, var_148);
        var_152 = wp::where(var_21, var_33, var_149);
        // light_xdir_out[worldid, lightid] = wp.normalize(light_xdir_out[worldid, lightid])       <L 951>
        var_153 = wp::address(var_light_xdir_out, var_0, var_1);
        var_155 = wp::load(var_153);
        var_154 = wp::normalize(var_155);
        wp::array_store(var_light_xdir_out, var_0, var_1, var_154);
    }
}



extern "C" __global__ void _subtree_vel_forward_1d7ed456_cuda_kernel_forward(
    wp::launch_bounds_t<2> dim,
    wp::array_t<wp::int32> var_body_rootid,
    wp::array_t<wp::float32> var_body_mass,
    wp::array_t<wp::vec_t<3, wp::float32>> var_body_inertia,
    wp::array_t<wp::vec_t<3, wp::float32>> var_xipos_in,
    wp::array_t<wp::mat_t<3, 3, wp::float32>> var_ximat_in,
    wp::array_t<wp::vec_t<3, wp::float32>> var_subtree_com_in,
    wp::array_t<wp::vec_t<6, wp::float32>> var_cvel_in,
    wp::array_t<wp::vec_t<3, wp::float32>> var_subtree_linvel_out,
    wp::array_t<wp::vec_t<3, wp::float32>> var_subtree_angmom_out,
    wp::array_t<wp::vec_t<6, wp::float32>> var_subtree_bodyvel_out)
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
        wp::vec_t<6, wp::float32>* var_12;
        wp::vec_t<6, wp::float32> var_13;
        wp::vec_t<6, wp::float32> var_14;
        wp::vec_t<3, wp::float32> var_15;
        wp::vec_t<3, wp::float32> var_16;
        wp::vec_t<3, wp::float32>* var_17;
        wp::vec_t<3, wp::float32> var_18;
        wp::vec_t<3, wp::float32> var_19;
        wp::mat_t<3, 3, wp::float32>* var_20;
        wp::mat_t<3, 3, wp::float32> var_21;
        wp::mat_t<3, 3, wp::float32> var_22;
        wp::int32* var_23;
        wp::int32 var_24;
        wp::vec_t<3, wp::float32>* var_25;
        wp::vec_t<3, wp::float32> var_26;
        wp::vec_t<3, wp::float32> var_27;
        wp::vec_t<3, wp::float32> var_28;
        wp::vec_t<3, wp::float32> var_29;
        wp::vec_t<3, wp::float32> var_30;
        wp::float32* var_31;
        wp::vec_t<3, wp::float32> var_32;
        wp::float32 var_33;
        wp::mat_t<3, 3, wp::float32> var_34;
        wp::vec_t<3, wp::float32> var_35;
        const wp::int32 var_36 = 0;
        wp::float32 var_37;
        wp::vec_t<3, wp::float32>* var_38;
        const wp::int32 var_39 = 0;
        wp::float32 var_40;
        wp::vec_t<3, wp::float32> var_41;
        wp::float32 var_42;
        const wp::int32 var_43 = 1;
        wp::float32 var_44;
        wp::vec_t<3, wp::float32>* var_45;
        const wp::int32 var_46 = 1;
        wp::float32 var_47;
        wp::vec_t<3, wp::float32> var_48;
        wp::float32 var_49;
        const wp::int32 var_50 = 2;
        wp::float32 var_51;
        wp::vec_t<3, wp::float32>* var_52;
        const wp::int32 var_53 = 2;
        wp::float32 var_54;
        wp::vec_t<3, wp::float32> var_55;
        wp::float32 var_56;
        wp::vec_t<3, wp::float32> var_57;
        wp::vec_t<6, wp::float32> var_58;
        //---------
        // forward
        // def _subtree_vel_forward(                                                              <L 3831>
        // worldid, bodyid = wp.tid()                                                             <L 3847>
        builtin_tid2d(var_0, var_1);
        // body_mass_id = worldid % body_mass.shape[0]                                            <L 3848>
        var_2 = &(var_body_mass.shape);
        var_5 = wp::load(var_2);
        var_4 = wp::extract(var_5, var_3);
        var_6 = wp::mod(var_0, var_4);
        // body_inertia_id = worldid % body_inertia.shape[0]                                      <L 3849>
        var_7 = &(var_body_inertia.shape);
        var_10 = wp::load(var_7);
        var_9 = wp::extract(var_10, var_8);
        var_11 = wp::mod(var_0, var_9);
        // cvel = cvel_in[worldid, bodyid]                                                        <L 3851>
        var_12 = wp::address(var_cvel_in, var_0, var_1);
        var_14 = wp::load(var_12);
        var_13 = wp::copy(var_14);
        // ang = wp.spatial_top(cvel)                                                             <L 3852>
        var_15 = wp::spatial_top(var_13);
        // lin = wp.spatial_bottom(cvel)                                                          <L 3853>
        var_16 = wp::spatial_bottom(var_13);
        // xipos = xipos_in[worldid, bodyid]                                                      <L 3854>
        var_17 = wp::address(var_xipos_in, var_0, var_1);
        var_19 = wp::load(var_17);
        var_18 = wp::copy(var_19);
        // ximat = ximat_in[worldid, bodyid]                                                      <L 3855>
        var_20 = wp::address(var_ximat_in, var_0, var_1);
        var_22 = wp::load(var_20);
        var_21 = wp::copy(var_22);
        // subtree_com_root = subtree_com_in[worldid, body_rootid[bodyid]]                        <L 3856>
        var_23 = wp::address(var_body_rootid, var_1);
        var_24 = wp::load(var_23);
        var_25 = wp::address(var_subtree_com_in, var_0, var_24);
        var_27 = wp::load(var_25);
        var_26 = wp::copy(var_27);
        // lin -= wp.cross(xipos - subtree_com_root, ang)                                         <L 3859>
        var_28 = wp::sub(var_18, var_26);
        var_29 = wp::cross(var_28, var_15);
        var_30 = wp::sub(var_16, var_29);
        // subtree_linvel_out[worldid, bodyid] = body_mass[body_mass_id, bodyid] * lin            <L 3861>
        var_31 = wp::address(var_body_mass, var_6, var_1);
        var_33 = wp::load(var_31);
        var_32 = wp::mul(var_33, var_30);
        wp::array_store(var_subtree_linvel_out, var_0, var_1, var_32);
        // dv = wp.transpose(ximat) @ ang                                                         <L 3862>
        var_34 = wp::transpose(var_21);
        var_35 = wp::mul(var_34, var_15);
        // dv[0] *= body_inertia[body_inertia_id, bodyid][0]                                      <L 3863>
        var_37 = wp::extract(var_35, var_36);
        var_38 = wp::address(var_body_inertia, var_11, var_1);
        var_41 = wp::load(var_38);
        var_40 = wp::extract(var_41, var_39);
        var_42 = wp::mul(var_37, var_40);
        wp::assign_inplace(var_35, var_36, var_42);
        // dv[1] *= body_inertia[body_inertia_id, bodyid][1]                                      <L 3864>
        var_44 = wp::extract(var_35, var_43);
        var_45 = wp::address(var_body_inertia, var_11, var_1);
        var_48 = wp::load(var_45);
        var_47 = wp::extract(var_48, var_46);
        var_49 = wp::mul(var_44, var_47);
        wp::assign_inplace(var_35, var_43, var_49);
        // dv[2] *= body_inertia[body_inertia_id, bodyid][2]                                      <L 3865>
        var_51 = wp::extract(var_35, var_50);
        var_52 = wp::address(var_body_inertia, var_11, var_1);
        var_55 = wp::load(var_52);
        var_54 = wp::extract(var_55, var_53);
        var_56 = wp::mul(var_51, var_54);
        wp::assign_inplace(var_35, var_50, var_56);
        // subtree_angmom_out[worldid, bodyid] = ximat @ dv                                       <L 3866>
        var_57 = wp::mul(var_21, var_35);
        wp::array_store(var_subtree_angmom_out, var_0, var_1, var_57);
        // subtree_bodyvel_out[worldid, bodyid] = wp.spatial_vector(ang, lin)                     <L 3867>
        var_58 = wp::vec_t<6, wp::float32>(var_15, var_30);
        wp::array_store(var_subtree_bodyvel_out, var_0, var_1, var_58);
    }
}



extern "C" __global__ void _tendon_bias_coef_2556dfce_cuda_kernel_forward(
    wp::launch_bounds_t<3> dim,
    wp::array_t<wp::int32> var_ten_J_rownnz,
    wp::array_t<wp::int32> var_ten_J_rowadr,
    wp::array_t<wp::int32> var_ten_J_colind,
    wp::array_t<wp::float32> var_tendon_armature,
    wp::array_t<wp::float32> var_qvel_in,
    wp::array_t<wp::float32> var_ten_Jdot_in,
    wp::array_t<wp::float32> var_ten_bias_coef_out)
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
        wp::shape_t* var_3;
        const wp::int32 var_4 = 0;
        wp::int32 var_5;
        wp::shape_t var_6;
        wp::int32 var_7;
        wp::float32* var_8;
        wp::float32 var_9;
        wp::float32 var_10;
        const wp::float32 var_11 = 0.0;
        bool var_12;
        wp::int32* var_13;
        wp::int32 var_14;
        wp::int32 var_15;
        bool var_16;
        wp::int32* var_17;
        wp::int32 var_18;
        wp::int32 var_19;
        wp::int32 var_20;
        wp::float32* var_21;
        wp::float32 var_22;
        wp::float32 var_23;
        const wp::float32 var_24 = 0.0;
        bool var_25;
        wp::int32* var_26;
        wp::int32 var_27;
        wp::int32 var_28;
        wp::array_t<wp::float32> var_29;
        wp::float32* var_30;
        wp::float32 var_31;
        wp::float32 var_32;
        wp::float32 var_33;
        //---------
        // forward
        // def _tendon_bias_coef(                                                                 <L 2383>
        // worldid, tenid, dofid_sparse = wp.tid()                                                <L 2396>
        builtin_tid3d(var_0, var_1, var_2);
        // armature = tendon_armature[worldid % tendon_armature.shape[0], tenid]                  <L 2398>
        var_3 = &(var_tendon_armature.shape);
        var_6 = wp::load(var_3);
        var_5 = wp::extract(var_6, var_4);
        var_7 = wp::mod(var_0, var_5);
        var_8 = wp::address(var_tendon_armature, var_7, var_1);
        var_10 = wp::load(var_8);
        var_9 = wp::copy(var_10);
        // if armature == 0.0:                                                                    <L 2399>
        var_12 = (var_9 == var_11);
        if (var_12) {
            // return                                                                             <L 2400>
            continue;
        }
        // rownnz = ten_J_rownnz[tenid]                                                           <L 2402>
        var_13 = wp::address(var_ten_J_rownnz, var_1);
        var_15 = wp::load(var_13);
        var_14 = wp::copy(var_15);
        // if dofid_sparse >= rownnz:                                                             <L 2403>
        var_16 = (var_2 >= var_14);
        if (var_16) {
            // return                                                                             <L 2404>
            continue;
        }
        // rowadr = ten_J_rowadr[tenid]                                                           <L 2405>
        var_17 = wp::address(var_ten_J_rowadr, var_1);
        var_19 = wp::load(var_17);
        var_18 = wp::copy(var_19);
        // sparseid = rowadr + dofid_sparse                                                       <L 2406>
        var_20 = wp::add(var_18, var_2);
        // ten_Jdot = ten_Jdot_in[worldid, sparseid]                                              <L 2407>
        var_21 = wp::address(var_ten_Jdot_in, var_0, var_20);
        var_23 = wp::load(var_21);
        var_22 = wp::copy(var_23);
        // if ten_Jdot == 0.0:                                                                    <L 2408>
        var_25 = (var_22 == var_24);
        if (var_25) {
            // return                                                                             <L 2409>
            continue;
        }
        // dofid = ten_J_colind[sparseid]                                                         <L 2411>
        var_26 = wp::address(var_ten_J_colind, var_20);
        var_28 = wp::load(var_26);
        var_27 = wp::copy(var_28);
        // wp.atomic_add(ten_bias_coef_out[worldid], tenid, ten_Jdot * qvel_in[worldid, dofid])       <L 2412>
        var_29 = wp::view(var_ten_bias_coef_out, var_0);
        var_30 = wp::address(var_qvel_in, var_0, var_27);
        var_32 = wp::load(var_30);
        var_31 = wp::mul(var_22, var_32);
        var_33 = wp::atomic_add(var_29, var_1, var_31);
    }
}



extern "C" __global__ void _compute_body_matrices_27760750_cuda_kernel_forward(
    wp::launch_bounds_t<2> dim,
    wp::array_t<wp::quat_t<wp::float32>> var_xquat_in,
    wp::array_t<wp::mat_t<3, 3, wp::float32>> var_xmat_out)
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
        wp::quat_t<wp::float32>* var_2;
        wp::mat_t<3, 3, wp::float32> var_3;
        wp::quat_t<wp::float32> var_4;
        //---------
        // forward
        // def _compute_body_matrices(                                                            <L 169>
        // worldid, bodyid = wp.tid()                                                             <L 175>
        builtin_tid2d(var_0, var_1);
        // xmat_out[worldid, bodyid] = math.quat_to_mat(xquat_in[worldid, bodyid])                <L 176>
        var_2 = wp::address(var_xquat_in, var_0, var_1);
        var_4 = wp::load(var_2);
        var_3 = quat_to_mat_0(var_4);
        wp::array_store(var_xmat_out, var_0, var_1, var_3);
    }
}



extern "C" __global__ void _cfrc_ext_spatial_tendon_294ed7a8_cuda_kernel_forward(
    wp::launch_bounds_t<2> dim,
    wp::array_t<wp::int32> var_body_rootid,
    wp::array_t<wp::int32> var_geom_bodyid,
    wp::array_t<wp::int32> var_site_bodyid,
    wp::array_t<wp::int32> var_tendon_adr,
    wp::array_t<wp::int32> var_tendon_num,
    wp::array_t<wp::int32> var_ten_J_rownnz,
    wp::array_t<wp::int32> var_ten_J_rowadr,
    wp::array_t<wp::int32> var_ten_J_colind,
    wp::array_t<wp::float32> var_tendon_stiffness,
    wp::array_t<wp::vec_t<2, wp::float32>> var_tendon_stiffnesspoly,
    wp::array_t<wp::float32> var_tendon_damping,
    wp::array_t<wp::vec_t<2, wp::float32>> var_tendon_dampingpoly,
    wp::array_t<wp::float32> var_tendon_armature,
    wp::array_t<wp::vec_t<2, wp::float32>> var_tendon_lengthspring,
    wp::array_t<wp::int32> var_wrap_type,
    wp::array_t<wp::int32> var_wrap_objid,
    wp::array_t<wp::float32> var_wrap_prm,
    wp::array_t<wp::float32> var_qacc_in,
    wp::array_t<wp::vec_t<3, wp::float32>> var_subtree_com_in,
    wp::array_t<wp::int32> var_ten_wrapadr_in,
    wp::array_t<wp::int32> var_ten_wrapnum_in,
    wp::array_t<wp::float32> var_ten_J_in,
    wp::array_t<wp::float32> var_ten_length_in,
    wp::array_t<wp::vec_t<2, wp::int32>> var_wrap_obj_in,
    wp::array_t<wp::vec_t<6, wp::float32>> var_wrap_xpos_in,
    wp::array_t<wp::float32> var_ten_velocity_in,
    wp::array_t<wp::float32> var_ten_bias_coef_in,
    bool var_dsbl_spring,
    bool var_dsbl_damper,
    wp::array_t<wp::float32> var_ten_frc_in,
    wp::array_t<wp::vec_t<6, wp::float32>> var_cfrc_ext_out)
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
        wp::int32* var_5;
        const wp::int32 var_6 = 1;
        bool var_7;
        wp::int32 var_8;
        wp::float32* var_9;
        wp::float32 var_10;
        wp::float32 var_11;
        bool var_12;
        wp::shape_t* var_13;
        const wp::int32 var_14 = 0;
        wp::int32 var_15;
        wp::shape_t var_16;
        wp::int32 var_17;
        wp::float32* var_18;
        wp::float32 var_19;
        wp::float32 var_20;
        wp::shape_t* var_21;
        const wp::int32 var_22 = 0;
        wp::int32 var_23;
        wp::shape_t var_24;
        wp::int32 var_25;
        wp::vec_t<2, wp::float32>* var_26;
        wp::vec_t<2, wp::float32> var_27;
        wp::vec_t<2, wp::float32> var_28;
        bool var_29;
        const wp::float32 var_30 = 0.0;
        bool var_31;
        const wp::int32 var_32 = 0;
        wp::float32 var_33;
        const wp::float32 var_34 = 0.0;
        bool var_35;
        const wp::int32 var_36 = 1;
        wp::float32 var_37;
        const wp::float32 var_38 = 0.0;
        bool var_39;
        wp::float32* var_40;
        wp::float32 var_41;
        wp::float32 var_42;
        wp::shape_t* var_43;
        const wp::int32 var_44 = 0;
        wp::int32 var_45;
        wp::shape_t var_46;
        wp::int32 var_47;
        wp::vec_t<2, wp::float32>* var_48;
        wp::vec_t<2, wp::float32> var_49;
        wp::vec_t<2, wp::float32> var_50;
        const wp::int32 var_51 = 0;
        wp::float32 var_52;
        const wp::int32 var_53 = 1;
        wp::float32 var_54;
        bool var_55;
        wp::float32 var_56;
        bool var_57;
        wp::float32 var_58;
        const wp::float32 var_59 = 0.0;
        wp::float32 var_60;
        wp::float32 var_61;
        wp::float32 var_62;
        const wp::int32 var_63 = 0;
        wp::float32 var_64;
        wp::float32 var_65;
        wp::float32 var_66;
        wp::float32 var_67;
        wp::float32 var_68;
        bool var_69;
        wp::shape_t* var_70;
        const wp::int32 var_71 = 0;
        wp::int32 var_72;
        wp::shape_t var_73;
        wp::int32 var_74;
        wp::float32* var_75;
        wp::float32 var_76;
        wp::float32 var_77;
        wp::shape_t* var_78;
        const wp::int32 var_79 = 0;
        wp::int32 var_80;
        wp::shape_t var_81;
        wp::int32 var_82;
        wp::vec_t<2, wp::float32>* var_83;
        wp::vec_t<2, wp::float32> var_84;
        wp::vec_t<2, wp::float32> var_85;
        bool var_86;
        const wp::float32 var_87 = 0.0;
        bool var_88;
        const wp::int32 var_89 = 0;
        wp::float32 var_90;
        const wp::float32 var_91 = 0.0;
        bool var_92;
        const wp::int32 var_93 = 1;
        wp::float32 var_94;
        const wp::float32 var_95 = 0.0;
        bool var_96;
        wp::float32* var_97;
        wp::float32 var_98;
        wp::float32 var_99;
        wp::float32 var_100;
        const wp::int32 var_101 = 1;
        wp::float32 var_102;
        wp::float32 var_103;
        wp::float32 var_104;
        wp::float32 var_105;
        wp::float32 var_106;
        wp::shape_t* var_107;
        const wp::int32 var_108 = 0;
        wp::int32 var_109;
        wp::shape_t var_110;
        wp::int32 var_111;
        wp::float32* var_112;
        wp::float32 var_113;
        wp::float32 var_114;
        const wp::float32 var_115 = 0.0;
        bool var_116;
        wp::float32* var_117;
        wp::float32 var_118;
        wp::float32 var_119;
        wp::int32* var_120;
        wp::int32 var_121;
        wp::int32 var_122;
        wp::int32* var_123;
        wp::int32 var_124;
        wp::int32 var_125;
        wp::range_t var_126;
        wp::int32 var_127;
        wp::int32 var_128;
        wp::float32* var_129;
        wp::int32* var_130;
        wp::int32 var_131;
        wp::float32* var_132;
        wp::float32 var_133;
        wp::float32 var_134;
        wp::float32 var_135;
        wp::float32 var_136;
        wp::float32 var_137;
        wp::float32 var_138;
        wp::float32 var_139;
        const wp::float32 var_140 = 0.0;
        bool var_141;
        wp::int32* var_142;
        wp::int32 var_143;
        wp::int32 var_144;
        wp::int32* var_145;
        wp::int32 var_146;
        wp::int32 var_147;
        wp::int32* var_148;
        wp::int32 var_149;
        wp::int32 var_150;
        wp::float32 var_151;
        const wp::int32 var_152 = -1;
        wp::int32 var_153;
        const wp::float32 var_154 = 0.0;
        wp::vec_t<3, wp::float32> var_155;
        const wp::float32 var_156 = 0.0;
        wp::vec_t<3, wp::float32> var_157;
        wp::range_t var_158;
        wp::int32 var_159;
        wp::int32 var_160;
        wp::int32* var_161;
        wp::int32 var_162;
        wp::int32 var_163;
        wp::int32* var_164;
        wp::int32 var_165;
        wp::int32 var_166;
        const wp::int32 var_167 = 2;
        bool var_168;
        wp::float32* var_169;
        wp::float32 var_170;
        wp::float32 var_171;
        const wp::int32 var_172 = -1;
        const wp::int32 var_173 = 1;
        wp::int32 var_174;
        const wp::int32 var_175 = 0;
        wp::int32 var_176;
        const wp::int32 var_177 = 0;
        wp::int32 var_178;
        const wp::int32 var_179 = 3;
        bool var_180;
        const wp::int32 var_181 = 1;
        wp::int32* var_182;
        wp::int32 var_183;
        wp::int32 var_184;
        wp::int32* var_185;
        wp::int32 var_186;
        wp::int32 var_187;
        bool var_188;
        bool var_189;
        const wp::int32 var_190 = 2;
        wp::int32 var_191;
        wp::vec_t<2, wp::int32>* var_192;
        const wp::int32 var_193 = 2;
        wp::int32 var_194;
        wp::int32 var_195;
        wp::vec_t<2, wp::int32> var_196;
        bool var_197;
        const wp::int32 var_198 = 2;
        wp::int32 var_199;
        wp::int32 var_200;
        wp::int32 var_201;
        const wp::float32 var_202 = 0.0;
        wp::vec_t<3, wp::float32> var_203;
        bool var_204;
        const wp::int32 var_205 = 0;
        bool var_206;
        const wp::int32 var_207 = 0;
        bool var_208;
        wp::int32* var_209;
        wp::int32 var_210;
        wp::vec_t<3, wp::float32>* var_211;
        wp::vec_t<3, wp::float32> var_212;
        wp::vec_t<3, wp::float32> var_213;
        wp::vec_t<3, wp::float32> var_214;
        const wp::int32 var_215 = 0;
        const bool var_216 = true;
        bool var_217;
        const bool var_218 = false;
        bool var_219;
        const wp::int32 var_220 = 2;
        wp::int32 var_221;
        wp::vec_t<6, wp::float32>* var_222;
        wp::vec_t<6, wp::float32> var_223;
        wp::vec_t<6, wp::float32> var_224;
        const wp::int32 var_225 = 2;
        wp::int32 var_226;
        const wp::int32 var_227 = 0;
        bool var_228;
        wp::vec_t<3, wp::float32> var_229;
        wp::vec_t<3, wp::float32> var_230;
        wp::vec_t<3, wp::float32> var_231;
        bool var_232;
        const wp::int32 var_233 = 0;
        bool var_234;
        bool var_235;
        wp::vec_t<3, wp::float32> var_236;
        wp::vec_t<3, wp::float32> var_237;
        wp::float32 var_238;
        const wp::float32 var_239 = 0.0;
        bool var_240;
        wp::vec_t<3, wp::float32> var_241;
        const wp::int32 var_242 = 0;
        bool var_243;
        wp::array_t<wp::vec_t<6, wp::float32>> var_244;
        wp::vec_t<3, wp::float32> var_245;
        wp::vec_t<3, wp::float32> var_246;
        wp::vec_t<3, wp::float32> var_247;
        wp::vec_t<6, wp::float32> var_248;
        wp::vec_t<6, wp::float32> var_249;
        const wp::int32 var_250 = 0;
        bool var_251;
        wp::array_t<wp::vec_t<6, wp::float32>> var_252;
        wp::vec_t<3, wp::float32> var_253;
        wp::vec_t<3, wp::float32> var_254;
        wp::vec_t<3, wp::float32> var_255;
        wp::vec_t<6, wp::float32> var_256;
        wp::vec_t<6, wp::float32> var_257;
        wp::int32 var_258;
        wp::vec_t<3, wp::float32> var_259;
        wp::vec_t<3, wp::float32> var_260;
        const wp::int32 var_261 = 1;
        wp::int32 var_262;
        wp::int32 var_263;
        wp::int32 var_264;
        wp::vec_t<3, wp::float32> var_265;
        wp::vec_t<3, wp::float32> var_266;
        const wp::int32 var_267 = 1;
        const bool var_268 = true;
        bool var_269;
        const bool var_270 = false;
        bool var_271;
        const wp::int32 var_272 = 2;
        wp::int32 var_273;
        wp::vec_t<6, wp::float32>* var_274;
        wp::vec_t<6, wp::float32> var_275;
        wp::vec_t<6, wp::float32> var_276;
        const wp::int32 var_277 = 2;
        wp::int32 var_278;
        const wp::int32 var_279 = 0;
        bool var_280;
        wp::vec_t<3, wp::float32> var_281;
        wp::vec_t<3, wp::float32> var_282;
        wp::vec_t<3, wp::float32> var_283;
        bool var_284;
        const wp::int32 var_285 = 0;
        bool var_286;
        bool var_287;
        wp::vec_t<3, wp::float32> var_288;
        wp::vec_t<3, wp::float32> var_289;
        wp::float32 var_290;
        const wp::float32 var_291 = 0.0;
        bool var_292;
        wp::vec_t<3, wp::float32> var_293;
        const wp::int32 var_294 = 0;
        bool var_295;
        wp::array_t<wp::vec_t<6, wp::float32>> var_296;
        wp::vec_t<3, wp::float32> var_297;
        wp::vec_t<3, wp::float32> var_298;
        wp::vec_t<3, wp::float32> var_299;
        wp::vec_t<6, wp::float32> var_300;
        wp::vec_t<6, wp::float32> var_301;
        const wp::int32 var_302 = 0;
        bool var_303;
        wp::array_t<wp::vec_t<6, wp::float32>> var_304;
        wp::vec_t<3, wp::float32> var_305;
        wp::vec_t<3, wp::float32> var_306;
        wp::vec_t<3, wp::float32> var_307;
        wp::vec_t<6, wp::float32> var_308;
        wp::vec_t<6, wp::float32> var_309;
        wp::vec_t<3, wp::float32> var_310;
        wp::vec_t<3, wp::float32> var_311;
        wp::vec_t<3, wp::float32> var_312;
        wp::float32 var_313;
        wp::vec_t<3, wp::float32> var_314;
        wp::int32 var_315;
        wp::vec_t<3, wp::float32> var_316;
        wp::vec_t<3, wp::float32> var_317;
        const wp::int32 var_318 = 1;
        wp::int32 var_319;
        wp::int32 var_320;
        wp::int32 var_321;
        wp::vec_t<3, wp::float32> var_322;
        wp::vec_t<3, wp::float32> var_323;
        wp::vec_t<6, wp::float32> var_324;
        wp::vec_t<3, wp::float32> var_325;
        wp::vec_t<3, wp::float32> var_326;
        wp::vec_t<3, wp::float32> var_327;
        wp::float32 var_328;
        wp::vec_t<3, wp::float32> var_329;
        //---------
        // forward
        // def _cfrc_ext_spatial_tendon(                                                          <L 1815>
        // worldid, tenid = wp.tid()                                                              <L 1852>
        builtin_tid2d(var_0, var_1);
        // adr = tendon_adr[tenid]                                                                <L 1855>
        var_2 = wp::address(var_tendon_adr, var_1);
        var_4 = wp::load(var_2);
        var_3 = wp::copy(var_4);
        // if wrap_type[adr] == WrapType.JOINT:                                                   <L 1856>
        var_5 = wp::address(var_wrap_type, var_3);
        var_8 = wp::load(var_5);
        var_7 = (var_8 == var_6);
        if (var_7) {
            // return                                                                             <L 1857>
            continue;
        }
        // frc = ten_frc_in[worldid, tenid]                                                       <L 1859>
        var_9 = wp::address(var_ten_frc_in, var_0, var_1);
        var_11 = wp::load(var_9);
        var_10 = wp::copy(var_11);
        // if not dsbl_spring:                                                                    <L 1862>
        var_12 = wp::unot(var_dsbl_spring);
        if (var_12) {
            // stiffness = tendon_stiffness[worldid % tendon_stiffness.shape[0], tenid]           <L 1863>
            var_13 = &(var_tendon_stiffness.shape);
            var_16 = wp::load(var_13);
            var_15 = wp::extract(var_16, var_14);
            var_17 = wp::mod(var_0, var_15);
            var_18 = wp::address(var_tendon_stiffness, var_17, var_1);
            var_20 = wp::load(var_18);
            var_19 = wp::copy(var_20);
            // spoly = tendon_stiffnesspoly[worldid % tendon_stiffnesspoly.shape[0], tenid]       <L 1864>
            var_21 = &(var_tendon_stiffnesspoly.shape);
            var_24 = wp::load(var_21);
            var_23 = wp::extract(var_24, var_22);
            var_25 = wp::mod(var_0, var_23);
            var_26 = wp::address(var_tendon_stiffnesspoly, var_25, var_1);
            var_28 = wp::load(var_26);
            var_27 = wp::copy(var_28);
            // if stiffness != 0.0 or spoly[0] != 0.0 or spoly[1] != 0.0:                         <L 1865>
            var_31 = (var_19 != var_30);
            var_29 = var_31;
            if (!var_29) {
                var_33 = wp::extract(var_27, var_32);
                var_35 = (var_33 != var_34);
                var_29 = var_29 || var_35;
            }
            if (!var_29) {
                var_37 = wp::extract(var_27, var_36);
                var_39 = (var_37 != var_38);
                var_29 = var_29 || var_39;
            }
            if (var_29) {
                // length = ten_length_in[worldid, tenid]                                         <L 1866>
                var_40 = wp::address(var_ten_length_in, var_0, var_1);
                var_42 = wp::load(var_40);
                var_41 = wp::copy(var_42);
                // lengthspring = tendon_lengthspring[worldid % tendon_lengthspring.shape[0], tenid]       <L 1867>
                var_43 = &(var_tendon_lengthspring.shape);
                var_46 = wp::load(var_43);
                var_45 = wp::extract(var_46, var_44);
                var_47 = wp::mod(var_0, var_45);
                var_48 = wp::address(var_tendon_lengthspring, var_47, var_1);
                var_50 = wp::load(var_48);
                var_49 = wp::copy(var_50);
                // lower = lengthspring[0]                                                        <L 1868>
                var_52 = wp::extract(var_49, var_51);
                // upper = lengthspring[1]                                                        <L 1869>
                var_54 = wp::extract(var_49, var_53);
                // x = wp.where(length > upper, length - upper, wp.where(length < lower, length - lower, 0.0))       <L 1870>
                var_55 = (var_41 > var_54);
                var_56 = wp::sub(var_41, var_54);
                var_57 = (var_41 < var_52);
                var_58 = wp::sub(var_41, var_52);
                var_60 = wp::where(var_57, var_58, var_59);
                var_61 = wp::where(var_55, var_56, var_60);
                // frc += -x * util_misc._poly_force(stiffness, spoly, x, 0)                      <L 1871>
                var_62 = wp::neg(var_61);
                var_64 = _poly_force_0(var_19, var_27, var_61, var_63);
                var_65 = wp::mul(var_62, var_64);
                var_66 = wp::add(var_10, var_65);
            }
            var_67 = wp::where(var_29, var_66, var_10);
        }
        var_68 = wp::where(var_12, var_67, var_10);
        // if not dsbl_damper:                                                                    <L 1873>
        var_69 = wp::unot(var_dsbl_damper);
        if (var_69) {
            // damping = tendon_damping[worldid % tendon_damping.shape[0], tenid]                 <L 1874>
            var_70 = &(var_tendon_damping.shape);
            var_73 = wp::load(var_70);
            var_72 = wp::extract(var_73, var_71);
            var_74 = wp::mod(var_0, var_72);
            var_75 = wp::address(var_tendon_damping, var_74, var_1);
            var_77 = wp::load(var_75);
            var_76 = wp::copy(var_77);
            // dpoly = tendon_dampingpoly[worldid % tendon_dampingpoly.shape[0], tenid]           <L 1875>
            var_78 = &(var_tendon_dampingpoly.shape);
            var_81 = wp::load(var_78);
            var_80 = wp::extract(var_81, var_79);
            var_82 = wp::mod(var_0, var_80);
            var_83 = wp::address(var_tendon_dampingpoly, var_82, var_1);
            var_85 = wp::load(var_83);
            var_84 = wp::copy(var_85);
            // if damping != 0.0 or dpoly[0] != 0.0 or dpoly[1] != 0.0:                           <L 1876>
            var_88 = (var_76 != var_87);
            var_86 = var_88;
            if (!var_86) {
                var_90 = wp::extract(var_84, var_89);
                var_92 = (var_90 != var_91);
                var_86 = var_86 || var_92;
            }
            if (!var_86) {
                var_94 = wp::extract(var_84, var_93);
                var_96 = (var_94 != var_95);
                var_86 = var_86 || var_96;
            }
            if (var_86) {
                // v = ten_velocity_in[worldid, tenid]                                            <L 1877>
                var_97 = wp::address(var_ten_velocity_in, var_0, var_1);
                var_99 = wp::load(var_97);
                var_98 = wp::copy(var_99);
                // frc += -v * util_misc._poly_force(damping, dpoly, v, 1)                        <L 1878>
                var_100 = wp::neg(var_98);
                var_102 = _poly_force_0(var_76, var_84, var_98, var_101);
                var_103 = wp::mul(var_100, var_102);
                var_104 = wp::add(var_68, var_103);
            }
            var_105 = wp::where(var_86, var_104, var_68);
        }
        var_106 = wp::where(var_69, var_105, var_68);
        // armature = tendon_armature[worldid % tendon_armature.shape[0], tenid]                  <L 1881>
        var_107 = &(var_tendon_armature.shape);
        var_110 = wp::load(var_107);
        var_109 = wp::extract(var_110, var_108);
        var_111 = wp::mod(var_0, var_109);
        var_112 = wp::address(var_tendon_armature, var_111, var_1);
        var_114 = wp::load(var_112);
        var_113 = wp::copy(var_114);
        // if armature != 0.0:                                                                    <L 1882>
        var_116 = (var_113 != var_115);
        if (var_116) {
            // acc = ten_bias_coef_in[worldid, tenid]                                             <L 1883>
            var_117 = wp::address(var_ten_bias_coef_in, var_0, var_1);
            var_119 = wp::load(var_117);
            var_118 = wp::copy(var_119);
            // rownnz = ten_J_rownnz[tenid]                                                       <L 1884>
            var_120 = wp::address(var_ten_J_rownnz, var_1);
            var_122 = wp::load(var_120);
            var_121 = wp::copy(var_122);
            // rowadr = ten_J_rowadr[tenid]                                                       <L 1885>
            var_123 = wp::address(var_ten_J_rowadr, var_1);
            var_125 = wp::load(var_123);
            var_124 = wp::copy(var_125);
            // for i in range(rownnz):                                                            <L 1886>
            var_126 = wp::range(var_121);
            start_for_1:;
                if (iter_cmp(var_126) == 0) goto end_for_1;
                var_127 = wp::iter_next(var_126);
                // sparseid = rowadr + i                                                          <L 1887>
                var_128 = wp::add(var_124, var_127);
                // acc += ten_J_in[worldid, sparseid] * qacc_in[worldid, ten_J_colind[sparseid]]       <L 1888>
                var_129 = wp::address(var_ten_J_in, var_0, var_128);
                var_130 = wp::address(var_ten_J_colind, var_128);
                var_131 = wp::load(var_130);
                var_132 = wp::address(var_qacc_in, var_0, var_131);
                var_134 = wp::load(var_129);
                var_135 = wp::load(var_132);
                var_133 = wp::mul(var_134, var_135);
                var_136 = wp::add(var_118, var_133);
                wp::assign(var_118, var_136);
                goto start_for_1;
            end_for_1:;
            // frc -= armature * acc                                                              <L 1889>
            var_137 = wp::mul(var_113, var_118);
            var_138 = wp::sub(var_106, var_137);
        }
        var_139 = wp::where(var_116, var_138, var_106);
        // if frc == 0.0:                                                                         <L 1891>
        var_141 = (var_139 == var_140);
        if (var_141) {
            // return                                                                             <L 1892>
            continue;
        }
        // num = tendon_num[tenid]                                                                <L 1894>
        var_142 = wp::address(var_tendon_num, var_1);
        var_144 = wp::load(var_142);
        var_143 = wp::copy(var_144);
        // p = ten_wrapadr_in[worldid, tenid]                                                     <L 1895>
        var_145 = wp::address(var_ten_wrapadr_in, var_0, var_1);
        var_147 = wp::load(var_145);
        var_146 = wp::copy(var_147);
        // pend = p + ten_wrapnum_in[worldid, tenid]                                              <L 1896>
        var_148 = wp::address(var_ten_wrapnum_in, var_0, var_1);
        var_150 = wp::load(var_148);
        var_149 = wp::add(var_146, var_150);
        // scaled_frc = frc                                                                       <L 1897>
        var_151 = wp::copy(var_139);
        // prevbody = int(-1)                                                                     <L 1898>
        var_153 = wp::int(var_152);
        // prevpnt = wp.vec3(0.0)                                                                 <L 1899>
        var_155 = wp::vec_t<3, wp::float32>(var_154);
        // prevcom = wp.vec3(0.0)                                                                 <L 1900>
        var_157 = wp::vec_t<3, wp::float32>(var_156);
        // for j_offset in range(num):                                                            <L 1902>
        var_158 = wp::range(var_143);
        start_for_4:;
            if (iter_cmp(var_158) == 0) goto end_for_4;
            var_159 = wp::iter_next(var_158);
            // j = adr + j_offset                                                                 <L 1903>
            var_160 = wp::add(var_3, var_159);
            // wtype = wrap_type[j]                                                               <L 1904>
            var_161 = wp::address(var_wrap_type, var_160);
            var_163 = wp::load(var_161);
            var_162 = wp::copy(var_163);
            // objid = wrap_objid[j]                                                              <L 1905>
            var_164 = wp::address(var_wrap_objid, var_160);
            var_166 = wp::load(var_164);
            var_165 = wp::copy(var_166);
            // if wtype == WrapType.PULLEY:                                                       <L 1908>
            var_168 = (var_162 == var_167);
            if (var_168) {
                // scaled_frc = frc / wrap_prm[j]                                                 <L 1909>
                var_169 = wp::address(var_wrap_prm, var_160);
                var_171 = wp::load(var_169);
                var_170 = wp::div(var_139, var_171);
                // prevbody = -1                                                                  <L 1910>
                // p += 1                                                                         <L 1911>
                var_174 = wp::add(var_146, var_173);
                // continue                                                                       <L 1912>
                wp::assign(var_146, var_174);
                wp::assign(var_151, var_170);
                wp::assign(var_153, var_172);
                goto start_for_4;
            }
            // npnt = int(0)                                                                      <L 1915>
            var_176 = wp::int(var_175);
            // body = int(0)                                                                      <L 1916>
            var_178 = wp::int(var_177);
            // if wtype == WrapType.SITE:                                                         <L 1917>
            var_180 = (var_162 == var_179);
            if (var_180) {
                // npnt = 1                                                                       <L 1918>
                // body = site_bodyid[objid]                                                      <L 1919>
                var_182 = wp::address(var_site_bodyid, var_165);
                var_184 = wp::load(var_182);
                var_183 = wp::copy(var_184);
            }
            if (!var_180) {
                // body = geom_bodyid[objid]                                                      <L 1921>
                var_185 = wp::address(var_geom_bodyid, var_165);
                var_187 = wp::load(var_185);
                var_186 = wp::copy(var_187);
                // if p < pend and wrap_obj_in[worldid, p // 2][p % 2] == objid:                  <L 1922>
                var_189 = (var_146 < var_149);
                var_188 = var_189;
                if (var_188) {
                    var_191 = wp::floordiv(var_146, var_190);
                    var_192 = wp::address(var_wrap_obj_in, var_0, var_191);
                    var_194 = wp::mod(var_146, var_193);
                    var_196 = wp::load(var_192);
                    var_195 = wp::extract(var_196, var_194);
                    var_197 = (var_195 == var_165);
                    var_188 = var_188 && var_197;
                }
                if (var_188) {
                    // npnt = 2                                                                   <L 1923>
                }
                var_199 = wp::where(var_188, var_198, var_176);
            }
            var_200 = wp::where(var_180, var_181, var_199);
            var_201 = wp::where(var_180, var_183, var_186);
            // com = wp.vec3(0.0)                                                                 <L 1925>
            var_203 = wp::vec_t<3, wp::float32>(var_202);
            // if npnt > 0 and body != 0:                                                         <L 1926>
            var_206 = (var_200 > var_205);
            var_204 = var_206;
            if (var_204) {
                var_208 = (var_201 != var_207);
                var_204 = var_204 && var_208;
            }
            if (var_204) {
                // com = subtree_com_in[worldid, body_rootid[body]]                               <L 1927>
                var_209 = wp::address(var_body_rootid, var_201);
                var_210 = wp::load(var_209);
                var_211 = wp::address(var_subtree_com_in, var_0, var_210);
                var_213 = wp::load(var_211);
                var_212 = wp::copy(var_213);
            }
            var_214 = wp::where(var_204, var_212, var_203);
            // for k in range(2):                                                                 <L 1929>
            // if k >= npnt:                                                                      <L 1930>
            var_217 = (var_215 >= var_200);
            if (var_217) {
                // continue                                                                       <L 1931>
            }
            var_219 = wp::where(var_217, var_218, var_216);
            // wrap_pos = wrap_xpos_in[worldid, p // 2]                                           <L 1933>
            if (var_219) {
                var_221 = wp::floordiv(var_146, var_220);
                var_222 = wp::address(var_wrap_xpos_in, var_0, var_221);
                var_224 = wp::load(var_222);
                var_223 = wp::copy(var_224);
                // pnt = wp.where(p % 2 == 0, wp.spatial_top(wrap_pos), wp.spatial_bottom(wrap_pos))       <L 1934>
                var_226 = wp::mod(var_146, var_225);
                var_228 = (var_226 == var_227);
                var_229 = wp::spatial_top(var_223);
                var_230 = wp::spatial_bottom(var_223);
                var_231 = wp::where(var_228, var_229, var_230);
                // if prevbody >= 0 and body != prevbody:                                         <L 1935>
                var_234 = (var_153 >= var_233);
                var_232 = var_234;
                if (var_232) {
                    var_235 = (var_201 != var_153);
                    var_232 = var_232 && var_235;
                }
                if (var_232) {
                    // diff = pnt - prevpnt                                                       <L 1936>
                    var_236 = wp::sub(var_231, var_155);
                    // vec, norm = math.normalize_with_norm(diff)                                 <L 1937>
                    normalize_with_norm_0(var_236, var_237, var_238);
                    // if norm > 0.0:                                                             <L 1938>
                    var_240 = (var_238 > var_239);
                    if (var_240) {
                        // force = vec * scaled_frc                                               <L 1939>
                        var_241 = wp::mul(var_237, var_151);
                        // if body != 0:                                                          <L 1940>
                        var_243 = (var_201 != var_242);
                        if (var_243) {
                            // wp.atomic_add(cfrc_ext_out[worldid], body, wp.spatial_vector(-wp.cross(com - pnt, force), force))       <L 1941>
                            var_244 = wp::view(var_cfrc_ext_out, var_0);
                            var_245 = wp::sub(var_214, var_231);
                            var_246 = wp::cross(var_245, var_241);
                            var_247 = wp::neg(var_246);
                            var_248 = wp::vec_t<6, wp::float32>(var_247, var_241);
                            var_249 = wp::atomic_add(var_244, var_201, var_248);
                        }
                        // if prevbody != 0:                                                      <L 1942>
                        var_251 = (var_153 != var_250);
                        if (var_251) {
                            // wp.atomic_sub(                                                     <L 1943>
                            // cfrc_ext_out[worldid],                                             <L 1944>
                            var_252 = wp::view(var_cfrc_ext_out, var_0);
                            // prevbody,                                                          <L 1945>
                            // wp.spatial_vector(-wp.cross(prevcom - prevpnt, force), force),       <L 1946>
                            var_253 = wp::sub(var_157, var_155);
                            var_254 = wp::cross(var_253, var_241);
                            var_255 = wp::neg(var_254);
                            var_256 = wp::vec_t<6, wp::float32>(var_255, var_241);
                            var_257 = wp::atomic_sub(var_252, var_153, var_256);
                        }
                    }
                }
                // prevbody = body                                                                <L 1948>
                var_258 = wp::copy(var_201);
                // prevpnt = pnt                                                                  <L 1949>
                var_259 = wp::copy(var_231);
                // prevcom = com                                                                  <L 1950>
                var_260 = wp::copy(var_214);
                // p += 1                                                                         <L 1951>
                var_262 = wp::add(var_146, var_261);
            }
            var_263 = wp::where(var_219, var_262, var_146);
            var_264 = wp::where(var_219, var_258, var_153);
            var_265 = wp::where(var_219, var_259, var_155);
            var_266 = wp::where(var_219, var_260, var_157);
            // if k >= npnt:                                                                      <L 1930>
            var_269 = (var_267 >= var_200);
            if (var_269) {
                // continue                                                                       <L 1931>
            }
            var_271 = wp::where(var_269, var_270, var_268);
            // wrap_pos = wrap_xpos_in[worldid, p // 2]                                           <L 1933>
            if (var_271) {
                var_273 = wp::floordiv(var_263, var_272);
                var_274 = wp::address(var_wrap_xpos_in, var_0, var_273);
                var_276 = wp::load(var_274);
                var_275 = wp::copy(var_276);
                // pnt = wp.where(p % 2 == 0, wp.spatial_top(wrap_pos), wp.spatial_bottom(wrap_pos))       <L 1934>
                var_278 = wp::mod(var_263, var_277);
                var_280 = (var_278 == var_279);
                var_281 = wp::spatial_top(var_275);
                var_282 = wp::spatial_bottom(var_275);
                var_283 = wp::where(var_280, var_281, var_282);
                // if prevbody >= 0 and body != prevbody:                                         <L 1935>
                var_286 = (var_264 >= var_285);
                var_284 = var_286;
                if (var_284) {
                    var_287 = (var_201 != var_264);
                    var_284 = var_284 && var_287;
                }
                if (var_284) {
                    // diff = pnt - prevpnt                                                       <L 1936>
                    var_288 = wp::sub(var_283, var_265);
                    // vec, norm = math.normalize_with_norm(diff)                                 <L 1937>
                    normalize_with_norm_0(var_288, var_289, var_290);
                    // if norm > 0.0:                                                             <L 1938>
                    var_292 = (var_290 > var_291);
                    if (var_292) {
                        // force = vec * scaled_frc                                               <L 1939>
                        var_293 = wp::mul(var_289, var_151);
                        // if body != 0:                                                          <L 1940>
                        var_295 = (var_201 != var_294);
                        if (var_295) {
                            // wp.atomic_add(cfrc_ext_out[worldid], body, wp.spatial_vector(-wp.cross(com - pnt, force), force))       <L 1941>
                            var_296 = wp::view(var_cfrc_ext_out, var_0);
                            var_297 = wp::sub(var_214, var_283);
                            var_298 = wp::cross(var_297, var_293);
                            var_299 = wp::neg(var_298);
                            var_300 = wp::vec_t<6, wp::float32>(var_299, var_293);
                            var_301 = wp::atomic_add(var_296, var_201, var_300);
                        }
                        // if prevbody != 0:                                                      <L 1942>
                        var_303 = (var_264 != var_302);
                        if (var_303) {
                            // wp.atomic_sub(                                                     <L 1943>
                            // cfrc_ext_out[worldid],                                             <L 1944>
                            var_304 = wp::view(var_cfrc_ext_out, var_0);
                            // prevbody,                                                          <L 1945>
                            // wp.spatial_vector(-wp.cross(prevcom - prevpnt, force), force),       <L 1946>
                            var_305 = wp::sub(var_266, var_265);
                            var_306 = wp::cross(var_305, var_293);
                            var_307 = wp::neg(var_306);
                            var_308 = wp::vec_t<6, wp::float32>(var_307, var_293);
                            var_309 = wp::atomic_sub(var_304, var_264, var_308);
                        }
                    }
                    var_310 = wp::where(var_292, var_293, var_241);
                }
                var_311 = wp::where(var_284, var_288, var_236);
                var_312 = wp::where(var_284, var_289, var_237);
                var_313 = wp::where(var_284, var_290, var_238);
                var_314 = wp::where(var_284, var_310, var_241);
                // prevbody = body                                                                <L 1948>
                var_315 = wp::copy(var_201);
                // prevpnt = pnt                                                                  <L 1949>
                var_316 = wp::copy(var_283);
                // prevcom = com                                                                  <L 1950>
                var_317 = wp::copy(var_214);
                // p += 1                                                                         <L 1951>
                var_319 = wp::add(var_263, var_318);
            }
            var_320 = wp::where(var_271, var_319, var_263);
            var_321 = wp::where(var_271, var_315, var_264);
            var_322 = wp::where(var_271, var_316, var_265);
            var_323 = wp::where(var_271, var_317, var_266);
            var_324 = wp::where(var_271, var_275, var_223);
            var_325 = wp::where(var_271, var_283, var_231);
            var_326 = wp::where(var_271, var_311, var_236);
            var_327 = wp::where(var_271, var_312, var_237);
            var_328 = wp::where(var_271, var_313, var_238);
            var_329 = wp::where(var_271, var_314, var_241);
            wp::assign(var_146, var_320);
            wp::assign(var_153, var_321);
            wp::assign(var_155, var_322);
            wp::assign(var_157, var_323);
            goto start_for_4;
        end_for_4:;
    }
}



extern "C" __global__ void _tendon_bias_qfrc_2ae21571_cuda_kernel_forward(
    wp::launch_bounds_t<3> dim,
    wp::array_t<wp::int32> var_ten_J_rownnz,
    wp::array_t<wp::int32> var_ten_J_rowadr,
    wp::array_t<wp::int32> var_ten_J_colind,
    wp::array_t<wp::float32> var_tendon_armature,
    wp::array_t<wp::float32> var_ten_J_in,
    wp::array_t<wp::float32> var_ten_bias_coef_in,
    wp::array_t<wp::float32> var_qfrc_out)
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
        wp::shape_t* var_3;
        const wp::int32 var_4 = 0;
        wp::int32 var_5;
        wp::shape_t var_6;
        wp::int32 var_7;
        wp::float32* var_8;
        wp::float32 var_9;
        wp::float32 var_10;
        const wp::float32 var_11 = 0.0;
        bool var_12;
        wp::int32* var_13;
        wp::int32 var_14;
        wp::int32 var_15;
        bool var_16;
        wp::int32* var_17;
        wp::int32 var_18;
        wp::int32 var_19;
        wp::int32 var_20;
        wp::float32* var_21;
        wp::float32 var_22;
        wp::float32 var_23;
        const wp::float32 var_24 = 0.0;
        bool var_25;
        wp::int32* var_26;
        wp::int32 var_27;
        wp::int32 var_28;
        wp::array_t<wp::float32> var_29;
        wp::float32 var_30;
        wp::float32* var_31;
        wp::float32 var_32;
        wp::float32 var_33;
        wp::float32 var_34;
        //---------
        // forward
        // def _tendon_bias_qfrc(                                                                 <L 2416>
        // worldid, tenid, dofid = wp.tid()                                                       <L 2429>
        builtin_tid3d(var_0, var_1, var_2);
        // armature = tendon_armature[worldid % tendon_armature.shape[0], tenid]                  <L 2431>
        var_3 = &(var_tendon_armature.shape);
        var_6 = wp::load(var_3);
        var_5 = wp::extract(var_6, var_4);
        var_7 = wp::mod(var_0, var_5);
        var_8 = wp::address(var_tendon_armature, var_7, var_1);
        var_10 = wp::load(var_8);
        var_9 = wp::copy(var_10);
        // if armature == 0.0:                                                                    <L 2432>
        var_12 = (var_9 == var_11);
        if (var_12) {
            // return                                                                             <L 2433>
            continue;
        }
        // rownnz = ten_J_rownnz[tenid]                                                           <L 2435>
        var_13 = wp::address(var_ten_J_rownnz, var_1);
        var_15 = wp::load(var_13);
        var_14 = wp::copy(var_15);
        // if dofid >= rownnz:                                                                    <L 2436>
        var_16 = (var_2 >= var_14);
        if (var_16) {
            // return                                                                             <L 2437>
            continue;
        }
        // rowadr = ten_J_rowadr[tenid]                                                           <L 2438>
        var_17 = wp::address(var_ten_J_rowadr, var_1);
        var_19 = wp::load(var_17);
        var_18 = wp::copy(var_19);
        // sparseid = rowadr + dofid                                                              <L 2439>
        var_20 = wp::add(var_18, var_2);
        // ten_J = ten_J_in[worldid, sparseid]                                                    <L 2440>
        var_21 = wp::address(var_ten_J_in, var_0, var_20);
        var_23 = wp::load(var_21);
        var_22 = wp::copy(var_23);
        // if ten_J == 0.0:                                                                       <L 2442>
        var_25 = (var_22 == var_24);
        if (var_25) {
            // return                                                                             <L 2443>
            continue;
        }
        // dofid = ten_J_colind[sparseid]                                                         <L 2445>
        var_26 = wp::address(var_ten_J_colind, var_20);
        var_28 = wp::load(var_26);
        var_27 = wp::copy(var_28);
        // wp.atomic_add(qfrc_out[worldid], dofid, ten_J * armature * ten_bias_coef_in[worldid, tenid])       <L 2447>
        var_29 = wp::view(var_qfrc_out, var_0);
        var_30 = wp::mul(var_22, var_9);
        var_31 = wp::address(var_ten_bias_coef_in, var_0, var_1);
        var_33 = wp::load(var_31);
        var_32 = wp::mul(var_30, var_33);
        var_34 = wp::atomic_add(var_29, var_27, var_32);
    }
}



extern "C" __global__ void _spatial_tendon_wrap_2bb6c667_cuda_kernel_forward(
    wp::launch_bounds_t<1> dim,
    wp::int32 var_ntendon,
    wp::array_t<wp::int32> var_tendon_adr,
    wp::array_t<wp::int32> var_tendon_num,
    wp::array_t<wp::int32> var_wrap_type,
    wp::array_t<wp::int32> var_wrap_objid,
    wp::array_t<wp::vec_t<3, wp::float32>> var_site_xpos_in,
    wp::array_t<wp::vec_t<6, wp::float32>> var_wrap_geom_xpos_in,
    wp::array_t<wp::int32> var_ten_wrapadr_out,
    wp::array_t<wp::int32> var_ten_wrapnum_out,
    wp::array_t<wp::vec_t<2, wp::int32>> var_wrap_obj_out,
    wp::array_t<wp::vec_t<6, wp::float32>> var_wrap_xpos_out)
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
        wp::int32 var_2;
        const wp::int32 var_3 = 0;
        wp::int32 var_4;
        wp::range_t var_5;
        wp::int32 var_6;
        wp::int32* var_7;
        wp::int32 var_8;
        wp::int32 var_9;
        const wp::int32 var_10 = 0;
        wp::int32 var_11;
        wp::int32* var_12;
        wp::int32 var_13;
        wp::int32 var_14;
        wp::int32* var_15;
        const wp::int32 var_16 = 1;
        bool var_17;
        wp::int32 var_18;
        const wp::int32 var_19 = 0;
        wp::int32 var_20;
        const wp::int32 var_21 = 1;
        wp::int32 var_22;
        bool var_23;
        wp::int32 var_24;
        const wp::int32 var_25 = 0;
        wp::int32 var_26;
        wp::int32* var_27;
        wp::int32 var_28;
        wp::int32 var_29;
        wp::int32 var_30;
        const wp::int32 var_31 = 1;
        wp::int32 var_32;
        wp::int32* var_33;
        wp::int32 var_34;
        wp::int32 var_35;
        wp::int32 var_36;
        const wp::int32 var_37 = 0;
        wp::int32 var_38;
        wp::int32* var_39;
        wp::int32 var_40;
        wp::int32 var_41;
        wp::int32 var_42;
        const wp::int32 var_43 = 1;
        wp::int32 var_44;
        wp::int32* var_45;
        wp::int32 var_46;
        wp::int32 var_47;
        const wp::int32 var_48 = 2;
        bool var_49;
        bool var_50;
        const wp::int32 var_51 = 2;
        bool var_52;
        const wp::int32 var_53 = 2;
        wp::int32 var_54;
        const wp::int32 var_55 = 2;
        wp::int32 var_56;
        const wp::float32 var_57 = 0.0;
        wp::vec_t<6, wp::float32>* var_58;
        const wp::int32 var_59 = 3;
        wp::int32 var_60;
        const wp::int32 var_61 = 0;
        wp::int32 var_62;
        const wp::float32 var_63 = 0.0;
        wp::vec_t<6, wp::float32>* var_64;
        const wp::int32 var_65 = 3;
        wp::int32 var_66;
        const wp::int32 var_67 = 1;
        wp::int32 var_68;
        const wp::float32 var_69 = 0.0;
        wp::vec_t<6, wp::float32>* var_70;
        const wp::int32 var_71 = 3;
        wp::int32 var_72;
        const wp::int32 var_73 = 2;
        wp::int32 var_74;
        const wp::int32 var_75 = -2;
        wp::vec_t<2, wp::int32>* var_76;
        const wp::int32 var_77 = 1;
        wp::int32 var_78;
        const wp::int32 var_79 = 1;
        wp::int32 var_80;
        wp::int32 var_81;
        wp::int32 var_82;
        const wp::int32 var_83 = 1;
        wp::int32 var_84;
        wp::vec_t<3, wp::float32>* var_85;
        wp::vec_t<3, wp::float32> var_86;
        wp::vec_t<3, wp::float32> var_87;
        bool var_88;
        const wp::int32 var_89 = 4;
        bool var_90;
        const wp::int32 var_91 = 5;
        bool var_92;
        wp::vec_t<6, wp::float32>* var_93;
        wp::vec_t<6, wp::float32> var_94;
        wp::vec_t<6, wp::float32> var_95;
        wp::vec_t<3, wp::float32> var_96;
        const wp::int32 var_97 = 1;
        wp::int32 var_98;
        wp::int32 var_99;
        wp::int32 var_100;
        const wp::int32 var_101 = 2;
        wp::int32 var_102;
        wp::int32* var_103;
        wp::int32 var_104;
        wp::int32 var_105;
        wp::float32 var_106;
        const wp::float32 var_107 = 10000000000.0;
        bool var_108;
        wp::vec_t<3, wp::float32> var_109;
        wp::vec_t<3, wp::float32>* var_110;
        wp::vec_t<3, wp::float32> var_111;
        wp::vec_t<3, wp::float32> var_112;
        const wp::int32 var_113 = 0;
        wp::int32 var_114;
        const wp::int32 var_115 = 2;
        wp::int32 var_116;
        const wp::int32 var_117 = 0;
        wp::int32 var_118;
        const wp::int32 var_119 = 2;
        wp::int32 var_120;
        const wp::int32 var_121 = 1;
        wp::int32 var_122;
        const wp::int32 var_123 = 2;
        wp::int32 var_124;
        const wp::int32 var_125 = 1;
        wp::int32 var_126;
        const wp::int32 var_127 = 2;
        wp::int32 var_128;
        const wp::int32 var_129 = 2;
        wp::int32 var_130;
        const wp::int32 var_131 = 2;
        wp::int32 var_132;
        const wp::int32 var_133 = 2;
        wp::int32 var_134;
        const wp::int32 var_135 = 2;
        wp::int32 var_136;
        const wp::int32 var_137 = 3;
        wp::int32 var_138;
        const wp::int32 var_139 = 2;
        wp::int32 var_140;
        const wp::int32 var_141 = 3;
        wp::int32 var_142;
        const wp::int32 var_143 = 2;
        wp::int32 var_144;
        const wp::int32 var_145 = 0;
        wp::float32 var_146;
        wp::vec_t<6, wp::float32>* var_147;
        const wp::int32 var_148 = 3;
        wp::int32 var_149;
        const wp::int32 var_150 = 0;
        wp::int32 var_151;
        const wp::int32 var_152 = 1;
        wp::float32 var_153;
        wp::vec_t<6, wp::float32>* var_154;
        const wp::int32 var_155 = 3;
        wp::int32 var_156;
        const wp::int32 var_157 = 1;
        wp::int32 var_158;
        const wp::int32 var_159 = 2;
        wp::float32 var_160;
        wp::vec_t<6, wp::float32>* var_161;
        const wp::int32 var_162 = 3;
        wp::int32 var_163;
        const wp::int32 var_164 = 2;
        wp::int32 var_165;
        const wp::int32 var_166 = 0;
        wp::float32 var_167;
        wp::vec_t<6, wp::float32>* var_168;
        const wp::int32 var_169 = 3;
        wp::int32 var_170;
        const wp::int32 var_171 = 0;
        wp::int32 var_172;
        const wp::int32 var_173 = 1;
        wp::float32 var_174;
        wp::vec_t<6, wp::float32>* var_175;
        const wp::int32 var_176 = 3;
        wp::int32 var_177;
        const wp::int32 var_178 = 1;
        wp::int32 var_179;
        const wp::int32 var_180 = 2;
        wp::float32 var_181;
        wp::vec_t<6, wp::float32>* var_182;
        const wp::int32 var_183 = 3;
        wp::int32 var_184;
        const wp::int32 var_185 = 2;
        wp::int32 var_186;
        const wp::int32 var_187 = 0;
        wp::float32 var_188;
        wp::vec_t<6, wp::float32>* var_189;
        const wp::int32 var_190 = 3;
        wp::int32 var_191;
        const wp::int32 var_192 = 0;
        wp::int32 var_193;
        const wp::int32 var_194 = 1;
        wp::float32 var_195;
        wp::vec_t<6, wp::float32>* var_196;
        const wp::int32 var_197 = 3;
        wp::int32 var_198;
        const wp::int32 var_199 = 1;
        wp::int32 var_200;
        const wp::int32 var_201 = 2;
        wp::float32 var_202;
        wp::vec_t<6, wp::float32>* var_203;
        const wp::int32 var_204 = 3;
        wp::int32 var_205;
        const wp::int32 var_206 = 2;
        wp::int32 var_207;
        const wp::int32 var_208 = 0;
        wp::float32 var_209;
        wp::vec_t<6, wp::float32>* var_210;
        const wp::int32 var_211 = 3;
        wp::int32 var_212;
        const wp::int32 var_213 = 0;
        wp::int32 var_214;
        const wp::int32 var_215 = 1;
        wp::float32 var_216;
        wp::vec_t<6, wp::float32>* var_217;
        const wp::int32 var_218 = 3;
        wp::int32 var_219;
        const wp::int32 var_220 = 1;
        wp::int32 var_221;
        const wp::int32 var_222 = 2;
        wp::float32 var_223;
        wp::vec_t<6, wp::float32>* var_224;
        const wp::int32 var_225 = 3;
        wp::int32 var_226;
        const wp::int32 var_227 = 2;
        wp::int32 var_228;
        const wp::int32 var_229 = -1;
        wp::vec_t<2, wp::int32>* var_230;
        wp::vec_t<2, wp::int32>* var_231;
        wp::vec_t<2, wp::int32>* var_232;
        const wp::int32 var_233 = 3;
        wp::int32 var_234;
        const wp::int32 var_235 = 3;
        wp::int32 var_236;
        const wp::int32 var_237 = 2;
        wp::int32 var_238;
        const wp::int32 var_239 = 0;
        wp::int32 var_240;
        const wp::int32 var_241 = 2;
        wp::int32 var_242;
        const wp::int32 var_243 = 0;
        wp::int32 var_244;
        const wp::int32 var_245 = 2;
        wp::int32 var_246;
        const wp::int32 var_247 = 0;
        wp::float32 var_248;
        wp::vec_t<6, wp::float32>* var_249;
        const wp::int32 var_250 = 3;
        wp::int32 var_251;
        const wp::int32 var_252 = 0;
        wp::int32 var_253;
        const wp::int32 var_254 = 1;
        wp::float32 var_255;
        wp::vec_t<6, wp::float32>* var_256;
        const wp::int32 var_257 = 3;
        wp::int32 var_258;
        const wp::int32 var_259 = 1;
        wp::int32 var_260;
        const wp::int32 var_261 = 2;
        wp::float32 var_262;
        wp::vec_t<6, wp::float32>* var_263;
        const wp::int32 var_264 = 3;
        wp::int32 var_265;
        const wp::int32 var_266 = 2;
        wp::int32 var_267;
        const wp::int32 var_268 = -1;
        wp::vec_t<2, wp::int32>* var_269;
        const wp::int32 var_270 = 1;
        wp::int32 var_271;
        const wp::int32 var_272 = 1;
        wp::int32 var_273;
        const wp::int32 var_274 = 2;
        wp::int32 var_275;
        wp::int32 var_276;
        wp::int32 var_277;
        wp::int32 var_278;
        wp::int32 var_279;
        wp::int32 var_280;
        const wp::int32 var_281 = 0;
        wp::int32 var_282;
        const wp::int32 var_283 = 2;
        wp::int32 var_284;
        const wp::int32 var_285 = 0;
        wp::int32 var_286;
        const wp::int32 var_287 = 2;
        wp::int32 var_288;
        const wp::int32 var_289 = 0;
        wp::float32 var_290;
        wp::vec_t<6, wp::float32>* var_291;
        const wp::int32 var_292 = 3;
        wp::int32 var_293;
        const wp::int32 var_294 = 0;
        wp::int32 var_295;
        const wp::int32 var_296 = 1;
        wp::float32 var_297;
        wp::vec_t<6, wp::float32>* var_298;
        const wp::int32 var_299 = 3;
        wp::int32 var_300;
        const wp::int32 var_301 = 1;
        wp::int32 var_302;
        const wp::int32 var_303 = 2;
        wp::float32 var_304;
        wp::vec_t<6, wp::float32>* var_305;
        const wp::int32 var_306 = 3;
        wp::int32 var_307;
        const wp::int32 var_308 = 2;
        wp::int32 var_309;
        const wp::int32 var_310 = -1;
        wp::vec_t<2, wp::int32>* var_311;
        const wp::int32 var_312 = 1;
        wp::int32 var_313;
        const wp::int32 var_314 = 1;
        wp::int32 var_315;
        const wp::int32 var_316 = 1;
        wp::int32 var_317;
        wp::int32 var_318;
        wp::int32 var_319;
        wp::int32 var_320;
        wp::int32 var_321;
        wp::int32 var_322;
        wp::int32 var_323;
        wp::int32 var_324;
        wp::int32 var_325;
        const wp::int32 var_326 = 1;
        wp::int32 var_327;
        wp::shape_t* var_328;
        const wp::int32 var_329 = 0;
        wp::int32 var_330;
        wp::shape_t var_331;
        bool var_332;
        wp::int32 var_333;
        const wp::int32 var_334 = 1;
        wp::int32 var_335;
        wp::int32* var_336;
        const wp::int32 var_337 = 2;
        bool var_338;
        wp::int32 var_339;
        const bool var_340 = false;
        bool var_341;
        bool var_342;
        const wp::int32 var_343 = 1;
        wp::int32 var_344;
        bool var_345;
        const wp::int32 var_346 = 0;
        wp::int32 var_347;
        const wp::int32 var_348 = 2;
        wp::int32 var_349;
        const wp::int32 var_350 = 0;
        wp::int32 var_351;
        const wp::int32 var_352 = 2;
        wp::int32 var_353;
        wp::vec_t<3, wp::float32>* var_354;
        wp::vec_t<3, wp::float32> var_355;
        wp::vec_t<3, wp::float32> var_356;
        const wp::int32 var_357 = 0;
        wp::float32 var_358;
        wp::vec_t<6, wp::float32>* var_359;
        const wp::int32 var_360 = 3;
        wp::int32 var_361;
        const wp::int32 var_362 = 0;
        wp::int32 var_363;
        const wp::int32 var_364 = 1;
        wp::float32 var_365;
        wp::vec_t<6, wp::float32>* var_366;
        const wp::int32 var_367 = 3;
        wp::int32 var_368;
        const wp::int32 var_369 = 1;
        wp::int32 var_370;
        const wp::int32 var_371 = 2;
        wp::float32 var_372;
        wp::vec_t<6, wp::float32>* var_373;
        const wp::int32 var_374 = 3;
        wp::int32 var_375;
        const wp::int32 var_376 = 2;
        wp::int32 var_377;
        const wp::int32 var_378 = -1;
        wp::vec_t<2, wp::int32>* var_379;
        const wp::int32 var_380 = 1;
        wp::int32 var_381;
        const wp::int32 var_382 = 1;
        wp::int32 var_383;
        wp::int32 var_384;
        wp::int32 var_385;
        wp::vec_t<3, wp::float32> var_386;
        wp::int32 var_387;
        wp::int32 var_388;
        //---------
        // forward
        // def _spatial_tendon_wrap(                                                              <L 4367>
        // worldid = wp.tid()                                                                     <L 4384>
        var_0 = builtin_tid1d();
        // wrapcount = int(0)                                                                     <L 4386>
        var_2 = wp::int(var_1);
        // wrapgeomid = int(0)                                                                    <L 4387>
        var_4 = wp::int(var_3);
        // for i in range(ntendon):                                                               <L 4388>
        var_5 = wp::range(var_ntendon);
        start_for_0:;
            if (iter_cmp(var_5) == 0) goto end_for_0;
            var_6 = wp::iter_next(var_5);
            // adr = tendon_adr[i]                                                                <L 4389>
            var_7 = wp::address(var_tendon_adr, var_6);
            var_9 = wp::load(var_7);
            var_8 = wp::copy(var_9);
            // ten_wrapadr_out[worldid, i] = wrapcount                                            <L 4390>
            wp::array_store(var_ten_wrapadr_out, var_0, var_6, var_2);
            // wrapnum = int(0)                                                                   <L 4391>
            var_11 = wp::int(var_10);
            // tendonnum = tendon_num[i]                                                          <L 4392>
            var_12 = wp::address(var_tendon_num, var_6);
            var_14 = wp::load(var_12);
            var_13 = wp::copy(var_14);
            // if wrap_type[adr] == WrapType.JOINT:                                               <L 4395>
            var_15 = wp::address(var_wrap_type, var_8);
            var_18 = wp::load(var_15);
            var_17 = (var_18 == var_16);
            if (var_17) {
                // continue                                                                       <L 4396>
                goto start_for_0;
            }
            // j = int(0)                                                                         <L 4399>
            var_20 = wp::int(var_19);
            // while j < tendonnum - 1:                                                           <L 4400>
        start_while_2:;
            var_22 = wp::sub(var_13, var_21);
            var_23 = (var_20 < var_22);
        if ((var_23) == false) goto end_while_2;
                // type0 = wrap_type[adr + j + 0]                                                 <L 4402>
                var_24 = wp::add(var_8, var_20);
                var_26 = wp::add(var_24, var_25);
                var_27 = wp::address(var_wrap_type, var_26);
                var_29 = wp::load(var_27);
                var_28 = wp::copy(var_29);
                // type1 = wrap_type[adr + j + 1]                                                 <L 4403>
                var_30 = wp::add(var_8, var_20);
                var_32 = wp::add(var_30, var_31);
                var_33 = wp::address(var_wrap_type, var_32);
                var_35 = wp::load(var_33);
                var_34 = wp::copy(var_35);
                // id0 = wrap_objid[adr + j + 0]                                                  <L 4404>
                var_36 = wp::add(var_8, var_20);
                var_38 = wp::add(var_36, var_37);
                var_39 = wp::address(var_wrap_objid, var_38);
                var_41 = wp::load(var_39);
                var_40 = wp::copy(var_41);
                // id1 = wrap_objid[adr + j + 1]                                                  <L 4405>
                var_42 = wp::add(var_8, var_20);
                var_44 = wp::add(var_42, var_43);
                var_45 = wp::address(var_wrap_objid, var_44);
                var_47 = wp::load(var_45);
                var_46 = wp::copy(var_47);
                // pulley0 = type0 == WrapType.PULLEY                                             <L 4408>
                var_49 = (var_28 == var_48);
                // if pulley0 or type1 == WrapType.PULLEY:                                        <L 4409>
                var_50 = var_49;
                if (!var_50) {
                    var_52 = (var_34 == var_51);
                    var_50 = var_50 || var_52;
                }
                if (var_50) {
                    // if pulley0:                                                                <L 4410>
                    if (var_49) {
                        // row = wrapcount // 2                                                   <L 4411>
                        var_54 = wp::floordiv(var_2, var_53);
                        // col = wrapcount % 2                                                    <L 4412>
                        var_56 = wp::mod(var_2, var_55);
                        // wrap_xpos_out[worldid, row][3 * col + 0] = 0.0                         <L 4413>
                        var_58 = &(wp::index(var_wrap_xpos_out, var_0, var_54));
                        var_60 = wp::mul(var_59, var_56);
                        var_62 = wp::add(var_60, var_61);
                        wp::store(&(wp::index(var_wrap_xpos_out, var_0, var_54).component_ref(var_62)), var_57);
                        // wrap_xpos_out[worldid, row][3 * col + 1] = 0.0                         <L 4414>
                        var_64 = &(wp::index(var_wrap_xpos_out, var_0, var_54));
                        var_66 = wp::mul(var_65, var_56);
                        var_68 = wp::add(var_66, var_67);
                        wp::store(&(wp::index(var_wrap_xpos_out, var_0, var_54).component_ref(var_68)), var_63);
                        // wrap_xpos_out[worldid, row][3 * col + 2] = 0.0                         <L 4415>
                        var_70 = &(wp::index(var_wrap_xpos_out, var_0, var_54));
                        var_72 = wp::mul(var_71, var_56);
                        var_74 = wp::add(var_72, var_73);
                        wp::store(&(wp::index(var_wrap_xpos_out, var_0, var_54).component_ref(var_74)), var_69);
                        // wrap_obj_out[worldid, row][col] = -2                                   <L 4417>
                        var_76 = &(wp::index(var_wrap_obj_out, var_0, var_54));
                        wp::store(&(wp::index(var_wrap_obj_out, var_0, var_54).component_ref(var_56)), var_75);
                        // wrapnum += 1                                                           <L 4419>
                        var_78 = wp::add(var_11, var_77);
                        // wrapcount += 1                                                         <L 4420>
                        var_80 = wp::add(var_2, var_79);
                    }
                    var_81 = wp::where(var_49, var_80, var_2);
                    var_82 = wp::where(var_49, var_78, var_11);
                    // j += 1                                                                     <L 4423>
                    var_84 = wp::add(var_20, var_83);
                    // continue                                                                   <L 4424>
                    wp::assign(var_2, var_81);
                    wp::assign(var_11, var_82);
                    wp::assign(var_20, var_84);
                    goto start_while_2;
                }
                // wpnt_site0 = site_xpos_in[worldid, id0]                                        <L 4427>
                var_85 = wp::address(var_site_xpos_in, var_0, var_40);
                var_87 = wp::load(var_85);
                var_86 = wp::copy(var_87);
                // if type1 == WrapType.SPHERE or type1 == WrapType.CYLINDER:                     <L 4430>
                var_90 = (var_34 == var_89);
                var_88 = var_90;
                if (!var_88) {
                    var_92 = (var_34 == var_91);
                    var_88 = var_88 || var_92;
                }
                if (var_88) {
                    // wrap_geom_xpos = wrap_geom_xpos_in[worldid, wrapgeomid]                    <L 4431>
                    var_93 = wp::address(var_wrap_geom_xpos_in, var_0, var_4);
                    var_95 = wp::load(var_93);
                    var_94 = wp::copy(var_95);
                    // wpnt_geom0 = wp.spatial_top(wrap_geom_xpos)                                <L 4432>
                    var_96 = wp::spatial_top(var_94);
                    // wrapgeomid += 1                                                            <L 4433>
                    var_98 = wp::add(var_4, var_97);
                    // wrapid = id1                                                               <L 4435>
                    var_99 = wp::copy(var_46);
                    // id1 = wrap_objid[adr + j + 2]                                              <L 4436>
                    var_100 = wp::add(var_8, var_20);
                    var_102 = wp::add(var_100, var_101);
                    var_103 = wp::address(var_wrap_objid, var_102);
                    var_105 = wp::load(var_103);
                    var_104 = wp::copy(var_105);
                    // if wp.norm_l2(wpnt_geom0) < MJ_MAXVAL:                                     <L 4437>
                    var_106 = norm_l2_0(var_96);
                    var_108 = (var_106 < var_107);
                    if (var_108) {
                        // wpnt_geom1 = wp.spatial_bottom(wrap_geom_xpos)                         <L 4438>
                        var_109 = wp::spatial_bottom(var_94);
                        // wpnt_site1 = site_xpos_in[worldid, id1]                                <L 4439>
                        var_110 = wp::address(var_site_xpos_in, var_0, var_104);
                        var_112 = wp::load(var_110);
                        var_111 = wp::copy(var_112);
                        // row0 = (wrapcount + 0) // 2                                            <L 4442>
                        var_114 = wp::add(var_2, var_113);
                        var_116 = wp::floordiv(var_114, var_115);
                        // col0 = (wrapcount + 0) % 2                                             <L 4443>
                        var_118 = wp::add(var_2, var_117);
                        var_120 = wp::mod(var_118, var_119);
                        // row1 = (wrapcount + 1) // 2                                            <L 4444>
                        var_122 = wp::add(var_2, var_121);
                        var_124 = wp::floordiv(var_122, var_123);
                        // col1 = (wrapcount + 1) % 2                                             <L 4445>
                        var_126 = wp::add(var_2, var_125);
                        var_128 = wp::mod(var_126, var_127);
                        // row2 = (wrapcount + 2) // 2                                            <L 4446>
                        var_130 = wp::add(var_2, var_129);
                        var_132 = wp::floordiv(var_130, var_131);
                        // col2 = (wrapcount + 2) % 2                                             <L 4447>
                        var_134 = wp::add(var_2, var_133);
                        var_136 = wp::mod(var_134, var_135);
                        // row3 = (wrapcount + 3) // 2                                            <L 4448>
                        var_138 = wp::add(var_2, var_137);
                        var_140 = wp::floordiv(var_138, var_139);
                        // col3 = (wrapcount + 3) % 2                                             <L 4449>
                        var_142 = wp::add(var_2, var_141);
                        var_144 = wp::mod(var_142, var_143);
                        // wrap_xpos_out[worldid, row0][3 * col0 + 0] = wpnt_site0[0]             <L 4451>
                        var_146 = wp::extract(var_86, var_145);
                        var_147 = &(wp::index(var_wrap_xpos_out, var_0, var_116));
                        var_149 = wp::mul(var_148, var_120);
                        var_151 = wp::add(var_149, var_150);
                        wp::store(&(wp::index(var_wrap_xpos_out, var_0, var_116).component_ref(var_151)), var_146);
                        // wrap_xpos_out[worldid, row0][3 * col0 + 1] = wpnt_site0[1]             <L 4452>
                        var_153 = wp::extract(var_86, var_152);
                        var_154 = &(wp::index(var_wrap_xpos_out, var_0, var_116));
                        var_156 = wp::mul(var_155, var_120);
                        var_158 = wp::add(var_156, var_157);
                        wp::store(&(wp::index(var_wrap_xpos_out, var_0, var_116).component_ref(var_158)), var_153);
                        // wrap_xpos_out[worldid, row0][3 * col0 + 2] = wpnt_site0[2]             <L 4453>
                        var_160 = wp::extract(var_86, var_159);
                        var_161 = &(wp::index(var_wrap_xpos_out, var_0, var_116));
                        var_163 = wp::mul(var_162, var_120);
                        var_165 = wp::add(var_163, var_164);
                        wp::store(&(wp::index(var_wrap_xpos_out, var_0, var_116).component_ref(var_165)), var_160);
                        // wrap_xpos_out[worldid, row1][3 * col1 + 0] = wpnt_geom0[0]             <L 4455>
                        var_167 = wp::extract(var_96, var_166);
                        var_168 = &(wp::index(var_wrap_xpos_out, var_0, var_124));
                        var_170 = wp::mul(var_169, var_128);
                        var_172 = wp::add(var_170, var_171);
                        wp::store(&(wp::index(var_wrap_xpos_out, var_0, var_124).component_ref(var_172)), var_167);
                        // wrap_xpos_out[worldid, row1][3 * col1 + 1] = wpnt_geom0[1]             <L 4456>
                        var_174 = wp::extract(var_96, var_173);
                        var_175 = &(wp::index(var_wrap_xpos_out, var_0, var_124));
                        var_177 = wp::mul(var_176, var_128);
                        var_179 = wp::add(var_177, var_178);
                        wp::store(&(wp::index(var_wrap_xpos_out, var_0, var_124).component_ref(var_179)), var_174);
                        // wrap_xpos_out[worldid, row1][3 * col1 + 2] = wpnt_geom0[2]             <L 4457>
                        var_181 = wp::extract(var_96, var_180);
                        var_182 = &(wp::index(var_wrap_xpos_out, var_0, var_124));
                        var_184 = wp::mul(var_183, var_128);
                        var_186 = wp::add(var_184, var_185);
                        wp::store(&(wp::index(var_wrap_xpos_out, var_0, var_124).component_ref(var_186)), var_181);
                        // wrap_xpos_out[worldid, row2][3 * col2 + 0] = wpnt_geom1[0]             <L 4459>
                        var_188 = wp::extract(var_109, var_187);
                        var_189 = &(wp::index(var_wrap_xpos_out, var_0, var_132));
                        var_191 = wp::mul(var_190, var_136);
                        var_193 = wp::add(var_191, var_192);
                        wp::store(&(wp::index(var_wrap_xpos_out, var_0, var_132).component_ref(var_193)), var_188);
                        // wrap_xpos_out[worldid, row2][3 * col2 + 1] = wpnt_geom1[1]             <L 4460>
                        var_195 = wp::extract(var_109, var_194);
                        var_196 = &(wp::index(var_wrap_xpos_out, var_0, var_132));
                        var_198 = wp::mul(var_197, var_136);
                        var_200 = wp::add(var_198, var_199);
                        wp::store(&(wp::index(var_wrap_xpos_out, var_0, var_132).component_ref(var_200)), var_195);
                        // wrap_xpos_out[worldid, row2][3 * col2 + 2] = wpnt_geom1[2]             <L 4461>
                        var_202 = wp::extract(var_109, var_201);
                        var_203 = &(wp::index(var_wrap_xpos_out, var_0, var_132));
                        var_205 = wp::mul(var_204, var_136);
                        var_207 = wp::add(var_205, var_206);
                        wp::store(&(wp::index(var_wrap_xpos_out, var_0, var_132).component_ref(var_207)), var_202);
                        // wrap_xpos_out[worldid, row3][3 * col3 + 0] = wpnt_site1[0]             <L 4463>
                        var_209 = wp::extract(var_111, var_208);
                        var_210 = &(wp::index(var_wrap_xpos_out, var_0, var_140));
                        var_212 = wp::mul(var_211, var_144);
                        var_214 = wp::add(var_212, var_213);
                        wp::store(&(wp::index(var_wrap_xpos_out, var_0, var_140).component_ref(var_214)), var_209);
                        // wrap_xpos_out[worldid, row3][3 * col3 + 1] = wpnt_site1[1]             <L 4464>
                        var_216 = wp::extract(var_111, var_215);
                        var_217 = &(wp::index(var_wrap_xpos_out, var_0, var_140));
                        var_219 = wp::mul(var_218, var_144);
                        var_221 = wp::add(var_219, var_220);
                        wp::store(&(wp::index(var_wrap_xpos_out, var_0, var_140).component_ref(var_221)), var_216);
                        // wrap_xpos_out[worldid, row3][3 * col3 + 2] = wpnt_site1[2]             <L 4465>
                        var_223 = wp::extract(var_111, var_222);
                        var_224 = &(wp::index(var_wrap_xpos_out, var_0, var_140));
                        var_226 = wp::mul(var_225, var_144);
                        var_228 = wp::add(var_226, var_227);
                        wp::store(&(wp::index(var_wrap_xpos_out, var_0, var_140).component_ref(var_228)), var_223);
                        // wrap_obj_out[worldid, row0][col0] = -1                                 <L 4467>
                        var_230 = &(wp::index(var_wrap_obj_out, var_0, var_116));
                        wp::store(&(wp::index(var_wrap_obj_out, var_0, var_116).component_ref(var_120)), var_229);
                        // wrap_obj_out[worldid, row1][col1] = wrapid                             <L 4468>
                        var_231 = &(wp::index(var_wrap_obj_out, var_0, var_124));
                        wp::store(&(wp::index(var_wrap_obj_out, var_0, var_124).component_ref(var_128)), var_99);
                        // wrap_obj_out[worldid, row2][col2] = wrapid                             <L 4469>
                        var_232 = &(wp::index(var_wrap_obj_out, var_0, var_132));
                        wp::store(&(wp::index(var_wrap_obj_out, var_0, var_132).component_ref(var_136)), var_99);
                        // wrapnum += 3                                                           <L 4471>
                        var_234 = wp::add(var_11, var_233);
                        // wrapcount += 3                                                         <L 4472>
                        var_236 = wp::add(var_2, var_235);
                        // j += 2                                                                 <L 4473>
                        var_238 = wp::add(var_20, var_237);
                    }
                    if (!var_108) {
                        // row0 = (wrapcount + 0) // 2                                            <L 4476>
                        var_240 = wp::add(var_2, var_239);
                        var_242 = wp::floordiv(var_240, var_241);
                        // col0 = (wrapcount + 0) % 2                                             <L 4477>
                        var_244 = wp::add(var_2, var_243);
                        var_246 = wp::mod(var_244, var_245);
                        // wrap_xpos_out[worldid, row0][3 * col0 + 0] = wpnt_site0[0]             <L 4479>
                        var_248 = wp::extract(var_86, var_247);
                        var_249 = &(wp::index(var_wrap_xpos_out, var_0, var_242));
                        var_251 = wp::mul(var_250, var_246);
                        var_253 = wp::add(var_251, var_252);
                        wp::store(&(wp::index(var_wrap_xpos_out, var_0, var_242).component_ref(var_253)), var_248);
                        // wrap_xpos_out[worldid, row0][3 * col0 + 1] = wpnt_site0[1]             <L 4480>
                        var_255 = wp::extract(var_86, var_254);
                        var_256 = &(wp::index(var_wrap_xpos_out, var_0, var_242));
                        var_258 = wp::mul(var_257, var_246);
                        var_260 = wp::add(var_258, var_259);
                        wp::store(&(wp::index(var_wrap_xpos_out, var_0, var_242).component_ref(var_260)), var_255);
                        // wrap_xpos_out[worldid, row0][3 * col0 + 2] = wpnt_site0[2]             <L 4481>
                        var_262 = wp::extract(var_86, var_261);
                        var_263 = &(wp::index(var_wrap_xpos_out, var_0, var_242));
                        var_265 = wp::mul(var_264, var_246);
                        var_267 = wp::add(var_265, var_266);
                        wp::store(&(wp::index(var_wrap_xpos_out, var_0, var_242).component_ref(var_267)), var_262);
                        // wrap_obj_out[worldid, row0][col0] = -1                                 <L 4483>
                        var_269 = &(wp::index(var_wrap_obj_out, var_0, var_242));
                        wp::store(&(wp::index(var_wrap_obj_out, var_0, var_242).component_ref(var_246)), var_268);
                        // wrapnum += 1                                                           <L 4485>
                        var_271 = wp::add(var_11, var_270);
                        // wrapcount += 1                                                         <L 4486>
                        var_273 = wp::add(var_2, var_272);
                        // j += 2                                                                 <L 4487>
                        var_275 = wp::add(var_20, var_274);
                    }
                    var_276 = wp::where(var_108, var_236, var_273);
                    var_277 = wp::where(var_108, var_234, var_271);
                    var_278 = wp::where(var_108, var_238, var_275);
                    var_279 = wp::where(var_108, var_116, var_242);
                    var_280 = wp::where(var_108, var_120, var_246);
                }
                if (!var_88) {
                    // row0 = (wrapcount + 0) // 2                                                <L 4490>
                    var_282 = wp::add(var_2, var_281);
                    var_284 = wp::floordiv(var_282, var_283);
                    // col0 = (wrapcount + 0) % 2                                                 <L 4491>
                    var_286 = wp::add(var_2, var_285);
                    var_288 = wp::mod(var_286, var_287);
                    // wrap_xpos_out[worldid, row0][3 * col0 + 0] = wpnt_site0[0]                 <L 4493>
                    var_290 = wp::extract(var_86, var_289);
                    var_291 = &(wp::index(var_wrap_xpos_out, var_0, var_284));
                    var_293 = wp::mul(var_292, var_288);
                    var_295 = wp::add(var_293, var_294);
                    wp::store(&(wp::index(var_wrap_xpos_out, var_0, var_284).component_ref(var_295)), var_290);
                    // wrap_xpos_out[worldid, row0][3 * col0 + 1] = wpnt_site0[1]                 <L 4494>
                    var_297 = wp::extract(var_86, var_296);
                    var_298 = &(wp::index(var_wrap_xpos_out, var_0, var_284));
                    var_300 = wp::mul(var_299, var_288);
                    var_302 = wp::add(var_300, var_301);
                    wp::store(&(wp::index(var_wrap_xpos_out, var_0, var_284).component_ref(var_302)), var_297);
                    // wrap_xpos_out[worldid, row0][3 * col0 + 2] = wpnt_site0[2]                 <L 4495>
                    var_304 = wp::extract(var_86, var_303);
                    var_305 = &(wp::index(var_wrap_xpos_out, var_0, var_284));
                    var_307 = wp::mul(var_306, var_288);
                    var_309 = wp::add(var_307, var_308);
                    wp::store(&(wp::index(var_wrap_xpos_out, var_0, var_284).component_ref(var_309)), var_304);
                    // wrap_obj_out[worldid, row0][col0] = -1                                     <L 4497>
                    var_311 = &(wp::index(var_wrap_obj_out, var_0, var_284));
                    wp::store(&(wp::index(var_wrap_obj_out, var_0, var_284).component_ref(var_288)), var_310);
                    // wrapnum += 1                                                               <L 4499>
                    var_313 = wp::add(var_11, var_312);
                    // wrapcount += 1                                                             <L 4500>
                    var_315 = wp::add(var_2, var_314);
                    // j += 1                                                                     <L 4501>
                    var_317 = wp::add(var_20, var_316);
                }
                var_318 = wp::where(var_88, var_276, var_315);
                var_319 = wp::where(var_88, var_98, var_4);
                var_320 = wp::where(var_88, var_277, var_313);
                var_321 = wp::where(var_88, var_278, var_317);
                var_322 = wp::where(var_88, var_104, var_46);
                var_323 = wp::where(var_88, var_279, var_284);
                var_324 = wp::where(var_88, var_280, var_288);
                // if adr + j + 1 < wrap_type.shape[0]:                                           <L 4504>
                var_325 = wp::add(var_8, var_321);
                var_327 = wp::add(var_325, var_326);
                var_328 = &(var_wrap_type.shape);
                var_331 = wp::load(var_328);
                var_330 = wp::extract(var_331, var_329);
                var_332 = (var_327 < var_330);
                if (var_332) {
                    // last_before_pulley = wrap_type[adr + j + 1] == WrapType.PULLEY             <L 4505>
                    var_333 = wp::add(var_8, var_321);
                    var_335 = wp::add(var_333, var_334);
                    var_336 = wp::address(var_wrap_type, var_335);
                    var_339 = wp::load(var_336);
                    var_338 = (var_339 == var_337);
                }
                if (!var_332) {
                    // last_before_pulley = False                                                 <L 4507>
                }
                var_341 = wp::where(var_332, var_338, var_340);
                // if j == tendonnum - 1 or last_before_pulley:                                   <L 4509>
                var_344 = wp::sub(var_13, var_343);
                var_345 = (var_321 == var_344);
                var_342 = var_345;
                if (!var_342) {
                    var_342 = var_342 || var_341;
                }
                if (var_342) {
                    // row0 = (wrapcount + 0) // 2                                                <L 4510>
                    var_347 = wp::add(var_318, var_346);
                    var_349 = wp::floordiv(var_347, var_348);
                    // col0 = (wrapcount + 0) % 2                                                 <L 4511>
                    var_351 = wp::add(var_318, var_350);
                    var_353 = wp::mod(var_351, var_352);
                    // wpnt_site1 = site_xpos_in[worldid, id1]                                    <L 4513>
                    var_354 = wp::address(var_site_xpos_in, var_0, var_322);
                    var_356 = wp::load(var_354);
                    var_355 = wp::copy(var_356);
                    // wrap_xpos_out[worldid, row0][3 * col0 + 0] = wpnt_site1[0]                 <L 4514>
                    var_358 = wp::extract(var_355, var_357);
                    var_359 = &(wp::index(var_wrap_xpos_out, var_0, var_349));
                    var_361 = wp::mul(var_360, var_353);
                    var_363 = wp::add(var_361, var_362);
                    wp::store(&(wp::index(var_wrap_xpos_out, var_0, var_349).component_ref(var_363)), var_358);
                    // wrap_xpos_out[worldid, row0][3 * col0 + 1] = wpnt_site1[1]                 <L 4515>
                    var_365 = wp::extract(var_355, var_364);
                    var_366 = &(wp::index(var_wrap_xpos_out, var_0, var_349));
                    var_368 = wp::mul(var_367, var_353);
                    var_370 = wp::add(var_368, var_369);
                    wp::store(&(wp::index(var_wrap_xpos_out, var_0, var_349).component_ref(var_370)), var_365);
                    // wrap_xpos_out[worldid, row0][3 * col0 + 2] = wpnt_site1[2]                 <L 4516>
                    var_372 = wp::extract(var_355, var_371);
                    var_373 = &(wp::index(var_wrap_xpos_out, var_0, var_349));
                    var_375 = wp::mul(var_374, var_353);
                    var_377 = wp::add(var_375, var_376);
                    wp::store(&(wp::index(var_wrap_xpos_out, var_0, var_349).component_ref(var_377)), var_372);
                    // wrap_obj_out[worldid, row0][col0] = -1                                     <L 4518>
                    var_379 = &(wp::index(var_wrap_obj_out, var_0, var_349));
                    wp::store(&(wp::index(var_wrap_obj_out, var_0, var_349).component_ref(var_353)), var_378);
                    // wrapnum += 1                                                               <L 4519>
                    var_381 = wp::add(var_320, var_380);
                    // wrapcount += 1                                                             <L 4520>
                    var_383 = wp::add(var_318, var_382);
                }
                var_384 = wp::where(var_342, var_383, var_318);
                var_385 = wp::where(var_342, var_381, var_320);
                var_386 = wp::where(var_342, var_355, var_111);
                var_387 = wp::where(var_342, var_349, var_323);
                var_388 = wp::where(var_342, var_353, var_324);
                wp::assign(var_2, var_384);
                wp::assign(var_4, var_319);
                wp::assign(var_11, var_385);
                wp::assign(var_20, var_321);
        goto start_while_2;
        end_while_2:;
            // ten_wrapnum_out[worldid, i] = wrapnum                                              <L 4522>
            wp::array_store(var_ten_wrapnum_out, var_0, var_6, var_11);
            goto start_for_0;
        end_for_0:;
    }
}



extern "C" __global__ void _qLDiag_div_3444613a_cuda_kernel_forward(
    wp::launch_bounds_t<2> dim,
    wp::array_t<wp::int32> var_M_rownnz,
    wp::array_t<wp::int32> var_M_rowadr,
    wp::array_t<wp::float32> var_L_in,
    wp::array_t<wp::float32> var_D_out)
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
        wp::int32* var_3;
        wp::int32 var_4;
        wp::int32 var_5;
        wp::int32 var_6;
        const wp::int32 var_7 = 1;
        wp::int32 var_8;
        const wp::float32 var_9 = 1.0;
        wp::float32* var_10;
        wp::float32 var_11;
        wp::float32 var_12;
        //---------
        // forward
        // def _qLDiag_div(                                                                       <L 1180>
        // worldid, dofid = wp.tid()                                                              <L 1189>
        builtin_tid2d(var_0, var_1);
        // diag_i = M_rowadr[dofid] + M_rownnz[dofid] - 1  # Address of diagonal element of i       <L 1190>
        var_2 = wp::address(var_M_rowadr, var_1);
        var_3 = wp::address(var_M_rownnz, var_1);
        var_5 = wp::load(var_2);
        var_6 = wp::load(var_3);
        var_4 = wp::add(var_5, var_6);
        var_8 = wp::sub(var_4, var_7);
        // D_out[worldid, dofid] = 1.0 / L_in[worldid, diag_i]                                    <L 1191>
        var_10 = wp::address(var_L_in, var_0, var_8);
        var_12 = wp::load(var_10);
        var_11 = wp::div(var_9, var_12);
        wp::array_store(var_D_out, var_0, var_1, var_11);
    }
}



extern "C" __global__ void _cfrc_ext_contact_348e18d8_cuda_kernel_forward(
    wp::launch_bounds_t<1> dim,
    wp::int32 var_opt_cone,
    wp::array_t<wp::int32> var_body_rootid,
    wp::array_t<wp::int32> var_geom_bodyid,
    wp::array_t<wp::vec_t<3, wp::float32>> var_subtree_com_in,
    wp::array_t<wp::vec_t<3, wp::float32>> var_contact_pos_in,
    wp::array_t<wp::mat_t<3, 3, wp::float32>> var_contact_frame_in,
    wp::array_t<wp::vec_t<5, wp::float32>> var_contact_friction_in,
    wp::array_t<wp::int32> var_contact_dim_in,
    wp::array_t<wp::vec_t<2, wp::int32>> var_contact_geom_in,
    wp::array_t<wp::int32> var_contact_efc_address_in,
    wp::array_t<wp::int32> var_contact_worldid_in,
    wp::array_t<wp::float32> var_contact_adhesion_in,
    wp::array_t<wp::float32> var_efc_force_in,
    wp::int32 var_njmax_in,
    wp::array_t<wp::int32> var_nacon_in,
    wp::array_t<wp::vec_t<6, wp::float32>> var_cfrc_ext_out)
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
        wp::vec_t<2, wp::int32>* var_5;
        wp::vec_t<2, wp::int32> var_6;
        wp::vec_t<2, wp::int32> var_7;
        const wp::int32 var_8 = 0;
        wp::int32 var_9;
        wp::int32* var_10;
        wp::int32 var_11;
        wp::int32 var_12;
        const wp::int32 var_13 = 1;
        wp::int32 var_14;
        wp::int32* var_15;
        wp::int32 var_16;
        wp::int32 var_17;
        bool var_18;
        const wp::int32 var_19 = 0;
        bool var_20;
        const wp::int32 var_21 = 0;
        bool var_22;
        wp::int32* var_23;
        wp::int32 var_24;
        wp::int32 var_25;
        const bool var_26 = true;
        wp::vec_t<6, wp::float32> var_27;
        wp::vec_t<3, wp::float32>* var_28;
        wp::vec_t<3, wp::float32> var_29;
        wp::vec_t<3, wp::float32> var_30;
        wp::int32* var_31;
        wp::int32 var_32;
        wp::vec_t<3, wp::float32>* var_33;
        wp::vec_t<3, wp::float32> var_34;
        wp::vec_t<3, wp::float32> var_35;
        wp::array_t<wp::vec_t<6, wp::float32>> var_36;
        wp::vec_t<3, wp::float32> var_37;
        wp::vec_t<6, wp::float32> var_38;
        wp::vec_t<6, wp::float32> var_39;
        wp::int32* var_40;
        wp::int32 var_41;
        wp::vec_t<3, wp::float32>* var_42;
        wp::vec_t<3, wp::float32> var_43;
        wp::vec_t<3, wp::float32> var_44;
        wp::array_t<wp::vec_t<6, wp::float32>> var_45;
        wp::vec_t<3, wp::float32> var_46;
        wp::vec_t<6, wp::float32> var_47;
        wp::vec_t<6, wp::float32> var_48;
        //---------
        // forward
        // def _cfrc_ext_contact(                                                                 <L 1670>
        // contactid = wp.tid()                                                                   <L 1691>
        var_0 = builtin_tid1d();
        // if contactid >= nacon_in[0]:                                                           <L 1693>
        var_2 = wp::address(var_nacon_in, var_1);
        var_4 = wp::load(var_2);
        var_3 = (var_0 >= var_4);
        if (var_3) {
            // return                                                                             <L 1694>
            continue;
        }
        // geom = contact_geom_in[contactid]                                                      <L 1696>
        var_5 = wp::address(var_contact_geom_in, var_0);
        var_7 = wp::load(var_5);
        var_6 = wp::copy(var_7);
        // id1 = geom_bodyid[geom[0]]                                                             <L 1697>
        var_9 = wp::extract(var_6, var_8);
        var_10 = wp::address(var_geom_bodyid, var_9);
        var_12 = wp::load(var_10);
        var_11 = wp::copy(var_12);
        // id2 = geom_bodyid[geom[1]]                                                             <L 1698>
        var_14 = wp::extract(var_6, var_13);
        var_15 = wp::address(var_geom_bodyid, var_14);
        var_17 = wp::load(var_15);
        var_16 = wp::copy(var_17);
        // if id1 == 0 and id2 == 0:                                                              <L 1700>
        var_20 = (var_11 == var_19);
        var_18 = var_20;
        if (var_18) {
            var_22 = (var_16 == var_21);
            var_18 = var_18 && var_22;
        }
        if (var_18) {
            // return                                                                             <L 1701>
            continue;
        }
        // worldid = contact_worldid_in[contactid]                                                <L 1703>
        var_23 = wp::address(var_contact_worldid_in, var_0);
        var_25 = wp::load(var_23);
        var_24 = wp::copy(var_25);
        // force = support.contact_force_fn(                                                      <L 1706>
        // opt_cone,                                                                              <L 1707>
        // contact_frame_in,                                                                      <L 1708>
        // contact_friction_in,                                                                   <L 1709>
        // contact_dim_in,                                                                        <L 1710>
        // contact_efc_address_in,                                                                <L 1711>
        // contact_adhesion_in,                                                                   <L 1712>
        // efc_force_in,                                                                          <L 1713>
        // njmax_in,                                                                              <L 1714>
        // nacon_in,                                                                              <L 1715>
        // worldid,                                                                               <L 1716>
        // contactid,                                                                             <L 1717>
        // to_world_frame=True,                                                                   <L 1718>
        var_27 = contact_force_fn_0(var_opt_cone, var_contact_frame_in, var_contact_friction_in, var_contact_dim_in, var_contact_efc_address_in, var_contact_adhesion_in, var_efc_force_in, var_njmax_in, var_nacon_in, var_24, var_0, var_26);
        // pos = contact_pos_in[contactid]                                                        <L 1721>
        var_28 = wp::address(var_contact_pos_in, var_0);
        var_30 = wp::load(var_28);
        var_29 = wp::copy(var_30);
        // if id1:                                                                                <L 1724>
        if (var_11) {
            // com1 = subtree_com_in[worldid, body_rootid[id1]]                                   <L 1725>
            var_31 = wp::address(var_body_rootid, var_11);
            var_32 = wp::load(var_31);
            var_33 = wp::address(var_subtree_com_in, var_24, var_32);
            var_35 = wp::load(var_33);
            var_34 = wp::copy(var_35);
            // wp.atomic_sub(cfrc_ext_out[worldid], id1, support.transform_force(force, com1 - pos))       <L 1726>
            var_36 = wp::view(var_cfrc_ext_out, var_24);
            var_37 = wp::sub(var_34, var_29);
            var_38 = transform_force_1(var_27, var_37);
            var_39 = wp::atomic_sub(var_36, var_11, var_38);
        }
        // if id2:                                                                                <L 1728>
        if (var_16) {
            // com2 = subtree_com_in[worldid, body_rootid[id2]]                                   <L 1729>
            var_40 = wp::address(var_body_rootid, var_16);
            var_41 = wp::load(var_40);
            var_42 = wp::address(var_subtree_com_in, var_24, var_41);
            var_44 = wp::load(var_42);
            var_43 = wp::copy(var_44);
            // wp.atomic_add(cfrc_ext_out[worldid], id2, support.transform_force(force, com2 - pos))       <L 1730>
            var_45 = wp::view(var_cfrc_ext_out, var_24);
            var_46 = wp::sub(var_43, var_29);
            var_47 = transform_force_1(var_27, var_46);
            var_48 = wp::atomic_add(var_45, var_16, var_47);
        }
    }
}



extern "C" __global__ void _subtree_com_acc_350f0560_cuda_kernel_forward(
    wp::launch_bounds_t<2> dim,
    wp::array_t<wp::int32> var_body_parentid,
    wp::array_t<wp::vec_t<3, wp::float32>> var_subtree_com_in,
    wp::array_t<wp::int32> var_body_tree_,
    wp::array_t<wp::vec_t<3, wp::float32>> var_subtree_com_out)
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
        wp::int32* var_5;
        wp::int32 var_6;
        wp::int32 var_7;
        const wp::int32 var_8 = 0;
        bool var_9;
        wp::vec_t<3, wp::float32>* var_10;
        wp::vec_t<3, wp::float32> var_11;
        wp::vec_t<3, wp::float32> var_12;
        //---------
        // forward
        // def _subtree_com_acc(                                                                  <L 670>
        // worldid, nodeid = wp.tid()                                                             <L 680>
        builtin_tid2d(var_0, var_1);
        // bodyid = body_tree_[nodeid]                                                            <L 681>
        var_2 = wp::address(var_body_tree_, var_1);
        var_4 = wp::load(var_2);
        var_3 = wp::copy(var_4);
        // pid = body_parentid[bodyid]                                                            <L 682>
        var_5 = wp::address(var_body_parentid, var_3);
        var_7 = wp::load(var_5);
        var_6 = wp::copy(var_7);
        // if bodyid != 0:                                                                        <L 683>
        var_9 = (var_3 != var_8);
        if (var_9) {
            // wp.atomic_add(subtree_com_out, worldid, pid, subtree_com_in[worldid, bodyid])       <L 684>
            var_10 = wp::address(var_subtree_com_in, var_0, var_3);
            var_12 = wp::load(var_10);
            var_11 = wp::atomic_add(var_subtree_com_out, var_0, var_6, var_12);
        }
    }
}



extern "C" __global__ void _angular_momentum_35b41762_cuda_kernel_forward(
    wp::launch_bounds_t<2> dim,
    wp::array_t<wp::int32> var_body_parentid,
    wp::array_t<wp::float32> var_body_mass,
    wp::array_t<wp::float32> var_body_subtreemass,
    wp::array_t<wp::vec_t<3, wp::float32>> var_xipos_in,
    wp::array_t<wp::vec_t<3, wp::float32>> var_subtree_com_in,
    wp::array_t<wp::vec_t<3, wp::float32>> var_subtree_linvel_in,
    wp::array_t<wp::vec_t<6, wp::float32>> var_subtree_bodyvel_in,
    wp::array_t<wp::int32> var_body_tree_,
    wp::array_t<wp::vec_t<3, wp::float32>> var_subtree_angmom_out)
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
        bool var_6;
        wp::int32* var_7;
        wp::int32 var_8;
        wp::int32 var_9;
        wp::vec_t<3, wp::float32>* var_10;
        wp::vec_t<3, wp::float32> var_11;
        wp::vec_t<3, wp::float32> var_12;
        wp::vec_t<3, wp::float32>* var_13;
        wp::vec_t<3, wp::float32> var_14;
        wp::vec_t<3, wp::float32> var_15;
        wp::vec_t<3, wp::float32>* var_16;
        wp::vec_t<3, wp::float32> var_17;
        wp::vec_t<3, wp::float32> var_18;
        wp::vec_t<6, wp::float32>* var_19;
        wp::vec_t<6, wp::float32> var_20;
        wp::vec_t<6, wp::float32> var_21;
        wp::vec_t<3, wp::float32>* var_22;
        wp::vec_t<3, wp::float32> var_23;
        wp::vec_t<3, wp::float32> var_24;
        wp::vec_t<3, wp::float32>* var_25;
        wp::vec_t<3, wp::float32> var_26;
        wp::vec_t<3, wp::float32> var_27;
        wp::shape_t* var_28;
        const wp::int32 var_29 = 0;
        wp::int32 var_30;
        wp::shape_t var_31;
        wp::int32 var_32;
        wp::float32* var_33;
        wp::float32 var_34;
        wp::float32 var_35;
        wp::shape_t* var_36;
        const wp::int32 var_37 = 0;
        wp::int32 var_38;
        wp::shape_t var_39;
        wp::int32 var_40;
        wp::float32* var_41;
        wp::float32 var_42;
        wp::float32 var_43;
        wp::vec_t<3, wp::float32> var_44;
        wp::vec_t<3, wp::float32> var_45;
        wp::vec_t<3, wp::float32> var_46;
        wp::vec_t<3, wp::float32> var_47;
        wp::vec_t<3, wp::float32> var_48;
        wp::vec_t<3, wp::float32> var_49;
        wp::array_t<wp::vec_t<3, wp::float32>> var_50;
        wp::vec_t<3, wp::float32>* var_51;
        wp::vec_t<3, wp::float32> var_52;
        wp::vec_t<3, wp::float32> var_53;
        wp::vec_t<3, wp::float32> var_54;
        wp::vec_t<3, wp::float32> var_55;
        wp::vec_t<3, wp::float32> var_56;
        wp::vec_t<3, wp::float32> var_57;
        wp::array_t<wp::vec_t<3, wp::float32>> var_58;
        wp::vec_t<3, wp::float32> var_59;
        //---------
        // forward
        // def _angular_momentum(                                                                 <L 3891>
        // worldid, nodeid = wp.tid()                                                             <L 3906>
        builtin_tid2d(var_0, var_1);
        // bodyid = body_tree_[nodeid]                                                            <L 3907>
        var_2 = wp::address(var_body_tree_, var_1);
        var_4 = wp::load(var_2);
        var_3 = wp::copy(var_4);
        // if bodyid == 0:                                                                        <L 3909>
        var_6 = (var_3 == var_5);
        if (var_6) {
            // return                                                                             <L 3910>
            continue;
        }
        // pid = body_parentid[bodyid]                                                            <L 3912>
        var_7 = wp::address(var_body_parentid, var_3);
        var_9 = wp::load(var_7);
        var_8 = wp::copy(var_9);
        // xipos = xipos_in[worldid, bodyid]                                                      <L 3914>
        var_10 = wp::address(var_xipos_in, var_0, var_3);
        var_12 = wp::load(var_10);
        var_11 = wp::copy(var_12);
        // com = subtree_com_in[worldid, bodyid]                                                  <L 3915>
        var_13 = wp::address(var_subtree_com_in, var_0, var_3);
        var_15 = wp::load(var_13);
        var_14 = wp::copy(var_15);
        // com_parent = subtree_com_in[worldid, pid]                                              <L 3916>
        var_16 = wp::address(var_subtree_com_in, var_0, var_8);
        var_18 = wp::load(var_16);
        var_17 = wp::copy(var_18);
        // vel = subtree_bodyvel_in[worldid, bodyid]                                              <L 3917>
        var_19 = wp::address(var_subtree_bodyvel_in, var_0, var_3);
        var_21 = wp::load(var_19);
        var_20 = wp::copy(var_21);
        // linvel = subtree_linvel_in[worldid, bodyid]                                            <L 3918>
        var_22 = wp::address(var_subtree_linvel_in, var_0, var_3);
        var_24 = wp::load(var_22);
        var_23 = wp::copy(var_24);
        // linvel_parent = subtree_linvel_in[worldid, pid]  # Data field                          <L 3919>
        var_25 = wp::address(var_subtree_linvel_in, var_0, var_8);
        var_27 = wp::load(var_25);
        var_26 = wp::copy(var_27);
        // mass = body_mass[worldid % body_mass.shape[0], bodyid]                                 <L 3920>
        var_28 = &(var_body_mass.shape);
        var_31 = wp::load(var_28);
        var_30 = wp::extract(var_31, var_29);
        var_32 = wp::mod(var_0, var_30);
        var_33 = wp::address(var_body_mass, var_32, var_3);
        var_35 = wp::load(var_33);
        var_34 = wp::copy(var_35);
        // subtreemass = body_subtreemass[worldid % body_subtreemass.shape[0], bodyid]            <L 3921>
        var_36 = &(var_body_subtreemass.shape);
        var_39 = wp::load(var_36);
        var_38 = wp::extract(var_39, var_37);
        var_40 = wp::mod(var_0, var_38);
        var_41 = wp::address(var_body_subtreemass, var_40, var_3);
        var_43 = wp::load(var_41);
        var_42 = wp::copy(var_43);
        // dx = xipos - com                                                                       <L 3924>
        var_44 = wp::sub(var_11, var_14);
        // dv = wp.spatial_bottom(vel) - linvel                                                   <L 3925>
        var_45 = wp::spatial_bottom(var_20);
        var_46 = wp::sub(var_45, var_23);
        // dp = dv * mass                                                                         <L 3926>
        var_47 = wp::mul(var_46, var_34);
        // dL = wp.cross(dx, dp)                                                                  <L 3927>
        var_48 = wp::cross(var_44, var_47);
        // subtree_angmom_out[worldid, bodyid] += dL                                              <L 3930>
        var_49 = wp::atomic_add(var_subtree_angmom_out, var_0, var_3, var_48);
        // wp.atomic_add(subtree_angmom_out[worldid], pid, subtree_angmom_out[worldid, bodyid])       <L 3933>
        var_50 = wp::view(var_subtree_angmom_out, var_0);
        var_51 = wp::address(var_subtree_angmom_out, var_0, var_3);
        var_53 = wp::load(var_51);
        var_52 = wp::atomic_add(var_50, var_8, var_53);
        // dx = com - com_parent                                                                  <L 3936>
        var_54 = wp::sub(var_14, var_17);
        // dv = linvel - linvel_parent                                                            <L 3937>
        var_55 = wp::sub(var_23, var_26);
        // dv *= subtreemass                                                                      <L 3938>
        var_56 = wp::mul(var_55, var_42);
        // dL = wp.cross(dx, dv)                                                                  <L 3939>
        var_57 = wp::cross(var_54, var_56);
        // wp.atomic_add(subtree_angmom_out[worldid], pid, dL)                                    <L 3940>
        var_58 = wp::view(var_subtree_angmom_out, var_0);
        var_59 = wp::atomic_add(var_58, var_8, var_57);
    }
}



extern "C" __global__ void _joint_tendon_35c34b14_cuda_kernel_forward(
    wp::launch_bounds_t<2> dim,
    wp::array_t<wp::int32> var_jnt_qposadr,
    wp::array_t<wp::int32> var_jnt_dofadr,
    wp::array_t<wp::int32> var_ten_J_rownnz,
    wp::array_t<wp::int32> var_ten_J_rowadr,
    wp::array_t<wp::int32> var_ten_J_colind,
    wp::array_t<wp::int32> var_wrap_objid,
    wp::array_t<wp::float32> var_wrap_prm,
    wp::array_t<wp::int32> var_tendon_jnt_adr,
    wp::array_t<wp::int32> var_wrap_jnt_adr,
    wp::array_t<wp::float32> var_qpos_in,
    wp::array_t<wp::float32> var_ten_J_out,
    wp::array_t<wp::float32> var_ten_length_out)
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
        wp::int32* var_5;
        wp::int32 var_6;
        wp::int32 var_7;
        wp::int32* var_8;
        wp::int32 var_9;
        wp::int32 var_10;
        wp::float32* var_11;
        wp::float32 var_12;
        wp::float32 var_13;
        wp::int32* var_14;
        wp::int32 var_15;
        wp::float32* var_16;
        wp::float32 var_17;
        wp::float32 var_18;
        wp::array_t<wp::float32> var_19;
        wp::float32 var_20;
        wp::int32* var_21;
        wp::int32 var_22;
        wp::int32 var_23;
        wp::int32* var_24;
        wp::int32 var_25;
        wp::int32 var_26;
        wp::int32* var_27;
        wp::int32 var_28;
        wp::int32 var_29;
        wp::range_t var_30;
        wp::int32 var_31;
        wp::int32 var_32;
        wp::int32* var_33;
        bool var_34;
        wp::int32 var_35;
        wp::int32 var_36;
        //---------
        // forward
        // def _joint_tendon(                                                                     <L 3987>
        // worldid, wrapid = wp.tid()                                                             <L 4004>
        builtin_tid2d(var_0, var_1);
        // tenid = tendon_jnt_adr[wrapid]                                                         <L 4006>
        var_2 = wp::address(var_tendon_jnt_adr, var_1);
        var_4 = wp::load(var_2);
        var_3 = wp::copy(var_4);
        // wrapjntid = wrap_jnt_adr[wrapid]                                                       <L 4007>
        var_5 = wp::address(var_wrap_jnt_adr, var_1);
        var_7 = wp::load(var_5);
        var_6 = wp::copy(var_7);
        // wrapobjid = wrap_objid[wrapjntid]                                                      <L 4008>
        var_8 = wp::address(var_wrap_objid, var_6);
        var_10 = wp::load(var_8);
        var_9 = wp::copy(var_10);
        // prm = wrap_prm[wrapjntid]                                                              <L 4009>
        var_11 = wp::address(var_wrap_prm, var_6);
        var_13 = wp::load(var_11);
        var_12 = wp::copy(var_13);
        // L = prm * qpos_in[worldid, jnt_qposadr[wrapobjid]]                                     <L 4012>
        var_14 = wp::address(var_jnt_qposadr, var_9);
        var_15 = wp::load(var_14);
        var_16 = wp::address(var_qpos_in, var_0, var_15);
        var_18 = wp::load(var_16);
        var_17 = wp::mul(var_12, var_18);
        // wp.atomic_add(ten_length_out[worldid], tenid, L)                                       <L 4013>
        var_19 = wp::view(var_ten_length_out, var_0);
        var_20 = wp::atomic_add(var_19, var_3, var_17);
        // dofadr = jnt_dofadr[wrapobjid]                                                         <L 4016>
        var_21 = wp::address(var_jnt_dofadr, var_9);
        var_23 = wp::load(var_21);
        var_22 = wp::copy(var_23);
        // rowadr = ten_J_rowadr[tenid]                                                           <L 4017>
        var_24 = wp::address(var_ten_J_rowadr, var_3);
        var_26 = wp::load(var_24);
        var_25 = wp::copy(var_26);
        // rownnz = ten_J_rownnz[tenid]                                                           <L 4018>
        var_27 = wp::address(var_ten_J_rownnz, var_3);
        var_29 = wp::load(var_27);
        var_28 = wp::copy(var_29);
        // for k in range(rownnz):                                                                <L 4019>
        var_30 = wp::range(var_28);
        start_for_0:;
            if (iter_cmp(var_30) == 0) goto end_for_0;
            var_31 = wp::iter_next(var_30);
            // if ten_J_colind[rowadr + k] == dofadr:                                             <L 4020>
            var_32 = wp::add(var_25, var_31);
            var_33 = wp::address(var_ten_J_colind, var_32);
            var_35 = wp::load(var_33);
            var_34 = (var_35 == var_22);
            if (var_34) {
                // ten_J_out[worldid, rowadr + k] = prm                                           <L 4021>
                var_36 = wp::add(var_25, var_31);
                wp::array_store(var_ten_J_out, var_0, var_36, var_12);
                // break                                                                          <L 4022>
                goto end_for_0;
            }
            goto start_for_0;
        end_for_0:;
    }
}



extern "C" __global__ void _cfrc_3922b605_cuda_kernel_forward(
    wp::launch_bounds_t<2> dim,
    wp::array_t<wp::vec_t<10, wp::float32>> var_cinert_in,
    wp::array_t<wp::vec_t<6, wp::float32>> var_cvel_in,
    wp::array_t<wp::vec_t<6, wp::float32>> var_cacc_in,
    wp::array_t<wp::vec_t<6, wp::float32>> var_cfrc_ext_in,
    bool var_flg_cfrc_ext,
    wp::array_t<wp::vec_t<6, wp::float32>> var_cfrc_int_out)
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
        const wp::int32 var_2 = 0;
        bool var_3;
        const wp::float32 var_4 = 0.0;
        const wp::float32 var_5 = 0.0;
        const wp::float32 var_6 = 0.0;
        const wp::float32 var_7 = 0.0;
        const wp::float32 var_8 = 0.0;
        const wp::float32 var_9 = 0.0;
        wp::vec_t<6, wp::float32> var_10;
        const wp::int32 var_11 = 0;
        wp::vec_t<6, wp::float32>* var_12;
        wp::vec_t<6, wp::float32> var_13;
        wp::vec_t<6, wp::float32> var_14;
        wp::vec_t<10, wp::float32>* var_15;
        wp::vec_t<10, wp::float32> var_16;
        wp::vec_t<10, wp::float32> var_17;
        wp::vec_t<6, wp::float32>* var_18;
        wp::vec_t<6, wp::float32> var_19;
        wp::vec_t<6, wp::float32> var_20;
        wp::vec_t<6, wp::float32> var_21;
        wp::vec_t<6, wp::float32> var_22;
        wp::vec_t<6, wp::float32> var_23;
        wp::vec_t<6, wp::float32> var_24;
        wp::vec_t<6, wp::float32>* var_25;
        wp::vec_t<6, wp::float32> var_26;
        wp::vec_t<6, wp::float32> var_27;
        wp::vec_t<6, wp::float32> var_28;
        //---------
        // forward
        // def _cfrc(                                                                             <L 1399>
        // worldid, bodyid = wp.tid()                                                             <L 1410>
        builtin_tid2d(var_0, var_1);
        // if bodyid == 0:                                                                        <L 1411>
        var_3 = (var_1 == var_2);
        if (var_3) {
            // cfrc_int_out[worldid, 0] = wp.spatial_vector(0.0, 0.0, 0.0, 0.0, 0.0, 0.0)         <L 1412>
            var_10 = wp::vec_t<6, wp::float32>({var_4, var_5, var_6, var_7, var_8, var_9});
            wp::array_store(var_cfrc_int_out, var_0, var_11, var_10);
            // return                                                                             <L 1413>
            continue;
        }
        // cacc = cacc_in[worldid, bodyid]                                                        <L 1414>
        var_12 = wp::address(var_cacc_in, var_0, var_1);
        var_14 = wp::load(var_12);
        var_13 = wp::copy(var_14);
        // cinert = cinert_in[worldid, bodyid]                                                    <L 1415>
        var_15 = wp::address(var_cinert_in, var_0, var_1);
        var_17 = wp::load(var_15);
        var_16 = wp::copy(var_17);
        // cvel = cvel_in[worldid, bodyid]                                                        <L 1416>
        var_18 = wp::address(var_cvel_in, var_0, var_1);
        var_20 = wp::load(var_18);
        var_19 = wp::copy(var_20);
        // frc = math.inert_vec(cinert, cacc)                                                     <L 1417>
        var_21 = inert_vec_0(var_16, var_13);
        // frc += math.motion_cross_force(cvel, math.inert_vec(cinert, cvel))                     <L 1418>
        var_22 = inert_vec_0(var_16, var_19);
        var_23 = motion_cross_force_0(var_19, var_22);
        var_24 = wp::add(var_21, var_23);
        // if flg_cfrc_ext:                                                                       <L 1419>
        if (var_flg_cfrc_ext) {
            // frc -= cfrc_ext_in[worldid, bodyid]                                                <L 1420>
            var_25 = wp::address(var_cfrc_ext_in, var_0, var_1);
            var_27 = wp::load(var_25);
            var_26 = wp::sub(var_24, var_27);
        }
        var_28 = wp::where(var_flg_cfrc_ext, var_26, var_24);
        // cfrc_int_out[worldid, bodyid] = frc                                                    <L 1422>
        wp::array_store(var_cfrc_int_out, var_0, var_1, var_28);
    }
}



extern "C" __global__ void _cdof_3e73eb78_cuda_kernel_forward(
    wp::launch_bounds_t<2> dim,
    wp::array_t<wp::int32> var_body_rootid,
    wp::array_t<wp::int32> var_jnt_type,
    wp::array_t<wp::int32> var_jnt_dofadr,
    wp::array_t<wp::int32> var_jnt_bodyid,
    wp::array_t<wp::mat_t<3, 3, wp::float32>> var_xmat_in,
    wp::array_t<wp::vec_t<3, wp::float32>> var_xanchor_in,
    wp::array_t<wp::vec_t<3, wp::float32>> var_xaxis_in,
    wp::array_t<wp::vec_t<3, wp::float32>> var_subtree_com_in,
    wp::array_t<wp::vec_t<6, wp::float32>> var_cdof_out)
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
        wp::int32* var_5;
        wp::int32 var_6;
        wp::int32 var_7;
        wp::int32* var_8;
        wp::int32 var_9;
        wp::int32 var_10;
        wp::vec_t<3, wp::float32>* var_11;
        wp::vec_t<3, wp::float32> var_12;
        wp::vec_t<3, wp::float32> var_13;
        wp::mat_t<3, 3, wp::float32>* var_14;
        wp::mat_t<3, 3, wp::float32> var_15;
        wp::mat_t<3, 3, wp::float32> var_16;
        wp::int32* var_17;
        wp::int32 var_18;
        wp::vec_t<3, wp::float32>* var_19;
        wp::vec_t<3, wp::float32>* var_20;
        wp::vec_t<3, wp::float32> var_21;
        wp::vec_t<3, wp::float32> var_22;
        wp::vec_t<3, wp::float32> var_23;
        wp::array_t<wp::vec_t<6, wp::float32>> var_24;
        const wp::int32 var_25 = 0;
        bool var_26;
        const wp::float32 var_27 = 0.0;
        const wp::float32 var_28 = 0.0;
        const wp::float32 var_29 = 0.0;
        const wp::float32 var_30 = 1.0;
        const wp::float32 var_31 = 0.0;
        const wp::float32 var_32 = 0.0;
        wp::vec_t<6, wp::float32> var_33;
        const wp::int32 var_34 = 0;
        wp::int32 var_35;
        const wp::float32 var_36 = 0.0;
        const wp::float32 var_37 = 0.0;
        const wp::float32 var_38 = 0.0;
        const wp::float32 var_39 = 0.0;
        const wp::float32 var_40 = 1.0;
        const wp::float32 var_41 = 0.0;
        wp::vec_t<6, wp::float32> var_42;
        const wp::int32 var_43 = 1;
        wp::int32 var_44;
        const wp::float32 var_45 = 0.0;
        const wp::float32 var_46 = 0.0;
        const wp::float32 var_47 = 0.0;
        const wp::float32 var_48 = 0.0;
        const wp::float32 var_49 = 0.0;
        const wp::float32 var_50 = 1.0;
        wp::vec_t<6, wp::float32> var_51;
        const wp::int32 var_52 = 2;
        wp::int32 var_53;
        const wp::int32 var_54 = 0;
        wp::vec_t<3, wp::float32> var_55;
        const wp::int32 var_56 = 0;
        wp::vec_t<3, wp::float32> var_57;
        wp::vec_t<3, wp::float32> var_58;
        wp::vec_t<6, wp::float32> var_59;
        const wp::int32 var_60 = 3;
        wp::int32 var_61;
        const wp::int32 var_62 = 1;
        wp::vec_t<3, wp::float32> var_63;
        const wp::int32 var_64 = 1;
        wp::vec_t<3, wp::float32> var_65;
        wp::vec_t<3, wp::float32> var_66;
        wp::vec_t<6, wp::float32> var_67;
        const wp::int32 var_68 = 4;
        wp::int32 var_69;
        const wp::int32 var_70 = 2;
        wp::vec_t<3, wp::float32> var_71;
        const wp::int32 var_72 = 2;
        wp::vec_t<3, wp::float32> var_73;
        wp::vec_t<3, wp::float32> var_74;
        wp::vec_t<6, wp::float32> var_75;
        const wp::int32 var_76 = 5;
        wp::int32 var_77;
        const wp::int32 var_78 = 1;
        bool var_79;
        const wp::int32 var_80 = 0;
        wp::vec_t<3, wp::float32> var_81;
        const wp::int32 var_82 = 0;
        wp::vec_t<3, wp::float32> var_83;
        wp::vec_t<3, wp::float32> var_84;
        wp::vec_t<6, wp::float32> var_85;
        const wp::int32 var_86 = 0;
        wp::int32 var_87;
        const wp::int32 var_88 = 1;
        wp::vec_t<3, wp::float32> var_89;
        const wp::int32 var_90 = 1;
        wp::vec_t<3, wp::float32> var_91;
        wp::vec_t<3, wp::float32> var_92;
        wp::vec_t<6, wp::float32> var_93;
        const wp::int32 var_94 = 1;
        wp::int32 var_95;
        const wp::int32 var_96 = 2;
        wp::vec_t<3, wp::float32> var_97;
        const wp::int32 var_98 = 2;
        wp::vec_t<3, wp::float32> var_99;
        wp::vec_t<3, wp::float32> var_100;
        wp::vec_t<6, wp::float32> var_101;
        const wp::int32 var_102 = 2;
        wp::int32 var_103;
        const wp::int32 var_104 = 2;
        bool var_105;
        const wp::float32 var_106 = 0.0;
        wp::vec_t<3, wp::float32> var_107;
        wp::vec_t<6, wp::float32> var_108;
        const wp::int32 var_109 = 3;
        bool var_110;
        wp::vec_t<3, wp::float32> var_111;
        wp::vec_t<6, wp::float32> var_112;
        //---------
        // forward
        // def _cdof(                                                                             <L 750>
        // worldid, jntid = wp.tid()                                                              <L 764>
        builtin_tid2d(var_0, var_1);
        // bodyid = jnt_bodyid[jntid]                                                             <L 765>
        var_2 = wp::address(var_jnt_bodyid, var_1);
        var_4 = wp::load(var_2);
        var_3 = wp::copy(var_4);
        // dofid = jnt_dofadr[jntid]                                                              <L 766>
        var_5 = wp::address(var_jnt_dofadr, var_1);
        var_7 = wp::load(var_5);
        var_6 = wp::copy(var_7);
        // jnt_type_ = jnt_type[jntid]                                                            <L 767>
        var_8 = wp::address(var_jnt_type, var_1);
        var_10 = wp::load(var_8);
        var_9 = wp::copy(var_10);
        // xaxis = xaxis_in[worldid, jntid]                                                       <L 768>
        var_11 = wp::address(var_xaxis_in, var_0, var_1);
        var_13 = wp::load(var_11);
        var_12 = wp::copy(var_13);
        // xmat = wp.transpose(xmat_in[worldid, bodyid])                                          <L 769>
        var_14 = wp::address(var_xmat_in, var_0, var_3);
        var_16 = wp::load(var_14);
        var_15 = wp::transpose(var_16);
        // offset = subtree_com_in[worldid, body_rootid[bodyid]] - xanchor_in[worldid, jntid]       <L 772>
        var_17 = wp::address(var_body_rootid, var_3);
        var_18 = wp::load(var_17);
        var_19 = wp::address(var_subtree_com_in, var_0, var_18);
        var_20 = wp::address(var_xanchor_in, var_0, var_1);
        var_22 = wp::load(var_19);
        var_23 = wp::load(var_20);
        var_21 = wp::sub(var_22, var_23);
        // res = cdof_out[worldid]                                                                <L 774>
        var_24 = wp::view(var_cdof_out, var_0);
        // if jnt_type_ == JointType.FREE:                                                        <L 775>
        var_26 = (var_9 == var_25);
        if (var_26) {
            // res[dofid + 0] = wp.spatial_vector(0.0, 0.0, 0.0, 1.0, 0.0, 0.0)                   <L 776>
            var_33 = wp::vec_t<6, wp::float32>({var_27, var_28, var_29, var_30, var_31, var_32});
            var_35 = wp::add(var_6, var_34);
            wp::array_store(var_24, var_35, var_33);
            // res[dofid + 1] = wp.spatial_vector(0.0, 0.0, 0.0, 0.0, 1.0, 0.0)                   <L 777>
            var_42 = wp::vec_t<6, wp::float32>({var_36, var_37, var_38, var_39, var_40, var_41});
            var_44 = wp::add(var_6, var_43);
            wp::array_store(var_24, var_44, var_42);
            // res[dofid + 2] = wp.spatial_vector(0.0, 0.0, 0.0, 0.0, 0.0, 1.0)                   <L 778>
            var_51 = wp::vec_t<6, wp::float32>({var_45, var_46, var_47, var_48, var_49, var_50});
            var_53 = wp::add(var_6, var_52);
            wp::array_store(var_24, var_53, var_51);
            // res[dofid + 3] = wp.spatial_vector(xmat[0], wp.cross(xmat[0], offset))             <L 780>
            var_55 = wp::extract(var_15, var_54);
            var_57 = wp::extract(var_15, var_56);
            var_58 = wp::cross(var_57, var_21);
            var_59 = wp::vec_t<6, wp::float32>(var_55, var_58);
            var_61 = wp::add(var_6, var_60);
            wp::array_store(var_24, var_61, var_59);
            // res[dofid + 4] = wp.spatial_vector(xmat[1], wp.cross(xmat[1], offset))             <L 781>
            var_63 = wp::extract(var_15, var_62);
            var_65 = wp::extract(var_15, var_64);
            var_66 = wp::cross(var_65, var_21);
            var_67 = wp::vec_t<6, wp::float32>(var_63, var_66);
            var_69 = wp::add(var_6, var_68);
            wp::array_store(var_24, var_69, var_67);
            // res[dofid + 5] = wp.spatial_vector(xmat[2], wp.cross(xmat[2], offset))             <L 782>
            var_71 = wp::extract(var_15, var_70);
            var_73 = wp::extract(var_15, var_72);
            var_74 = wp::cross(var_73, var_21);
            var_75 = wp::vec_t<6, wp::float32>(var_71, var_74);
            var_77 = wp::add(var_6, var_76);
            wp::array_store(var_24, var_77, var_75);
        }
        if (!var_26) {
            // elif jnt_type_ == JointType.BALL:  # ball                                          <L 783>
            var_79 = (var_9 == var_78);
            if (var_79) {
                // res[dofid + 0] = wp.spatial_vector(xmat[0], wp.cross(xmat[0], offset))         <L 785>
                var_81 = wp::extract(var_15, var_80);
                var_83 = wp::extract(var_15, var_82);
                var_84 = wp::cross(var_83, var_21);
                var_85 = wp::vec_t<6, wp::float32>(var_81, var_84);
                var_87 = wp::add(var_6, var_86);
                wp::array_store(var_24, var_87, var_85);
                // res[dofid + 1] = wp.spatial_vector(xmat[1], wp.cross(xmat[1], offset))         <L 786>
                var_89 = wp::extract(var_15, var_88);
                var_91 = wp::extract(var_15, var_90);
                var_92 = wp::cross(var_91, var_21);
                var_93 = wp::vec_t<6, wp::float32>(var_89, var_92);
                var_95 = wp::add(var_6, var_94);
                wp::array_store(var_24, var_95, var_93);
                // res[dofid + 2] = wp.spatial_vector(xmat[2], wp.cross(xmat[2], offset))         <L 787>
                var_97 = wp::extract(var_15, var_96);
                var_99 = wp::extract(var_15, var_98);
                var_100 = wp::cross(var_99, var_21);
                var_101 = wp::vec_t<6, wp::float32>(var_97, var_100);
                var_103 = wp::add(var_6, var_102);
                wp::array_store(var_24, var_103, var_101);
            }
            if (!var_79) {
                // elif jnt_type_ == JointType.SLIDE:                                             <L 788>
                var_105 = (var_9 == var_104);
                if (var_105) {
                    // res[dofid] = wp.spatial_vector(wp.vec3(0.0), xaxis)                        <L 789>
                    var_107 = wp::vec_t<3, wp::float32>(var_106);
                    var_108 = wp::vec_t<6, wp::float32>(var_107, var_12);
                    wp::array_store(var_24, var_6, var_108);
                }
                if (!var_105) {
                    // elif jnt_type_ == JointType.HINGE:  # hinge                                <L 790>
                    var_110 = (var_9 == var_109);
                    if (var_110) {
                        // res[dofid] = wp.spatial_vector(xaxis, wp.cross(xaxis, offset))         <L 791>
                        var_111 = wp::cross(var_12, var_21);
                        var_112 = wp::vec_t<6, wp::float32>(var_12, var_111);
                        wp::array_store(var_24, var_6, var_112);
                    }
                }
            }
        }
    }
}



extern "C" __global__ void _qfrc_bias_4085b6de_cuda_kernel_forward(
    wp::launch_bounds_t<2> dim,
    wp::array_t<wp::int32> var_dof_bodyid,
    wp::array_t<wp::vec_t<6, wp::float32>> var_cdof_in,
    wp::array_t<wp::vec_t<6, wp::float32>> var_cfrc_int_in,
    wp::array_t<wp::float32> var_qfrc_bias_out)
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
        wp::vec_t<6, wp::float32>* var_5;
        wp::vec_t<6, wp::float32>* var_6;
        wp::float32 var_7;
        wp::vec_t<6, wp::float32> var_8;
        wp::vec_t<6, wp::float32> var_9;
        //---------
        // forward
        // def _qfrc_bias(                                                                        <L 1455>
        // worldid, dofid = wp.tid()                                                              <L 1464>
        builtin_tid2d(var_0, var_1);
        // bodyid = dof_bodyid[dofid]                                                             <L 1465>
        var_2 = wp::address(var_dof_bodyid, var_1);
        var_4 = wp::load(var_2);
        var_3 = wp::copy(var_4);
        // qfrc_bias_out[worldid, dofid] = wp.dot(cdof_in[worldid, dofid], cfrc_int_in[worldid, bodyid])       <L 1466>
        var_5 = wp::address(var_cdof_in, var_0, var_1);
        var_6 = wp::address(var_cfrc_int_in, var_0, var_3);
        var_8 = wp::load(var_5);
        var_9 = wp::load(var_6);
        var_7 = wp::dot(var_8, var_9);
        wp::array_store(var_qfrc_bias_out, var_0, var_1, var_7);
    }
}



extern "C" __global__ void _cfrc_ext_equality_460e8932_cuda_kernel_forward(
    wp::launch_bounds_t<2> dim,
    wp::array_t<wp::int32> var_body_rootid,
    wp::array_t<wp::int32> var_site_bodyid,
    wp::array_t<wp::vec_t<3, wp::float32>> var_site_pos,
    wp::array_t<wp::quat_t<wp::float32>> var_site_quat,
    wp::array_t<wp::int32> var_eq_obj1id,
    wp::array_t<wp::int32> var_eq_obj2id,
    wp::array_t<wp::int32> var_eq_objtype,
    wp::array_t<wp::vec_t<11, wp::float32>> var_eq_data,
    wp::array_t<wp::vec_t<3, wp::float32>> var_xpos_in,
    wp::array_t<wp::quat_t<wp::float32>> var_xquat_in,
    wp::array_t<wp::mat_t<3, 3, wp::float32>> var_xmat_in,
    wp::array_t<wp::vec_t<3, wp::float32>> var_subtree_com_in,
    wp::array_t<wp::int32> var_efc_id_in,
    wp::array_t<wp::float32> var_efc_force_in,
    wp::array_t<wp::int32> var_ne_connect_in,
    wp::array_t<wp::int32> var_ne_weld_in,
    wp::array_t<wp::vec_t<6, wp::float32>> var_cfrc_ext_out)
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
        wp::int32* var_5;
        wp::int32 var_6;
        wp::int32 var_7;
        const wp::int32 var_8 = 3;
        wp::int32 var_9;
        const wp::int32 var_10 = 6;
        wp::int32 var_11;
        wp::int32 var_12;
        bool var_13;
        bool var_14;
        const wp::int32 var_15 = 3;
        wp::int32 var_16;
        const wp::int32 var_17 = 6;
        wp::int32 var_18;
        wp::int32 var_19;
        wp::int32 var_20;
        const wp::int32 var_21 = 0;
        wp::int32 var_22;
        wp::float32* var_23;
        const wp::int32 var_24 = 1;
        wp::int32 var_25;
        wp::float32* var_26;
        const wp::int32 var_27 = 2;
        wp::int32 var_28;
        wp::float32* var_29;
        wp::vec_t<3, wp::float32> var_30;
        wp::float32 var_31;
        wp::float32 var_32;
        wp::float32 var_33;
        wp::int32* var_34;
        wp::int32 var_35;
        wp::int32 var_36;
        wp::shape_t* var_37;
        const wp::int32 var_38 = 0;
        wp::int32 var_39;
        wp::shape_t var_40;
        wp::int32 var_41;
        wp::vec_t<11, wp::float32>* var_42;
        wp::vec_t<11, wp::float32> var_43;
        wp::vec_t<11, wp::float32> var_44;
        wp::int32* var_45;
        const wp::int32 var_46 = 1;
        bool var_47;
        wp::int32 var_48;
        wp::int32* var_49;
        wp::int32 var_50;
        wp::int32 var_51;
        wp::int32* var_52;
        wp::int32 var_53;
        wp::int32 var_54;
        wp::int32 var_55;
        wp::int32 var_56;
        wp::int32* var_57;
        wp::int32 var_58;
        wp::int32 var_59;
        wp::int32* var_60;
        wp::int32 var_61;
        wp::int32 var_62;
        wp::int32 var_63;
        wp::int32 var_64;
        const wp::float32 var_65 = 0.0;
        const wp::float32 var_66 = 0.0;
        const wp::float32 var_67 = 0.0;
        wp::vec_t<3, wp::float32> var_68;
        const wp::int32 var_69 = 3;
        wp::int32 var_70;
        wp::float32* var_71;
        const wp::int32 var_72 = 4;
        wp::int32 var_73;
        wp::float32* var_74;
        const wp::int32 var_75 = 5;
        wp::int32 var_76;
        wp::float32* var_77;
        wp::vec_t<3, wp::float32> var_78;
        wp::float32 var_79;
        wp::float32 var_80;
        wp::float32 var_81;
        const wp::int32 var_82 = 6;
        wp::float32 var_83;
        const wp::int32 var_84 = 7;
        wp::float32 var_85;
        const wp::int32 var_86 = 8;
        wp::float32 var_87;
        const wp::int32 var_88 = 9;
        wp::float32 var_89;
        wp::quat_t<wp::float32> var_90;
        wp::quat_t<wp::float32>* var_91;
        wp::quat_t<wp::float32> var_92;
        wp::quat_t<wp::float32> var_93;
        wp::quat_t<wp::float32>* var_94;
        wp::quat_t<wp::float32> var_95;
        wp::quat_t<wp::float32> var_96;
        wp::shape_t* var_97;
        const wp::int32 var_98 = 0;
        wp::int32 var_99;
        wp::shape_t var_100;
        wp::int32 var_101;
        wp::quat_t<wp::float32>* var_102;
        wp::quat_t<wp::float32>* var_103;
        wp::quat_t<wp::float32> var_104;
        wp::quat_t<wp::float32> var_105;
        wp::quat_t<wp::float32> var_106;
        wp::quat_t<wp::float32>* var_107;
        wp::quat_t<wp::float32>* var_108;
        wp::quat_t<wp::float32> var_109;
        wp::quat_t<wp::float32> var_110;
        wp::quat_t<wp::float32> var_111;
        wp::quat_t<wp::float32> var_112;
        wp::quat_t<wp::float32> var_113;
        wp::quat_t<wp::float32> var_114;
        wp::quat_t<wp::float32> var_115;
        wp::quat_t<wp::float32> var_116;
        const wp::float32 var_117 = 0.5;
        const wp::int32 var_118 = 10;
        wp::float32 var_119;
        wp::float32 var_120;
        const wp::int32 var_121 = 1;
        wp::float32 var_122;
        const wp::int32 var_123 = 2;
        wp::float32 var_124;
        const wp::int32 var_125 = 3;
        wp::float32 var_126;
        wp::vec_t<3, wp::float32> var_127;
        wp::vec_t<3, wp::float32> var_128;
        wp::vec_t<3, wp::float32> var_129;
        const wp::int32 var_130 = 0;
        wp::float32 var_131;
        const wp::int32 var_132 = 1;
        wp::float32 var_133;
        const wp::int32 var_134 = 2;
        wp::float32 var_135;
        wp::vec_t<3, wp::float32> var_136;
        const wp::int32 var_137 = 3;
        wp::float32 var_138;
        const wp::int32 var_139 = 4;
        wp::float32 var_140;
        const wp::int32 var_141 = 5;
        wp::float32 var_142;
        wp::vec_t<3, wp::float32> var_143;
        wp::vec_t<3, wp::float32> var_144;
        wp::shape_t* var_145;
        const wp::int32 var_146 = 0;
        wp::int32 var_147;
        wp::shape_t var_148;
        wp::int32 var_149;
        wp::vec_t<3, wp::float32>* var_150;
        wp::vec_t<3, wp::float32> var_151;
        wp::vec_t<3, wp::float32> var_152;
        wp::vec_t<3, wp::float32> var_153;
        wp::mat_t<3, 3, wp::float32>* var_154;
        wp::vec_t<3, wp::float32> var_155;
        wp::mat_t<3, 3, wp::float32> var_156;
        wp::vec_t<3, wp::float32>* var_157;
        wp::vec_t<3, wp::float32> var_158;
        wp::vec_t<3, wp::float32> var_159;
        wp::int32* var_160;
        wp::int32 var_161;
        wp::vec_t<3, wp::float32>* var_162;
        wp::vec_t<3, wp::float32> var_163;
        wp::vec_t<3, wp::float32> var_164;
        wp::vec_t<3, wp::float32> var_165;
        wp::vec_t<3, wp::float32> var_166;
        wp::vec_t<3, wp::float32> var_167;
        wp::vec_t<6, wp::float32> var_168;
        wp::array_t<wp::vec_t<6, wp::float32>> var_169;
        wp::vec_t<6, wp::float32> var_170;
        const wp::int32 var_171 = 3;
        wp::float32 var_172;
        const wp::int32 var_173 = 4;
        wp::float32 var_174;
        const wp::int32 var_175 = 5;
        wp::float32 var_176;
        wp::vec_t<3, wp::float32> var_177;
        const wp::int32 var_178 = 0;
        wp::float32 var_179;
        const wp::int32 var_180 = 1;
        wp::float32 var_181;
        const wp::int32 var_182 = 2;
        wp::float32 var_183;
        wp::vec_t<3, wp::float32> var_184;
        wp::vec_t<3, wp::float32> var_185;
        wp::shape_t* var_186;
        const wp::int32 var_187 = 0;
        wp::int32 var_188;
        wp::shape_t var_189;
        wp::int32 var_190;
        wp::vec_t<3, wp::float32>* var_191;
        wp::vec_t<3, wp::float32> var_192;
        wp::vec_t<3, wp::float32> var_193;
        wp::vec_t<3, wp::float32> var_194;
        wp::mat_t<3, 3, wp::float32>* var_195;
        wp::vec_t<3, wp::float32> var_196;
        wp::mat_t<3, 3, wp::float32> var_197;
        wp::vec_t<3, wp::float32>* var_198;
        wp::vec_t<3, wp::float32> var_199;
        wp::vec_t<3, wp::float32> var_200;
        wp::int32* var_201;
        wp::int32 var_202;
        wp::vec_t<3, wp::float32>* var_203;
        wp::vec_t<3, wp::float32> var_204;
        wp::vec_t<3, wp::float32> var_205;
        wp::vec_t<3, wp::float32> var_206;
        wp::vec_t<3, wp::float32> var_207;
        wp::vec_t<3, wp::float32> var_208;
        wp::vec_t<6, wp::float32> var_209;
        wp::array_t<wp::vec_t<6, wp::float32>> var_210;
        wp::vec_t<6, wp::float32> var_211;
        wp::vec_t<3, wp::float32> var_212;
        wp::vec_t<3, wp::float32> var_213;
        wp::vec_t<3, wp::float32> var_214;
        wp::vec_t<3, wp::float32> var_215;
        wp::vec_t<6, wp::float32> var_216;
        //---------
        // forward
        // def _cfrc_ext_equality(                                                                <L 1540>
        // worldid, eqid = wp.tid()                                                               <L 1563>
        builtin_tid2d(var_0, var_1);
        // ne_connect = ne_connect_in[worldid]                                                    <L 1565>
        var_2 = wp::address(var_ne_connect_in, var_0);
        var_4 = wp::load(var_2);
        var_3 = wp::copy(var_4);
        // ne_weld = ne_weld_in[worldid]                                                          <L 1566>
        var_5 = wp::address(var_ne_weld_in, var_0);
        var_7 = wp::load(var_5);
        var_6 = wp::copy(var_7);
        // num_connect = ne_connect // 3                                                          <L 1567>
        var_9 = wp::floordiv(var_3, var_8);
        // if eqid >= num_connect + ne_weld // 6:                                                 <L 1569>
        var_11 = wp::floordiv(var_6, var_10);
        var_12 = wp::add(var_9, var_11);
        var_13 = (var_1 >= var_12);
        if (var_13) {
            // return                                                                             <L 1570>
            continue;
        }
        // is_connect = eqid < num_connect                                                        <L 1572>
        var_14 = (var_1 < var_9);
        // if is_connect:                                                                         <L 1573>
        if (var_14) {
            // efcid = 3 * eqid                                                                   <L 1574>
            var_16 = wp::mul(var_15, var_1);
        }
        if (!var_14) {
            // efcid = 6 * eqid - ne_connect                                                      <L 1576>
            var_18 = wp::mul(var_17, var_1);
            var_19 = wp::sub(var_18, var_3);
        }
        var_20 = wp::where(var_14, var_16, var_19);
        // cfrc_force = wp.vec3(                                                                  <L 1578>
        // efc_force_in[worldid, efcid + 0],                                                      <L 1579>
        var_22 = wp::add(var_20, var_21);
        var_23 = wp::address(var_efc_force_in, var_0, var_22);
        // efc_force_in[worldid, efcid + 1],                                                      <L 1580>
        var_25 = wp::add(var_20, var_24);
        var_26 = wp::address(var_efc_force_in, var_0, var_25);
        // efc_force_in[worldid, efcid + 2],                                                      <L 1581>
        var_28 = wp::add(var_20, var_27);
        var_29 = wp::address(var_efc_force_in, var_0, var_28);
        var_31 = wp::load(var_23);
        var_32 = wp::load(var_26);
        var_33 = wp::load(var_29);
        var_30 = wp::vec_t<3, wp::float32>(var_31, var_32, var_33);
        // id = efc_id_in[worldid, efcid]                                                         <L 1584>
        var_34 = wp::address(var_efc_id_in, var_0, var_20);
        var_36 = wp::load(var_34);
        var_35 = wp::copy(var_36);
        // eq_data_ = eq_data[worldid % eq_data.shape[0], id]                                     <L 1585>
        var_37 = &(var_eq_data.shape);
        var_40 = wp::load(var_37);
        var_39 = wp::extract(var_40, var_38);
        var_41 = wp::mod(var_0, var_39);
        var_42 = wp::address(var_eq_data, var_41, var_35);
        var_44 = wp::load(var_42);
        var_43 = wp::copy(var_44);
        // body_semantic = eq_objtype[id] == ObjType.BODY                                         <L 1586>
        var_45 = wp::address(var_eq_objtype, var_35);
        var_48 = wp::load(var_45);
        var_47 = (var_48 == var_46);
        // obj1 = eq_obj1id[id]                                                                   <L 1588>
        var_49 = wp::address(var_eq_obj1id, var_35);
        var_51 = wp::load(var_49);
        var_50 = wp::copy(var_51);
        // obj2 = eq_obj2id[id]                                                                   <L 1589>
        var_52 = wp::address(var_eq_obj2id, var_35);
        var_54 = wp::load(var_52);
        var_53 = wp::copy(var_54);
        // if body_semantic:                                                                      <L 1591>
        if (var_47) {
            // bodyid1 = obj1                                                                     <L 1592>
            var_55 = wp::copy(var_50);
            // bodyid2 = obj2                                                                     <L 1593>
            var_56 = wp::copy(var_53);
        }
        if (!var_47) {
            // bodyid1 = site_bodyid[obj1]                                                        <L 1595>
            var_57 = wp::address(var_site_bodyid, var_50);
            var_59 = wp::load(var_57);
            var_58 = wp::copy(var_59);
            // bodyid2 = site_bodyid[obj2]                                                        <L 1596>
            var_60 = wp::address(var_site_bodyid, var_53);
            var_62 = wp::load(var_60);
            var_61 = wp::copy(var_62);
        }
        var_63 = wp::where(var_47, var_55, var_58);
        var_64 = wp::where(var_47, var_56, var_61);
        // if is_connect:                                                                         <L 1598>
        if (var_14) {
            // cfrc_torque = wp.vec3(0.0, 0.0, 0.0)  # no torque from connect                     <L 1599>
            var_68 = wp::vec_t<3, wp::float32>(var_65, var_66, var_67);
        }
        if (!var_14) {
            // force_torque = wp.vec3(                                                            <L 1601>
            // efc_force_in[worldid, efcid + 3],                                                  <L 1602>
            var_70 = wp::add(var_20, var_69);
            var_71 = wp::address(var_efc_force_in, var_0, var_70);
            // efc_force_in[worldid, efcid + 4],                                                  <L 1603>
            var_73 = wp::add(var_20, var_72);
            var_74 = wp::address(var_efc_force_in, var_0, var_73);
            // efc_force_in[worldid, efcid + 5],                                                  <L 1604>
            var_76 = wp::add(var_20, var_75);
            var_77 = wp::address(var_efc_force_in, var_0, var_76);
            var_79 = wp::load(var_71);
            var_80 = wp::load(var_74);
            var_81 = wp::load(var_77);
            var_78 = wp::vec_t<3, wp::float32>(var_79, var_80, var_81);
            // if body_semantic:                                                                  <L 1606>
            if (var_47) {
                // relpose = wp.quat(eq_data_[6], eq_data_[7], eq_data_[8], eq_data_[9])          <L 1607>
                var_83 = wp::extract(var_43, var_82);
                var_85 = wp::extract(var_43, var_84);
                var_87 = wp::extract(var_43, var_86);
                var_89 = wp::extract(var_43, var_88);
                var_90 = wp::quat_t<wp::float32>(var_83, var_85, var_87, var_89);
                // q0 = math.mul_quat(xquat_in[worldid, bodyid1], relpose)                        <L 1608>
                var_91 = wp::address(var_xquat_in, var_0, var_63);
                var_93 = wp::load(var_91);
                var_92 = mul_quat_0(var_93, var_90);
                // q1 = xquat_in[worldid, bodyid2]                                                <L 1609>
                var_94 = wp::address(var_xquat_in, var_0, var_64);
                var_96 = wp::load(var_94);
                var_95 = wp::copy(var_96);
            }
            if (!var_47) {
                // site_quat_id = worldid % site_quat.shape[0]                                    <L 1611>
                var_97 = &(var_site_quat.shape);
                var_100 = wp::load(var_97);
                var_99 = wp::extract(var_100, var_98);
                var_101 = wp::mod(var_0, var_99);
                // q0 = math.mul_quat(xquat_in[worldid, bodyid1], site_quat[site_quat_id, obj1])       <L 1612>
                var_102 = wp::address(var_xquat_in, var_0, var_63);
                var_103 = wp::address(var_site_quat, var_101, var_50);
                var_105 = wp::load(var_102);
                var_106 = wp::load(var_103);
                var_104 = mul_quat_0(var_105, var_106);
                // q1 = math.mul_quat(xquat_in[worldid, bodyid2], site_quat[site_quat_id, obj2])       <L 1613>
                var_107 = wp::address(var_xquat_in, var_0, var_64);
                var_108 = wp::address(var_site_quat, var_101, var_53);
                var_110 = wp::load(var_107);
                var_111 = wp::load(var_108);
                var_109 = mul_quat_0(var_110, var_111);
            }
            var_112 = wp::where(var_47, var_92, var_104);
            var_113 = wp::where(var_47, var_95, var_109);
            // quat = math.mul_quat(math.quat_mul_axis(q1, force_torque), math.quat_inv(q0))       <L 1615>
            var_114 = quat_mul_axis_0(var_113, var_78);
            var_115 = quat_inv_0(var_112);
            var_116 = mul_quat_0(var_114, var_115);
            // cfrc_torque = (0.5 * eq_data_[10]) * wp.vec3(quat[1], quat[2], quat[3])            <L 1616>
            var_119 = wp::extract(var_43, var_118);
            var_120 = wp::mul(var_117, var_119);
            var_122 = wp::extract(var_116, var_121);
            var_124 = wp::extract(var_116, var_123);
            var_126 = wp::extract(var_116, var_125);
            var_127 = wp::vec_t<3, wp::float32>(var_122, var_124, var_126);
            var_128 = wp::mul(var_120, var_127);
        }
        var_129 = wp::where(var_14, var_68, var_128);
        // if bodyid1:                                                                            <L 1619>
        if (var_63) {
            // if body_semantic:                                                                  <L 1620>
            if (var_47) {
                // if is_connect:                                                                 <L 1621>
                if (var_14) {
                    // offset = wp.vec3(eq_data_[0], eq_data_[1], eq_data_[2])                    <L 1622>
                    var_131 = wp::extract(var_43, var_130);
                    var_133 = wp::extract(var_43, var_132);
                    var_135 = wp::extract(var_43, var_134);
                    var_136 = wp::vec_t<3, wp::float32>(var_131, var_133, var_135);
                }
                if (!var_14) {
                    // offset = wp.vec3(eq_data_[3], eq_data_[4], eq_data_[5])                    <L 1624>
                    var_138 = wp::extract(var_43, var_137);
                    var_140 = wp::extract(var_43, var_139);
                    var_142 = wp::extract(var_43, var_141);
                    var_143 = wp::vec_t<3, wp::float32>(var_138, var_140, var_142);
                }
                var_144 = wp::where(var_14, var_136, var_143);
            }
            if (!var_47) {
                // offset = site_pos[worldid % site_pos.shape[0], obj1]                           <L 1626>
                var_145 = &(var_site_pos.shape);
                var_148 = wp::load(var_145);
                var_147 = wp::extract(var_148, var_146);
                var_149 = wp::mod(var_0, var_147);
                var_150 = wp::address(var_site_pos, var_149, var_50);
                var_152 = wp::load(var_150);
                var_151 = wp::copy(var_152);
            }
            var_153 = wp::where(var_47, var_144, var_151);
            // pos = xmat_in[worldid, bodyid1] @ offset + xpos_in[worldid, bodyid1]               <L 1629>
            var_154 = wp::address(var_xmat_in, var_0, var_63);
            var_156 = wp::load(var_154);
            var_155 = wp::mul(var_156, var_153);
            var_157 = wp::address(var_xpos_in, var_0, var_63);
            var_159 = wp::load(var_157);
            var_158 = wp::add(var_155, var_159);
            // newpos = subtree_com_in[worldid, body_rootid[bodyid1]]                             <L 1632>
            var_160 = wp::address(var_body_rootid, var_63);
            var_161 = wp::load(var_160);
            var_162 = wp::address(var_subtree_com_in, var_0, var_161);
            var_164 = wp::load(var_162);
            var_163 = wp::copy(var_164);
            // dif = newpos - pos                                                                 <L 1634>
            var_165 = wp::sub(var_163, var_158);
            // cfrc_com = wp.spatial_vector(cfrc_torque - wp.cross(dif, cfrc_force), cfrc_force)       <L 1635>
            var_166 = wp::cross(var_165, var_30);
            var_167 = wp::sub(var_129, var_166);
            var_168 = wp::vec_t<6, wp::float32>(var_167, var_30);
            // wp.atomic_add(cfrc_ext_out[worldid], bodyid1, cfrc_com)                            <L 1638>
            var_169 = wp::view(var_cfrc_ext_out, var_0);
            var_170 = wp::atomic_add(var_169, var_63, var_168);
        }
        // if bodyid2:                                                                            <L 1641>
        if (var_64) {
            // if body_semantic:                                                                  <L 1642>
            if (var_47) {
                // if is_connect:                                                                 <L 1643>
                if (var_14) {
                    // offset = wp.vec3(eq_data_[3], eq_data_[4], eq_data_[5])                    <L 1644>
                    var_172 = wp::extract(var_43, var_171);
                    var_174 = wp::extract(var_43, var_173);
                    var_176 = wp::extract(var_43, var_175);
                    var_177 = wp::vec_t<3, wp::float32>(var_172, var_174, var_176);
                }
                if (!var_14) {
                    // offset = wp.vec3(eq_data_[0], eq_data_[1], eq_data_[2])                    <L 1646>
                    var_179 = wp::extract(var_43, var_178);
                    var_181 = wp::extract(var_43, var_180);
                    var_183 = wp::extract(var_43, var_182);
                    var_184 = wp::vec_t<3, wp::float32>(var_179, var_181, var_183);
                }
                var_185 = wp::where(var_14, var_177, var_184);
            }
            if (!var_47) {
                // offset = site_pos[worldid % site_pos.shape[0], obj2]                           <L 1648>
                var_186 = &(var_site_pos.shape);
                var_189 = wp::load(var_186);
                var_188 = wp::extract(var_189, var_187);
                var_190 = wp::mod(var_0, var_188);
                var_191 = wp::address(var_site_pos, var_190, var_53);
                var_193 = wp::load(var_191);
                var_192 = wp::copy(var_193);
            }
            var_194 = wp::where(var_47, var_185, var_192);
            // pos = xmat_in[worldid, bodyid2] @ offset + xpos_in[worldid, bodyid2]               <L 1651>
            var_195 = wp::address(var_xmat_in, var_0, var_64);
            var_197 = wp::load(var_195);
            var_196 = wp::mul(var_197, var_194);
            var_198 = wp::address(var_xpos_in, var_0, var_64);
            var_200 = wp::load(var_198);
            var_199 = wp::add(var_196, var_200);
            // newpos = subtree_com_in[worldid, body_rootid[bodyid2]]                             <L 1654>
            var_201 = wp::address(var_body_rootid, var_64);
            var_202 = wp::load(var_201);
            var_203 = wp::address(var_subtree_com_in, var_0, var_202);
            var_205 = wp::load(var_203);
            var_204 = wp::copy(var_205);
            // dif = newpos - pos                                                                 <L 1656>
            var_206 = wp::sub(var_204, var_199);
            // cfrc_com = wp.spatial_vector(cfrc_torque - wp.cross(dif, cfrc_force), cfrc_force)       <L 1657>
            var_207 = wp::cross(var_206, var_30);
            var_208 = wp::sub(var_129, var_207);
            var_209 = wp::vec_t<6, wp::float32>(var_208, var_30);
            // wp.atomic_sub(cfrc_ext_out[worldid], bodyid2, cfrc_com)                            <L 1660>
            var_210 = wp::view(var_cfrc_ext_out, var_0);
            var_211 = wp::atomic_sub(var_210, var_64, var_209);
        }
        var_212 = wp::where(var_64, var_194, var_153);
        var_213 = wp::where(var_64, var_199, var_158);
        var_214 = wp::where(var_64, var_204, var_163);
        var_215 = wp::where(var_64, var_206, var_165);
        var_216 = wp::where(var_64, var_209, var_168);
    }
}



extern "C" __global__ void _flex_edges_4aa58739_cuda_kernel_forward(
    wp::launch_bounds_t<2> dim,
    wp::array_t<wp::int32> var_body_rootid,
    wp::array_t<wp::int32> var_flex_vertadr,
    wp::array_t<wp::int32> var_flex_vertbodyid,
    wp::array_t<wp::vec_t<2, wp::int32>> var_flex_edge,
    wp::array_t<wp::int32> var_flexedge_J_rownnz,
    wp::array_t<wp::int32> var_flexedge_J_rowadr,
    wp::array_t<wp::int32> var_flexedge_J_colind,
    wp::array_t<wp::int32> var_body_isdofancestor,
    wp::array_t<wp::int32> var_flex_edgeflexid,
    wp::array_t<wp::float32> var_qvel_in,
    wp::array_t<wp::vec_t<3, wp::float32>> var_subtree_com_in,
    wp::array_t<wp::vec_t<6, wp::float32>> var_cdof_in,
    wp::array_t<wp::vec_t<3, wp::float32>> var_flexvert_xpos_in,
    wp::array_t<wp::float32> var_flexedge_J_out,
    wp::array_t<wp::float32> var_flexedge_length_out,
    wp::array_t<wp::float32> var_flexedge_velocity_out)
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
        wp::int32* var_5;
        wp::int32 var_6;
        wp::int32 var_7;
        wp::vec_t<2, wp::int32>* var_8;
        wp::vec_t<2, wp::int32> var_9;
        wp::vec_t<2, wp::int32> var_10;
        const wp::int32 var_11 = 0;
        wp::int32 var_12;
        wp::int32 var_13;
        const wp::int32 var_14 = 1;
        wp::int32 var_15;
        wp::int32 var_16;
        wp::vec_t<3, wp::float32>* var_17;
        wp::vec_t<3, wp::float32> var_18;
        wp::vec_t<3, wp::float32> var_19;
        wp::vec_t<3, wp::float32>* var_20;
        wp::vec_t<3, wp::float32> var_21;
        wp::vec_t<3, wp::float32> var_22;
        wp::vec_t<3, wp::float32> var_23;
        wp::vec_t<3, wp::float32> var_24;
        wp::float32 var_25;
        wp::int32* var_26;
        wp::int32 var_27;
        wp::int32 var_28;
        wp::int32* var_29;
        wp::int32 var_30;
        wp::int32 var_31;
        wp::int32* var_32;
        wp::int32 var_33;
        wp::int32 var_34;
        bool var_35;
        const wp::int32 var_36 = 0;
        bool var_37;
        const wp::int32 var_38 = 0;
        bool var_39;
        const wp::int32 var_40 = 0;
        bool var_41;
        const wp::float32 var_42 = 0.0;
        wp::int32* var_43;
        wp::int32 var_44;
        wp::int32 var_45;
        wp::int32* var_46;
        wp::int32 var_47;
        wp::int32 var_48;
        wp::int32* var_49;
        wp::int32 var_50;
        wp::int32 var_51;
        wp::vec_t<3, wp::float32>* var_52;
        wp::vec_t<3, wp::float32> var_53;
        wp::vec_t<3, wp::float32> var_54;
        bool var_55;
        wp::vec_t<3, wp::float32>* var_56;
        wp::vec_t<3, wp::float32> var_57;
        wp::vec_t<3, wp::float32> var_58;
        wp::vec_t<3, wp::float32> var_59;
        wp::vec_t<3, wp::float32> var_60;
        wp::vec_t<3, wp::float32> var_61;
        const wp::float32 var_62 = 0.0;
        wp::float32 var_63;
        wp::range_t var_64;
        wp::int32 var_65;
        wp::int32 var_66;
        wp::int32* var_67;
        wp::int32 var_68;
        wp::int32 var_69;
        wp::vec_t<6, wp::float32>* var_70;
        wp::vec_t<6, wp::float32> var_71;
        wp::vec_t<6, wp::float32> var_72;
        wp::vec_t<3, wp::float32> var_73;
        wp::vec_t<3, wp::float32> var_74;
        const wp::float32 var_75 = 0.0;
        wp::float32 var_76;
        wp::int32* var_77;
        const wp::int32 var_78 = 0;
        bool var_79;
        wp::int32 var_80;
        wp::vec_t<3, wp::float32> var_81;
        wp::vec_t<3, wp::float32> var_82;
        wp::float32 var_83;
        wp::float32 var_84;
        wp::float32 var_85;
        wp::int32* var_86;
        const wp::int32 var_87 = 0;
        bool var_88;
        wp::int32 var_89;
        wp::vec_t<3, wp::float32> var_90;
        wp::vec_t<3, wp::float32> var_91;
        wp::float32 var_92;
        wp::float32 var_93;
        wp::float32 var_94;
        wp::int32 var_95;
        wp::float32* var_96;
        wp::float32 var_97;
        wp::float32 var_98;
        wp::float32 var_99;
        //---------
        // forward
        // def _flex_edges(                                                                       <L 347>
        // worldid, edgeid = wp.tid()                                                             <L 368>
        builtin_tid2d(var_0, var_1);
        // f = flex_edgeflexid[edgeid]                                                            <L 369>
        var_2 = wp::address(var_flex_edgeflexid, var_1);
        var_4 = wp::load(var_2);
        var_3 = wp::copy(var_4);
        // vbase = flex_vertadr[f]                                                                <L 371>
        var_5 = wp::address(var_flex_vertadr, var_3);
        var_7 = wp::load(var_5);
        var_6 = wp::copy(var_7);
        // v = flex_edge[edgeid]                                                                  <L 372>
        var_8 = wp::address(var_flex_edge, var_1);
        var_10 = wp::load(var_8);
        var_9 = wp::copy(var_10);
        // vbase0 = vbase + v[0]                                                                  <L 373>
        var_12 = wp::extract(var_9, var_11);
        var_13 = wp::add(var_6, var_12);
        // vbase1 = vbase + v[1]                                                                  <L 374>
        var_15 = wp::extract(var_9, var_14);
        var_16 = wp::add(var_6, var_15);
        // pos1 = flexvert_xpos_in[worldid, vbase0]                                               <L 376>
        var_17 = wp::address(var_flexvert_xpos_in, var_0, var_13);
        var_19 = wp::load(var_17);
        var_18 = wp::copy(var_19);
        // pos2 = flexvert_xpos_in[worldid, vbase1]                                               <L 377>
        var_20 = wp::address(var_flexvert_xpos_in, var_0, var_16);
        var_22 = wp::load(var_20);
        var_21 = wp::copy(var_22);
        // vec = pos2 - pos1                                                                      <L 378>
        var_23 = wp::sub(var_21, var_18);
        // edge, edge_length = math.normalize_with_norm(vec)                                      <L 379>
        normalize_with_norm_0(var_23, var_24, var_25);
        // flexedge_length_out[worldid, edgeid] = edge_length                                     <L 380>
        wp::array_store(var_flexedge_length_out, var_0, var_1, var_25);
        // b1 = flex_vertbodyid[vbase0]                                                           <L 382>
        var_26 = wp::address(var_flex_vertbodyid, var_13);
        var_28 = wp::load(var_26);
        var_27 = wp::copy(var_28);
        // b2 = flex_vertbodyid[vbase1]                                                           <L 383>
        var_29 = wp::address(var_flex_vertbodyid, var_16);
        var_31 = wp::load(var_29);
        var_30 = wp::copy(var_31);
        // rownnz = flexedge_J_rownnz[edgeid]                                                     <L 385>
        var_32 = wp::address(var_flexedge_J_rownnz, var_1);
        var_34 = wp::load(var_32);
        var_33 = wp::copy(var_34);
        // if b1 < 0 or b2 < 0 or rownnz == 0:                                                    <L 386>
        var_37 = (var_27 < var_36);
        var_35 = var_37;
        if (!var_35) {
            var_39 = (var_30 < var_38);
            var_35 = var_35 || var_39;
        }
        if (!var_35) {
            var_41 = (var_33 == var_40);
            var_35 = var_35 || var_41;
        }
        if (var_35) {
            // flexedge_velocity_out[worldid, edgeid] = 0.0                                       <L 387>
            wp::array_store(var_flexedge_velocity_out, var_0, var_1, var_42);
            // return                                                                             <L 388>
            continue;
        }
        // rowadr = flexedge_J_rowadr[edgeid]                                                     <L 390>
        var_43 = wp::address(var_flexedge_J_rowadr, var_1);
        var_45 = wp::load(var_43);
        var_44 = wp::copy(var_45);
        // root1 = body_rootid[b1]                                                                <L 391>
        var_46 = wp::address(var_body_rootid, var_27);
        var_48 = wp::load(var_46);
        var_47 = wp::copy(var_48);
        // root2 = body_rootid[b2]                                                                <L 392>
        var_49 = wp::address(var_body_rootid, var_30);
        var_51 = wp::load(var_49);
        var_50 = wp::copy(var_51);
        // com1 = subtree_com_in[worldid, root1]                                                  <L 393>
        var_52 = wp::address(var_subtree_com_in, var_0, var_47);
        var_54 = wp::load(var_52);
        var_53 = wp::copy(var_54);
        // com2 = com1 if root2 == root1 else subtree_com_in[worldid, root2]                      <L 394>
        var_55 = (var_50 == var_47);
        if (var_55) {
        }
        if (!var_55) {
            var_56 = wp::address(var_subtree_com_in, var_0, var_50);
            var_58 = wp::load(var_56);
            var_57 = wp::copy(var_58);
        }
        var_59 = wp::where(var_55, var_53, var_57);
        // offset1 = pos1 - com1                                                                  <L 395>
        var_60 = wp::sub(var_18, var_53);
        // offset2 = pos2 - com2                                                                  <L 396>
        var_61 = wp::sub(var_21, var_59);
        // vel = float(0.0)                                                                       <L 397>
        var_63 = wp::float(var_62);
        // for i in range(rownnz):                                                                <L 399>
        var_64 = wp::range(var_33);
        start_for_1:;
            if (iter_cmp(var_64) == 0) goto end_for_1;
            var_65 = wp::iter_next(var_64);
            // dofid = flexedge_J_colind[rowadr + i]                                              <L 400>
            var_66 = wp::add(var_44, var_65);
            var_67 = wp::address(var_flexedge_J_colind, var_66);
            var_69 = wp::load(var_67);
            var_68 = wp::copy(var_69);
            // cdof = cdof_in[worldid, dofid]                                                     <L 401>
            var_70 = wp::address(var_cdof_in, var_0, var_68);
            var_72 = wp::load(var_70);
            var_71 = wp::copy(var_72);
            // cdof_ang = wp.spatial_top(cdof)                                                    <L 402>
            var_73 = wp::spatial_top(var_71);
            // cdof_lin = wp.spatial_bottom(cdof)                                                 <L 403>
            var_74 = wp::spatial_bottom(var_71);
            // J_val = float(0.0)                                                                 <L 404>
            var_76 = wp::float(var_75);
            // if body_isdofancestor[b1, dofid] != 0:                                             <L 405>
            var_77 = wp::address(var_body_isdofancestor, var_27, var_68);
            var_80 = wp::load(var_77);
            var_79 = (var_80 != var_78);
            if (var_79) {
                // jacp1 = cdof_lin + wp.cross(cdof_ang, offset1)                                 <L 406>
                var_81 = wp::cross(var_73, var_60);
                var_82 = wp::add(var_74, var_81);
                // J_val -= wp.dot(jacp1, edge)                                                   <L 407>
                var_83 = wp::dot(var_82, var_24);
                var_84 = wp::sub(var_76, var_83);
            }
            var_85 = wp::where(var_79, var_84, var_76);
            // if body_isdofancestor[b2, dofid] != 0:                                             <L 408>
            var_86 = wp::address(var_body_isdofancestor, var_30, var_68);
            var_89 = wp::load(var_86);
            var_88 = (var_89 != var_87);
            if (var_88) {
                // jacp2 = cdof_lin + wp.cross(cdof_ang, offset2)                                 <L 409>
                var_90 = wp::cross(var_73, var_61);
                var_91 = wp::add(var_74, var_90);
                // J_val += wp.dot(jacp2, edge)                                                   <L 410>
                var_92 = wp::dot(var_91, var_24);
                var_93 = wp::add(var_85, var_92);
            }
            var_94 = wp::where(var_88, var_93, var_85);
            // flexedge_J_out[worldid, rowadr + i] = J_val                                        <L 411>
            var_95 = wp::add(var_44, var_65);
            wp::array_store(var_flexedge_J_out, var_0, var_95, var_94);
            // vel += J_val * qvel_in[worldid, dofid]                                             <L 412>
            var_96 = wp::address(var_qvel_in, var_0, var_68);
            var_98 = wp::load(var_96);
            var_97 = wp::mul(var_94, var_98);
            var_99 = wp::add(var_63, var_97);
            wp::assign(var_63, var_99);
            goto start_for_1;
        end_for_1:;
        // flexedge_velocity_out[worldid, edgeid] = vel                                           <L 414>
        wp::array_store(var_flexedge_velocity_out, var_0, var_1, var_63);
    }
}



extern "C" __global__ void _count_equality_constraints_54fc2a85_cuda_kernel_forward(
    wp::launch_bounds_t<2> dim,
    wp::array_t<wp::int32> var_eq_type,
    wp::array_t<wp::int32> var_ne_in,
    wp::array_t<wp::int32> var_efc_type_in,
    wp::array_t<wp::int32> var_efc_id_in,
    wp::array_t<wp::int32> var_ne_connect_out,
    wp::array_t<wp::int32> var_ne_weld_out)
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
        bool var_3;
        wp::int32 var_4;
        wp::int32* var_5;
        wp::int32 var_6;
        wp::int32 var_7;
        wp::int32* var_8;
        wp::int32 var_9;
        wp::int32 var_10;
        const wp::int32 var_11 = 0;
        bool var_12;
        const wp::int32 var_13 = 1;
        wp::int32 var_14;
        const wp::int32 var_15 = 1;
        bool var_16;
        const wp::int32 var_17 = 1;
        wp::int32 var_18;
        //---------
        // forward
        // def _count_equality_constraints(                                                       <L 1510>
        // worldid, efcid = wp.tid()                                                              <L 1522>
        builtin_tid2d(var_0, var_1);
        // if efcid >= ne_in[worldid]:                                                            <L 1525>
        var_2 = wp::address(var_ne_in, var_0);
        var_4 = wp::load(var_2);
        var_3 = (var_1 >= var_4);
        if (var_3) {
            // return                                                                             <L 1526>
            continue;
        }
        // eq_id = efc_id_in[worldid, efcid]                                                      <L 1529>
        var_5 = wp::address(var_efc_id_in, var_0, var_1);
        var_7 = wp::load(var_5);
        var_6 = wp::copy(var_7);
        // eq_constraint_type = eq_type[eq_id]                                                    <L 1530>
        var_8 = wp::address(var_eq_type, var_6);
        var_10 = wp::load(var_8);
        var_9 = wp::copy(var_10);
        // if eq_constraint_type == EqType.CONNECT:                                               <L 1533>
        var_12 = (var_9 == var_11);
        if (var_12) {
            // wp.atomic_add(ne_connect_out, worldid, 1)                                          <L 1534>
            var_14 = wp::atomic_add(var_ne_connect_out, var_0, var_13);
        }
        if (!var_12) {
            // elif eq_constraint_type == EqType.WELD:                                            <L 1535>
            var_16 = (var_9 == var_15);
            if (var_16) {
                // wp.atomic_add(ne_weld_out, worldid, 1)                                         <L 1536>
                var_18 = wp::atomic_add(var_ne_weld_out, var_0, var_17);
            }
        }
    }
}



extern "C" __global__ void _tendon_dot_62db264d_cuda_kernel_forward(
    wp::launch_bounds_t<2> dim,
    wp::array_t<wp::int32> var_body_parentid,
    wp::array_t<wp::int32> var_body_rootid,
    wp::array_t<wp::int32> var_body_dofnum,
    wp::array_t<wp::int32> var_body_dofadr,
    wp::array_t<wp::int32> var_jnt_type,
    wp::array_t<wp::int32> var_jnt_dofadr,
    wp::array_t<wp::int32> var_dof_jntid,
    wp::array_t<wp::int32> var_site_bodyid,
    wp::array_t<wp::int32> var_tendon_adr,
    wp::array_t<wp::int32> var_tendon_num,
    wp::array_t<wp::int32> var_ten_J_rownnz,
    wp::array_t<wp::int32> var_ten_J_rowadr,
    wp::array_t<wp::int32> var_ten_J_colind,
    wp::array_t<wp::float32> var_tendon_armature,
    wp::array_t<wp::int32> var_wrap_type,
    wp::array_t<wp::int32> var_wrap_objid,
    wp::array_t<wp::float32> var_wrap_prm,
    wp::array_t<wp::vec_t<3, wp::float32>> var_site_xpos_in,
    wp::array_t<wp::vec_t<3, wp::float32>> var_subtree_com_in,
    wp::array_t<wp::vec_t<6, wp::float32>> var_cdof_in,
    wp::array_t<wp::vec_t<6, wp::float32>> var_cvel_in,
    wp::array_t<wp::vec_t<6, wp::float32>> var_cdof_dot_in,
    wp::array_t<wp::float32> var_ten_Jdot_out)
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
        wp::shape_t* var_2;
        const wp::int32 var_3 = 0;
        wp::int32 var_4;
        wp::shape_t var_5;
        wp::int32 var_6;
        wp::float32* var_7;
        wp::float32 var_8;
        wp::float32 var_9;
        const wp::float32 var_10 = 0.0;
        bool var_11;
        wp::int32* var_12;
        wp::int32 var_13;
        wp::int32 var_14;
        wp::int32* var_15;
        const wp::int32 var_16 = 1;
        bool var_17;
        wp::int32 var_18;
        const wp::float32 var_19 = 1.0;
        wp::float32 var_20;
        wp::int32* var_21;
        wp::int32 var_22;
        wp::int32 var_23;
        const wp::int32 var_24 = 0;
        wp::int32 var_25;
        const wp::int32 var_26 = 1;
        wp::int32 var_27;
        bool var_28;
        wp::int32 var_29;
        const wp::int32 var_30 = 0;
        wp::int32 var_31;
        wp::int32* var_32;
        wp::int32 var_33;
        wp::int32 var_34;
        wp::int32 var_35;
        const wp::int32 var_36 = 1;
        wp::int32 var_37;
        wp::int32* var_38;
        wp::int32 var_39;
        wp::int32 var_40;
        wp::int32 var_41;
        const wp::int32 var_42 = 0;
        wp::int32 var_43;
        wp::int32* var_44;
        wp::int32 var_45;
        wp::int32 var_46;
        wp::int32 var_47;
        const wp::int32 var_48 = 1;
        wp::int32 var_49;
        wp::int32* var_50;
        wp::int32 var_51;
        wp::int32 var_52;
        const wp::int32 var_53 = 2;
        bool var_54;
        bool var_55;
        bool var_56;
        bool var_57;
        wp::int32 var_58;
        wp::float32* var_59;
        wp::float32 var_60;
        wp::float32 var_61;
        wp::float32 var_62;
        const wp::int32 var_63 = 1;
        wp::int32 var_64;
        wp::vec_t<3, wp::float32>* var_65;
        wp::vec_t<3, wp::float32> var_66;
        wp::vec_t<3, wp::float32> var_67;
        wp::int32* var_68;
        wp::int32 var_69;
        wp::int32 var_70;
        wp::vec_t<6, wp::float32>* var_71;
        wp::vec_t<6, wp::float32> var_72;
        wp::vec_t<6, wp::float32> var_73;
        wp::int32* var_74;
        wp::int32 var_75;
        wp::vec_t<3, wp::float32>* var_76;
        wp::vec_t<3, wp::float32> var_77;
        wp::vec_t<3, wp::float32> var_78;
        wp::vec_t<3, wp::float32> var_79;
        wp::vec_t<3, wp::float32> var_80;
        wp::vec_t<3, wp::float32> var_81;
        wp::vec_t<3, wp::float32> var_82;
        wp::vec_t<3, wp::float32> var_83;
        bool var_84;
        const wp::int32 var_85 = 4;
        bool var_86;
        const wp::int32 var_87 = 5;
        bool var_88;
        wp::int32* var_89;
        wp::int32 var_90;
        wp::int32 var_91;
        wp::vec_t<3, wp::float32>* var_92;
        wp::vec_t<3, wp::float32> var_93;
        wp::vec_t<3, wp::float32> var_94;
        wp::vec_t<6, wp::float32>* var_95;
        wp::vec_t<6, wp::float32> var_96;
        wp::vec_t<6, wp::float32> var_97;
        wp::int32* var_98;
        wp::int32 var_99;
        wp::vec_t<3, wp::float32>* var_100;
        wp::vec_t<3, wp::float32> var_101;
        wp::vec_t<3, wp::float32> var_102;
        wp::vec_t<3, wp::float32> var_103;
        wp::vec_t<3, wp::float32> var_104;
        wp::vec_t<3, wp::float32> var_105;
        wp::vec_t<3, wp::float32> var_106;
        wp::vec_t<3, wp::float32> var_107;
        bool var_108;
        wp::vec_t<3, wp::float32> var_109;
        wp::vec_t<3, wp::float32> var_110;
        wp::float32 var_111;
        wp::vec_t<3, wp::float32> var_112;
        wp::vec_t<3, wp::float32> var_113;
        wp::vec_t<3, wp::float32> var_114;
        wp::vec_t<3, wp::float32> var_115;
        wp::vec_t<3, wp::float32> var_116;
        wp::vec_t<3, wp::float32> var_117;
        wp::vec_t<3, wp::float32> var_118;
        wp::vec_t<3, wp::float32> var_119;
        wp::vec_t<3, wp::float32> var_120;
        wp::vec_t<3, wp::float32> var_121;
        wp::vec_t<3, wp::float32> var_122;
        wp::float32 var_123;
        wp::float32 var_124;
        wp::vec_t<3, wp::float32> var_125;
        wp::vec_t<3, wp::float32> var_126;
        const wp::float32 var_127 = 1e-15;
        bool var_128;
        wp::vec_t<3, wp::float32> var_129;
        const wp::float32 var_130 = 0.0;
        wp::vec_t<3, wp::float32> var_131;
        wp::vec_t<3, wp::float32> var_132;
        wp::int32* var_133;
        wp::int32 var_134;
        wp::int32 var_135;
        wp::int32* var_136;
        wp::int32 var_137;
        wp::int32 var_138;
        const wp::float32 var_139 = 1.0;
        wp::float32 var_140;
        wp::float32 var_141;
        wp::float32 var_142;
        const wp::int32 var_143 = 1;
        wp::int32 var_144;
        //---------
        // forward
        // def _tendon_dot(                                                                       <L 2229>
        // worldid, tenid = wp.tid()                                                              <L 2257>
        builtin_tid2d(var_0, var_1);
        // armature = tendon_armature[worldid % tendon_armature.shape[0], tenid]                  <L 2259>
        var_2 = &(var_tendon_armature.shape);
        var_5 = wp::load(var_2);
        var_4 = wp::extract(var_5, var_3);
        var_6 = wp::mod(var_0, var_4);
        var_7 = wp::address(var_tendon_armature, var_6, var_1);
        var_9 = wp::load(var_7);
        var_8 = wp::copy(var_9);
        // if armature == 0.0:                                                                    <L 2260>
        var_11 = (var_8 == var_10);
        if (var_11) {
            // return                                                                             <L 2261>
            continue;
        }
        // adr = tendon_adr[tenid]                                                                <L 2264>
        var_12 = wp::address(var_tendon_adr, var_1);
        var_14 = wp::load(var_12);
        var_13 = wp::copy(var_14);
        // if wrap_type[adr] == WrapType.JOINT:                                                   <L 2265>
        var_15 = wp::address(var_wrap_type, var_13);
        var_18 = wp::load(var_15);
        var_17 = (var_18 == var_16);
        if (var_17) {
            // return                                                                             <L 2266>
            continue;
        }
        // divisor = float(1.0)                                                                   <L 2269>
        var_20 = wp::float(var_19);
        // num = tendon_num[tenid]                                                                <L 2270>
        var_21 = wp::address(var_tendon_num, var_1);
        var_23 = wp::load(var_21);
        var_22 = wp::copy(var_23);
        // j = int(0)                                                                             <L 2271>
        var_25 = wp::int(var_24);
        // while j < num - 1:                                                                     <L 2272>
        start_while_2:;
        var_27 = wp::sub(var_22, var_26);
        var_28 = (var_25 < var_27);
        if ((var_28) == false) goto end_while_2;
            // type0 = wrap_type[adr + j + 0]                                                     <L 2274>
            var_29 = wp::add(var_13, var_25);
            var_31 = wp::add(var_29, var_30);
            var_32 = wp::address(var_wrap_type, var_31);
            var_34 = wp::load(var_32);
            var_33 = wp::copy(var_34);
            // type1 = wrap_type[adr + j + 1]                                                     <L 2275>
            var_35 = wp::add(var_13, var_25);
            var_37 = wp::add(var_35, var_36);
            var_38 = wp::address(var_wrap_type, var_37);
            var_40 = wp::load(var_38);
            var_39 = wp::copy(var_40);
            // id0 = wrap_objid[adr + j + 0]                                                      <L 2276>
            var_41 = wp::add(var_13, var_25);
            var_43 = wp::add(var_41, var_42);
            var_44 = wp::address(var_wrap_objid, var_43);
            var_46 = wp::load(var_44);
            var_45 = wp::copy(var_46);
            // id1 = wrap_objid[adr + j + 1]                                                      <L 2277>
            var_47 = wp::add(var_13, var_25);
            var_49 = wp::add(var_47, var_48);
            var_50 = wp::address(var_wrap_objid, var_49);
            var_52 = wp::load(var_50);
            var_51 = wp::copy(var_52);
            // pulley = WrapType.PULLEY                                                           <L 2280>
            // if (type0 == pulley) or (type1 == pulley):                                         <L 2281>
            var_55 = (var_33 == var_53);
            var_54 = var_55;
            if (!var_54) {
                var_56 = (var_39 == var_53);
                var_54 = var_54 || var_56;
            }
            if (var_54) {
                // if type0 == pulley:                                                            <L 2283>
                var_57 = (var_33 == var_53);
                if (var_57) {
                    // divisor = wrap_prm[adr + j]                                                <L 2284>
                    var_58 = wp::add(var_13, var_25);
                    var_59 = wp::address(var_wrap_prm, var_58);
                    var_61 = wp::load(var_59);
                    var_60 = wp::copy(var_61);
                }
                var_62 = wp::where(var_57, var_60, var_20);
                // j += 1                                                                         <L 2286>
                var_64 = wp::add(var_25, var_63);
                // continue                                                                       <L 2287>
                wp::assign(var_20, var_62);
                wp::assign(var_25, var_64);
                goto start_while_2;
            }
            // wpnt0 = site_xpos_in[worldid, id0]                                                 <L 2290>
            var_65 = wp::address(var_site_xpos_in, var_0, var_45);
            var_67 = wp::load(var_65);
            var_66 = wp::copy(var_67);
            // wbody0 = site_bodyid[id0]                                                          <L 2292>
            var_68 = wp::address(var_site_bodyid, var_45);
            var_70 = wp::load(var_68);
            var_69 = wp::copy(var_70);
            // cvel0 = cvel_in[worldid, wbody0]                                                   <L 2293>
            var_71 = wp::address(var_cvel_in, var_0, var_69);
            var_73 = wp::load(var_71);
            var_72 = wp::copy(var_73);
            // subtree_com0 = subtree_com_in[worldid, body_rootid[wbody0]]                        <L 2294>
            var_74 = wp::address(var_body_rootid, var_69);
            var_75 = wp::load(var_74);
            var_76 = wp::address(var_subtree_com_in, var_0, var_75);
            var_78 = wp::load(var_76);
            var_77 = wp::copy(var_78);
            // offset0 = wpnt0 - subtree_com0                                                     <L 2295>
            var_79 = wp::sub(var_66, var_77);
            // pvel_lin0 = wp.spatial_bottom(cvel0) - wp.cross(offset0, wp.spatial_top(cvel0))       <L 2296>
            var_80 = wp::spatial_bottom(var_72);
            var_81 = wp::spatial_top(var_72);
            var_82 = wp::cross(var_79, var_81);
            var_83 = wp::sub(var_80, var_82);
            // if (type1 == WrapType.SPHERE) or (type1 == WrapType.CYLINDER):                     <L 2299>
            var_86 = (var_39 == var_85);
            var_84 = var_86;
            if (!var_84) {
                var_88 = (var_39 == var_87);
                var_84 = var_84 || var_88;
            }
            if (var_84) {
                // return                                                                         <L 2301>
                continue;
            }
            // wbody1 = site_bodyid[id1]                                                          <L 2304>
            var_89 = wp::address(var_site_bodyid, var_51);
            var_91 = wp::load(var_89);
            var_90 = wp::copy(var_91);
            // wpnt1 = site_xpos_in[worldid, id1]                                                 <L 2305>
            var_92 = wp::address(var_site_xpos_in, var_0, var_51);
            var_94 = wp::load(var_92);
            var_93 = wp::copy(var_94);
            // cvel1 = cvel_in[worldid, wbody1]                                                   <L 2307>
            var_95 = wp::address(var_cvel_in, var_0, var_90);
            var_97 = wp::load(var_95);
            var_96 = wp::copy(var_97);
            // subtree_com1 = subtree_com_in[worldid, body_rootid[wbody1]]                        <L 2308>
            var_98 = wp::address(var_body_rootid, var_90);
            var_99 = wp::load(var_98);
            var_100 = wp::address(var_subtree_com_in, var_0, var_99);
            var_102 = wp::load(var_100);
            var_101 = wp::copy(var_102);
            // offset1 = wpnt1 - subtree_com1                                                     <L 2309>
            var_103 = wp::sub(var_93, var_101);
            // pvel_lin1 = wp.spatial_bottom(cvel1) - wp.cross(offset1, wp.spatial_top(cvel1))       <L 2310>
            var_104 = wp::spatial_bottom(var_96);
            var_105 = wp::spatial_top(var_96);
            var_106 = wp::cross(var_103, var_105);
            var_107 = wp::sub(var_104, var_106);
            // if wbody0 != wbody1:                                                               <L 2313>
            var_108 = (var_69 != var_90);
            if (var_108) {
                // dpnt, norm = math.normalize_with_norm(wpnt1 - wpnt0)                           <L 2315>
                var_109 = wp::sub(var_93, var_66);
                normalize_with_norm_0(var_109, var_110, var_111);
                // wvel0 = wp.spatial_bottom(cvel0) - wp.cross(wpnt0 - subtree_com0, wp.spatial_top(cvel0))       <L 2318>
                var_112 = wp::spatial_bottom(var_72);
                var_113 = wp::sub(var_66, var_77);
                var_114 = wp::spatial_top(var_72);
                var_115 = wp::cross(var_113, var_114);
                var_116 = wp::sub(var_112, var_115);
                // wvel1 = wp.spatial_bottom(cvel1) - wp.cross(wpnt1 - subtree_com1, wp.spatial_top(cvel1))       <L 2319>
                var_117 = wp::spatial_bottom(var_96);
                var_118 = wp::sub(var_93, var_101);
                var_119 = wp::spatial_top(var_96);
                var_120 = wp::cross(var_118, var_119);
                var_121 = wp::sub(var_117, var_120);
                // dvel = wvel1 - wvel0                                                           <L 2320>
                var_122 = wp::sub(var_121, var_116);
                // dot = wp.dot(dpnt, dvel)                                                       <L 2321>
                var_123 = wp::dot(var_110, var_122);
                // dvel += dpnt * (-dot)                                                          <L 2322>
                var_124 = wp::neg(var_123);
                var_125 = wp::mul(var_110, var_124);
                var_126 = wp::add(var_122, var_125);
                // if norm > MJ_MINVAL:                                                           <L 2323>
                var_128 = (var_111 > var_127);
                if (var_128) {
                    // dvel /= norm                                                               <L 2324>
                    var_129 = wp::div(var_126, var_111);
                }
                if (!var_128) {
                    // dvel = wp.vec3(0.0)                                                        <L 2326>
                    var_131 = wp::vec_t<3, wp::float32>(var_130);
                }
                var_132 = wp::where(var_128, var_129, var_131);
                // rownnz = ten_J_rownnz[tenid]                                                   <L 2328>
                var_133 = wp::address(var_ten_J_rownnz, var_1);
                var_135 = wp::load(var_133);
                var_134 = wp::copy(var_135);
                // rowadr = ten_J_rowadr[tenid]                                                   <L 2329>
                var_136 = wp::address(var_ten_J_rowadr, var_1);
                var_138 = wp::load(var_136);
                var_137 = wp::copy(var_138);
                // inv_divisor = math.safe_div(float(1.0), divisor)                               <L 2330>
                var_140 = wp::float(var_139);
                var_141 = safe_div_0(var_140, var_20);
                // _accumulate_jac_dot_chain(                                                     <L 2333>
                // body_parentid,                                                                 <L 2334>
                // body_dofnum,                                                                   <L 2335>
                // body_dofadr,                                                                   <L 2336>
                // jnt_type,                                                                      <L 2337>
                // jnt_dofadr,                                                                    <L 2338>
                // dof_jntid,                                                                     <L 2339>
                // ten_J_colind,                                                                  <L 2340>
                // cdof_in,                                                                       <L 2341>
                // cvel_in,                                                                       <L 2342>
                // cdof_dot_in,                                                                   <L 2343>
                // offset0,                                                                       <L 2344>
                // pvel_lin0,                                                                     <L 2345>
                // dpnt,                                                                          <L 2346>
                // dvel,                                                                          <L 2347>
                // wbody0,                                                                        <L 2348>
                // rowadr,                                                                        <L 2349>
                // rownnz,                                                                        <L 2350>
                // -inv_divisor,                                                                  <L 2351>
                var_142 = wp::neg(var_141);
                // worldid,                                                                       <L 2352>
                // ten_Jdot_out,                                                                  <L 2353>
                _accumulate_jac_dot_chain_0(var_body_parentid, var_body_dofnum, var_body_dofadr, var_jnt_type, var_jnt_dofadr, var_dof_jntid, var_ten_J_colind, var_cdof_in, var_cvel_in, var_cdof_dot_in, var_79, var_83, var_110, var_132, var_69, var_137, var_134, var_142, var_0, var_ten_Jdot_out);
                // _accumulate_jac_dot_chain(                                                     <L 2355>
                // body_parentid,                                                                 <L 2356>
                // body_dofnum,                                                                   <L 2357>
                // body_dofadr,                                                                   <L 2358>
                // jnt_type,                                                                      <L 2359>
                // jnt_dofadr,                                                                    <L 2360>
                // dof_jntid,                                                                     <L 2361>
                // ten_J_colind,                                                                  <L 2362>
                // cdof_in,                                                                       <L 2363>
                // cvel_in,                                                                       <L 2364>
                // cdof_dot_in,                                                                   <L 2365>
                // offset1,                                                                       <L 2366>
                // pvel_lin1,                                                                     <L 2367>
                // dpnt,                                                                          <L 2368>
                // dvel,                                                                          <L 2369>
                // wbody1,                                                                        <L 2370>
                // rowadr,                                                                        <L 2371>
                // rownnz,                                                                        <L 2372>
                // inv_divisor,                                                                   <L 2373>
                // worldid,                                                                       <L 2374>
                // ten_Jdot_out,                                                                  <L 2375>
                _accumulate_jac_dot_chain_0(var_body_parentid, var_body_dofnum, var_body_dofadr, var_jnt_type, var_jnt_dofadr, var_dof_jntid, var_ten_J_colind, var_cdof_in, var_cvel_in, var_cdof_dot_in, var_103, var_107, var_110, var_132, var_90, var_137, var_134, var_141, var_0, var_ten_Jdot_out);
            }
            // j += 1                                                                             <L 2379>
            var_144 = wp::add(var_25, var_143);
            wp::assign(var_25, var_144);
        goto start_while_2;
        end_while_2:;
    }
}



extern "C" __global__ void _flex_vertices_68b379d4_cuda_kernel_forward(
    wp::launch_bounds_t<2> dim,
    wp::int32 var_nflex,
    wp::array_t<wp::int32> var_flex_interp,
    wp::array_t<wp::vec_t<3, wp::int32>> var_flex_cellnum,
    wp::array_t<wp::int32> var_flex_nodeadr,
    wp::array_t<wp::int32> var_flex_vertadr,
    wp::array_t<wp::int32> var_flex_vertnum,
    wp::array_t<wp::int32> var_flex_vertbodyid,
    wp::array_t<wp::vec_t<3, wp::float32>> var_flex_vert,
    wp::array_t<wp::vec_t<3, wp::float32>> var_flex_vert0,
    wp::array_t<bool> var_flex_centered,
    wp::array_t<wp::vec_t<3, wp::float32>> var_xpos_in,
    wp::array_t<wp::mat_t<3, 3, wp::float32>> var_xmat_in,
    wp::array_t<wp::vec_t<3, wp::float32>> var_flexnode_xpos_in,
    wp::array_t<wp::vec_t<3, wp::float32>> var_flexvert_xpos_out)
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
        wp::range_t var_2;
        wp::int32 var_3;
        wp::int32* var_4;
        wp::int32 var_5;
        wp::int32 var_6;
        bool var_7;
        const wp::int32 var_8 = 0;
        bool var_9;
        wp::int32* var_10;
        bool var_11;
        wp::int32 var_12;
        wp::int32* var_13;
        const wp::int32 var_14 = 0;
        bool var_15;
        wp::int32 var_16;
        wp::vec_t<3, wp::float32>* var_17;
        wp::vec_t<3, wp::float32> var_18;
        wp::vec_t<3, wp::float32> var_19;
        wp::vec_t<3, wp::int32>* var_20;
        wp::vec_t<3, wp::int32> var_21;
        wp::vec_t<3, wp::int32> var_22;
        const wp::int32 var_23 = 0;
        wp::int32 var_24;
        const wp::int32 var_25 = 1;
        wp::int32 var_26;
        const wp::int32 var_27 = 2;
        wp::int32 var_28;
        const wp::int32 var_29 = 0;
        wp::float32 var_30;
        wp::float32 var_31;
        wp::float32 var_32;
        wp::int32 var_33;
        const wp::int32 var_34 = 1;
        wp::int32 var_35;
        wp::int32 var_36;
        const wp::int32 var_37 = 0;
        wp::int32 var_38;
        const wp::int32 var_39 = 1;
        wp::float32 var_40;
        wp::float32 var_41;
        wp::float32 var_42;
        wp::int32 var_43;
        const wp::int32 var_44 = 1;
        wp::int32 var_45;
        wp::int32 var_46;
        const wp::int32 var_47 = 0;
        wp::int32 var_48;
        const wp::int32 var_49 = 2;
        wp::float32 var_50;
        wp::float32 var_51;
        wp::float32 var_52;
        wp::int32 var_53;
        const wp::int32 var_54 = 1;
        wp::int32 var_55;
        wp::int32 var_56;
        const wp::int32 var_57 = 0;
        wp::int32 var_58;
        const wp::int32 var_59 = 0;
        wp::float32 var_60;
        wp::float32 var_61;
        wp::float32 var_62;
        wp::float32 var_63;
        wp::float32 var_64;
        const wp::float32 var_65 = 0.0;
        const wp::float32 var_66 = 1.0;
        wp::float32 var_67;
        const wp::int32 var_68 = 1;
        wp::float32 var_69;
        wp::float32 var_70;
        wp::float32 var_71;
        wp::float32 var_72;
        wp::float32 var_73;
        const wp::float32 var_74 = 0.0;
        const wp::float32 var_75 = 1.0;
        wp::float32 var_76;
        const wp::int32 var_77 = 2;
        wp::float32 var_78;
        wp::float32 var_79;
        wp::float32 var_80;
        wp::float32 var_81;
        wp::float32 var_82;
        const wp::float32 var_83 = 0.0;
        const wp::float32 var_84 = 1.0;
        wp::float32 var_85;
        wp::vec_t<3, wp::float32> var_86;
        const wp::int32 var_87 = 1;
        wp::int32 var_88;
        const wp::int32 var_89 = 1;
        wp::int32 var_90;
        wp::int32* var_91;
        wp::int32 var_92;
        wp::int32 var_93;
        const wp::float32 var_94 = 0.0;
        const wp::float32 var_95 = 0.0;
        const wp::float32 var_96 = 0.0;
        wp::vec_t<3, wp::float32> var_97;
        const wp::int32 var_98 = 0;
        const wp::int32 var_99 = 0;
        const wp::int32 var_100 = 0;
        const wp::int32 var_101 = 4;
        wp::int32 var_102;
        const wp::int32 var_103 = 2;
        wp::int32 var_104;
        wp::int32 var_105;
        wp::int32 var_106;
        wp::float32 var_107;
        wp::int32 var_108;
        wp::int32 var_109;
        wp::int32 var_110;
        wp::int32 var_111;
        wp::int32 var_112;
        wp::int32 var_113;
        wp::int32 var_114;
        wp::int32 var_115;
        wp::int32 var_116;
        wp::vec_t<3, wp::float32>* var_117;
        wp::vec_t<3, wp::float32> var_118;
        wp::vec_t<3, wp::float32> var_119;
        wp::vec_t<3, wp::float32> var_120;
        const wp::int32 var_121 = 1;
        const wp::int32 var_122 = 4;
        wp::int32 var_123;
        const wp::int32 var_124 = 2;
        wp::int32 var_125;
        wp::int32 var_126;
        wp::int32 var_127;
        wp::float32 var_128;
        wp::int32 var_129;
        wp::int32 var_130;
        wp::int32 var_131;
        wp::int32 var_132;
        wp::int32 var_133;
        wp::int32 var_134;
        wp::int32 var_135;
        wp::int32 var_136;
        wp::int32 var_137;
        wp::vec_t<3, wp::float32>* var_138;
        wp::vec_t<3, wp::float32> var_139;
        wp::vec_t<3, wp::float32> var_140;
        wp::vec_t<3, wp::float32> var_141;
        const wp::int32 var_142 = 1;
        const wp::int32 var_143 = 0;
        const wp::int32 var_144 = 4;
        wp::int32 var_145;
        const wp::int32 var_146 = 2;
        wp::int32 var_147;
        wp::int32 var_148;
        wp::int32 var_149;
        wp::float32 var_150;
        wp::int32 var_151;
        wp::int32 var_152;
        wp::int32 var_153;
        wp::int32 var_154;
        wp::int32 var_155;
        wp::int32 var_156;
        wp::int32 var_157;
        wp::int32 var_158;
        wp::int32 var_159;
        wp::vec_t<3, wp::float32>* var_160;
        wp::vec_t<3, wp::float32> var_161;
        wp::vec_t<3, wp::float32> var_162;
        wp::vec_t<3, wp::float32> var_163;
        const wp::int32 var_164 = 1;
        const wp::int32 var_165 = 4;
        wp::int32 var_166;
        const wp::int32 var_167 = 2;
        wp::int32 var_168;
        wp::int32 var_169;
        wp::int32 var_170;
        wp::float32 var_171;
        wp::int32 var_172;
        wp::int32 var_173;
        wp::int32 var_174;
        wp::int32 var_175;
        wp::int32 var_176;
        wp::int32 var_177;
        wp::int32 var_178;
        wp::int32 var_179;
        wp::int32 var_180;
        wp::vec_t<3, wp::float32>* var_181;
        wp::vec_t<3, wp::float32> var_182;
        wp::vec_t<3, wp::float32> var_183;
        wp::vec_t<3, wp::float32> var_184;
        const wp::int32 var_185 = 1;
        const wp::int32 var_186 = 0;
        const wp::int32 var_187 = 0;
        const wp::int32 var_188 = 4;
        wp::int32 var_189;
        const wp::int32 var_190 = 2;
        wp::int32 var_191;
        wp::int32 var_192;
        wp::int32 var_193;
        wp::float32 var_194;
        wp::int32 var_195;
        wp::int32 var_196;
        wp::int32 var_197;
        wp::int32 var_198;
        wp::int32 var_199;
        wp::int32 var_200;
        wp::int32 var_201;
        wp::int32 var_202;
        wp::int32 var_203;
        wp::vec_t<3, wp::float32>* var_204;
        wp::vec_t<3, wp::float32> var_205;
        wp::vec_t<3, wp::float32> var_206;
        wp::vec_t<3, wp::float32> var_207;
        const wp::int32 var_208 = 1;
        const wp::int32 var_209 = 4;
        wp::int32 var_210;
        const wp::int32 var_211 = 2;
        wp::int32 var_212;
        wp::int32 var_213;
        wp::int32 var_214;
        wp::float32 var_215;
        wp::int32 var_216;
        wp::int32 var_217;
        wp::int32 var_218;
        wp::int32 var_219;
        wp::int32 var_220;
        wp::int32 var_221;
        wp::int32 var_222;
        wp::int32 var_223;
        wp::int32 var_224;
        wp::vec_t<3, wp::float32>* var_225;
        wp::vec_t<3, wp::float32> var_226;
        wp::vec_t<3, wp::float32> var_227;
        wp::vec_t<3, wp::float32> var_228;
        const wp::int32 var_229 = 1;
        const wp::int32 var_230 = 0;
        const wp::int32 var_231 = 4;
        wp::int32 var_232;
        const wp::int32 var_233 = 2;
        wp::int32 var_234;
        wp::int32 var_235;
        wp::int32 var_236;
        wp::float32 var_237;
        wp::int32 var_238;
        wp::int32 var_239;
        wp::int32 var_240;
        wp::int32 var_241;
        wp::int32 var_242;
        wp::int32 var_243;
        wp::int32 var_244;
        wp::int32 var_245;
        wp::int32 var_246;
        wp::vec_t<3, wp::float32>* var_247;
        wp::vec_t<3, wp::float32> var_248;
        wp::vec_t<3, wp::float32> var_249;
        wp::vec_t<3, wp::float32> var_250;
        const wp::int32 var_251 = 1;
        const wp::int32 var_252 = 4;
        wp::int32 var_253;
        const wp::int32 var_254 = 2;
        wp::int32 var_255;
        wp::int32 var_256;
        wp::int32 var_257;
        wp::float32 var_258;
        wp::int32 var_259;
        wp::int32 var_260;
        wp::int32 var_261;
        wp::int32 var_262;
        wp::int32 var_263;
        wp::int32 var_264;
        wp::int32 var_265;
        wp::int32 var_266;
        wp::int32 var_267;
        wp::vec_t<3, wp::float32>* var_268;
        wp::vec_t<3, wp::float32> var_269;
        wp::vec_t<3, wp::float32> var_270;
        wp::vec_t<3, wp::float32> var_271;
        wp::int32* var_272;
        wp::int32 var_273;
        wp::int32 var_274;
        wp::vec_t<3, wp::float32>* var_275;
        wp::vec_t<3, wp::float32> var_276;
        wp::vec_t<3, wp::float32> var_277;
        bool* var_278;
        bool var_279;
        bool var_280;
        bool var_281;
        wp::mat_t<3, 3, wp::float32>* var_282;
        wp::mat_t<3, 3, wp::float32> var_283;
        wp::mat_t<3, 3, wp::float32> var_284;
        wp::vec_t<3, wp::float32>* var_285;
        wp::vec_t<3, wp::float32> var_286;
        wp::vec_t<3, wp::float32> var_287;
        wp::vec_t<3, wp::float32> var_288;
        wp::vec_t<3, wp::float32> var_289;
        bool var_290;
        //---------
        // forward
        // def _flex_vertices(                                                                    <L 231>
        // worldid, vertid = wp.tid()                                                             <L 250>
        builtin_tid2d(var_0, var_1);
        // for f in range(nflex):                                                                 <L 252>
        var_2 = wp::range(var_nflex);
        start_for_0:;
            if (iter_cmp(var_2) == 0) goto end_for_0;
            var_3 = wp::iter_next(var_2);
            // locid = vertid - flex_vertadr[f]                                                   <L 253>
            var_4 = wp::address(var_flex_vertadr, var_3);
            var_6 = wp::load(var_4);
            var_5 = wp::sub(var_1, var_6);
            // if locid >= 0 and locid < flex_vertnum[f]:                                         <L 254>
            var_9 = (var_5 >= var_8);
            var_7 = var_9;
            if (var_7) {
                var_10 = wp::address(var_flex_vertnum, var_3);
                var_12 = wp::load(var_10);
                var_11 = (var_5 < var_12);
                var_7 = var_7 && var_11;
            }
            if (var_7) {
                // break                                                                          <L 255>
                goto end_for_0;
            }
            goto start_for_0;
        end_for_0:;
        // if flex_interp[f] != 0:                                                                <L 257>
        var_13 = wp::address(var_flex_interp, var_3);
        var_16 = wp::load(var_13);
        var_15 = (var_16 != var_14);
        if (var_15) {
            // coord = flex_vert0[vertid]                                                         <L 259>
            var_17 = wp::address(var_flex_vert0, var_1);
            var_19 = wp::load(var_17);
            var_18 = wp::copy(var_19);
            // cn = flex_cellnum[f]                                                               <L 260>
            var_20 = wp::address(var_flex_cellnum, var_3);
            var_22 = wp::load(var_20);
            var_21 = wp::copy(var_22);
            // cx = cn[0]                                                                         <L 261>
            var_24 = wp::extract(var_21, var_23);
            // cy = cn[1]                                                                         <L 262>
            var_26 = wp::extract(var_21, var_25);
            // cz = cn[2]                                                                         <L 263>
            var_28 = wp::extract(var_21, var_27);
            // ci = wp.min(int(coord[0] * float(cx)), cx - 1)                                     <L 266>
            var_30 = wp::extract(var_18, var_29);
            var_31 = wp::float(var_24);
            var_32 = wp::mul(var_30, var_31);
            var_33 = wp::int(var_32);
            var_35 = wp::sub(var_24, var_34);
            var_36 = wp::min(var_33, var_35);
            // ci = wp.max(ci, 0)                                                                 <L 267>
            var_38 = wp::max(var_36, var_37);
            // cj = wp.min(int(coord[1] * float(cy)), cy - 1)                                     <L 268>
            var_40 = wp::extract(var_18, var_39);
            var_41 = wp::float(var_26);
            var_42 = wp::mul(var_40, var_41);
            var_43 = wp::int(var_42);
            var_45 = wp::sub(var_26, var_44);
            var_46 = wp::min(var_43, var_45);
            // cj = wp.max(cj, 0)                                                                 <L 269>
            var_48 = wp::max(var_46, var_47);
            // ck = wp.min(int(coord[2] * float(cz)), cz - 1)                                     <L 270>
            var_50 = wp::extract(var_18, var_49);
            var_51 = wp::float(var_28);
            var_52 = wp::mul(var_50, var_51);
            var_53 = wp::int(var_52);
            var_55 = wp::sub(var_28, var_54);
            var_56 = wp::min(var_53, var_55);
            // ck = wp.max(ck, 0)                                                                 <L 271>
            var_58 = wp::max(var_56, var_57);
            // local_x = wp.clamp(coord[0] * float(cx) - float(ci), 0.0, 1.0)                     <L 274>
            var_60 = wp::extract(var_18, var_59);
            var_61 = wp::float(var_24);
            var_62 = wp::mul(var_60, var_61);
            var_63 = wp::float(var_38);
            var_64 = wp::sub(var_62, var_63);
            var_67 = wp::clamp(var_64, var_65, var_66);
            // local_y = wp.clamp(coord[1] * float(cy) - float(cj), 0.0, 1.0)                     <L 275>
            var_69 = wp::extract(var_18, var_68);
            var_70 = wp::float(var_26);
            var_71 = wp::mul(var_69, var_70);
            var_72 = wp::float(var_48);
            var_73 = wp::sub(var_71, var_72);
            var_76 = wp::clamp(var_73, var_74, var_75);
            // local_z = wp.clamp(coord[2] * float(cz) - float(ck), 0.0, 1.0)                     <L 276>
            var_78 = wp::extract(var_18, var_77);
            var_79 = wp::float(var_28);
            var_80 = wp::mul(var_78, var_79);
            var_81 = wp::float(var_58);
            var_82 = wp::sub(var_80, var_81);
            var_85 = wp::clamp(var_82, var_83, var_84);
            // local = wp.vec3(local_x, local_y, local_z)                                         <L 277>
            var_86 = wp::vec_t<3, wp::float32>(var_67, var_76, var_85);
            // ny_g = cy + 1                                                                      <L 280>
            var_88 = wp::add(var_26, var_87);
            // nz_g = cz + 1                                                                      <L 281>
            var_90 = wp::add(var_28, var_89);
            // nstart = flex_nodeadr[f]                                                           <L 282>
            var_91 = wp::address(var_flex_nodeadr, var_3);
            var_93 = wp::load(var_91);
            var_92 = wp::copy(var_93);
            // result = wp.vec3(0.0, 0.0, 0.0)                                                    <L 285>
            var_97 = wp::vec_t<3, wp::float32>(var_94, var_95, var_96);
            // for li in range(2):                                                                <L 286>
            // for lj in range(2):                                                                <L 287>
            // for lk in range(2):                                                                <L 288>
            // w = support.eval_basis_trilinear(local, li * 4 + lj * 2 + lk)                      <L 289>
            var_102 = wp::mul(var_98, var_101);
            var_104 = wp::mul(var_99, var_103);
            var_105 = wp::add(var_102, var_104);
            var_106 = wp::add(var_105, var_100);
            var_107 = eval_basis_trilinear_0(var_86, var_106);
            // gi = ci + li                                                                       <L 290>
            var_108 = wp::add(var_38, var_98);
            // gj = cj + lj                                                                       <L 291>
            var_109 = wp::add(var_48, var_99);
            // gk = ck + lk                                                                       <L 292>
            var_110 = wp::add(var_58, var_100);
            // node_idx = gi * ny_g * nz_g + gj * nz_g + gk                                       <L 293>
            var_111 = wp::mul(var_108, var_88);
            var_112 = wp::mul(var_111, var_90);
            var_113 = wp::mul(var_109, var_90);
            var_114 = wp::add(var_112, var_113);
            var_115 = wp::add(var_114, var_110);
            // result += w * flexnode_xpos_in[worldid, nstart + node_idx]                         <L 294>
            var_116 = wp::add(var_92, var_115);
            var_117 = wp::address(var_flexnode_xpos_in, var_0, var_116);
            var_119 = wp::load(var_117);
            var_118 = wp::mul(var_107, var_119);
            var_120 = wp::add(var_97, var_118);
            // w = support.eval_basis_trilinear(local, li * 4 + lj * 2 + lk)                      <L 289>
            var_123 = wp::mul(var_98, var_122);
            var_125 = wp::mul(var_99, var_124);
            var_126 = wp::add(var_123, var_125);
            var_127 = wp::add(var_126, var_121);
            var_128 = eval_basis_trilinear_0(var_86, var_127);
            // gi = ci + li                                                                       <L 290>
            var_129 = wp::add(var_38, var_98);
            // gj = cj + lj                                                                       <L 291>
            var_130 = wp::add(var_48, var_99);
            // gk = ck + lk                                                                       <L 292>
            var_131 = wp::add(var_58, var_121);
            // node_idx = gi * ny_g * nz_g + gj * nz_g + gk                                       <L 293>
            var_132 = wp::mul(var_129, var_88);
            var_133 = wp::mul(var_132, var_90);
            var_134 = wp::mul(var_130, var_90);
            var_135 = wp::add(var_133, var_134);
            var_136 = wp::add(var_135, var_131);
            // result += w * flexnode_xpos_in[worldid, nstart + node_idx]                         <L 294>
            var_137 = wp::add(var_92, var_136);
            var_138 = wp::address(var_flexnode_xpos_in, var_0, var_137);
            var_140 = wp::load(var_138);
            var_139 = wp::mul(var_128, var_140);
            var_141 = wp::add(var_120, var_139);
            // for lk in range(2):                                                                <L 288>
            // w = support.eval_basis_trilinear(local, li * 4 + lj * 2 + lk)                      <L 289>
            var_145 = wp::mul(var_98, var_144);
            var_147 = wp::mul(var_142, var_146);
            var_148 = wp::add(var_145, var_147);
            var_149 = wp::add(var_148, var_143);
            var_150 = eval_basis_trilinear_0(var_86, var_149);
            // gi = ci + li                                                                       <L 290>
            var_151 = wp::add(var_38, var_98);
            // gj = cj + lj                                                                       <L 291>
            var_152 = wp::add(var_48, var_142);
            // gk = ck + lk                                                                       <L 292>
            var_153 = wp::add(var_58, var_143);
            // node_idx = gi * ny_g * nz_g + gj * nz_g + gk                                       <L 293>
            var_154 = wp::mul(var_151, var_88);
            var_155 = wp::mul(var_154, var_90);
            var_156 = wp::mul(var_152, var_90);
            var_157 = wp::add(var_155, var_156);
            var_158 = wp::add(var_157, var_153);
            // result += w * flexnode_xpos_in[worldid, nstart + node_idx]                         <L 294>
            var_159 = wp::add(var_92, var_158);
            var_160 = wp::address(var_flexnode_xpos_in, var_0, var_159);
            var_162 = wp::load(var_160);
            var_161 = wp::mul(var_150, var_162);
            var_163 = wp::add(var_141, var_161);
            // w = support.eval_basis_trilinear(local, li * 4 + lj * 2 + lk)                      <L 289>
            var_166 = wp::mul(var_98, var_165);
            var_168 = wp::mul(var_142, var_167);
            var_169 = wp::add(var_166, var_168);
            var_170 = wp::add(var_169, var_164);
            var_171 = eval_basis_trilinear_0(var_86, var_170);
            // gi = ci + li                                                                       <L 290>
            var_172 = wp::add(var_38, var_98);
            // gj = cj + lj                                                                       <L 291>
            var_173 = wp::add(var_48, var_142);
            // gk = ck + lk                                                                       <L 292>
            var_174 = wp::add(var_58, var_164);
            // node_idx = gi * ny_g * nz_g + gj * nz_g + gk                                       <L 293>
            var_175 = wp::mul(var_172, var_88);
            var_176 = wp::mul(var_175, var_90);
            var_177 = wp::mul(var_173, var_90);
            var_178 = wp::add(var_176, var_177);
            var_179 = wp::add(var_178, var_174);
            // result += w * flexnode_xpos_in[worldid, nstart + node_idx]                         <L 294>
            var_180 = wp::add(var_92, var_179);
            var_181 = wp::address(var_flexnode_xpos_in, var_0, var_180);
            var_183 = wp::load(var_181);
            var_182 = wp::mul(var_171, var_183);
            var_184 = wp::add(var_163, var_182);
            // for lj in range(2):                                                                <L 287>
            // for lk in range(2):                                                                <L 288>
            // w = support.eval_basis_trilinear(local, li * 4 + lj * 2 + lk)                      <L 289>
            var_189 = wp::mul(var_185, var_188);
            var_191 = wp::mul(var_186, var_190);
            var_192 = wp::add(var_189, var_191);
            var_193 = wp::add(var_192, var_187);
            var_194 = eval_basis_trilinear_0(var_86, var_193);
            // gi = ci + li                                                                       <L 290>
            var_195 = wp::add(var_38, var_185);
            // gj = cj + lj                                                                       <L 291>
            var_196 = wp::add(var_48, var_186);
            // gk = ck + lk                                                                       <L 292>
            var_197 = wp::add(var_58, var_187);
            // node_idx = gi * ny_g * nz_g + gj * nz_g + gk                                       <L 293>
            var_198 = wp::mul(var_195, var_88);
            var_199 = wp::mul(var_198, var_90);
            var_200 = wp::mul(var_196, var_90);
            var_201 = wp::add(var_199, var_200);
            var_202 = wp::add(var_201, var_197);
            // result += w * flexnode_xpos_in[worldid, nstart + node_idx]                         <L 294>
            var_203 = wp::add(var_92, var_202);
            var_204 = wp::address(var_flexnode_xpos_in, var_0, var_203);
            var_206 = wp::load(var_204);
            var_205 = wp::mul(var_194, var_206);
            var_207 = wp::add(var_184, var_205);
            // w = support.eval_basis_trilinear(local, li * 4 + lj * 2 + lk)                      <L 289>
            var_210 = wp::mul(var_185, var_209);
            var_212 = wp::mul(var_186, var_211);
            var_213 = wp::add(var_210, var_212);
            var_214 = wp::add(var_213, var_208);
            var_215 = eval_basis_trilinear_0(var_86, var_214);
            // gi = ci + li                                                                       <L 290>
            var_216 = wp::add(var_38, var_185);
            // gj = cj + lj                                                                       <L 291>
            var_217 = wp::add(var_48, var_186);
            // gk = ck + lk                                                                       <L 292>
            var_218 = wp::add(var_58, var_208);
            // node_idx = gi * ny_g * nz_g + gj * nz_g + gk                                       <L 293>
            var_219 = wp::mul(var_216, var_88);
            var_220 = wp::mul(var_219, var_90);
            var_221 = wp::mul(var_217, var_90);
            var_222 = wp::add(var_220, var_221);
            var_223 = wp::add(var_222, var_218);
            // result += w * flexnode_xpos_in[worldid, nstart + node_idx]                         <L 294>
            var_224 = wp::add(var_92, var_223);
            var_225 = wp::address(var_flexnode_xpos_in, var_0, var_224);
            var_227 = wp::load(var_225);
            var_226 = wp::mul(var_215, var_227);
            var_228 = wp::add(var_207, var_226);
            // for lk in range(2):                                                                <L 288>
            // w = support.eval_basis_trilinear(local, li * 4 + lj * 2 + lk)                      <L 289>
            var_232 = wp::mul(var_185, var_231);
            var_234 = wp::mul(var_229, var_233);
            var_235 = wp::add(var_232, var_234);
            var_236 = wp::add(var_235, var_230);
            var_237 = eval_basis_trilinear_0(var_86, var_236);
            // gi = ci + li                                                                       <L 290>
            var_238 = wp::add(var_38, var_185);
            // gj = cj + lj                                                                       <L 291>
            var_239 = wp::add(var_48, var_229);
            // gk = ck + lk                                                                       <L 292>
            var_240 = wp::add(var_58, var_230);
            // node_idx = gi * ny_g * nz_g + gj * nz_g + gk                                       <L 293>
            var_241 = wp::mul(var_238, var_88);
            var_242 = wp::mul(var_241, var_90);
            var_243 = wp::mul(var_239, var_90);
            var_244 = wp::add(var_242, var_243);
            var_245 = wp::add(var_244, var_240);
            // result += w * flexnode_xpos_in[worldid, nstart + node_idx]                         <L 294>
            var_246 = wp::add(var_92, var_245);
            var_247 = wp::address(var_flexnode_xpos_in, var_0, var_246);
            var_249 = wp::load(var_247);
            var_248 = wp::mul(var_237, var_249);
            var_250 = wp::add(var_228, var_248);
            // w = support.eval_basis_trilinear(local, li * 4 + lj * 2 + lk)                      <L 289>
            var_253 = wp::mul(var_185, var_252);
            var_255 = wp::mul(var_229, var_254);
            var_256 = wp::add(var_253, var_255);
            var_257 = wp::add(var_256, var_251);
            var_258 = eval_basis_trilinear_0(var_86, var_257);
            // gi = ci + li                                                                       <L 290>
            var_259 = wp::add(var_38, var_185);
            // gj = cj + lj                                                                       <L 291>
            var_260 = wp::add(var_48, var_229);
            // gk = ck + lk                                                                       <L 292>
            var_261 = wp::add(var_58, var_251);
            // node_idx = gi * ny_g * nz_g + gj * nz_g + gk                                       <L 293>
            var_262 = wp::mul(var_259, var_88);
            var_263 = wp::mul(var_262, var_90);
            var_264 = wp::mul(var_260, var_90);
            var_265 = wp::add(var_263, var_264);
            var_266 = wp::add(var_265, var_261);
            // result += w * flexnode_xpos_in[worldid, nstart + node_idx]                         <L 294>
            var_267 = wp::add(var_92, var_266);
            var_268 = wp::address(var_flexnode_xpos_in, var_0, var_267);
            var_270 = wp::load(var_268);
            var_269 = wp::mul(var_258, var_270);
            var_271 = wp::add(var_250, var_269);
            // flexvert_xpos_out[worldid, vertid] = result                                        <L 296>
            wp::array_store(var_flexvert_xpos_out, var_0, var_1, var_271);
        }
        if (!var_15) {
            // bodyid = flex_vertbodyid[vertid]                                                   <L 299>
            var_272 = wp::address(var_flex_vertbodyid, var_1);
            var_274 = wp::load(var_272);
            var_273 = wp::copy(var_274);
            // xpos = xpos_in[worldid, bodyid]                                                    <L 300>
            var_275 = wp::address(var_xpos_in, var_0, var_273);
            var_277 = wp::load(var_275);
            var_276 = wp::copy(var_277);
            // if flex_centered[f]:                                                               <L 302>
            var_278 = wp::address(var_flex_centered, var_3);
            var_279 = wp::load(var_278);
            if (var_279) {
                // flexvert_xpos_out[worldid, vertid] = xpos                                      <L 303>
                wp::array_store(var_flexvert_xpos_out, var_0, var_1, var_276);
            }
            var_280 = wp::load(var_278);
            var_281 = wp::load(var_278);
            if (!var_281) {
                // xmat = xmat_in[worldid, bodyid]                                                <L 305>
                var_282 = wp::address(var_xmat_in, var_0, var_273);
                var_284 = wp::load(var_282);
                var_283 = wp::copy(var_284);
                // local_pos = flex_vert[vertid]                                                  <L 306>
                var_285 = wp::address(var_flex_vert, var_1);
                var_287 = wp::load(var_285);
                var_286 = wp::copy(var_287);
                // flexvert_xpos_out[worldid, vertid] = xmat @ local_pos + xpos                   <L 307>
                var_288 = wp::mul(var_283, var_286);
                var_289 = wp::add(var_288, var_276);
                wp::array_store(var_flexvert_xpos_out, var_0, var_1, var_289);
            }
            var_290 = wp::load(var_278);
        }
    }
}



extern "C" __global__ void _cacc_world_6ec26fdb_cuda_kernel_forward(
    wp::launch_bounds_t<1> dim,
    wp::array_t<wp::vec_t<3, wp::float32>> var_gravity,
    wp::array_t<wp::vec_t<6, wp::float32>> var_cacc_out)
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
        const wp::float32 var_1 = 0.0;
        wp::vec_t<3, wp::float32> var_2;
        wp::shape_t* var_3;
        const wp::int32 var_4 = 0;
        wp::int32 var_5;
        wp::shape_t var_6;
        wp::int32 var_7;
        wp::vec_t<3, wp::float32>* var_8;
        wp::vec_t<3, wp::float32> var_9;
        wp::vec_t<3, wp::float32> var_10;
        wp::vec_t<6, wp::float32> var_11;
        const wp::int32 var_12 = 0;
        //---------
        // forward
        // def _cacc_world(                                                                       <L 1324>
        // worldid = wp.tid()                                                                     <L 1330>
        var_0 = builtin_tid1d();
        // cacc_out[worldid, 0] = wp.spatial_vector(wp.vec3(0.0), -gravity[worldid % gravity.shape[0]])       <L 1331>
        var_2 = wp::vec_t<3, wp::float32>(var_1);
        var_3 = &(var_gravity.shape);
        var_6 = wp::load(var_3);
        var_5 = wp::extract(var_6, var_4);
        var_7 = wp::mod(var_0, var_5);
        var_8 = wp::address(var_gravity, var_7);
        var_10 = wp::load(var_8);
        var_9 = wp::neg(var_10);
        var_11 = wp::vec_t<6, wp::float32>(var_2, var_9);
        wp::array_store(var_cacc_out, var_0, var_12, var_11);
    }
}



extern "C" __global__ void _cfrc_backward_72237d38_cuda_kernel_forward(
    wp::launch_bounds_t<2> dim,
    wp::array_t<wp::int32> var_body_parentid,
    wp::array_t<wp::vec_t<6, wp::float32>> var_cfrc_int_in,
    wp::array_t<wp::int32> var_body_tree_,
    wp::array_t<wp::vec_t<6, wp::float32>> var_cfrc_int_out)
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
        wp::int32* var_5;
        wp::int32 var_6;
        wp::int32 var_7;
        const wp::int32 var_8 = 0;
        bool var_9;
        wp::array_t<wp::vec_t<6, wp::float32>> var_10;
        wp::vec_t<6, wp::float32>* var_11;
        wp::vec_t<6, wp::float32> var_12;
        wp::vec_t<6, wp::float32> var_13;
        //---------
        // forward
        // def _cfrc_backward(                                                                    <L 1430>
        // worldid, nodeid = wp.tid()                                                             <L 1440>
        builtin_tid2d(var_0, var_1);
        // bodyid = body_tree_[nodeid]                                                            <L 1441>
        var_2 = wp::address(var_body_tree_, var_1);
        var_4 = wp::load(var_2);
        var_3 = wp::copy(var_4);
        // pid = body_parentid[bodyid]                                                            <L 1442>
        var_5 = wp::address(var_body_parentid, var_3);
        var_7 = wp::load(var_5);
        var_6 = wp::copy(var_7);
        // if bodyid != 0:                                                                        <L 1443>
        var_9 = (var_3 != var_8);
        if (var_9) {
            // wp.atomic_add(cfrc_int_out[worldid], pid, cfrc_int_in[worldid, bodyid])            <L 1444>
            var_10 = wp::view(var_cfrc_int_out, var_0);
            var_11 = wp::address(var_cfrc_int_in, var_0, var_3);
            var_13 = wp::load(var_11);
            var_12 = wp::atomic_add(var_10, var_6, var_13);
        }
    }
}



extern "C" __global__ void _flex_nodes_7c3ae771_cuda_kernel_forward(
    wp::launch_bounds_t<2> dim,
    wp::int32 var_nflex,
    wp::array_t<wp::int32> var_flex_nodeadr,
    wp::array_t<wp::int32> var_flex_nodenum,
    wp::array_t<wp::int32> var_flex_nodebodyid,
    wp::array_t<wp::vec_t<3, wp::float32>> var_flex_node,
    wp::array_t<bool> var_flex_centered,
    wp::array_t<wp::vec_t<3, wp::float32>> var_xpos_in,
    wp::array_t<wp::mat_t<3, 3, wp::float32>> var_xmat_in,
    wp::array_t<wp::vec_t<3, wp::float32>> var_flexnode_xpos_out)
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
        wp::range_t var_2;
        wp::int32 var_3;
        wp::int32* var_4;
        wp::int32 var_5;
        wp::int32 var_6;
        bool var_7;
        const wp::int32 var_8 = 0;
        bool var_9;
        wp::int32* var_10;
        bool var_11;
        wp::int32 var_12;
        wp::int32* var_13;
        wp::int32 var_14;
        wp::int32 var_15;
        wp::vec_t<3, wp::float32>* var_16;
        wp::vec_t<3, wp::float32> var_17;
        wp::vec_t<3, wp::float32> var_18;
        bool* var_19;
        bool var_20;
        bool var_21;
        bool var_22;
        wp::vec_t<3, wp::float32>* var_23;
        wp::vec_t<3, wp::float32> var_24;
        wp::vec_t<3, wp::float32> var_25;
        bool var_26;
        const wp::int32 var_27 = 0;
        wp::float32 var_28;
        const wp::float32 var_29 = 0.0;
        bool var_30;
        const wp::int32 var_31 = 1;
        wp::float32 var_32;
        const wp::float32 var_33 = 0.0;
        bool var_34;
        const wp::int32 var_35 = 2;
        wp::float32 var_36;
        const wp::float32 var_37 = 0.0;
        bool var_38;
        wp::mat_t<3, 3, wp::float32>* var_39;
        wp::mat_t<3, 3, wp::float32> var_40;
        wp::mat_t<3, 3, wp::float32> var_41;
        wp::vec_t<3, wp::float32> var_42;
        wp::vec_t<3, wp::float32> var_43;
        bool var_44;
        //---------
        // forward
        // def _flex_nodes(                                                                       <L 311>
        // worldid, nodeid = wp.tid()                                                             <L 325>
        builtin_tid2d(var_0, var_1);
        // for f in range(nflex):                                                                 <L 327>
        var_2 = wp::range(var_nflex);
        start_for_0:;
            if (iter_cmp(var_2) == 0) goto end_for_0;
            var_3 = wp::iter_next(var_2);
            // locid = nodeid - flex_nodeadr[f]                                                   <L 328>
            var_4 = wp::address(var_flex_nodeadr, var_3);
            var_6 = wp::load(var_4);
            var_5 = wp::sub(var_1, var_6);
            // if locid >= 0 and locid < flex_nodenum[f]:                                         <L 329>
            var_9 = (var_5 >= var_8);
            var_7 = var_9;
            if (var_7) {
                var_10 = wp::address(var_flex_nodenum, var_3);
                var_12 = wp::load(var_10);
                var_11 = (var_5 < var_12);
                var_7 = var_7 && var_11;
            }
            if (var_7) {
                // break                                                                          <L 330>
                goto end_for_0;
            }
            goto start_for_0;
        end_for_0:;
        // bodyid = flex_nodebodyid[nodeid]                                                       <L 332>
        var_13 = wp::address(var_flex_nodebodyid, var_1);
        var_15 = wp::load(var_13);
        var_14 = wp::copy(var_15);
        // xpos = xpos_in[worldid, bodyid]                                                        <L 333>
        var_16 = wp::address(var_xpos_in, var_0, var_14);
        var_18 = wp::load(var_16);
        var_17 = wp::copy(var_18);
        // if flex_centered[f]:                                                                   <L 335>
        var_19 = wp::address(var_flex_centered, var_3);
        var_20 = wp::load(var_19);
        if (var_20) {
            // flexnode_xpos_out[worldid, nodeid] = xpos                                          <L 336>
            wp::array_store(var_flexnode_xpos_out, var_0, var_1, var_17);
        }
        var_21 = wp::load(var_19);
        var_22 = wp::load(var_19);
        if (!var_22) {
            // local_pos = flex_node[nodeid]                                                      <L 338>
            var_23 = wp::address(var_flex_node, var_1);
            var_25 = wp::load(var_23);
            var_24 = wp::copy(var_25);
            // if local_pos[0] == 0.0 and local_pos[1] == 0.0 and local_pos[2] == 0.0:            <L 339>
            var_28 = wp::extract(var_24, var_27);
            var_30 = (var_28 == var_29);
            var_26 = var_30;
            if (var_26) {
                var_32 = wp::extract(var_24, var_31);
                var_34 = (var_32 == var_33);
                var_26 = var_26 && var_34;
            }
            if (var_26) {
                var_36 = wp::extract(var_24, var_35);
                var_38 = (var_36 == var_37);
                var_26 = var_26 && var_38;
            }
            if (var_26) {
                // flexnode_xpos_out[worldid, nodeid] = xpos                                      <L 340>
                wp::array_store(var_flexnode_xpos_out, var_0, var_1, var_17);
            }
            if (!var_26) {
                // xmat = xmat_in[worldid, bodyid]                                                <L 342>
                var_39 = wp::address(var_xmat_in, var_0, var_14);
                var_41 = wp::load(var_39);
                var_40 = wp::copy(var_41);
                // flexnode_xpos_out[worldid, nodeid] = xmat @ local_pos + xpos                   <L 343>
                var_42 = wp::mul(var_40, var_24);
                var_43 = wp::add(var_42, var_17);
                wp::array_store(var_flexnode_xpos_out, var_0, var_1, var_43);
            }
        }
        var_44 = wp::load(var_19);
    }
}



extern "C" __global__ void _spatial_geom_tendon_7f51691f_cuda_kernel_forward(
    wp::launch_bounds_t<2> dim,
    wp::array_t<wp::int32> var_body_parentid,
    wp::array_t<wp::int32> var_body_rootid,
    wp::array_t<wp::int32> var_body_dofnum,
    wp::array_t<wp::int32> var_body_dofadr,
    wp::array_t<wp::int32> var_geom_bodyid,
    wp::array_t<wp::vec_t<3, wp::float32>> var_geom_size,
    wp::array_t<wp::int32> var_site_bodyid,
    wp::array_t<wp::int32> var_ten_J_rownnz,
    wp::array_t<wp::int32> var_ten_J_rowadr,
    wp::array_t<wp::int32> var_ten_J_colind,
    wp::array_t<wp::int32> var_wrap_type,
    wp::array_t<wp::int32> var_wrap_objid,
    wp::array_t<wp::float32> var_wrap_prm,
    wp::array_t<wp::int32> var_tendon_geom_adr,
    wp::array_t<wp::int32> var_wrap_geom_adr,
    wp::array_t<wp::float32> var_wrap_pulley_scale,
    wp::array_t<wp::vec_t<3, wp::float32>> var_geom_xpos_in,
    wp::array_t<wp::mat_t<3, 3, wp::float32>> var_geom_xmat_in,
    wp::array_t<wp::vec_t<3, wp::float32>> var_site_xpos_in,
    wp::array_t<wp::vec_t<3, wp::float32>> var_subtree_com_in,
    wp::array_t<wp::vec_t<6, wp::float32>> var_cdof_in,
    wp::array_t<wp::float32> var_ten_J_out,
    wp::array_t<wp::float32> var_ten_length_out,
    wp::array_t<wp::vec_t<6, wp::float32>> var_wrap_geom_xpos_out)
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
        wp::int32* var_5;
        wp::int32 var_6;
        wp::int32 var_7;
        wp::float32* var_8;
        wp::float32 var_9;
        wp::float32 var_10;
        const wp::int32 var_11 = 1;
        wp::int32 var_12;
        wp::int32* var_13;
        wp::int32 var_14;
        wp::int32 var_15;
        const wp::int32 var_16 = 0;
        wp::int32 var_17;
        wp::int32* var_18;
        wp::int32 var_19;
        wp::int32 var_20;
        const wp::int32 var_21 = 1;
        wp::int32 var_22;
        wp::int32* var_23;
        wp::int32 var_24;
        wp::int32 var_25;
        wp::vec_t<3, wp::float32>* var_26;
        wp::vec_t<3, wp::float32> var_27;
        wp::vec_t<3, wp::float32> var_28;
        wp::vec_t<3, wp::float32>* var_29;
        wp::vec_t<3, wp::float32> var_30;
        wp::vec_t<3, wp::float32> var_31;
        wp::vec_t<3, wp::float32>* var_32;
        wp::vec_t<3, wp::float32> var_33;
        wp::vec_t<3, wp::float32> var_34;
        wp::mat_t<3, 3, wp::float32>* var_35;
        wp::mat_t<3, 3, wp::float32> var_36;
        wp::mat_t<3, 3, wp::float32> var_37;
        wp::shape_t* var_38;
        const wp::int32 var_39 = 0;
        wp::int32 var_40;
        wp::shape_t var_41;
        wp::int32 var_42;
        wp::vec_t<3, wp::float32>* var_43;
        const wp::int32 var_44 = 0;
        wp::float32 var_45;
        wp::vec_t<3, wp::float32> var_46;
        wp::int32* var_47;
        wp::int32 var_48;
        wp::int32 var_49;
        wp::int32* var_50;
        wp::int32 var_51;
        wp::int32 var_52;
        wp::int32* var_53;
        wp::int32 var_54;
        wp::int32 var_55;
        wp::int32* var_56;
        wp::int32 var_57;
        wp::int32 var_58;
        wp::float32* var_59;
        wp::float32 var_60;
        wp::float32 var_61;
        wp::int32 var_62;
        const wp::int32 var_63 = 0;
        bool var_64;
        wp::vec_t<3, wp::float32>* var_65;
        wp::vec_t<3, wp::float32> var_66;
        wp::vec_t<3, wp::float32> var_67;
        const wp::float32 var_68 = 10000000000.0;
        wp::vec_t<3, wp::float32> var_69;
        wp::vec_t<3, wp::float32> var_70;
        wp::float32 var_71;
        wp::vec_t<3, wp::float32> var_72;
        wp::vec_t<3, wp::float32> var_73;
        wp::vec_t<6, wp::float32> var_74;
        wp::int32* var_75;
        wp::int32 var_76;
        wp::int32 var_77;
        wp::int32* var_78;
        wp::int32 var_79;
        wp::int32 var_80;
        const wp::float32 var_81 = 0.0;
        bool var_82;
        wp::vec_t<3, wp::float32> var_83;
        wp::vec_t<3, wp::float32> var_84;
        wp::vec_t<3, wp::float32> var_85;
        wp::float32 var_86;
        wp::vec_t<3, wp::float32> var_87;
        wp::float32 var_88;
        wp::float32 var_89;
        wp::float32 var_90;
        wp::array_t<wp::float32> var_91;
        wp::float32 var_92;
        wp::float32 var_93;
        const wp::float32 var_94 = 1e-15;
        bool var_95;
        const wp::float32 var_96 = 1.0;
        const wp::float32 var_97 = 0.0;
        const wp::float32 var_98 = 0.0;
        wp::vec_t<3, wp::float32> var_99;
        wp::vec_t<3, wp::float32> var_100;
        bool var_101;
        const wp::float32 var_102 = 1.0;
        const wp::float32 var_103 = 0.0;
        const wp::float32 var_104 = 0.0;
        wp::vec_t<3, wp::float32> var_105;
        wp::vec_t<3, wp::float32> var_106;
        bool var_107;
        bool var_108;
        wp::int32* var_109;
        wp::int32 var_110;
        wp::vec_t<3, wp::float32>* var_111;
        wp::vec_t<3, wp::float32> var_112;
        wp::vec_t<3, wp::float32> var_113;
        wp::int32* var_114;
        wp::int32 var_115;
        wp::vec_t<3, wp::float32>* var_116;
        wp::vec_t<3, wp::float32> var_117;
        wp::vec_t<3, wp::float32> var_118;
        wp::float32 var_119;
        wp::int32* var_120;
        wp::int32 var_121;
        wp::vec_t<3, wp::float32>* var_122;
        wp::vec_t<3, wp::float32> var_123;
        wp::vec_t<3, wp::float32> var_124;
        wp::int32* var_125;
        wp::int32 var_126;
        wp::vec_t<3, wp::float32>* var_127;
        wp::vec_t<3, wp::float32> var_128;
        wp::vec_t<3, wp::float32> var_129;
        wp::float32 var_130;
        wp::vec_t<3, wp::float32> var_131;
        wp::vec_t<3, wp::float32> var_132;
        wp::float32 var_133;
        wp::array_t<wp::float32> var_134;
        wp::float32 var_135;
        wp::float32 var_136;
        const wp::float32 var_137 = 1e-15;
        bool var_138;
        const wp::float32 var_139 = 1.0;
        const wp::float32 var_140 = 0.0;
        const wp::float32 var_141 = 0.0;
        wp::vec_t<3, wp::float32> var_142;
        wp::vec_t<3, wp::float32> var_143;
        bool var_144;
        wp::int32* var_145;
        wp::int32 var_146;
        wp::vec_t<3, wp::float32>* var_147;
        wp::vec_t<3, wp::float32> var_148;
        wp::vec_t<3, wp::float32> var_149;
        wp::int32* var_150;
        wp::int32 var_151;
        wp::vec_t<3, wp::float32>* var_152;
        wp::vec_t<3, wp::float32> var_153;
        wp::vec_t<3, wp::float32> var_154;
        wp::float32 var_155;
        wp::float32 var_156;
        wp::vec_t<3, wp::float32> var_157;
        wp::vec_t<3, wp::float32> var_158;
        //---------
        // forward
        // def _spatial_geom_tendon(                                                              <L 4155>
        // worldid, elementid = wp.tid()                                                          <L 4185>
        builtin_tid2d(var_0, var_1);
        // wrap_adr = wrap_geom_adr[elementid]                                                    <L 4186>
        var_2 = wp::address(var_wrap_geom_adr, var_1);
        var_4 = wp::load(var_2);
        var_3 = wp::copy(var_4);
        // tenid = tendon_geom_adr[elementid]                                                     <L 4187>
        var_5 = wp::address(var_tendon_geom_adr, var_1);
        var_7 = wp::load(var_5);
        var_6 = wp::copy(var_7);
        // pulley_scale = wrap_pulley_scale[wrap_adr]                                             <L 4190>
        var_8 = wp::address(var_wrap_pulley_scale, var_3);
        var_10 = wp::load(var_8);
        var_9 = wp::copy(var_10);
        // wrap_objid_site0 = wrap_objid[wrap_adr - 1]                                            <L 4193>
        var_12 = wp::sub(var_3, var_11);
        var_13 = wp::address(var_wrap_objid, var_12);
        var_15 = wp::load(var_13);
        var_14 = wp::copy(var_15);
        // wrap_objid_geom = wrap_objid[wrap_adr + 0]                                             <L 4194>
        var_17 = wp::add(var_3, var_16);
        var_18 = wp::address(var_wrap_objid, var_17);
        var_20 = wp::load(var_18);
        var_19 = wp::copy(var_20);
        // wrap_objid_site1 = wrap_objid[wrap_adr + 1]                                            <L 4195>
        var_22 = wp::add(var_3, var_21);
        var_23 = wp::address(var_wrap_objid, var_22);
        var_25 = wp::load(var_23);
        var_24 = wp::copy(var_25);
        // site_pnt0 = site_xpos_in[worldid, wrap_objid_site0]                                    <L 4198>
        var_26 = wp::address(var_site_xpos_in, var_0, var_14);
        var_28 = wp::load(var_26);
        var_27 = wp::copy(var_28);
        // site_pnt1 = site_xpos_in[worldid, wrap_objid_site1]                                    <L 4199>
        var_29 = wp::address(var_site_xpos_in, var_0, var_24);
        var_31 = wp::load(var_29);
        var_30 = wp::copy(var_31);
        // geom_xpos = geom_xpos_in[worldid, wrap_objid_geom]                                     <L 4202>
        var_32 = wp::address(var_geom_xpos_in, var_0, var_19);
        var_34 = wp::load(var_32);
        var_33 = wp::copy(var_34);
        // geom_xmat = geom_xmat_in[worldid, wrap_objid_geom]                                     <L 4203>
        var_35 = wp::address(var_geom_xmat_in, var_0, var_19);
        var_37 = wp::load(var_35);
        var_36 = wp::copy(var_37);
        // geomsize = geom_size[worldid % geom_size.shape[0], wrap_objid_geom][0]                 <L 4204>
        var_38 = &(var_geom_size.shape);
        var_41 = wp::load(var_38);
        var_40 = wp::extract(var_41, var_39);
        var_42 = wp::mod(var_0, var_40);
        var_43 = wp::address(var_geom_size, var_42, var_19);
        var_46 = wp::load(var_43);
        var_45 = wp::extract(var_46, var_44);
        // geom_type = wrap_type[wrap_adr]                                                        <L 4205>
        var_47 = wp::address(var_wrap_type, var_3);
        var_49 = wp::load(var_47);
        var_48 = wp::copy(var_49);
        // bodyid_site0 = site_bodyid[wrap_objid_site0]                                           <L 4208>
        var_50 = wp::address(var_site_bodyid, var_14);
        var_52 = wp::load(var_50);
        var_51 = wp::copy(var_52);
        // bodyid_geom = geom_bodyid[wrap_objid_geom]                                             <L 4209>
        var_53 = wp::address(var_geom_bodyid, var_19);
        var_55 = wp::load(var_53);
        var_54 = wp::copy(var_55);
        // bodyid_site1 = site_bodyid[wrap_objid_site1]                                           <L 4210>
        var_56 = wp::address(var_site_bodyid, var_24);
        var_58 = wp::load(var_56);
        var_57 = wp::copy(var_58);
        // sideid = int(wp.round(wrap_prm[wrap_adr]))                                             <L 4213>
        var_59 = wp::address(var_wrap_prm, var_3);
        var_61 = wp::load(var_59);
        var_60 = wp::round(var_61);
        var_62 = wp::int(var_60);
        // if sideid >= 0:                                                                        <L 4214>
        var_64 = (var_62 >= var_63);
        if (var_64) {
            // side = site_xpos_in[worldid, sideid]                                               <L 4215>
            var_65 = wp::address(var_site_xpos_in, var_0, var_62);
            var_67 = wp::load(var_65);
            var_66 = wp::copy(var_67);
        }
        if (!var_64) {
            // side = wp.vec3(MJ_MAXVAL)                                                          <L 4217>
            var_69 = wp::vec_t<3, wp::float32>(var_68);
        }
        var_70 = wp::where(var_64, var_66, var_69);
        // length_geomgeom, geom_pnt0, geom_pnt1 = util_misc.wrap(site_pnt0, site_pnt1, geom_xpos, geom_xmat, geomsize, geom_type, side)       <L 4220>
        wrap_0(var_27, var_30, var_33, var_36, var_45, var_48, var_70, var_71, var_72, var_73);
        // wrap_geom_xpos_out[worldid, elementid] = wp.spatial_vector(geom_pnt0, geom_pnt1)       <L 4223>
        var_74 = wp::vec_t<6, wp::float32>(var_72, var_73);
        wp::array_store(var_wrap_geom_xpos_out, var_0, var_1, var_74);
        // rownnz = ten_J_rownnz[tenid]                                                           <L 4225>
        var_75 = wp::address(var_ten_J_rownnz, var_6);
        var_77 = wp::load(var_75);
        var_76 = wp::copy(var_77);
        // rowadr = ten_J_rowadr[tenid]                                                           <L 4226>
        var_78 = wp::address(var_ten_J_rowadr, var_6);
        var_80 = wp::load(var_78);
        var_79 = wp::copy(var_80);
        // if length_geomgeom >= 0.0:                                                             <L 4228>
        var_82 = (var_71 >= var_81);
        if (var_82) {
            // dif_sitegeom = geom_pnt0 - site_pnt0                                               <L 4229>
            var_83 = wp::sub(var_72, var_27);
            // dif_geomsite = site_pnt1 - geom_pnt1                                               <L 4230>
            var_84 = wp::sub(var_30, var_73);
            // vec_sitegeom, length_sitegeom = math.normalize_with_norm(dif_sitegeom)             <L 4231>
            normalize_with_norm_0(var_83, var_85, var_86);
            // vec_geomsite, length_geomsite = math.normalize_with_norm(dif_geomsite)             <L 4232>
            normalize_with_norm_0(var_84, var_87, var_88);
            // length_sitegeomsite = length_sitegeom + length_geomgeom + length_geomsite          <L 4235>
            var_89 = wp::add(var_86, var_71);
            var_90 = wp::add(var_89, var_88);
            // if length_sitegeomsite:                                                            <L 4237>
            if (var_90) {
                // wp.atomic_add(ten_length_out[worldid], tenid, length_sitegeomsite * pulley_scale)       <L 4238>
                var_91 = wp::view(var_ten_length_out, var_0);
                var_92 = wp::mul(var_90, var_9);
                var_93 = wp::atomic_add(var_91, var_6, var_92);
            }
            // if length_sitegeom < MJ_MINVAL:                                                    <L 4241>
            var_95 = (var_86 < var_94);
            if (var_95) {
                // vec_sitegeom = wp.vec3(1.0, 0.0, 0.0)                                          <L 4242>
                var_99 = wp::vec_t<3, wp::float32>(var_96, var_97, var_98);
            }
            var_100 = wp::where(var_95, var_99, var_85);
            // if length_geomsite < MJ_MINVAL:                                                    <L 4244>
            var_101 = (var_88 < var_94);
            if (var_101) {
                // vec_geomsite = wp.vec3(1.0, 0.0, 0.0)                                          <L 4245>
                var_105 = wp::vec_t<3, wp::float32>(var_102, var_103, var_104);
            }
            var_106 = wp::where(var_101, var_105, var_87);
            // dif_body_sitegeom = bodyid_site0 != bodyid_geom                                    <L 4247>
            var_107 = (var_51 != var_54);
            // dif_body_geomsite = bodyid_geom != bodyid_site1                                    <L 4248>
            var_108 = (var_54 != var_57);
            // if dif_body_sitegeom:                                                              <L 4251>
            if (var_107) {
                // offset_site0 = site_pnt0 - subtree_com_in[worldid, body_rootid[bodyid_site0]]       <L 4252>
                var_109 = wp::address(var_body_rootid, var_51);
                var_110 = wp::load(var_109);
                var_111 = wp::address(var_subtree_com_in, var_0, var_110);
                var_113 = wp::load(var_111);
                var_112 = wp::sub(var_27, var_113);
                // offset_geom0 = geom_pnt0 - subtree_com_in[worldid, body_rootid[bodyid_geom]]       <L 4253>
                var_114 = wp::address(var_body_rootid, var_54);
                var_115 = wp::load(var_114);
                var_116 = wp::address(var_subtree_com_in, var_0, var_115);
                var_118 = wp::load(var_116);
                var_117 = wp::sub(var_72, var_118);
                // _accumulate_jac_chain(                                                         <L 4254>
                // body_parentid,                                                                 <L 4255>
                // body_dofnum,                                                                   <L 4256>
                // body_dofadr,                                                                   <L 4257>
                // ten_J_colind,                                                                  <L 4258>
                // cdof_in,                                                                       <L 4259>
                // offset_site0,                                                                  <L 4260>
                // vec_sitegeom,                                                                  <L 4261>
                // bodyid_site0,                                                                  <L 4262>
                // rowadr,                                                                        <L 4263>
                // rownnz,                                                                        <L 4264>
                // -pulley_scale,                                                                 <L 4265>
                var_119 = wp::neg(var_9);
                // worldid,                                                                       <L 4266>
                // ten_J_out,                                                                     <L 4267>
                _accumulate_jac_chain_0(var_body_parentid, var_body_dofnum, var_body_dofadr, var_ten_J_colind, var_cdof_in, var_112, var_100, var_51, var_79, var_76, var_119, var_0, var_ten_J_out);
                // _accumulate_jac_chain(                                                         <L 4269>
                // body_parentid,                                                                 <L 4270>
                // body_dofnum,                                                                   <L 4271>
                // body_dofadr,                                                                   <L 4272>
                // ten_J_colind,                                                                  <L 4273>
                // cdof_in,                                                                       <L 4274>
                // offset_geom0,                                                                  <L 4275>
                // vec_sitegeom,                                                                  <L 4276>
                // bodyid_geom,                                                                   <L 4277>
                // rowadr,                                                                        <L 4278>
                // rownnz,                                                                        <L 4279>
                // pulley_scale,                                                                  <L 4280>
                // worldid,                                                                       <L 4281>
                // ten_J_out,                                                                     <L 4282>
                _accumulate_jac_chain_0(var_body_parentid, var_body_dofnum, var_body_dofadr, var_ten_J_colind, var_cdof_in, var_117, var_100, var_54, var_79, var_76, var_9, var_0, var_ten_J_out);
            }
            // if dif_body_geomsite:                                                              <L 4286>
            if (var_108) {
                // offset_geom1 = geom_pnt1 - subtree_com_in[worldid, body_rootid[bodyid_geom]]       <L 4287>
                var_120 = wp::address(var_body_rootid, var_54);
                var_121 = wp::load(var_120);
                var_122 = wp::address(var_subtree_com_in, var_0, var_121);
                var_124 = wp::load(var_122);
                var_123 = wp::sub(var_73, var_124);
                // offset_site1 = site_pnt1 - subtree_com_in[worldid, body_rootid[bodyid_site1]]       <L 4288>
                var_125 = wp::address(var_body_rootid, var_57);
                var_126 = wp::load(var_125);
                var_127 = wp::address(var_subtree_com_in, var_0, var_126);
                var_129 = wp::load(var_127);
                var_128 = wp::sub(var_30, var_129);
                // _accumulate_jac_chain(                                                         <L 4289>
                // body_parentid,                                                                 <L 4290>
                // body_dofnum,                                                                   <L 4291>
                // body_dofadr,                                                                   <L 4292>
                // ten_J_colind,                                                                  <L 4293>
                // cdof_in,                                                                       <L 4294>
                // offset_geom1,                                                                  <L 4295>
                // vec_geomsite,                                                                  <L 4296>
                // bodyid_geom,                                                                   <L 4297>
                // rowadr,                                                                        <L 4298>
                // rownnz,                                                                        <L 4299>
                // -pulley_scale,                                                                 <L 4300>
                var_130 = wp::neg(var_9);
                // worldid,                                                                       <L 4301>
                // ten_J_out,                                                                     <L 4302>
                _accumulate_jac_chain_0(var_body_parentid, var_body_dofnum, var_body_dofadr, var_ten_J_colind, var_cdof_in, var_123, var_106, var_54, var_79, var_76, var_130, var_0, var_ten_J_out);
                // _accumulate_jac_chain(                                                         <L 4304>
                // body_parentid,                                                                 <L 4305>
                // body_dofnum,                                                                   <L 4306>
                // body_dofadr,                                                                   <L 4307>
                // ten_J_colind,                                                                  <L 4308>
                // cdof_in,                                                                       <L 4309>
                // offset_site1,                                                                  <L 4310>
                // vec_geomsite,                                                                  <L 4311>
                // bodyid_site1,                                                                  <L 4312>
                // rowadr,                                                                        <L 4313>
                // rownnz,                                                                        <L 4314>
                // pulley_scale,                                                                  <L 4315>
                // worldid,                                                                       <L 4316>
                // ten_J_out,                                                                     <L 4317>
                _accumulate_jac_chain_0(var_body_parentid, var_body_dofnum, var_body_dofadr, var_ten_J_colind, var_cdof_in, var_128, var_106, var_57, var_79, var_76, var_9, var_0, var_ten_J_out);
            }
        }
        if (!var_82) {
            // dif_sitesite = site_pnt1 - site_pnt0                                               <L 4320>
            var_131 = wp::sub(var_30, var_27);
            // vec_sitesite, length_sitesite = math.normalize_with_norm(dif_sitesite)             <L 4321>
            normalize_with_norm_0(var_131, var_132, var_133);
            // if length_sitesite:                                                                <L 4324>
            if (var_133) {
                // wp.atomic_add(ten_length_out[worldid], tenid, length_sitesite * pulley_scale)       <L 4325>
                var_134 = wp::view(var_ten_length_out, var_0);
                var_135 = wp::mul(var_133, var_9);
                var_136 = wp::atomic_add(var_134, var_6, var_135);
            }
            // if length_sitesite < MJ_MINVAL:                                                    <L 4328>
            var_138 = (var_133 < var_137);
            if (var_138) {
                // vec_sitesite = wp.vec3(1.0, 0.0, 0.0)                                          <L 4329>
                var_142 = wp::vec_t<3, wp::float32>(var_139, var_140, var_141);
            }
            var_143 = wp::where(var_138, var_142, var_132);
            // if bodyid_site0 != bodyid_site1:                                                   <L 4331>
            var_144 = (var_51 != var_57);
            if (var_144) {
                // offset_site0 = site_pnt0 - subtree_com_in[worldid, body_rootid[bodyid_site0]]       <L 4332>
                var_145 = wp::address(var_body_rootid, var_51);
                var_146 = wp::load(var_145);
                var_147 = wp::address(var_subtree_com_in, var_0, var_146);
                var_149 = wp::load(var_147);
                var_148 = wp::sub(var_27, var_149);
                // offset_site1 = site_pnt1 - subtree_com_in[worldid, body_rootid[bodyid_site1]]       <L 4333>
                var_150 = wp::address(var_body_rootid, var_57);
                var_151 = wp::load(var_150);
                var_152 = wp::address(var_subtree_com_in, var_0, var_151);
                var_154 = wp::load(var_152);
                var_153 = wp::sub(var_30, var_154);
                // _accumulate_jac_chain(                                                         <L 4334>
                // body_parentid,                                                                 <L 4335>
                // body_dofnum,                                                                   <L 4336>
                // body_dofadr,                                                                   <L 4337>
                // ten_J_colind,                                                                  <L 4338>
                // cdof_in,                                                                       <L 4339>
                // offset_site0,                                                                  <L 4340>
                // vec_sitesite,                                                                  <L 4341>
                // bodyid_site0,                                                                  <L 4342>
                // rowadr,                                                                        <L 4343>
                // rownnz,                                                                        <L 4344>
                // -pulley_scale,                                                                 <L 4345>
                var_155 = wp::neg(var_9);
                // worldid,                                                                       <L 4346>
                // ten_J_out,                                                                     <L 4347>
                _accumulate_jac_chain_0(var_body_parentid, var_body_dofnum, var_body_dofadr, var_ten_J_colind, var_cdof_in, var_148, var_143, var_51, var_79, var_76, var_155, var_0, var_ten_J_out);
                // _accumulate_jac_chain(                                                         <L 4349>
                // body_parentid,                                                                 <L 4350>
                // body_dofnum,                                                                   <L 4351>
                // body_dofadr,                                                                   <L 4352>
                // ten_J_colind,                                                                  <L 4353>
                // cdof_in,                                                                       <L 4354>
                // offset_site1,                                                                  <L 4355>
                // vec_sitesite,                                                                  <L 4356>
                // bodyid_site1,                                                                  <L 4357>
                // rowadr,                                                                        <L 4358>
                // rownnz,                                                                        <L 4359>
                // pulley_scale,                                                                  <L 4360>
                // worldid,                                                                       <L 4361>
                // ten_J_out,                                                                     <L 4362>
                _accumulate_jac_chain_0(var_body_parentid, var_body_dofnum, var_body_dofadr, var_ten_J_colind, var_cdof_in, var_153, var_143, var_57, var_79, var_76, var_9, var_0, var_ten_J_out);
            }
        }
        var_156 = wp::where(var_82, var_94, var_137);
        var_157 = wp::where(var_82, var_112, var_148);
        var_158 = wp::where(var_82, var_128, var_153);
    }
}



extern "C" __global__ void _linear_momentum_813ce6af_cuda_kernel_forward(
    wp::launch_bounds_t<2> dim,
    wp::array_t<wp::int32> var_body_parentid,
    wp::array_t<wp::float32> var_body_subtreemass,
    wp::array_t<wp::vec_t<3, wp::float32>> var_subtree_linvel_in,
    wp::array_t<wp::int32> var_body_tree_,
    wp::array_t<wp::vec_t<3, wp::float32>> var_subtree_linvel_out)
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
        wp::int32* var_5;
        wp::int32 var_6;
        wp::int32 var_7;
        wp::array_t<wp::vec_t<3, wp::float32>> var_8;
        wp::vec_t<3, wp::float32>* var_9;
        wp::vec_t<3, wp::float32> var_10;
        wp::vec_t<3, wp::float32> var_11;
        wp::vec_t<3, wp::float32>* var_12;
        const wp::float32 var_13 = 1e-15;
        wp::shape_t* var_14;
        const wp::int32 var_15 = 0;
        wp::int32 var_16;
        wp::shape_t var_17;
        wp::int32 var_18;
        wp::float32* var_19;
        wp::float32 var_20;
        wp::float32 var_21;
        wp::vec_t<3, wp::float32> var_22;
        wp::vec_t<3, wp::float32> var_23;
        //---------
        // forward
        // def _linear_momentum(                                                                  <L 3871>
        // worldid, nodeid = wp.tid()                                                             <L 3882>
        builtin_tid2d(var_0, var_1);
        // bodyid = body_tree_[nodeid]                                                            <L 3883>
        var_2 = wp::address(var_body_tree_, var_1);
        var_4 = wp::load(var_2);
        var_3 = wp::copy(var_4);
        // if bodyid:                                                                             <L 3884>
        if (var_3) {
            // pid = body_parentid[bodyid]                                                        <L 3885>
            var_5 = wp::address(var_body_parentid, var_3);
            var_7 = wp::load(var_5);
            var_6 = wp::copy(var_7);
            // wp.atomic_add(subtree_linvel_out[worldid], pid, subtree_linvel_in[worldid, bodyid])       <L 3886>
            var_8 = wp::view(var_subtree_linvel_out, var_0);
            var_9 = wp::address(var_subtree_linvel_in, var_0, var_3);
            var_11 = wp::load(var_9);
            var_10 = wp::atomic_add(var_8, var_6, var_11);
        }
        // subtree_linvel_out[worldid, bodyid] /= wp.max(MJ_MINVAL, body_subtreemass[worldid % body_subtreemass.shape[0], bodyid])       <L 3887>
        var_12 = wp::address(var_subtree_linvel_out, var_0, var_3);
        var_14 = &(var_body_subtreemass.shape);
        var_17 = wp::load(var_14);
        var_16 = wp::extract(var_17, var_15);
        var_18 = wp::mod(var_0, var_16);
        var_19 = wp::address(var_body_subtreemass, var_18, var_3);
        var_21 = wp::load(var_19);
        var_20 = wp::max(var_13, var_21);
        var_23 = wp::load(var_12);
        var_22 = wp::div(var_23, var_20);
        wp::array_store(var_subtree_linvel_out, var_0, var_3, var_22);
    }
}



extern "C" __global__ void _kinematics_branch_95f8028e_cuda_kernel_forward(
    wp::launch_bounds_t<2> dim,
    wp::array_t<wp::float32> var_qpos0,
    wp::array_t<wp::int32> var_body_parentid,
    wp::array_t<wp::int32> var_body_mocapid,
    wp::array_t<wp::int32> var_body_jntnum,
    wp::array_t<wp::int32> var_body_jntadr,
    wp::array_t<wp::vec_t<3, wp::float32>> var_body_pos,
    wp::array_t<wp::quat_t<wp::float32>> var_body_quat,
    wp::array_t<wp::int32> var_jnt_type,
    wp::array_t<wp::int32> var_jnt_qposadr,
    wp::array_t<wp::vec_t<3, wp::float32>> var_jnt_pos,
    wp::array_t<wp::vec_t<3, wp::float32>> var_jnt_axis,
    wp::array_t<wp::int32> var_body_branches,
    wp::array_t<wp::int32> var_body_branch_start,
    wp::array_t<wp::float32> var_qpos_in,
    wp::array_t<wp::vec_t<3, wp::float32>> var_mocap_pos_in,
    wp::array_t<wp::quat_t<wp::float32>> var_mocap_quat_in,
    wp::array_t<wp::vec_t<3, wp::float32>> var_xpos_out,
    wp::array_t<wp::quat_t<wp::float32>> var_xquat_out,
    wp::array_t<wp::vec_t<3, wp::float32>> var_xanchor_out,
    wp::array_t<wp::vec_t<3, wp::float32>> var_xaxis_out)
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
        const wp::int32 var_5 = 1;
        wp::int32 var_6;
        wp::int32* var_7;
        wp::int32 var_8;
        wp::int32 var_9;
        wp::array_t<wp::float32> var_10;
        wp::range_t var_11;
        wp::int32 var_12;
        wp::int32* var_13;
        wp::int32 var_14;
        wp::int32 var_15;
        wp::int32* var_16;
        wp::int32 var_17;
        wp::int32 var_18;
        wp::int32* var_19;
        wp::int32 var_20;
        wp::int32 var_21;
        wp::int32* var_22;
        wp::int32 var_23;
        wp::int32 var_24;
        const wp::int32 var_25 = 1;
        bool var_26;
        wp::int32* var_27;
        wp::int32 var_28;
        wp::int32 var_29;
        const wp::int32 var_30 = 0;
        bool var_31;
        wp::int32* var_32;
        wp::int32 var_33;
        wp::int32 var_34;
        wp::float32* var_35;
        const wp::int32 var_36 = 1;
        wp::int32 var_37;
        wp::float32* var_38;
        const wp::int32 var_39 = 2;
        wp::int32 var_40;
        wp::float32* var_41;
        wp::vec_t<3, wp::float32> var_42;
        wp::float32 var_43;
        wp::float32 var_44;
        wp::float32 var_45;
        const wp::int32 var_46 = 3;
        wp::int32 var_47;
        wp::float32* var_48;
        const wp::int32 var_49 = 4;
        wp::int32 var_50;
        wp::float32* var_51;
        const wp::int32 var_52 = 5;
        wp::int32 var_53;
        wp::float32* var_54;
        const wp::int32 var_55 = 6;
        wp::int32 var_56;
        wp::float32* var_57;
        wp::quat_t<wp::float32> var_58;
        wp::float32 var_59;
        wp::float32 var_60;
        wp::float32 var_61;
        wp::float32 var_62;
        wp::quat_t<wp::float32> var_63;
        wp::shape_t* var_64;
        const wp::int32 var_65 = 0;
        wp::int32 var_66;
        wp::shape_t var_67;
        wp::int32 var_68;
        wp::vec_t<3, wp::float32>* var_69;
        wp::vec_t<3, wp::float32> var_70;
        wp::vec_t<3, wp::float32> var_71;
        wp::shape_t* var_72;
        const wp::int32 var_73 = 0;
        wp::int32 var_74;
        wp::shape_t var_75;
        wp::int32 var_76;
        wp::int32* var_77;
        wp::int32 var_78;
        wp::int32 var_79;
        wp::int32* var_80;
        wp::int32 var_81;
        wp::int32 var_82;
        const wp::int32 var_83 = 0;
        bool var_84;
        wp::vec_t<3, wp::float32>* var_85;
        wp::vec_t<3, wp::float32> var_86;
        wp::vec_t<3, wp::float32> var_87;
        wp::quat_t<wp::float32>* var_88;
        wp::quat_t<wp::float32> var_89;
        wp::quat_t<wp::float32> var_90;
        wp::shape_t* var_91;
        const wp::int32 var_92 = 0;
        wp::int32 var_93;
        wp::shape_t var_94;
        wp::int32 var_95;
        wp::vec_t<3, wp::float32>* var_96;
        wp::vec_t<3, wp::float32> var_97;
        wp::vec_t<3, wp::float32> var_98;
        wp::shape_t* var_99;
        const wp::int32 var_100 = 0;
        wp::int32 var_101;
        wp::shape_t var_102;
        wp::int32 var_103;
        wp::quat_t<wp::float32>* var_104;
        wp::quat_t<wp::float32> var_105;
        wp::quat_t<wp::float32> var_106;
        wp::vec_t<3, wp::float32> var_107;
        wp::quat_t<wp::float32> var_108;
        const wp::int32 var_109 = 0;
        bool var_110;
        wp::quat_t<wp::float32>* var_111;
        wp::vec_t<3, wp::float32> var_112;
        wp::quat_t<wp::float32> var_113;
        wp::vec_t<3, wp::float32>* var_114;
        wp::vec_t<3, wp::float32> var_115;
        wp::vec_t<3, wp::float32> var_116;
        wp::quat_t<wp::float32>* var_117;
        wp::quat_t<wp::float32> var_118;
        wp::quat_t<wp::float32> var_119;
        wp::vec_t<3, wp::float32> var_120;
        wp::quat_t<wp::float32> var_121;
        wp::range_t var_122;
        wp::int32 var_123;
        wp::int32* var_124;
        wp::int32 var_125;
        wp::int32 var_126;
        wp::int32* var_127;
        wp::int32 var_128;
        wp::int32 var_129;
        wp::shape_t* var_130;
        const wp::int32 var_131 = 0;
        wp::int32 var_132;
        wp::shape_t var_133;
        wp::int32 var_134;
        wp::vec_t<3, wp::float32>* var_135;
        wp::vec_t<3, wp::float32> var_136;
        wp::vec_t<3, wp::float32> var_137;
        wp::vec_t<3, wp::float32>* var_138;
        wp::vec_t<3, wp::float32> var_139;
        wp::vec_t<3, wp::float32> var_140;
        wp::vec_t<3, wp::float32> var_141;
        wp::vec_t<3, wp::float32> var_142;
        const wp::int32 var_143 = 1;
        bool var_144;
        const wp::int32 var_145 = 0;
        wp::int32 var_146;
        wp::float32* var_147;
        const wp::int32 var_148 = 1;
        wp::int32 var_149;
        wp::float32* var_150;
        const wp::int32 var_151 = 2;
        wp::int32 var_152;
        wp::float32* var_153;
        const wp::int32 var_154 = 3;
        wp::int32 var_155;
        wp::float32* var_156;
        wp::quat_t<wp::float32> var_157;
        wp::float32 var_158;
        wp::float32 var_159;
        wp::float32 var_160;
        wp::float32 var_161;
        wp::quat_t<wp::float32> var_162;
        wp::quat_t<wp::float32> var_163;
        wp::vec_t<3, wp::float32>* var_164;
        wp::vec_t<3, wp::float32> var_165;
        wp::vec_t<3, wp::float32> var_166;
        wp::vec_t<3, wp::float32> var_167;
        const wp::int32 var_168 = 2;
        bool var_169;
        wp::float32* var_170;
        wp::shape_t* var_171;
        const wp::int32 var_172 = 0;
        wp::int32 var_173;
        wp::shape_t var_174;
        wp::int32 var_175;
        wp::float32* var_176;
        wp::float32 var_177;
        wp::float32 var_178;
        wp::float32 var_179;
        wp::vec_t<3, wp::float32> var_180;
        wp::vec_t<3, wp::float32> var_181;
        const wp::int32 var_182 = 3;
        bool var_183;
        wp::shape_t* var_184;
        const wp::int32 var_185 = 0;
        wp::int32 var_186;
        wp::shape_t var_187;
        wp::int32 var_188;
        wp::float32* var_189;
        wp::float32 var_190;
        wp::float32 var_191;
        wp::float32* var_192;
        wp::float32 var_193;
        wp::float32 var_194;
        wp::quat_t<wp::float32> var_195;
        wp::quat_t<wp::float32> var_196;
        wp::vec_t<3, wp::float32>* var_197;
        wp::vec_t<3, wp::float32> var_198;
        wp::vec_t<3, wp::float32> var_199;
        wp::vec_t<3, wp::float32> var_200;
        wp::vec_t<3, wp::float32> var_201;
        wp::quat_t<wp::float32> var_202;
        wp::vec_t<3, wp::float32> var_203;
        wp::quat_t<wp::float32> var_204;
        wp::vec_t<3, wp::float32> var_205;
        wp::quat_t<wp::float32> var_206;
        const wp::int32 var_207 = 1;
        wp::int32 var_208;
        wp::quat_t<wp::float32> var_209;
        //---------
        // forward
        // def _kinematics_branch(                                                                <L 48>
        // worldid, branchid = wp.tid()                                                           <L 73>
        builtin_tid2d(var_0, var_1);
        // start = body_branch_start[branchid]                                                    <L 75>
        var_2 = wp::address(var_body_branch_start, var_1);
        var_4 = wp::load(var_2);
        var_3 = wp::copy(var_4);
        // end = body_branch_start[branchid + 1]                                                  <L 76>
        var_6 = wp::add(var_1, var_5);
        var_7 = wp::address(var_body_branch_start, var_6);
        var_9 = wp::load(var_7);
        var_8 = wp::copy(var_9);
        // qpos = qpos_in[worldid]                                                                <L 78>
        var_10 = wp::view(var_qpos_in, var_0);
        // for i in range(start, end):                                                            <L 80>
        var_11 = wp::range(var_3, var_8);
        start_for_0:;
            if (iter_cmp(var_11) == 0) goto end_for_0;
            var_12 = wp::iter_next(var_11);
            // bodyid = body_branches[i]                                                          <L 81>
            var_13 = wp::address(var_body_branches, var_12);
            var_15 = wp::load(var_13);
            var_14 = wp::copy(var_15);
            // pid = body_parentid[bodyid]                                                        <L 82>
            var_16 = wp::address(var_body_parentid, var_14);
            var_18 = wp::load(var_16);
            var_17 = wp::copy(var_18);
            // jntadr = body_jntadr[bodyid]                                                       <L 83>
            var_19 = wp::address(var_body_jntadr, var_14);
            var_21 = wp::load(var_19);
            var_20 = wp::copy(var_21);
            // jntnum = body_jntnum[bodyid]                                                       <L 84>
            var_22 = wp::address(var_body_jntnum, var_14);
            var_24 = wp::load(var_22);
            var_23 = wp::copy(var_24);
            // if jntnum == 1:                                                                    <L 86>
            var_26 = (var_23 == var_25);
            if (var_26) {
                // jnt_type_ = jnt_type[jntadr]                                                   <L 87>
                var_27 = wp::address(var_jnt_type, var_20);
                var_29 = wp::load(var_27);
                var_28 = wp::copy(var_29);
                // if jnt_type_ == JointType.FREE:                                                <L 88>
                var_31 = (var_28 == var_30);
                if (var_31) {
                    // qadr = jnt_qposadr[jntadr]                                                 <L 89>
                    var_32 = wp::address(var_jnt_qposadr, var_20);
                    var_34 = wp::load(var_32);
                    var_33 = wp::copy(var_34);
                    // xpos = wp.vec3(qpos[qadr], qpos[qadr + 1], qpos[qadr + 2])                 <L 90>
                    var_35 = wp::address(var_10, var_33);
                    var_37 = wp::add(var_33, var_36);
                    var_38 = wp::address(var_10, var_37);
                    var_40 = wp::add(var_33, var_39);
                    var_41 = wp::address(var_10, var_40);
                    var_43 = wp::load(var_35);
                    var_44 = wp::load(var_38);
                    var_45 = wp::load(var_41);
                    var_42 = wp::vec_t<3, wp::float32>(var_43, var_44, var_45);
                    // xquat = wp.quat(qpos[qadr + 3], qpos[qadr + 4], qpos[qadr + 5], qpos[qadr + 6])       <L 91>
                    var_47 = wp::add(var_33, var_46);
                    var_48 = wp::address(var_10, var_47);
                    var_50 = wp::add(var_33, var_49);
                    var_51 = wp::address(var_10, var_50);
                    var_53 = wp::add(var_33, var_52);
                    var_54 = wp::address(var_10, var_53);
                    var_56 = wp::add(var_33, var_55);
                    var_57 = wp::address(var_10, var_56);
                    var_59 = wp::load(var_48);
                    var_60 = wp::load(var_51);
                    var_61 = wp::load(var_54);
                    var_62 = wp::load(var_57);
                    var_58 = wp::quat_t<wp::float32>(var_59, var_60, var_61, var_62);
                    // xquat = wp.normalize(xquat)                                                <L 92>
                    var_63 = wp::normalize(var_58);
                    // xpos_out[worldid, bodyid] = xpos                                           <L 94>
                    wp::array_store(var_xpos_out, var_0, var_14, var_42);
                    // xquat_out[worldid, bodyid] = xquat                                         <L 95>
                    wp::array_store(var_xquat_out, var_0, var_14, var_63);
                    // xanchor_out[worldid, jntadr] = xpos                                        <L 96>
                    wp::array_store(var_xanchor_out, var_0, var_20, var_42);
                    // xaxis_out[worldid, jntadr] = jnt_axis[worldid % jnt_axis.shape[0], jntadr]       <L 97>
                    var_64 = &(var_jnt_axis.shape);
                    var_67 = wp::load(var_64);
                    var_66 = wp::extract(var_67, var_65);
                    var_68 = wp::mod(var_0, var_66);
                    var_69 = wp::address(var_jnt_axis, var_68, var_20);
                    var_71 = wp::load(var_69);
                    var_70 = wp::copy(var_71);
                    wp::array_store(var_xaxis_out, var_0, var_20, var_70);
                    // continue                                                                   <L 98>
                    goto start_for_0;
                }
            }
            // jnt_pos_id = worldid % jnt_pos.shape[0]                                            <L 102>
            var_72 = &(var_jnt_pos.shape);
            var_75 = wp::load(var_72);
            var_74 = wp::extract(var_75, var_73);
            var_76 = wp::mod(var_0, var_74);
            // pid = body_parentid[bodyid]                                                        <L 103>
            var_77 = wp::address(var_body_parentid, var_14);
            var_79 = wp::load(var_77);
            var_78 = wp::copy(var_79);
            // mocapid = body_mocapid[bodyid]                                                     <L 106>
            var_80 = wp::address(var_body_mocapid, var_14);
            var_82 = wp::load(var_80);
            var_81 = wp::copy(var_82);
            // if mocapid >= 0:                                                                   <L 107>
            var_84 = (var_81 >= var_83);
            if (var_84) {
                // xpos = mocap_pos_in[worldid, mocapid]                                          <L 108>
                var_85 = wp::address(var_mocap_pos_in, var_0, var_81);
                var_87 = wp::load(var_85);
                var_86 = wp::copy(var_87);
                // xquat = mocap_quat_in[worldid, mocapid]                                        <L 109>
                var_88 = wp::address(var_mocap_quat_in, var_0, var_81);
                var_90 = wp::load(var_88);
                var_89 = wp::copy(var_90);
            }
            if (!var_84) {
                // xpos = body_pos[worldid % body_pos.shape[0], bodyid]                           <L 111>
                var_91 = &(var_body_pos.shape);
                var_94 = wp::load(var_91);
                var_93 = wp::extract(var_94, var_92);
                var_95 = wp::mod(var_0, var_93);
                var_96 = wp::address(var_body_pos, var_95, var_14);
                var_98 = wp::load(var_96);
                var_97 = wp::copy(var_98);
                // xquat = body_quat[worldid % body_quat.shape[0], bodyid]                        <L 112>
                var_99 = &(var_body_quat.shape);
                var_102 = wp::load(var_99);
                var_101 = wp::extract(var_102, var_100);
                var_103 = wp::mod(var_0, var_101);
                var_104 = wp::address(var_body_quat, var_103, var_14);
                var_106 = wp::load(var_104);
                var_105 = wp::copy(var_106);
            }
            var_107 = wp::where(var_84, var_86, var_97);
            var_108 = wp::where(var_84, var_89, var_105);
            // if pid >= 0:                                                                       <L 114>
            var_110 = (var_78 >= var_109);
            if (var_110) {
                // xpos = math.rot_vec_quat(xpos, xquat_out[worldid, pid]) + xpos_out[worldid, pid]       <L 115>
                var_111 = wp::address(var_xquat_out, var_0, var_78);
                var_113 = wp::load(var_111);
                var_112 = rot_vec_quat_0(var_107, var_113);
                var_114 = wp::address(var_xpos_out, var_0, var_78);
                var_116 = wp::load(var_114);
                var_115 = wp::add(var_112, var_116);
                // xquat = math.mul_quat(xquat_out[worldid, pid], xquat)                          <L 116>
                var_117 = wp::address(var_xquat_out, var_0, var_78);
                var_119 = wp::load(var_117);
                var_118 = mul_quat_0(var_119, var_108);
            }
            var_120 = wp::where(var_110, var_115, var_107);
            var_121 = wp::where(var_110, var_118, var_108);
            // for _ in range(jntnum):                                                            <L 118>
            var_122 = wp::range(var_23);
            start_for_2:;
                if (iter_cmp(var_122) == 0) goto end_for_2;
                var_123 = wp::iter_next(var_122);
                // qadr = jnt_qposadr[jntadr]                                                     <L 119>
                var_124 = wp::address(var_jnt_qposadr, var_20);
                var_126 = wp::load(var_124);
                var_125 = wp::copy(var_126);
                // jnt_type_ = jnt_type[jntadr]                                                   <L 120>
                var_127 = wp::address(var_jnt_type, var_20);
                var_129 = wp::load(var_127);
                var_128 = wp::copy(var_129);
                // jnt_axis_ = jnt_axis[worldid % jnt_axis.shape[0], jntadr]                      <L 121>
                var_130 = &(var_jnt_axis.shape);
                var_133 = wp::load(var_130);
                var_132 = wp::extract(var_133, var_131);
                var_134 = wp::mod(var_0, var_132);
                var_135 = wp::address(var_jnt_axis, var_134, var_20);
                var_137 = wp::load(var_135);
                var_136 = wp::copy(var_137);
                // xanchor = math.rot_vec_quat(jnt_pos[jnt_pos_id, jntadr], xquat) + xpos         <L 122>
                var_138 = wp::address(var_jnt_pos, var_76, var_20);
                var_140 = wp::load(var_138);
                var_139 = rot_vec_quat_0(var_140, var_121);
                var_141 = wp::add(var_139, var_120);
                // xaxis = math.rot_vec_quat(jnt_axis_, xquat)                                    <L 123>
                var_142 = rot_vec_quat_0(var_136, var_121);
                // if jnt_type_ == JointType.BALL:                                                <L 125>
                var_144 = (var_128 == var_143);
                if (var_144) {
                    // qloc = wp.quat(qpos[qadr + 0], qpos[qadr + 1], qpos[qadr + 2], qpos[qadr + 3])       <L 126>
                    var_146 = wp::add(var_125, var_145);
                    var_147 = wp::address(var_10, var_146);
                    var_149 = wp::add(var_125, var_148);
                    var_150 = wp::address(var_10, var_149);
                    var_152 = wp::add(var_125, var_151);
                    var_153 = wp::address(var_10, var_152);
                    var_155 = wp::add(var_125, var_154);
                    var_156 = wp::address(var_10, var_155);
                    var_158 = wp::load(var_147);
                    var_159 = wp::load(var_150);
                    var_160 = wp::load(var_153);
                    var_161 = wp::load(var_156);
                    var_157 = wp::quat_t<wp::float32>(var_158, var_159, var_160, var_161);
                    // qloc = wp.normalize(qloc)                                                  <L 127>
                    var_162 = wp::normalize(var_157);
                    // xquat = math.mul_quat(xquat, qloc)                                         <L 128>
                    var_163 = mul_quat_0(var_121, var_162);
                    // xpos = xanchor - math.rot_vec_quat(jnt_pos[jnt_pos_id, jntadr], xquat)       <L 130>
                    var_164 = wp::address(var_jnt_pos, var_76, var_20);
                    var_166 = wp::load(var_164);
                    var_165 = rot_vec_quat_0(var_166, var_163);
                    var_167 = wp::sub(var_141, var_165);
                }
                if (!var_144) {
                    // elif jnt_type_ == JointType.SLIDE:                                         <L 131>
                    var_169 = (var_128 == var_168);
                    if (var_169) {
                        // xpos += xaxis * (qpos[qadr] - qpos0[worldid % qpos0.shape[0], qadr])       <L 132>
                        var_170 = wp::address(var_10, var_125);
                        var_171 = &(var_qpos0.shape);
                        var_174 = wp::load(var_171);
                        var_173 = wp::extract(var_174, var_172);
                        var_175 = wp::mod(var_0, var_173);
                        var_176 = wp::address(var_qpos0, var_175, var_125);
                        var_178 = wp::load(var_170);
                        var_179 = wp::load(var_176);
                        var_177 = wp::sub(var_178, var_179);
                        var_180 = wp::mul(var_142, var_177);
                        var_181 = wp::add(var_120, var_180);
                    }
                    if (!var_169) {
                        // elif jnt_type_ == JointType.HINGE:                                     <L 133>
                        var_183 = (var_128 == var_182);
                        if (var_183) {
                            // qpos0_ = qpos0[worldid % qpos0.shape[0], qadr]                     <L 134>
                            var_184 = &(var_qpos0.shape);
                            var_187 = wp::load(var_184);
                            var_186 = wp::extract(var_187, var_185);
                            var_188 = wp::mod(var_0, var_186);
                            var_189 = wp::address(var_qpos0, var_188, var_125);
                            var_191 = wp::load(var_189);
                            var_190 = wp::copy(var_191);
                            // qloc_ = math.axis_angle_to_quat(jnt_axis_, qpos[qadr] - qpos0_)       <L 135>
                            var_192 = wp::address(var_10, var_125);
                            var_194 = wp::load(var_192);
                            var_193 = wp::sub(var_194, var_190);
                            var_195 = axis_angle_to_quat_0(var_136, var_193);
                            // xquat = math.mul_quat(xquat, qloc_)                                <L 136>
                            var_196 = mul_quat_0(var_121, var_195);
                            // xpos = xanchor - math.rot_vec_quat(jnt_pos[jnt_pos_id, jntadr], xquat)       <L 138>
                            var_197 = wp::address(var_jnt_pos, var_76, var_20);
                            var_199 = wp::load(var_197);
                            var_198 = rot_vec_quat_0(var_199, var_196);
                            var_200 = wp::sub(var_141, var_198);
                        }
                        var_201 = wp::where(var_183, var_200, var_120);
                        var_202 = wp::where(var_183, var_196, var_121);
                    }
                    var_203 = wp::where(var_169, var_181, var_201);
                    var_204 = wp::where(var_169, var_121, var_202);
                }
                var_205 = wp::where(var_144, var_167, var_203);
                var_206 = wp::where(var_144, var_163, var_204);
                // xanchor_out[worldid, jntadr] = xanchor                                         <L 140>
                wp::array_store(var_xanchor_out, var_0, var_20, var_141);
                // xaxis_out[worldid, jntadr] = xaxis                                             <L 141>
                wp::array_store(var_xaxis_out, var_0, var_20, var_142);
                // jntadr += 1                                                                    <L 142>
                var_208 = wp::add(var_20, var_207);
                wp::assign(var_20, var_208);
                wp::assign(var_28, var_128);
                wp::assign(var_33, var_125);
                wp::assign(var_120, var_205);
                wp::assign(var_121, var_206);
                goto start_for_2;
            end_for_2:;
            // xquat = wp.normalize(xquat)                                                        <L 144>
            var_209 = wp::normalize(var_121);
            // xpos_out[worldid, bodyid] = xpos                                                   <L 145>
            wp::array_store(var_xpos_out, var_0, var_14, var_120);
            // xquat_out[worldid, bodyid] = xquat                                                 <L 146>
            wp::array_store(var_xquat_out, var_0, var_14, var_209);
            goto start_for_0;
        end_for_0:;
    }
}



extern "C" __global__ void _flex_face_kinematics_9a1620ae_cuda_kernel_forward(
    wp::launch_bounds_t<2> dim,
    wp::array_t<wp::int32> var_flex_interp,
    wp::array_t<wp::vec_t<3, wp::int32>> var_flex_cellnum,
    wp::array_t<wp::vec_t<2, wp::int32>> var_flex_face_map,
    wp::array_t<wp::int32> var_flex_face,
    wp::array_t<wp::vec_t<3, wp::float32>> var_flexnode_xpos_in,
    wp::array_t<wp::vec_t<3, wp::float32>> var_face_xpos_out,
    wp::array_t<wp::quat_t<wp::float32>> var_face_quat_out)
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
        wp::vec_t<2, wp::int32>* var_2;
        wp::vec_t<2, wp::int32> var_3;
        wp::vec_t<2, wp::int32> var_4;
        const wp::int32 var_5 = 0;
        wp::int32 var_6;
        const wp::int32 var_7 = 1;
        wp::int32 var_8;
        wp::int32* var_9;
        wp::int32 var_10;
        wp::int32 var_11;
        wp::int32 var_12;
        wp::vec_t<3, wp::int32>* var_13;
        wp::vec_t<3, wp::int32> var_14;
        wp::vec_t<3, wp::int32> var_15;
        const wp::int32 var_16 = 0;
        wp::int32 var_17;
        const wp::int32 var_18 = 1;
        wp::int32 var_19;
        const wp::int32 var_20 = 2;
        wp::int32 var_21;
        const wp::float32 var_22 = 0.0;
        wp::vec_t<3, wp::float32> var_23;
        const wp::float32 var_24 = 0.0;
        wp::vec_t<3, wp::float32> var_25;
        const wp::int32 var_26 = 0;
        wp::int32* var_27;
        wp::int32 var_28;
        wp::int32 var_29;
        const wp::int32 var_30 = -1;
        bool var_31;
        wp::vec_t<3, wp::float32>* var_32;
        wp::vec_t<3, wp::float32> var_33;
        wp::vec_t<3, wp::float32> var_34;
        const wp::int32 var_35 = 1;
        wp::int32 var_36;
        wp::int32 var_37;
        const wp::int32 var_38 = 1;
        wp::int32 var_39;
        wp::int32 var_40;
        const wp::int32 var_41 = 2;
        bool var_42;
        const wp::int32 var_43 = 1;
        wp::int32 var_44;
        wp::float32 var_45;
        const wp::float32 var_46 = -1.0;
        const wp::float32 var_47 = 2.0;
        wp::float32 var_48;
        wp::float32 var_49;
        wp::float32 var_50;
        wp::float32 var_51;
        const wp::int32 var_52 = 2;
        bool var_53;
        const wp::int32 var_54 = 1;
        wp::int32 var_55;
        wp::float32 var_56;
        const wp::float32 var_57 = -1.0;
        const wp::float32 var_58 = 2.0;
        wp::float32 var_59;
        wp::float32 var_60;
        wp::float32 var_61;
        wp::float32 var_62;
        const wp::int32 var_63 = 2;
        bool var_64;
        const wp::int32 var_65 = 1;
        bool var_66;
        const wp::float32 var_67 = 1.0;
        const wp::float32 var_68 = 0.0;
        wp::float32 var_69;
        const wp::float32 var_70 = 0.5;
        wp::float32 var_71;
        const wp::int32 var_72 = 2;
        bool var_73;
        const wp::int32 var_74 = 1;
        bool var_75;
        const wp::float32 var_76 = 1.0;
        const wp::float32 var_77 = 0.0;
        wp::float32 var_78;
        const wp::float32 var_79 = 0.5;
        wp::float32 var_80;
        wp::float32 var_81;
        wp::float32 var_82;
        wp::vec_t<3, wp::float32> var_83;
        wp::vec_t<3, wp::float32> var_84;
        wp::vec_t<3, wp::float32> var_85;
        wp::vec_t<3, wp::float32> var_86;
        const wp::float32 var_87 = 0.0;
        wp::vec_t<3, wp::float32> var_88;
        wp::vec_t<3, wp::float32> var_89;
        wp::vec_t<3, wp::float32> var_90;
        const wp::int32 var_91 = 1;
        wp::int32* var_92;
        wp::int32 var_93;
        wp::int32 var_94;
        const wp::int32 var_95 = -1;
        bool var_96;
        wp::vec_t<3, wp::float32>* var_97;
        wp::vec_t<3, wp::float32> var_98;
        wp::vec_t<3, wp::float32> var_99;
        const wp::int32 var_100 = 1;
        wp::int32 var_101;
        wp::int32 var_102;
        const wp::int32 var_103 = 1;
        wp::int32 var_104;
        wp::int32 var_105;
        const wp::int32 var_106 = 2;
        bool var_107;
        const wp::int32 var_108 = 1;
        wp::int32 var_109;
        wp::float32 var_110;
        const wp::float32 var_111 = -1.0;
        const wp::float32 var_112 = 2.0;
        wp::float32 var_113;
        wp::float32 var_114;
        wp::float32 var_115;
        wp::float32 var_116;
        const wp::int32 var_117 = 2;
        bool var_118;
        const wp::int32 var_119 = 1;
        wp::int32 var_120;
        wp::float32 var_121;
        const wp::float32 var_122 = -1.0;
        const wp::float32 var_123 = 2.0;
        wp::float32 var_124;
        wp::float32 var_125;
        wp::float32 var_126;
        wp::float32 var_127;
        const wp::int32 var_128 = 2;
        bool var_129;
        const wp::int32 var_130 = 1;
        bool var_131;
        const wp::float32 var_132 = 1.0;
        const wp::float32 var_133 = 0.0;
        wp::float32 var_134;
        const wp::float32 var_135 = 0.5;
        wp::float32 var_136;
        const wp::int32 var_137 = 2;
        bool var_138;
        const wp::int32 var_139 = 1;
        bool var_140;
        const wp::float32 var_141 = 1.0;
        const wp::float32 var_142 = 0.0;
        wp::float32 var_143;
        const wp::float32 var_144 = 0.5;
        wp::float32 var_145;
        wp::float32 var_146;
        wp::float32 var_147;
        wp::vec_t<3, wp::float32> var_148;
        wp::vec_t<3, wp::float32> var_149;
        wp::vec_t<3, wp::float32> var_150;
        wp::vec_t<3, wp::float32> var_151;
        const wp::float32 var_152 = 0.0;
        wp::vec_t<3, wp::float32> var_153;
        wp::vec_t<3, wp::float32> var_154;
        wp::vec_t<3, wp::float32> var_155;
        wp::vec_t<3, wp::float32> var_156;
        wp::int32 var_157;
        wp::int32 var_158;
        wp::float32 var_159;
        wp::float32 var_160;
        wp::float32 var_161;
        wp::float32 var_162;
        wp::float32 var_163;
        wp::float32 var_164;
        const wp::int32 var_165 = 2;
        wp::int32* var_166;
        wp::int32 var_167;
        wp::int32 var_168;
        const wp::int32 var_169 = -1;
        bool var_170;
        wp::vec_t<3, wp::float32>* var_171;
        wp::vec_t<3, wp::float32> var_172;
        wp::vec_t<3, wp::float32> var_173;
        const wp::int32 var_174 = 1;
        wp::int32 var_175;
        wp::int32 var_176;
        const wp::int32 var_177 = 1;
        wp::int32 var_178;
        wp::int32 var_179;
        const wp::int32 var_180 = 2;
        bool var_181;
        const wp::int32 var_182 = 1;
        wp::int32 var_183;
        wp::float32 var_184;
        const wp::float32 var_185 = -1.0;
        const wp::float32 var_186 = 2.0;
        wp::float32 var_187;
        wp::float32 var_188;
        wp::float32 var_189;
        wp::float32 var_190;
        const wp::int32 var_191 = 2;
        bool var_192;
        const wp::int32 var_193 = 1;
        wp::int32 var_194;
        wp::float32 var_195;
        const wp::float32 var_196 = -1.0;
        const wp::float32 var_197 = 2.0;
        wp::float32 var_198;
        wp::float32 var_199;
        wp::float32 var_200;
        wp::float32 var_201;
        const wp::int32 var_202 = 2;
        bool var_203;
        const wp::int32 var_204 = 1;
        bool var_205;
        const wp::float32 var_206 = 1.0;
        const wp::float32 var_207 = 0.0;
        wp::float32 var_208;
        const wp::float32 var_209 = 0.5;
        wp::float32 var_210;
        const wp::int32 var_211 = 2;
        bool var_212;
        const wp::int32 var_213 = 1;
        bool var_214;
        const wp::float32 var_215 = 1.0;
        const wp::float32 var_216 = 0.0;
        wp::float32 var_217;
        const wp::float32 var_218 = 0.5;
        wp::float32 var_219;
        wp::float32 var_220;
        wp::float32 var_221;
        wp::vec_t<3, wp::float32> var_222;
        wp::vec_t<3, wp::float32> var_223;
        wp::vec_t<3, wp::float32> var_224;
        wp::vec_t<3, wp::float32> var_225;
        const wp::float32 var_226 = 0.0;
        wp::vec_t<3, wp::float32> var_227;
        wp::vec_t<3, wp::float32> var_228;
        wp::vec_t<3, wp::float32> var_229;
        wp::vec_t<3, wp::float32> var_230;
        wp::int32 var_231;
        wp::int32 var_232;
        wp::float32 var_233;
        wp::float32 var_234;
        wp::float32 var_235;
        wp::float32 var_236;
        wp::float32 var_237;
        wp::float32 var_238;
        const wp::int32 var_239 = 3;
        wp::int32* var_240;
        wp::int32 var_241;
        wp::int32 var_242;
        const wp::int32 var_243 = -1;
        bool var_244;
        wp::vec_t<3, wp::float32>* var_245;
        wp::vec_t<3, wp::float32> var_246;
        wp::vec_t<3, wp::float32> var_247;
        const wp::int32 var_248 = 1;
        wp::int32 var_249;
        wp::int32 var_250;
        const wp::int32 var_251 = 1;
        wp::int32 var_252;
        wp::int32 var_253;
        const wp::int32 var_254 = 2;
        bool var_255;
        const wp::int32 var_256 = 1;
        wp::int32 var_257;
        wp::float32 var_258;
        const wp::float32 var_259 = -1.0;
        const wp::float32 var_260 = 2.0;
        wp::float32 var_261;
        wp::float32 var_262;
        wp::float32 var_263;
        wp::float32 var_264;
        const wp::int32 var_265 = 2;
        bool var_266;
        const wp::int32 var_267 = 1;
        wp::int32 var_268;
        wp::float32 var_269;
        const wp::float32 var_270 = -1.0;
        const wp::float32 var_271 = 2.0;
        wp::float32 var_272;
        wp::float32 var_273;
        wp::float32 var_274;
        wp::float32 var_275;
        const wp::int32 var_276 = 2;
        bool var_277;
        const wp::int32 var_278 = 1;
        bool var_279;
        const wp::float32 var_280 = 1.0;
        const wp::float32 var_281 = 0.0;
        wp::float32 var_282;
        const wp::float32 var_283 = 0.5;
        wp::float32 var_284;
        const wp::int32 var_285 = 2;
        bool var_286;
        const wp::int32 var_287 = 1;
        bool var_288;
        const wp::float32 var_289 = 1.0;
        const wp::float32 var_290 = 0.0;
        wp::float32 var_291;
        const wp::float32 var_292 = 0.5;
        wp::float32 var_293;
        wp::float32 var_294;
        wp::float32 var_295;
        wp::vec_t<3, wp::float32> var_296;
        wp::vec_t<3, wp::float32> var_297;
        wp::vec_t<3, wp::float32> var_298;
        wp::vec_t<3, wp::float32> var_299;
        const wp::float32 var_300 = 0.0;
        wp::vec_t<3, wp::float32> var_301;
        wp::vec_t<3, wp::float32> var_302;
        wp::vec_t<3, wp::float32> var_303;
        wp::vec_t<3, wp::float32> var_304;
        wp::int32 var_305;
        wp::int32 var_306;
        wp::float32 var_307;
        wp::float32 var_308;
        wp::float32 var_309;
        wp::float32 var_310;
        wp::float32 var_311;
        wp::float32 var_312;
        const wp::int32 var_313 = 4;
        wp::int32* var_314;
        wp::int32 var_315;
        wp::int32 var_316;
        const wp::int32 var_317 = -1;
        bool var_318;
        wp::vec_t<3, wp::float32>* var_319;
        wp::vec_t<3, wp::float32> var_320;
        wp::vec_t<3, wp::float32> var_321;
        const wp::int32 var_322 = 1;
        wp::int32 var_323;
        wp::int32 var_324;
        const wp::int32 var_325 = 1;
        wp::int32 var_326;
        wp::int32 var_327;
        const wp::int32 var_328 = 2;
        bool var_329;
        const wp::int32 var_330 = 1;
        wp::int32 var_331;
        wp::float32 var_332;
        const wp::float32 var_333 = -1.0;
        const wp::float32 var_334 = 2.0;
        wp::float32 var_335;
        wp::float32 var_336;
        wp::float32 var_337;
        wp::float32 var_338;
        const wp::int32 var_339 = 2;
        bool var_340;
        const wp::int32 var_341 = 1;
        wp::int32 var_342;
        wp::float32 var_343;
        const wp::float32 var_344 = -1.0;
        const wp::float32 var_345 = 2.0;
        wp::float32 var_346;
        wp::float32 var_347;
        wp::float32 var_348;
        wp::float32 var_349;
        const wp::int32 var_350 = 2;
        bool var_351;
        const wp::int32 var_352 = 1;
        bool var_353;
        const wp::float32 var_354 = 1.0;
        const wp::float32 var_355 = 0.0;
        wp::float32 var_356;
        const wp::float32 var_357 = 0.5;
        wp::float32 var_358;
        const wp::int32 var_359 = 2;
        bool var_360;
        const wp::int32 var_361 = 1;
        bool var_362;
        const wp::float32 var_363 = 1.0;
        const wp::float32 var_364 = 0.0;
        wp::float32 var_365;
        const wp::float32 var_366 = 0.5;
        wp::float32 var_367;
        wp::float32 var_368;
        wp::float32 var_369;
        wp::vec_t<3, wp::float32> var_370;
        wp::vec_t<3, wp::float32> var_371;
        wp::vec_t<3, wp::float32> var_372;
        wp::vec_t<3, wp::float32> var_373;
        const wp::float32 var_374 = 0.0;
        wp::vec_t<3, wp::float32> var_375;
        wp::vec_t<3, wp::float32> var_376;
        wp::vec_t<3, wp::float32> var_377;
        wp::vec_t<3, wp::float32> var_378;
        wp::int32 var_379;
        wp::int32 var_380;
        wp::float32 var_381;
        wp::float32 var_382;
        wp::float32 var_383;
        wp::float32 var_384;
        wp::float32 var_385;
        wp::float32 var_386;
        const wp::int32 var_387 = 5;
        wp::int32* var_388;
        wp::int32 var_389;
        wp::int32 var_390;
        const wp::int32 var_391 = -1;
        bool var_392;
        wp::vec_t<3, wp::float32>* var_393;
        wp::vec_t<3, wp::float32> var_394;
        wp::vec_t<3, wp::float32> var_395;
        const wp::int32 var_396 = 1;
        wp::int32 var_397;
        wp::int32 var_398;
        const wp::int32 var_399 = 1;
        wp::int32 var_400;
        wp::int32 var_401;
        const wp::int32 var_402 = 2;
        bool var_403;
        const wp::int32 var_404 = 1;
        wp::int32 var_405;
        wp::float32 var_406;
        const wp::float32 var_407 = -1.0;
        const wp::float32 var_408 = 2.0;
        wp::float32 var_409;
        wp::float32 var_410;
        wp::float32 var_411;
        wp::float32 var_412;
        const wp::int32 var_413 = 2;
        bool var_414;
        const wp::int32 var_415 = 1;
        wp::int32 var_416;
        wp::float32 var_417;
        const wp::float32 var_418 = -1.0;
        const wp::float32 var_419 = 2.0;
        wp::float32 var_420;
        wp::float32 var_421;
        wp::float32 var_422;
        wp::float32 var_423;
        const wp::int32 var_424 = 2;
        bool var_425;
        const wp::int32 var_426 = 1;
        bool var_427;
        const wp::float32 var_428 = 1.0;
        const wp::float32 var_429 = 0.0;
        wp::float32 var_430;
        const wp::float32 var_431 = 0.5;
        wp::float32 var_432;
        const wp::int32 var_433 = 2;
        bool var_434;
        const wp::int32 var_435 = 1;
        bool var_436;
        const wp::float32 var_437 = 1.0;
        const wp::float32 var_438 = 0.0;
        wp::float32 var_439;
        const wp::float32 var_440 = 0.5;
        wp::float32 var_441;
        wp::float32 var_442;
        wp::float32 var_443;
        wp::vec_t<3, wp::float32> var_444;
        wp::vec_t<3, wp::float32> var_445;
        wp::vec_t<3, wp::float32> var_446;
        wp::vec_t<3, wp::float32> var_447;
        const wp::float32 var_448 = 0.0;
        wp::vec_t<3, wp::float32> var_449;
        wp::vec_t<3, wp::float32> var_450;
        wp::vec_t<3, wp::float32> var_451;
        wp::vec_t<3, wp::float32> var_452;
        wp::int32 var_453;
        wp::int32 var_454;
        wp::float32 var_455;
        wp::float32 var_456;
        wp::float32 var_457;
        wp::float32 var_458;
        wp::float32 var_459;
        wp::float32 var_460;
        const wp::int32 var_461 = 6;
        wp::int32* var_462;
        wp::int32 var_463;
        wp::int32 var_464;
        const wp::int32 var_465 = -1;
        bool var_466;
        wp::vec_t<3, wp::float32>* var_467;
        wp::vec_t<3, wp::float32> var_468;
        wp::vec_t<3, wp::float32> var_469;
        const wp::int32 var_470 = 1;
        wp::int32 var_471;
        wp::int32 var_472;
        const wp::int32 var_473 = 1;
        wp::int32 var_474;
        wp::int32 var_475;
        const wp::int32 var_476 = 2;
        bool var_477;
        const wp::int32 var_478 = 1;
        wp::int32 var_479;
        wp::float32 var_480;
        const wp::float32 var_481 = -1.0;
        const wp::float32 var_482 = 2.0;
        wp::float32 var_483;
        wp::float32 var_484;
        wp::float32 var_485;
        wp::float32 var_486;
        const wp::int32 var_487 = 2;
        bool var_488;
        const wp::int32 var_489 = 1;
        wp::int32 var_490;
        wp::float32 var_491;
        const wp::float32 var_492 = -1.0;
        const wp::float32 var_493 = 2.0;
        wp::float32 var_494;
        wp::float32 var_495;
        wp::float32 var_496;
        wp::float32 var_497;
        const wp::int32 var_498 = 2;
        bool var_499;
        const wp::int32 var_500 = 1;
        bool var_501;
        const wp::float32 var_502 = 1.0;
        const wp::float32 var_503 = 0.0;
        wp::float32 var_504;
        const wp::float32 var_505 = 0.5;
        wp::float32 var_506;
        const wp::int32 var_507 = 2;
        bool var_508;
        const wp::int32 var_509 = 1;
        bool var_510;
        const wp::float32 var_511 = 1.0;
        const wp::float32 var_512 = 0.0;
        wp::float32 var_513;
        const wp::float32 var_514 = 0.5;
        wp::float32 var_515;
        wp::float32 var_516;
        wp::float32 var_517;
        wp::vec_t<3, wp::float32> var_518;
        wp::vec_t<3, wp::float32> var_519;
        wp::vec_t<3, wp::float32> var_520;
        wp::vec_t<3, wp::float32> var_521;
        const wp::float32 var_522 = 0.0;
        wp::vec_t<3, wp::float32> var_523;
        wp::vec_t<3, wp::float32> var_524;
        wp::vec_t<3, wp::float32> var_525;
        wp::vec_t<3, wp::float32> var_526;
        wp::int32 var_527;
        wp::int32 var_528;
        wp::float32 var_529;
        wp::float32 var_530;
        wp::float32 var_531;
        wp::float32 var_532;
        wp::float32 var_533;
        wp::float32 var_534;
        const wp::int32 var_535 = 7;
        wp::int32* var_536;
        wp::int32 var_537;
        wp::int32 var_538;
        const wp::int32 var_539 = -1;
        bool var_540;
        wp::vec_t<3, wp::float32>* var_541;
        wp::vec_t<3, wp::float32> var_542;
        wp::vec_t<3, wp::float32> var_543;
        const wp::int32 var_544 = 1;
        wp::int32 var_545;
        wp::int32 var_546;
        const wp::int32 var_547 = 1;
        wp::int32 var_548;
        wp::int32 var_549;
        const wp::int32 var_550 = 2;
        bool var_551;
        const wp::int32 var_552 = 1;
        wp::int32 var_553;
        wp::float32 var_554;
        const wp::float32 var_555 = -1.0;
        const wp::float32 var_556 = 2.0;
        wp::float32 var_557;
        wp::float32 var_558;
        wp::float32 var_559;
        wp::float32 var_560;
        const wp::int32 var_561 = 2;
        bool var_562;
        const wp::int32 var_563 = 1;
        wp::int32 var_564;
        wp::float32 var_565;
        const wp::float32 var_566 = -1.0;
        const wp::float32 var_567 = 2.0;
        wp::float32 var_568;
        wp::float32 var_569;
        wp::float32 var_570;
        wp::float32 var_571;
        const wp::int32 var_572 = 2;
        bool var_573;
        const wp::int32 var_574 = 1;
        bool var_575;
        const wp::float32 var_576 = 1.0;
        const wp::float32 var_577 = 0.0;
        wp::float32 var_578;
        const wp::float32 var_579 = 0.5;
        wp::float32 var_580;
        const wp::int32 var_581 = 2;
        bool var_582;
        const wp::int32 var_583 = 1;
        bool var_584;
        const wp::float32 var_585 = 1.0;
        const wp::float32 var_586 = 0.0;
        wp::float32 var_587;
        const wp::float32 var_588 = 0.5;
        wp::float32 var_589;
        wp::float32 var_590;
        wp::float32 var_591;
        wp::vec_t<3, wp::float32> var_592;
        wp::vec_t<3, wp::float32> var_593;
        wp::vec_t<3, wp::float32> var_594;
        wp::vec_t<3, wp::float32> var_595;
        const wp::float32 var_596 = 0.0;
        wp::vec_t<3, wp::float32> var_597;
        wp::vec_t<3, wp::float32> var_598;
        wp::vec_t<3, wp::float32> var_599;
        wp::vec_t<3, wp::float32> var_600;
        wp::int32 var_601;
        wp::int32 var_602;
        wp::float32 var_603;
        wp::float32 var_604;
        wp::float32 var_605;
        wp::float32 var_606;
        wp::float32 var_607;
        wp::float32 var_608;
        const wp::int32 var_609 = 8;
        wp::int32* var_610;
        wp::int32 var_611;
        wp::int32 var_612;
        const wp::int32 var_613 = -1;
        bool var_614;
        wp::vec_t<3, wp::float32>* var_615;
        wp::vec_t<3, wp::float32> var_616;
        wp::vec_t<3, wp::float32> var_617;
        const wp::int32 var_618 = 1;
        wp::int32 var_619;
        wp::int32 var_620;
        const wp::int32 var_621 = 1;
        wp::int32 var_622;
        wp::int32 var_623;
        const wp::int32 var_624 = 2;
        bool var_625;
        const wp::int32 var_626 = 1;
        wp::int32 var_627;
        wp::float32 var_628;
        const wp::float32 var_629 = -1.0;
        const wp::float32 var_630 = 2.0;
        wp::float32 var_631;
        wp::float32 var_632;
        wp::float32 var_633;
        wp::float32 var_634;
        const wp::int32 var_635 = 2;
        bool var_636;
        const wp::int32 var_637 = 1;
        wp::int32 var_638;
        wp::float32 var_639;
        const wp::float32 var_640 = -1.0;
        const wp::float32 var_641 = 2.0;
        wp::float32 var_642;
        wp::float32 var_643;
        wp::float32 var_644;
        wp::float32 var_645;
        const wp::int32 var_646 = 2;
        bool var_647;
        const wp::int32 var_648 = 1;
        bool var_649;
        const wp::float32 var_650 = 1.0;
        const wp::float32 var_651 = 0.0;
        wp::float32 var_652;
        const wp::float32 var_653 = 0.5;
        wp::float32 var_654;
        const wp::int32 var_655 = 2;
        bool var_656;
        const wp::int32 var_657 = 1;
        bool var_658;
        const wp::float32 var_659 = 1.0;
        const wp::float32 var_660 = 0.0;
        wp::float32 var_661;
        const wp::float32 var_662 = 0.5;
        wp::float32 var_663;
        wp::float32 var_664;
        wp::float32 var_665;
        wp::vec_t<3, wp::float32> var_666;
        wp::vec_t<3, wp::float32> var_667;
        wp::vec_t<3, wp::float32> var_668;
        wp::vec_t<3, wp::float32> var_669;
        const wp::float32 var_670 = 0.0;
        wp::vec_t<3, wp::float32> var_671;
        wp::vec_t<3, wp::float32> var_672;
        wp::vec_t<3, wp::float32> var_673;
        wp::vec_t<3, wp::float32> var_674;
        wp::int32 var_675;
        wp::int32 var_676;
        wp::float32 var_677;
        wp::float32 var_678;
        wp::float32 var_679;
        wp::float32 var_680;
        wp::float32 var_681;
        wp::float32 var_682;
        wp::vec_t<3, wp::float32> var_683;
        wp::int32 var_684;
        wp::int32 var_685;
        wp::int32 var_686;
        wp::int32 var_687;
        wp::int32 var_688;
        wp::int32 var_689;
        const wp::float32 var_690 = 0.0;
        wp::mat_t<3, 3, wp::float32> var_691;
        const wp::int32 var_692 = 0;
        bool var_693;
        const wp::int32 var_694 = 0;
        wp::float32 var_695;
        const wp::int32 var_696 = 0;
        wp::float32 var_697;
        const wp::int32 var_698 = 0;
        wp::float32 var_699;
        const wp::int32 var_700 = 1;
        wp::float32 var_701;
        const wp::int32 var_702 = 1;
        wp::float32 var_703;
        const wp::int32 var_704 = 1;
        wp::float32 var_705;
        const wp::int32 var_706 = 2;
        wp::float32 var_707;
        const wp::int32 var_708 = 2;
        wp::float32 var_709;
        const wp::int32 var_710 = 2;
        wp::float32 var_711;
        wp::mat_t<3, 3, wp::float32> var_712;
        const wp::int32 var_713 = 1;
        bool var_714;
        const wp::int32 var_715 = 0;
        wp::float32 var_716;
        const wp::int32 var_717 = 0;
        wp::float32 var_718;
        const wp::int32 var_719 = 0;
        wp::float32 var_720;
        const wp::int32 var_721 = 1;
        wp::float32 var_722;
        const wp::int32 var_723 = 1;
        wp::float32 var_724;
        const wp::int32 var_725 = 1;
        wp::float32 var_726;
        const wp::int32 var_727 = 2;
        wp::float32 var_728;
        const wp::int32 var_729 = 2;
        wp::float32 var_730;
        const wp::int32 var_731 = 2;
        wp::float32 var_732;
        wp::mat_t<3, 3, wp::float32> var_733;
        const wp::int32 var_734 = 0;
        wp::float32 var_735;
        const wp::int32 var_736 = 0;
        wp::float32 var_737;
        const wp::int32 var_738 = 0;
        wp::float32 var_739;
        const wp::int32 var_740 = 1;
        wp::float32 var_741;
        const wp::int32 var_742 = 1;
        wp::float32 var_743;
        const wp::int32 var_744 = 1;
        wp::float32 var_745;
        const wp::int32 var_746 = 2;
        wp::float32 var_747;
        const wp::int32 var_748 = 2;
        wp::float32 var_749;
        const wp::int32 var_750 = 2;
        wp::float32 var_751;
        wp::mat_t<3, 3, wp::float32> var_752;
        wp::mat_t<3, 3, wp::float32> var_753;
        wp::mat_t<3, 3, wp::float32> var_754;
        wp::quat_t<wp::float32> var_755;
        //---------
        // forward
        // def _flex_face_kinematics(                                                             <L 479>
        // worldid, face_id = wp.tid()                                                            <L 491>
        builtin_tid2d(var_0, var_1);
        // mapping = flex_face_map[face_id]                                                       <L 493>
        var_2 = wp::address(var_flex_face_map, var_1);
        var_4 = wp::load(var_2);
        var_3 = wp::copy(var_4);
        // f = mapping[0]                                                                         <L 494>
        var_6 = wp::extract(var_3, var_5);
        // face_elem_idx = mapping[1]                                                             <L 495>
        var_8 = wp::extract(var_3, var_7);
        // order = flex_interp[f]                                                                 <L 497>
        var_9 = wp::address(var_flex_interp, var_6);
        var_11 = wp::load(var_9);
        var_10 = wp::copy(var_11);
        // order_abs = -order                                                                     <L 498>
        var_12 = wp::neg(var_10);
        // cellnum = flex_cellnum[f]                                                              <L 499>
        var_13 = wp::address(var_flex_cellnum, var_6);
        var_15 = wp::load(var_13);
        var_14 = wp::copy(var_15);
        // cx = cellnum[0]                                                                        <L 500>
        var_17 = wp::extract(var_14, var_16);
        // cy = cellnum[1]                                                                        <L 501>
        var_19 = wp::extract(var_14, var_18);
        // cz = cellnum[2]                                                                        <L 502>
        var_21 = wp::extract(var_14, var_20);
        // t1 = wp.vec3(0.0)                                                                      <L 503>
        var_23 = wp::vec_t<3, wp::float32>(var_22);
        // t2 = wp.vec3(0.0)                                                                      <L 504>
        var_25 = wp::vec_t<3, wp::float32>(var_24);
        // for local_idx in range(9):                                                             <L 505>
        // gidx = flex_face[face_id, local_idx]                                                   <L 506>
        var_27 = wp::address(var_flex_face, var_1, var_26);
        var_29 = wp::load(var_27);
        var_28 = wp::copy(var_29);
        // if gidx != -1:                                                                         <L 507>
        var_31 = (var_28 != var_30);
        if (var_31) {
            // node_pos = flexnode_xpos_in[worldid, gidx]                                         <L 508>
            var_32 = wp::address(var_flexnode_xpos_in, var_0, var_28);
            var_34 = wp::load(var_32);
            var_33 = wp::copy(var_34);
            // face_xpos_out[worldid, face_id, local_idx] = node_pos                              <L 510>
            wp::array_store(var_face_xpos_out, var_0, var_1, var_26, var_33);
            // l0 = local_idx // (order_abs + 1)                                                  <L 512>
            var_36 = wp::add(var_12, var_35);
            var_37 = wp::floordiv(var_26, var_36);
            // l1 = local_idx % (order_abs + 1)                                                   <L 513>
            var_39 = wp::add(var_12, var_38);
            var_40 = wp::mod(var_26, var_39);
            // dphi0 = float(l0 - 1) if order_abs == 2 else (-1.0 + 2.0 * float(l0))              <L 515>
            var_42 = (var_12 == var_41);
            if (var_42) {
                var_44 = wp::sub(var_37, var_43);
                var_45 = wp::float(var_44);
            }
            if (!var_42) {
                var_48 = wp::float(var_37);
                var_49 = wp::mul(var_47, var_48);
                var_50 = wp::add(var_46, var_49);
            }
            var_51 = wp::where(var_42, var_45, var_50);
            // dphi1 = float(l1 - 1) if order_abs == 2 else (-1.0 + 2.0 * float(l1))              <L 516>
            var_53 = (var_12 == var_52);
            if (var_53) {
                var_55 = wp::sub(var_40, var_54);
                var_56 = wp::float(var_55);
            }
            if (!var_53) {
                var_59 = wp::float(var_40);
                var_60 = wp::mul(var_58, var_59);
                var_61 = wp::add(var_57, var_60);
            }
            var_62 = wp::where(var_53, var_56, var_61);
            // phi0 = wp.where(l0 == 1, 1.0, 0.0) if order_abs == 2 else 0.5                      <L 517>
            var_64 = (var_12 == var_63);
            if (var_64) {
                var_66 = (var_37 == var_65);
                var_69 = wp::where(var_66, var_67, var_68);
            }
            if (!var_64) {
            }
            var_71 = wp::where(var_64, var_69, var_70);
            // phi1 = wp.where(l1 == 1, 1.0, 0.0) if order_abs == 2 else 0.5                      <L 518>
            var_73 = (var_12 == var_72);
            if (var_73) {
                var_75 = (var_40 == var_74);
                var_78 = wp::where(var_75, var_76, var_77);
            }
            if (!var_73) {
            }
            var_80 = wp::where(var_73, var_78, var_79);
            // grad0 = dphi0 * phi1                                                               <L 520>
            var_81 = wp::mul(var_51, var_80);
            // grad1 = phi0 * dphi1                                                               <L 521>
            var_82 = wp::mul(var_71, var_62);
            // t1 += node_pos * grad0                                                             <L 523>
            var_83 = wp::mul(var_33, var_81);
            var_84 = wp::add(var_23, var_83);
            // t2 += node_pos * grad1                                                             <L 524>
            var_85 = wp::mul(var_33, var_82);
            var_86 = wp::add(var_25, var_85);
        }
        if (!var_31) {
            // face_xpos_out[worldid, face_id, local_idx] = wp.vec3(0.0)                          <L 526>
            var_88 = wp::vec_t<3, wp::float32>(var_87);
            wp::array_store(var_face_xpos_out, var_0, var_1, var_26, var_88);
        }
        var_89 = wp::where(var_31, var_84, var_23);
        var_90 = wp::where(var_31, var_86, var_25);
        // gidx = flex_face[face_id, local_idx]                                                   <L 506>
        var_92 = wp::address(var_flex_face, var_1, var_91);
        var_94 = wp::load(var_92);
        var_93 = wp::copy(var_94);
        // if gidx != -1:                                                                         <L 507>
        var_96 = (var_93 != var_95);
        if (var_96) {
            // node_pos = flexnode_xpos_in[worldid, gidx]                                         <L 508>
            var_97 = wp::address(var_flexnode_xpos_in, var_0, var_93);
            var_99 = wp::load(var_97);
            var_98 = wp::copy(var_99);
            // face_xpos_out[worldid, face_id, local_idx] = node_pos                              <L 510>
            wp::array_store(var_face_xpos_out, var_0, var_1, var_91, var_98);
            // l0 = local_idx // (order_abs + 1)                                                  <L 512>
            var_101 = wp::add(var_12, var_100);
            var_102 = wp::floordiv(var_91, var_101);
            // l1 = local_idx % (order_abs + 1)                                                   <L 513>
            var_104 = wp::add(var_12, var_103);
            var_105 = wp::mod(var_91, var_104);
            // dphi0 = float(l0 - 1) if order_abs == 2 else (-1.0 + 2.0 * float(l0))              <L 515>
            var_107 = (var_12 == var_106);
            if (var_107) {
                var_109 = wp::sub(var_102, var_108);
                var_110 = wp::float(var_109);
            }
            if (!var_107) {
                var_113 = wp::float(var_102);
                var_114 = wp::mul(var_112, var_113);
                var_115 = wp::add(var_111, var_114);
            }
            var_116 = wp::where(var_107, var_110, var_115);
            // dphi1 = float(l1 - 1) if order_abs == 2 else (-1.0 + 2.0 * float(l1))              <L 516>
            var_118 = (var_12 == var_117);
            if (var_118) {
                var_120 = wp::sub(var_105, var_119);
                var_121 = wp::float(var_120);
            }
            if (!var_118) {
                var_124 = wp::float(var_105);
                var_125 = wp::mul(var_123, var_124);
                var_126 = wp::add(var_122, var_125);
            }
            var_127 = wp::where(var_118, var_121, var_126);
            // phi0 = wp.where(l0 == 1, 1.0, 0.0) if order_abs == 2 else 0.5                      <L 517>
            var_129 = (var_12 == var_128);
            if (var_129) {
                var_131 = (var_102 == var_130);
                var_134 = wp::where(var_131, var_132, var_133);
            }
            if (!var_129) {
            }
            var_136 = wp::where(var_129, var_134, var_135);
            // phi1 = wp.where(l1 == 1, 1.0, 0.0) if order_abs == 2 else 0.5                      <L 518>
            var_138 = (var_12 == var_137);
            if (var_138) {
                var_140 = (var_105 == var_139);
                var_143 = wp::where(var_140, var_141, var_142);
            }
            if (!var_138) {
            }
            var_145 = wp::where(var_138, var_143, var_144);
            // grad0 = dphi0 * phi1                                                               <L 520>
            var_146 = wp::mul(var_116, var_145);
            // grad1 = phi0 * dphi1                                                               <L 521>
            var_147 = wp::mul(var_136, var_127);
            // t1 += node_pos * grad0                                                             <L 523>
            var_148 = wp::mul(var_98, var_146);
            var_149 = wp::add(var_89, var_148);
            // t2 += node_pos * grad1                                                             <L 524>
            var_150 = wp::mul(var_98, var_147);
            var_151 = wp::add(var_90, var_150);
        }
        if (!var_96) {
            // face_xpos_out[worldid, face_id, local_idx] = wp.vec3(0.0)                          <L 526>
            var_153 = wp::vec_t<3, wp::float32>(var_152);
            wp::array_store(var_face_xpos_out, var_0, var_1, var_91, var_153);
        }
        var_154 = wp::where(var_96, var_149, var_89);
        var_155 = wp::where(var_96, var_151, var_90);
        var_156 = wp::where(var_96, var_98, var_33);
        var_157 = wp::where(var_96, var_102, var_37);
        var_158 = wp::where(var_96, var_105, var_40);
        var_159 = wp::where(var_96, var_116, var_51);
        var_160 = wp::where(var_96, var_127, var_62);
        var_161 = wp::where(var_96, var_136, var_71);
        var_162 = wp::where(var_96, var_145, var_80);
        var_163 = wp::where(var_96, var_146, var_81);
        var_164 = wp::where(var_96, var_147, var_82);
        // gidx = flex_face[face_id, local_idx]                                                   <L 506>
        var_166 = wp::address(var_flex_face, var_1, var_165);
        var_168 = wp::load(var_166);
        var_167 = wp::copy(var_168);
        // if gidx != -1:                                                                         <L 507>
        var_170 = (var_167 != var_169);
        if (var_170) {
            // node_pos = flexnode_xpos_in[worldid, gidx]                                         <L 508>
            var_171 = wp::address(var_flexnode_xpos_in, var_0, var_167);
            var_173 = wp::load(var_171);
            var_172 = wp::copy(var_173);
            // face_xpos_out[worldid, face_id, local_idx] = node_pos                              <L 510>
            wp::array_store(var_face_xpos_out, var_0, var_1, var_165, var_172);
            // l0 = local_idx // (order_abs + 1)                                                  <L 512>
            var_175 = wp::add(var_12, var_174);
            var_176 = wp::floordiv(var_165, var_175);
            // l1 = local_idx % (order_abs + 1)                                                   <L 513>
            var_178 = wp::add(var_12, var_177);
            var_179 = wp::mod(var_165, var_178);
            // dphi0 = float(l0 - 1) if order_abs == 2 else (-1.0 + 2.0 * float(l0))              <L 515>
            var_181 = (var_12 == var_180);
            if (var_181) {
                var_183 = wp::sub(var_176, var_182);
                var_184 = wp::float(var_183);
            }
            if (!var_181) {
                var_187 = wp::float(var_176);
                var_188 = wp::mul(var_186, var_187);
                var_189 = wp::add(var_185, var_188);
            }
            var_190 = wp::where(var_181, var_184, var_189);
            // dphi1 = float(l1 - 1) if order_abs == 2 else (-1.0 + 2.0 * float(l1))              <L 516>
            var_192 = (var_12 == var_191);
            if (var_192) {
                var_194 = wp::sub(var_179, var_193);
                var_195 = wp::float(var_194);
            }
            if (!var_192) {
                var_198 = wp::float(var_179);
                var_199 = wp::mul(var_197, var_198);
                var_200 = wp::add(var_196, var_199);
            }
            var_201 = wp::where(var_192, var_195, var_200);
            // phi0 = wp.where(l0 == 1, 1.0, 0.0) if order_abs == 2 else 0.5                      <L 517>
            var_203 = (var_12 == var_202);
            if (var_203) {
                var_205 = (var_176 == var_204);
                var_208 = wp::where(var_205, var_206, var_207);
            }
            if (!var_203) {
            }
            var_210 = wp::where(var_203, var_208, var_209);
            // phi1 = wp.where(l1 == 1, 1.0, 0.0) if order_abs == 2 else 0.5                      <L 518>
            var_212 = (var_12 == var_211);
            if (var_212) {
                var_214 = (var_179 == var_213);
                var_217 = wp::where(var_214, var_215, var_216);
            }
            if (!var_212) {
            }
            var_219 = wp::where(var_212, var_217, var_218);
            // grad0 = dphi0 * phi1                                                               <L 520>
            var_220 = wp::mul(var_190, var_219);
            // grad1 = phi0 * dphi1                                                               <L 521>
            var_221 = wp::mul(var_210, var_201);
            // t1 += node_pos * grad0                                                             <L 523>
            var_222 = wp::mul(var_172, var_220);
            var_223 = wp::add(var_154, var_222);
            // t2 += node_pos * grad1                                                             <L 524>
            var_224 = wp::mul(var_172, var_221);
            var_225 = wp::add(var_155, var_224);
        }
        if (!var_170) {
            // face_xpos_out[worldid, face_id, local_idx] = wp.vec3(0.0)                          <L 526>
            var_227 = wp::vec_t<3, wp::float32>(var_226);
            wp::array_store(var_face_xpos_out, var_0, var_1, var_165, var_227);
        }
        var_228 = wp::where(var_170, var_223, var_154);
        var_229 = wp::where(var_170, var_225, var_155);
        var_230 = wp::where(var_170, var_172, var_156);
        var_231 = wp::where(var_170, var_176, var_157);
        var_232 = wp::where(var_170, var_179, var_158);
        var_233 = wp::where(var_170, var_190, var_159);
        var_234 = wp::where(var_170, var_201, var_160);
        var_235 = wp::where(var_170, var_210, var_161);
        var_236 = wp::where(var_170, var_219, var_162);
        var_237 = wp::where(var_170, var_220, var_163);
        var_238 = wp::where(var_170, var_221, var_164);
        // gidx = flex_face[face_id, local_idx]                                                   <L 506>
        var_240 = wp::address(var_flex_face, var_1, var_239);
        var_242 = wp::load(var_240);
        var_241 = wp::copy(var_242);
        // if gidx != -1:                                                                         <L 507>
        var_244 = (var_241 != var_243);
        if (var_244) {
            // node_pos = flexnode_xpos_in[worldid, gidx]                                         <L 508>
            var_245 = wp::address(var_flexnode_xpos_in, var_0, var_241);
            var_247 = wp::load(var_245);
            var_246 = wp::copy(var_247);
            // face_xpos_out[worldid, face_id, local_idx] = node_pos                              <L 510>
            wp::array_store(var_face_xpos_out, var_0, var_1, var_239, var_246);
            // l0 = local_idx // (order_abs + 1)                                                  <L 512>
            var_249 = wp::add(var_12, var_248);
            var_250 = wp::floordiv(var_239, var_249);
            // l1 = local_idx % (order_abs + 1)                                                   <L 513>
            var_252 = wp::add(var_12, var_251);
            var_253 = wp::mod(var_239, var_252);
            // dphi0 = float(l0 - 1) if order_abs == 2 else (-1.0 + 2.0 * float(l0))              <L 515>
            var_255 = (var_12 == var_254);
            if (var_255) {
                var_257 = wp::sub(var_250, var_256);
                var_258 = wp::float(var_257);
            }
            if (!var_255) {
                var_261 = wp::float(var_250);
                var_262 = wp::mul(var_260, var_261);
                var_263 = wp::add(var_259, var_262);
            }
            var_264 = wp::where(var_255, var_258, var_263);
            // dphi1 = float(l1 - 1) if order_abs == 2 else (-1.0 + 2.0 * float(l1))              <L 516>
            var_266 = (var_12 == var_265);
            if (var_266) {
                var_268 = wp::sub(var_253, var_267);
                var_269 = wp::float(var_268);
            }
            if (!var_266) {
                var_272 = wp::float(var_253);
                var_273 = wp::mul(var_271, var_272);
                var_274 = wp::add(var_270, var_273);
            }
            var_275 = wp::where(var_266, var_269, var_274);
            // phi0 = wp.where(l0 == 1, 1.0, 0.0) if order_abs == 2 else 0.5                      <L 517>
            var_277 = (var_12 == var_276);
            if (var_277) {
                var_279 = (var_250 == var_278);
                var_282 = wp::where(var_279, var_280, var_281);
            }
            if (!var_277) {
            }
            var_284 = wp::where(var_277, var_282, var_283);
            // phi1 = wp.where(l1 == 1, 1.0, 0.0) if order_abs == 2 else 0.5                      <L 518>
            var_286 = (var_12 == var_285);
            if (var_286) {
                var_288 = (var_253 == var_287);
                var_291 = wp::where(var_288, var_289, var_290);
            }
            if (!var_286) {
            }
            var_293 = wp::where(var_286, var_291, var_292);
            // grad0 = dphi0 * phi1                                                               <L 520>
            var_294 = wp::mul(var_264, var_293);
            // grad1 = phi0 * dphi1                                                               <L 521>
            var_295 = wp::mul(var_284, var_275);
            // t1 += node_pos * grad0                                                             <L 523>
            var_296 = wp::mul(var_246, var_294);
            var_297 = wp::add(var_228, var_296);
            // t2 += node_pos * grad1                                                             <L 524>
            var_298 = wp::mul(var_246, var_295);
            var_299 = wp::add(var_229, var_298);
        }
        if (!var_244) {
            // face_xpos_out[worldid, face_id, local_idx] = wp.vec3(0.0)                          <L 526>
            var_301 = wp::vec_t<3, wp::float32>(var_300);
            wp::array_store(var_face_xpos_out, var_0, var_1, var_239, var_301);
        }
        var_302 = wp::where(var_244, var_297, var_228);
        var_303 = wp::where(var_244, var_299, var_229);
        var_304 = wp::where(var_244, var_246, var_230);
        var_305 = wp::where(var_244, var_250, var_231);
        var_306 = wp::where(var_244, var_253, var_232);
        var_307 = wp::where(var_244, var_264, var_233);
        var_308 = wp::where(var_244, var_275, var_234);
        var_309 = wp::where(var_244, var_284, var_235);
        var_310 = wp::where(var_244, var_293, var_236);
        var_311 = wp::where(var_244, var_294, var_237);
        var_312 = wp::where(var_244, var_295, var_238);
        // gidx = flex_face[face_id, local_idx]                                                   <L 506>
        var_314 = wp::address(var_flex_face, var_1, var_313);
        var_316 = wp::load(var_314);
        var_315 = wp::copy(var_316);
        // if gidx != -1:                                                                         <L 507>
        var_318 = (var_315 != var_317);
        if (var_318) {
            // node_pos = flexnode_xpos_in[worldid, gidx]                                         <L 508>
            var_319 = wp::address(var_flexnode_xpos_in, var_0, var_315);
            var_321 = wp::load(var_319);
            var_320 = wp::copy(var_321);
            // face_xpos_out[worldid, face_id, local_idx] = node_pos                              <L 510>
            wp::array_store(var_face_xpos_out, var_0, var_1, var_313, var_320);
            // l0 = local_idx // (order_abs + 1)                                                  <L 512>
            var_323 = wp::add(var_12, var_322);
            var_324 = wp::floordiv(var_313, var_323);
            // l1 = local_idx % (order_abs + 1)                                                   <L 513>
            var_326 = wp::add(var_12, var_325);
            var_327 = wp::mod(var_313, var_326);
            // dphi0 = float(l0 - 1) if order_abs == 2 else (-1.0 + 2.0 * float(l0))              <L 515>
            var_329 = (var_12 == var_328);
            if (var_329) {
                var_331 = wp::sub(var_324, var_330);
                var_332 = wp::float(var_331);
            }
            if (!var_329) {
                var_335 = wp::float(var_324);
                var_336 = wp::mul(var_334, var_335);
                var_337 = wp::add(var_333, var_336);
            }
            var_338 = wp::where(var_329, var_332, var_337);
            // dphi1 = float(l1 - 1) if order_abs == 2 else (-1.0 + 2.0 * float(l1))              <L 516>
            var_340 = (var_12 == var_339);
            if (var_340) {
                var_342 = wp::sub(var_327, var_341);
                var_343 = wp::float(var_342);
            }
            if (!var_340) {
                var_346 = wp::float(var_327);
                var_347 = wp::mul(var_345, var_346);
                var_348 = wp::add(var_344, var_347);
            }
            var_349 = wp::where(var_340, var_343, var_348);
            // phi0 = wp.where(l0 == 1, 1.0, 0.0) if order_abs == 2 else 0.5                      <L 517>
            var_351 = (var_12 == var_350);
            if (var_351) {
                var_353 = (var_324 == var_352);
                var_356 = wp::where(var_353, var_354, var_355);
            }
            if (!var_351) {
            }
            var_358 = wp::where(var_351, var_356, var_357);
            // phi1 = wp.where(l1 == 1, 1.0, 0.0) if order_abs == 2 else 0.5                      <L 518>
            var_360 = (var_12 == var_359);
            if (var_360) {
                var_362 = (var_327 == var_361);
                var_365 = wp::where(var_362, var_363, var_364);
            }
            if (!var_360) {
            }
            var_367 = wp::where(var_360, var_365, var_366);
            // grad0 = dphi0 * phi1                                                               <L 520>
            var_368 = wp::mul(var_338, var_367);
            // grad1 = phi0 * dphi1                                                               <L 521>
            var_369 = wp::mul(var_358, var_349);
            // t1 += node_pos * grad0                                                             <L 523>
            var_370 = wp::mul(var_320, var_368);
            var_371 = wp::add(var_302, var_370);
            // t2 += node_pos * grad1                                                             <L 524>
            var_372 = wp::mul(var_320, var_369);
            var_373 = wp::add(var_303, var_372);
        }
        if (!var_318) {
            // face_xpos_out[worldid, face_id, local_idx] = wp.vec3(0.0)                          <L 526>
            var_375 = wp::vec_t<3, wp::float32>(var_374);
            wp::array_store(var_face_xpos_out, var_0, var_1, var_313, var_375);
        }
        var_376 = wp::where(var_318, var_371, var_302);
        var_377 = wp::where(var_318, var_373, var_303);
        var_378 = wp::where(var_318, var_320, var_304);
        var_379 = wp::where(var_318, var_324, var_305);
        var_380 = wp::where(var_318, var_327, var_306);
        var_381 = wp::where(var_318, var_338, var_307);
        var_382 = wp::where(var_318, var_349, var_308);
        var_383 = wp::where(var_318, var_358, var_309);
        var_384 = wp::where(var_318, var_367, var_310);
        var_385 = wp::where(var_318, var_368, var_311);
        var_386 = wp::where(var_318, var_369, var_312);
        // gidx = flex_face[face_id, local_idx]                                                   <L 506>
        var_388 = wp::address(var_flex_face, var_1, var_387);
        var_390 = wp::load(var_388);
        var_389 = wp::copy(var_390);
        // if gidx != -1:                                                                         <L 507>
        var_392 = (var_389 != var_391);
        if (var_392) {
            // node_pos = flexnode_xpos_in[worldid, gidx]                                         <L 508>
            var_393 = wp::address(var_flexnode_xpos_in, var_0, var_389);
            var_395 = wp::load(var_393);
            var_394 = wp::copy(var_395);
            // face_xpos_out[worldid, face_id, local_idx] = node_pos                              <L 510>
            wp::array_store(var_face_xpos_out, var_0, var_1, var_387, var_394);
            // l0 = local_idx // (order_abs + 1)                                                  <L 512>
            var_397 = wp::add(var_12, var_396);
            var_398 = wp::floordiv(var_387, var_397);
            // l1 = local_idx % (order_abs + 1)                                                   <L 513>
            var_400 = wp::add(var_12, var_399);
            var_401 = wp::mod(var_387, var_400);
            // dphi0 = float(l0 - 1) if order_abs == 2 else (-1.0 + 2.0 * float(l0))              <L 515>
            var_403 = (var_12 == var_402);
            if (var_403) {
                var_405 = wp::sub(var_398, var_404);
                var_406 = wp::float(var_405);
            }
            if (!var_403) {
                var_409 = wp::float(var_398);
                var_410 = wp::mul(var_408, var_409);
                var_411 = wp::add(var_407, var_410);
            }
            var_412 = wp::where(var_403, var_406, var_411);
            // dphi1 = float(l1 - 1) if order_abs == 2 else (-1.0 + 2.0 * float(l1))              <L 516>
            var_414 = (var_12 == var_413);
            if (var_414) {
                var_416 = wp::sub(var_401, var_415);
                var_417 = wp::float(var_416);
            }
            if (!var_414) {
                var_420 = wp::float(var_401);
                var_421 = wp::mul(var_419, var_420);
                var_422 = wp::add(var_418, var_421);
            }
            var_423 = wp::where(var_414, var_417, var_422);
            // phi0 = wp.where(l0 == 1, 1.0, 0.0) if order_abs == 2 else 0.5                      <L 517>
            var_425 = (var_12 == var_424);
            if (var_425) {
                var_427 = (var_398 == var_426);
                var_430 = wp::where(var_427, var_428, var_429);
            }
            if (!var_425) {
            }
            var_432 = wp::where(var_425, var_430, var_431);
            // phi1 = wp.where(l1 == 1, 1.0, 0.0) if order_abs == 2 else 0.5                      <L 518>
            var_434 = (var_12 == var_433);
            if (var_434) {
                var_436 = (var_401 == var_435);
                var_439 = wp::where(var_436, var_437, var_438);
            }
            if (!var_434) {
            }
            var_441 = wp::where(var_434, var_439, var_440);
            // grad0 = dphi0 * phi1                                                               <L 520>
            var_442 = wp::mul(var_412, var_441);
            // grad1 = phi0 * dphi1                                                               <L 521>
            var_443 = wp::mul(var_432, var_423);
            // t1 += node_pos * grad0                                                             <L 523>
            var_444 = wp::mul(var_394, var_442);
            var_445 = wp::add(var_376, var_444);
            // t2 += node_pos * grad1                                                             <L 524>
            var_446 = wp::mul(var_394, var_443);
            var_447 = wp::add(var_377, var_446);
        }
        if (!var_392) {
            // face_xpos_out[worldid, face_id, local_idx] = wp.vec3(0.0)                          <L 526>
            var_449 = wp::vec_t<3, wp::float32>(var_448);
            wp::array_store(var_face_xpos_out, var_0, var_1, var_387, var_449);
        }
        var_450 = wp::where(var_392, var_445, var_376);
        var_451 = wp::where(var_392, var_447, var_377);
        var_452 = wp::where(var_392, var_394, var_378);
        var_453 = wp::where(var_392, var_398, var_379);
        var_454 = wp::where(var_392, var_401, var_380);
        var_455 = wp::where(var_392, var_412, var_381);
        var_456 = wp::where(var_392, var_423, var_382);
        var_457 = wp::where(var_392, var_432, var_383);
        var_458 = wp::where(var_392, var_441, var_384);
        var_459 = wp::where(var_392, var_442, var_385);
        var_460 = wp::where(var_392, var_443, var_386);
        // gidx = flex_face[face_id, local_idx]                                                   <L 506>
        var_462 = wp::address(var_flex_face, var_1, var_461);
        var_464 = wp::load(var_462);
        var_463 = wp::copy(var_464);
        // if gidx != -1:                                                                         <L 507>
        var_466 = (var_463 != var_465);
        if (var_466) {
            // node_pos = flexnode_xpos_in[worldid, gidx]                                         <L 508>
            var_467 = wp::address(var_flexnode_xpos_in, var_0, var_463);
            var_469 = wp::load(var_467);
            var_468 = wp::copy(var_469);
            // face_xpos_out[worldid, face_id, local_idx] = node_pos                              <L 510>
            wp::array_store(var_face_xpos_out, var_0, var_1, var_461, var_468);
            // l0 = local_idx // (order_abs + 1)                                                  <L 512>
            var_471 = wp::add(var_12, var_470);
            var_472 = wp::floordiv(var_461, var_471);
            // l1 = local_idx % (order_abs + 1)                                                   <L 513>
            var_474 = wp::add(var_12, var_473);
            var_475 = wp::mod(var_461, var_474);
            // dphi0 = float(l0 - 1) if order_abs == 2 else (-1.0 + 2.0 * float(l0))              <L 515>
            var_477 = (var_12 == var_476);
            if (var_477) {
                var_479 = wp::sub(var_472, var_478);
                var_480 = wp::float(var_479);
            }
            if (!var_477) {
                var_483 = wp::float(var_472);
                var_484 = wp::mul(var_482, var_483);
                var_485 = wp::add(var_481, var_484);
            }
            var_486 = wp::where(var_477, var_480, var_485);
            // dphi1 = float(l1 - 1) if order_abs == 2 else (-1.0 + 2.0 * float(l1))              <L 516>
            var_488 = (var_12 == var_487);
            if (var_488) {
                var_490 = wp::sub(var_475, var_489);
                var_491 = wp::float(var_490);
            }
            if (!var_488) {
                var_494 = wp::float(var_475);
                var_495 = wp::mul(var_493, var_494);
                var_496 = wp::add(var_492, var_495);
            }
            var_497 = wp::where(var_488, var_491, var_496);
            // phi0 = wp.where(l0 == 1, 1.0, 0.0) if order_abs == 2 else 0.5                      <L 517>
            var_499 = (var_12 == var_498);
            if (var_499) {
                var_501 = (var_472 == var_500);
                var_504 = wp::where(var_501, var_502, var_503);
            }
            if (!var_499) {
            }
            var_506 = wp::where(var_499, var_504, var_505);
            // phi1 = wp.where(l1 == 1, 1.0, 0.0) if order_abs == 2 else 0.5                      <L 518>
            var_508 = (var_12 == var_507);
            if (var_508) {
                var_510 = (var_475 == var_509);
                var_513 = wp::where(var_510, var_511, var_512);
            }
            if (!var_508) {
            }
            var_515 = wp::where(var_508, var_513, var_514);
            // grad0 = dphi0 * phi1                                                               <L 520>
            var_516 = wp::mul(var_486, var_515);
            // grad1 = phi0 * dphi1                                                               <L 521>
            var_517 = wp::mul(var_506, var_497);
            // t1 += node_pos * grad0                                                             <L 523>
            var_518 = wp::mul(var_468, var_516);
            var_519 = wp::add(var_450, var_518);
            // t2 += node_pos * grad1                                                             <L 524>
            var_520 = wp::mul(var_468, var_517);
            var_521 = wp::add(var_451, var_520);
        }
        if (!var_466) {
            // face_xpos_out[worldid, face_id, local_idx] = wp.vec3(0.0)                          <L 526>
            var_523 = wp::vec_t<3, wp::float32>(var_522);
            wp::array_store(var_face_xpos_out, var_0, var_1, var_461, var_523);
        }
        var_524 = wp::where(var_466, var_519, var_450);
        var_525 = wp::where(var_466, var_521, var_451);
        var_526 = wp::where(var_466, var_468, var_452);
        var_527 = wp::where(var_466, var_472, var_453);
        var_528 = wp::where(var_466, var_475, var_454);
        var_529 = wp::where(var_466, var_486, var_455);
        var_530 = wp::where(var_466, var_497, var_456);
        var_531 = wp::where(var_466, var_506, var_457);
        var_532 = wp::where(var_466, var_515, var_458);
        var_533 = wp::where(var_466, var_516, var_459);
        var_534 = wp::where(var_466, var_517, var_460);
        // gidx = flex_face[face_id, local_idx]                                                   <L 506>
        var_536 = wp::address(var_flex_face, var_1, var_535);
        var_538 = wp::load(var_536);
        var_537 = wp::copy(var_538);
        // if gidx != -1:                                                                         <L 507>
        var_540 = (var_537 != var_539);
        if (var_540) {
            // node_pos = flexnode_xpos_in[worldid, gidx]                                         <L 508>
            var_541 = wp::address(var_flexnode_xpos_in, var_0, var_537);
            var_543 = wp::load(var_541);
            var_542 = wp::copy(var_543);
            // face_xpos_out[worldid, face_id, local_idx] = node_pos                              <L 510>
            wp::array_store(var_face_xpos_out, var_0, var_1, var_535, var_542);
            // l0 = local_idx // (order_abs + 1)                                                  <L 512>
            var_545 = wp::add(var_12, var_544);
            var_546 = wp::floordiv(var_535, var_545);
            // l1 = local_idx % (order_abs + 1)                                                   <L 513>
            var_548 = wp::add(var_12, var_547);
            var_549 = wp::mod(var_535, var_548);
            // dphi0 = float(l0 - 1) if order_abs == 2 else (-1.0 + 2.0 * float(l0))              <L 515>
            var_551 = (var_12 == var_550);
            if (var_551) {
                var_553 = wp::sub(var_546, var_552);
                var_554 = wp::float(var_553);
            }
            if (!var_551) {
                var_557 = wp::float(var_546);
                var_558 = wp::mul(var_556, var_557);
                var_559 = wp::add(var_555, var_558);
            }
            var_560 = wp::where(var_551, var_554, var_559);
            // dphi1 = float(l1 - 1) if order_abs == 2 else (-1.0 + 2.0 * float(l1))              <L 516>
            var_562 = (var_12 == var_561);
            if (var_562) {
                var_564 = wp::sub(var_549, var_563);
                var_565 = wp::float(var_564);
            }
            if (!var_562) {
                var_568 = wp::float(var_549);
                var_569 = wp::mul(var_567, var_568);
                var_570 = wp::add(var_566, var_569);
            }
            var_571 = wp::where(var_562, var_565, var_570);
            // phi0 = wp.where(l0 == 1, 1.0, 0.0) if order_abs == 2 else 0.5                      <L 517>
            var_573 = (var_12 == var_572);
            if (var_573) {
                var_575 = (var_546 == var_574);
                var_578 = wp::where(var_575, var_576, var_577);
            }
            if (!var_573) {
            }
            var_580 = wp::where(var_573, var_578, var_579);
            // phi1 = wp.where(l1 == 1, 1.0, 0.0) if order_abs == 2 else 0.5                      <L 518>
            var_582 = (var_12 == var_581);
            if (var_582) {
                var_584 = (var_549 == var_583);
                var_587 = wp::where(var_584, var_585, var_586);
            }
            if (!var_582) {
            }
            var_589 = wp::where(var_582, var_587, var_588);
            // grad0 = dphi0 * phi1                                                               <L 520>
            var_590 = wp::mul(var_560, var_589);
            // grad1 = phi0 * dphi1                                                               <L 521>
            var_591 = wp::mul(var_580, var_571);
            // t1 += node_pos * grad0                                                             <L 523>
            var_592 = wp::mul(var_542, var_590);
            var_593 = wp::add(var_524, var_592);
            // t2 += node_pos * grad1                                                             <L 524>
            var_594 = wp::mul(var_542, var_591);
            var_595 = wp::add(var_525, var_594);
        }
        if (!var_540) {
            // face_xpos_out[worldid, face_id, local_idx] = wp.vec3(0.0)                          <L 526>
            var_597 = wp::vec_t<3, wp::float32>(var_596);
            wp::array_store(var_face_xpos_out, var_0, var_1, var_535, var_597);
        }
        var_598 = wp::where(var_540, var_593, var_524);
        var_599 = wp::where(var_540, var_595, var_525);
        var_600 = wp::where(var_540, var_542, var_526);
        var_601 = wp::where(var_540, var_546, var_527);
        var_602 = wp::where(var_540, var_549, var_528);
        var_603 = wp::where(var_540, var_560, var_529);
        var_604 = wp::where(var_540, var_571, var_530);
        var_605 = wp::where(var_540, var_580, var_531);
        var_606 = wp::where(var_540, var_589, var_532);
        var_607 = wp::where(var_540, var_590, var_533);
        var_608 = wp::where(var_540, var_591, var_534);
        // gidx = flex_face[face_id, local_idx]                                                   <L 506>
        var_610 = wp::address(var_flex_face, var_1, var_609);
        var_612 = wp::load(var_610);
        var_611 = wp::copy(var_612);
        // if gidx != -1:                                                                         <L 507>
        var_614 = (var_611 != var_613);
        if (var_614) {
            // node_pos = flexnode_xpos_in[worldid, gidx]                                         <L 508>
            var_615 = wp::address(var_flexnode_xpos_in, var_0, var_611);
            var_617 = wp::load(var_615);
            var_616 = wp::copy(var_617);
            // face_xpos_out[worldid, face_id, local_idx] = node_pos                              <L 510>
            wp::array_store(var_face_xpos_out, var_0, var_1, var_609, var_616);
            // l0 = local_idx // (order_abs + 1)                                                  <L 512>
            var_619 = wp::add(var_12, var_618);
            var_620 = wp::floordiv(var_609, var_619);
            // l1 = local_idx % (order_abs + 1)                                                   <L 513>
            var_622 = wp::add(var_12, var_621);
            var_623 = wp::mod(var_609, var_622);
            // dphi0 = float(l0 - 1) if order_abs == 2 else (-1.0 + 2.0 * float(l0))              <L 515>
            var_625 = (var_12 == var_624);
            if (var_625) {
                var_627 = wp::sub(var_620, var_626);
                var_628 = wp::float(var_627);
            }
            if (!var_625) {
                var_631 = wp::float(var_620);
                var_632 = wp::mul(var_630, var_631);
                var_633 = wp::add(var_629, var_632);
            }
            var_634 = wp::where(var_625, var_628, var_633);
            // dphi1 = float(l1 - 1) if order_abs == 2 else (-1.0 + 2.0 * float(l1))              <L 516>
            var_636 = (var_12 == var_635);
            if (var_636) {
                var_638 = wp::sub(var_623, var_637);
                var_639 = wp::float(var_638);
            }
            if (!var_636) {
                var_642 = wp::float(var_623);
                var_643 = wp::mul(var_641, var_642);
                var_644 = wp::add(var_640, var_643);
            }
            var_645 = wp::where(var_636, var_639, var_644);
            // phi0 = wp.where(l0 == 1, 1.0, 0.0) if order_abs == 2 else 0.5                      <L 517>
            var_647 = (var_12 == var_646);
            if (var_647) {
                var_649 = (var_620 == var_648);
                var_652 = wp::where(var_649, var_650, var_651);
            }
            if (!var_647) {
            }
            var_654 = wp::where(var_647, var_652, var_653);
            // phi1 = wp.where(l1 == 1, 1.0, 0.0) if order_abs == 2 else 0.5                      <L 518>
            var_656 = (var_12 == var_655);
            if (var_656) {
                var_658 = (var_623 == var_657);
                var_661 = wp::where(var_658, var_659, var_660);
            }
            if (!var_656) {
            }
            var_663 = wp::where(var_656, var_661, var_662);
            // grad0 = dphi0 * phi1                                                               <L 520>
            var_664 = wp::mul(var_634, var_663);
            // grad1 = phi0 * dphi1                                                               <L 521>
            var_665 = wp::mul(var_654, var_645);
            // t1 += node_pos * grad0                                                             <L 523>
            var_666 = wp::mul(var_616, var_664);
            var_667 = wp::add(var_598, var_666);
            // t2 += node_pos * grad1                                                             <L 524>
            var_668 = wp::mul(var_616, var_665);
            var_669 = wp::add(var_599, var_668);
        }
        if (!var_614) {
            // face_xpos_out[worldid, face_id, local_idx] = wp.vec3(0.0)                          <L 526>
            var_671 = wp::vec_t<3, wp::float32>(var_670);
            wp::array_store(var_face_xpos_out, var_0, var_1, var_609, var_671);
        }
        var_672 = wp::where(var_614, var_667, var_598);
        var_673 = wp::where(var_614, var_669, var_599);
        var_674 = wp::where(var_614, var_616, var_600);
        var_675 = wp::where(var_614, var_620, var_601);
        var_676 = wp::where(var_614, var_623, var_602);
        var_677 = wp::where(var_614, var_634, var_603);
        var_678 = wp::where(var_614, var_645, var_604);
        var_679 = wp::where(var_614, var_654, var_605);
        var_680 = wp::where(var_614, var_663, var_606);
        var_681 = wp::where(var_614, var_664, var_607);
        var_682 = wp::where(var_614, var_665, var_608);
        // normal = wp.cross(t1, t2)                                                              <L 528>
        var_683 = wp::cross(var_672, var_673);
        // normal_axis, _, _, _, _, _ = support.get_face_metadata(cx, cy, cz, face_elem_idx, order_abs)       <L 530>
        get_face_metadata_0(var_17, var_19, var_21, var_8, var_12, var_684, var_685, var_686, var_687, var_688, var_689);
        // F = wp.mat33(0.0)                                                                      <L 532>
        var_691 = wp::mat_t<3, 3, wp::float32>(var_690);
        // if normal_axis == 0:                                                                   <L 533>
        var_693 = (var_684 == var_692);
        if (var_693) {
            // F = wp.mat33(                                                                      <L 534>
            // normal[0],                                                                         <L 535>
            var_695 = wp::extract(var_683, var_694);
            // t1[0],                                                                             <L 536>
            var_697 = wp::extract(var_672, var_696);
            // t2[0],                                                                             <L 537>
            var_699 = wp::extract(var_673, var_698);
            // normal[1],                                                                         <L 538>
            var_701 = wp::extract(var_683, var_700);
            // t1[1],                                                                             <L 539>
            var_703 = wp::extract(var_672, var_702);
            // t2[1],                                                                             <L 540>
            var_705 = wp::extract(var_673, var_704);
            // normal[2],                                                                         <L 541>
            var_707 = wp::extract(var_683, var_706);
            // t1[2],                                                                             <L 542>
            var_709 = wp::extract(var_672, var_708);
            // t2[2],                                                                             <L 543>
            var_711 = wp::extract(var_673, var_710);
            var_712 = wp::mat_t<3, 3, wp::float32>(var_695, var_697, var_699, var_701, var_703, var_705, var_707, var_709, var_711);
        }
        if (!var_693) {
            // elif normal_axis == 1:                                                             <L 545>
            var_714 = (var_684 == var_713);
            if (var_714) {
                // F = wp.mat33(                                                                  <L 546>
                // t2[0],                                                                         <L 547>
                var_716 = wp::extract(var_673, var_715);
                // normal[0],                                                                     <L 548>
                var_718 = wp::extract(var_683, var_717);
                // t1[0],                                                                         <L 549>
                var_720 = wp::extract(var_672, var_719);
                // t2[1],                                                                         <L 550>
                var_722 = wp::extract(var_673, var_721);
                // normal[1],                                                                     <L 551>
                var_724 = wp::extract(var_683, var_723);
                // t1[1],                                                                         <L 552>
                var_726 = wp::extract(var_672, var_725);
                // t2[2],                                                                         <L 553>
                var_728 = wp::extract(var_673, var_727);
                // normal[2],                                                                     <L 554>
                var_730 = wp::extract(var_683, var_729);
                // t1[2],                                                                         <L 555>
                var_732 = wp::extract(var_672, var_731);
                var_733 = wp::mat_t<3, 3, wp::float32>(var_716, var_718, var_720, var_722, var_724, var_726, var_728, var_730, var_732);
            }
            if (!var_714) {
                // F = wp.mat33(                                                                  <L 558>
                // t1[0],                                                                         <L 559>
                var_735 = wp::extract(var_672, var_734);
                // t2[0],                                                                         <L 560>
                var_737 = wp::extract(var_673, var_736);
                // normal[0],                                                                     <L 561>
                var_739 = wp::extract(var_683, var_738);
                // t1[1],                                                                         <L 562>
                var_741 = wp::extract(var_672, var_740);
                // t2[1],                                                                         <L 563>
                var_743 = wp::extract(var_673, var_742);
                // normal[1],                                                                     <L 564>
                var_745 = wp::extract(var_683, var_744);
                // t1[2],                                                                         <L 565>
                var_747 = wp::extract(var_672, var_746);
                // t2[2],                                                                         <L 566>
                var_749 = wp::extract(var_673, var_748);
                // normal[2],                                                                     <L 567>
                var_751 = wp::extract(var_683, var_750);
                var_752 = wp::mat_t<3, 3, wp::float32>(var_735, var_737, var_739, var_741, var_743, var_745, var_747, var_749, var_751);
            }
            var_753 = wp::where(var_714, var_733, var_752);
        }
        var_754 = wp::where(var_693, var_712, var_753);
        // face_quat_out[worldid, face_id] = support.mat33_to_quat_polar(F)                       <L 570>
        var_755 = mat33_to_quat_polar_0(var_754);
        wp::array_store(var_face_quat_out, var_0, var_1, var_755);
    }
}



extern "C" __global__ void _transmission_body_moment_af66f321_cuda_kernel_forward(
    wp::launch_bounds_t<3> dim,
    wp::int32 var_opt_cone,
    wp::array_t<wp::int32> var_body_parentid,
    wp::array_t<wp::int32> var_body_rootid,
    wp::array_t<wp::int32> var_dof_bodyid,
    wp::array_t<wp::int32> var_geom_bodyid,
    wp::array_t<wp::vec_t<2, wp::int32>> var_actuator_trnid,
    wp::array_t<wp::int32> var_body_isdofancestor,
    wp::array_t<wp::int32> var_actuator_trntype_body_adr,
    wp::array_t<wp::vec_t<3, wp::float32>> var_subtree_com_in,
    wp::array_t<wp::vec_t<6, wp::float32>> var_cdof_in,
    wp::array_t<wp::int32> var_moment_rowadr_in,
    wp::array_t<wp::float32> var_contact_dist_in,
    wp::array_t<wp::vec_t<3, wp::float32>> var_contact_pos_in,
    wp::array_t<wp::mat_t<3, 3, wp::float32>> var_contact_frame_in,
    wp::array_t<wp::float32> var_contact_includemargin_in,
    wp::array_t<wp::int32> var_contact_dim_in,
    wp::array_t<wp::vec_t<2, wp::int32>> var_contact_geom_in,
    wp::array_t<wp::int32> var_contact_efc_address_in,
    wp::array_t<wp::int32> var_contact_worldid_in,
    wp::array_t<wp::int32> var_efc_J_rownnz_in,
    wp::array_t<wp::int32> var_efc_J_rowadr_in,
    wp::array_t<wp::int32> var_efc_J_colind_in,
    wp::array_t<wp::float32> var_efc_J_in,
    wp::array_t<wp::int32> var_nacon_in,
    bool var_efc_is_sparse,
    wp::array_t<wp::float32> var_actuator_moment_out,
    wp::array_t<wp::int32> var_actuator_trntype_body_ncon_out)
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
        wp::int32 var_4;
        wp::int32 var_5;
        wp::vec_t<2, wp::int32>* var_6;
        const wp::int32 var_7 = 0;
        wp::int32 var_8;
        wp::vec_t<2, wp::int32> var_9;
        const wp::int32 var_10 = 0;
        wp::int32* var_11;
        bool var_12;
        wp::int32 var_13;
        wp::int32* var_14;
        wp::int32 var_15;
        wp::int32 var_16;
        wp::vec_t<2, wp::int32>* var_17;
        wp::vec_t<2, wp::int32> var_18;
        wp::vec_t<2, wp::int32> var_19;
        const wp::int32 var_20 = 0;
        wp::int32 var_21;
        const wp::int32 var_22 = 1;
        wp::int32 var_23;
        bool var_24;
        const wp::int32 var_25 = 0;
        bool var_26;
        const wp::int32 var_27 = 0;
        bool var_28;
        wp::int32* var_29;
        wp::int32 var_30;
        wp::int32 var_31;
        wp::int32* var_32;
        wp::int32 var_33;
        wp::int32 var_34;
        bool var_35;
        bool var_36;
        bool var_37;
        wp::float32* var_38;
        wp::float32* var_39;
        bool var_40;
        wp::float32 var_41;
        wp::float32 var_42;
        wp::int32 var_43;
        const wp::int32 var_44 = 0;
        bool var_45;
        wp::array_t<wp::int32> var_46;
        const wp::int32 var_47 = 1;
        wp::int32 var_48;
        wp::int32* var_49;
        wp::int32 var_50;
        wp::int32 var_51;
        const wp::int32 var_52 = 0;
        bool var_53;
        wp::int32* var_54;
        wp::int32 var_55;
        wp::int32 var_56;
        wp::array_t<wp::int32> var_57;
        bool var_58;
        const wp::int32 var_59 = 1;
        bool var_60;
        const wp::int32 var_61 = 1;
        bool var_62;
        const wp::int32 var_63 = 0;
        wp::int32* var_64;
        wp::int32 var_65;
        wp::int32 var_66;
        wp::int32* var_67;
        wp::int32 var_68;
        wp::int32 var_69;
        bool var_70;
        wp::int32* var_71;
        wp::int32 var_72;
        wp::int32 var_73;
        wp::int32 var_74;
        const wp::int32 var_75 = 0;
        wp::int32* var_76;
        wp::int32 var_77;
        wp::int32 var_78;
        wp::array_t<wp::float32> var_79;
        wp::int32 var_80;
        const wp::int32 var_81 = 0;
        wp::float32* var_82;
        wp::float32 var_83;
        wp::float32 var_84;
        wp::int32 var_85;
        wp::array_t<wp::float32> var_86;
        wp::int32 var_87;
        wp::float32* var_88;
        wp::float32 var_89;
        wp::float32 var_90;
        wp::int32 var_91;
        const wp::int32 var_92 = 1;
        wp::int32 var_93;
        const wp::float32 var_94 = 0.5;
        wp::float32 var_95;
        wp::float32 var_96;
        const wp::int32 var_97 = 2;
        wp::int32 var_98;
        wp::range_t var_99;
        wp::int32 var_100;
        wp::int32* var_101;
        wp::int32 var_102;
        wp::int32 var_103;
        wp::int32* var_104;
        wp::int32 var_105;
        wp::int32 var_106;
        bool var_107;
        wp::int32* var_108;
        wp::int32 var_109;
        wp::int32 var_110;
        wp::int32 var_111;
        const wp::int32 var_112 = 0;
        wp::int32* var_113;
        wp::int32 var_114;
        wp::int32 var_115;
        wp::array_t<wp::float32> var_116;
        wp::int32 var_117;
        const wp::int32 var_118 = 0;
        wp::float32* var_119;
        wp::float32 var_120;
        wp::float32 var_121;
        wp::float32 var_122;
        wp::int32 var_123;
        wp::array_t<wp::float32> var_124;
        wp::int32 var_125;
        wp::float32* var_126;
        wp::float32 var_127;
        wp::float32 var_128;
        wp::float32 var_129;
        wp::int32 var_130;
        wp::int32 var_131;
        wp::int32 var_132;
        wp::int32 var_133;
        wp::int32 var_134;
        const wp::int32 var_135 = 1;
        bool var_136;
        wp::vec_t<3, wp::float32>* var_137;
        wp::vec_t<3, wp::float32> var_138;
        wp::vec_t<3, wp::float32> var_139;
        wp::mat_t<3, 3, wp::float32>* var_140;
        wp::mat_t<3, 3, wp::float32> var_141;
        wp::mat_t<3, 3, wp::float32> var_142;
        const wp::int32 var_143 = 0;
        const wp::int32 var_144 = 0;
        wp::float32 var_145;
        const wp::int32 var_146 = 0;
        const wp::int32 var_147 = 1;
        wp::float32 var_148;
        const wp::int32 var_149 = 0;
        const wp::int32 var_150 = 2;
        wp::float32 var_151;
        wp::vec_t<3, wp::float32> var_152;
        const wp::int32 var_153 = 0;
        wp::int32* var_154;
        wp::int32 var_155;
        wp::int32 var_156;
        bool var_157;
        const wp::int32 var_158 = 0;
        bool var_159;
        wp::int32* var_160;
        bool var_161;
        wp::int32 var_162;
        wp::int32* var_163;
        wp::int32 var_164;
        wp::int32 var_165;
        const wp::int32 var_166 = 0;
        wp::int32* var_167;
        wp::int32 var_168;
        wp::int32 var_169;
        wp::int32 var_170;
        wp::int32 var_171;
        wp::vec_t<3, wp::float32> var_172;
        wp::vec_t<3, wp::float32> var_173;
        wp::vec_t<3, wp::float32> var_174;
        wp::vec_t<3, wp::float32> var_175;
        wp::vec_t<3, wp::float32> var_176;
        wp::array_t<wp::float32> var_177;
        wp::int32 var_178;
        wp::float32 var_179;
        wp::float32 var_180;
        wp::int32 var_181;
        wp::int32 var_182;
        //---------
        // forward
        // def _transmission_body_moment(                                                         <L 3062>
        // trnbodyid, conid, dofid = wp.tid()                                                     <L 3096>
        builtin_tid3d(var_0, var_1, var_2);
        // actid = actuator_trntype_body_adr[trnbodyid]                                           <L 3097>
        var_3 = wp::address(var_actuator_trntype_body_adr, var_0);
        var_5 = wp::load(var_3);
        var_4 = wp::copy(var_5);
        // bodyid = actuator_trnid[actid][0]                                                      <L 3098>
        var_6 = wp::address(var_actuator_trnid, var_4);
        var_9 = wp::load(var_6);
        var_8 = wp::extract(var_9, var_7);
        // if conid >= nacon_in[0]:                                                               <L 3100>
        var_11 = wp::address(var_nacon_in, var_10);
        var_13 = wp::load(var_11);
        var_12 = (var_1 >= var_13);
        if (var_12) {
            // return                                                                             <L 3101>
            continue;
        }
        // worldid = contact_worldid_in[conid]                                                    <L 3103>
        var_14 = wp::address(var_contact_worldid_in, var_1);
        var_16 = wp::load(var_14);
        var_15 = wp::copy(var_16);
        // geom = contact_geom_in[conid]                                                          <L 3106>
        var_17 = wp::address(var_contact_geom_in, var_1);
        var_19 = wp::load(var_17);
        var_18 = wp::copy(var_19);
        // g1 = geom[0]                                                                           <L 3107>
        var_21 = wp::extract(var_18, var_20);
        // g2 = geom[1]                                                                           <L 3108>
        var_23 = wp::extract(var_18, var_22);
        // if g1 < 0 or g2 < 0:                                                                   <L 3111>
        var_26 = (var_21 < var_25);
        var_24 = var_26;
        if (!var_24) {
            var_28 = (var_23 < var_27);
            var_24 = var_24 || var_28;
        }
        if (var_24) {
            // return                                                                             <L 3112>
            continue;
        }
        // b1 = geom_bodyid[g1]                                                                   <L 3115>
        var_29 = wp::address(var_geom_bodyid, var_21);
        var_31 = wp::load(var_29);
        var_30 = wp::copy(var_31);
        // b2 = geom_bodyid[g2]                                                                   <L 3116>
        var_32 = wp::address(var_geom_bodyid, var_23);
        var_34 = wp::load(var_32);
        var_33 = wp::copy(var_34);
        // if b1 != bodyid and b2 != bodyid:                                                      <L 3119>
        var_36 = (var_30 != var_8);
        var_35 = var_36;
        if (var_35) {
            var_37 = (var_33 != var_8);
            var_35 = var_35 && var_37;
        }
        if (var_35) {
            // return                                                                             <L 3120>
            continue;
        }
        // contact_exclude = int(contact_dist_in[conid] >= contact_includemargin_in[conid])       <L 3122>
        var_38 = wp::address(var_contact_dist_in, var_1);
        var_39 = wp::address(var_contact_includemargin_in, var_1);
        var_41 = wp::load(var_38);
        var_42 = wp::load(var_39);
        var_40 = (var_41 >= var_42);
        var_43 = wp::int(var_40);
        // if dofid == 0:                                                                         <L 3124>
        var_45 = (var_2 == var_44);
        if (var_45) {
            // wp.atomic_add(actuator_trntype_body_ncon_out[worldid], trnbodyid, 1)               <L 3125>
            var_46 = wp::view(var_actuator_trntype_body_ncon_out, var_15);
            var_48 = wp::atomic_add(var_46, var_0, var_47);
        }
        // rowadr = moment_rowadr_in[worldid, actid]                                              <L 3127>
        var_49 = wp::address(var_moment_rowadr_in, var_15, var_4);
        var_51 = wp::load(var_49);
        var_50 = wp::copy(var_51);
        // if contact_exclude == 0:                                                               <L 3130>
        var_53 = (var_43 == var_52);
        if (var_53) {
            // contact_dim = contact_dim_in[conid]                                                <L 3131>
            var_54 = wp::address(var_contact_dim_in, var_1);
            var_56 = wp::load(var_54);
            var_55 = wp::copy(var_56);
            // contact_efc_address = contact_efc_address_in[conid]                                <L 3132>
            var_57 = wp::view(var_contact_efc_address_in, var_1);
            // if contact_dim == 1 or opt_cone == ConeType.ELLIPTIC:                              <L 3134>
            var_60 = (var_55 == var_59);
            var_58 = var_60;
            if (!var_58) {
                var_62 = (var_opt_cone == var_61);
                var_58 = var_58 || var_62;
            }
            if (var_58) {
                // efcid0 = contact_efc_address[0]                                                <L 3135>
                var_64 = wp::address(var_57, var_63);
                var_66 = wp::load(var_64);
                var_65 = wp::copy(var_66);
                // if efc_is_sparse:                                                              <L 3136>
                if (var_efc_is_sparse) {
                    // rownnz = efc_J_rownnz_in[worldid, efcid0]                                  <L 3137>
                    var_67 = wp::address(var_efc_J_rownnz_in, var_15, var_65);
                    var_69 = wp::load(var_67);
                    var_68 = wp::copy(var_69);
                    // if dofid < rownnz:                                                         <L 3138>
                    var_70 = (var_2 < var_68);
                    if (var_70) {
                        // efc_rowadr = efc_J_rowadr_in[worldid, efcid0]                          <L 3139>
                        var_71 = wp::address(var_efc_J_rowadr_in, var_15, var_65);
                        var_73 = wp::load(var_71);
                        var_72 = wp::copy(var_73);
                        // efc_sparseid = efc_rowadr + dofid                                      <L 3140>
                        var_74 = wp::add(var_72, var_2);
                        // colind = efc_J_colind_in[worldid, 0, efc_sparseid]                     <L 3141>
                        var_76 = wp::address(var_efc_J_colind_in, var_15, var_75, var_74);
                        var_78 = wp::load(var_76);
                        var_77 = wp::copy(var_78);
                        // wp.atomic_add(actuator_moment_out[worldid], rowadr + colind, efc_J_in[worldid, 0, efc_sparseid])       <L 3142>
                        var_79 = wp::view(var_actuator_moment_out, var_15);
                        var_80 = wp::add(var_50, var_77);
                        var_82 = wp::address(var_efc_J_in, var_15, var_81, var_74);
                        var_84 = wp::load(var_82);
                        var_83 = wp::atomic_add(var_79, var_80, var_84);
                    }
                    if (!var_70) {
                        // return                                                                 <L 3144>
                        continue;
                    }
                }
                if (!var_efc_is_sparse) {
                    // colind = dofid                                                             <L 3146>
                    var_85 = wp::copy(var_2);
                    // wp.atomic_add(actuator_moment_out[worldid], rowadr + colind, efc_J_in[worldid, efcid0, dofid])       <L 3147>
                    var_86 = wp::view(var_actuator_moment_out, var_15);
                    var_87 = wp::add(var_50, var_85);
                    var_88 = wp::address(var_efc_J_in, var_15, var_65, var_2);
                    var_90 = wp::load(var_88);
                    var_89 = wp::atomic_add(var_86, var_87, var_90);
                }
                var_91 = wp::where(var_efc_is_sparse, var_77, var_85);
            }
            if (!var_58) {
                // npyramid = contact_dim - 1  # number of frictional directions                  <L 3149>
                var_93 = wp::sub(var_55, var_92);
                // efc_force = 0.5 / float(npyramid)                                              <L 3150>
                var_95 = wp::float(var_93);
                var_96 = wp::div(var_94, var_95);
                // for j in range(2 * npyramid):                                                  <L 3152>
                var_98 = wp::mul(var_97, var_93);
                var_99 = wp::range(var_98);
                start_for_4:;
                    if (iter_cmp(var_99) == 0) goto end_for_4;
                    var_100 = wp::iter_next(var_99);
                    // efcid = contact_efc_address[j]                                             <L 3153>
                    var_101 = wp::address(var_57, var_100);
                    var_103 = wp::load(var_101);
                    var_102 = wp::copy(var_103);
                    // if efc_is_sparse:                                                          <L 3154>
                    if (var_efc_is_sparse) {
                        // rownnz = efc_J_rownnz_in[worldid, efcid]                               <L 3155>
                        var_104 = wp::address(var_efc_J_rownnz_in, var_15, var_102);
                        var_106 = wp::load(var_104);
                        var_105 = wp::copy(var_106);
                        // if dofid < rownnz:                                                     <L 3156>
                        var_107 = (var_2 < var_105);
                        if (var_107) {
                            // efc_rowadr = efc_J_rowadr_in[worldid, efcid]                       <L 3157>
                            var_108 = wp::address(var_efc_J_rowadr_in, var_15, var_102);
                            var_110 = wp::load(var_108);
                            var_109 = wp::copy(var_110);
                            // efc_sparseid = efc_rowadr + dofid                                  <L 3158>
                            var_111 = wp::add(var_109, var_2);
                            // colind = efc_J_colind_in[worldid, 0, efc_sparseid]                 <L 3159>
                            var_113 = wp::address(var_efc_J_colind_in, var_15, var_112, var_111);
                            var_115 = wp::load(var_113);
                            var_114 = wp::copy(var_115);
                            // wp.atomic_add(actuator_moment_out[worldid], rowadr + colind, efc_J_in[worldid, 0, efc_sparseid] * efc_force)       <L 3160>
                            var_116 = wp::view(var_actuator_moment_out, var_15);
                            var_117 = wp::add(var_50, var_114);
                            var_119 = wp::address(var_efc_J_in, var_15, var_118, var_111);
                            var_121 = wp::load(var_119);
                            var_120 = wp::mul(var_121, var_96);
                            var_122 = wp::atomic_add(var_116, var_117, var_120);
                        }
                        if (!var_107) {
                            // return                                                             <L 3162>
                            continue;
                        }
                    }
                    if (!var_efc_is_sparse) {
                        // colind = dofid                                                         <L 3164>
                        var_123 = wp::copy(var_2);
                        // wp.atomic_add(actuator_moment_out[worldid], rowadr + colind, efc_J_in[worldid, efcid, dofid] * efc_force)       <L 3165>
                        var_124 = wp::view(var_actuator_moment_out, var_15);
                        var_125 = wp::add(var_50, var_123);
                        var_126 = wp::address(var_efc_J_in, var_15, var_102, var_2);
                        var_128 = wp::load(var_126);
                        var_127 = wp::mul(var_128, var_96);
                        var_129 = wp::atomic_add(var_124, var_125, var_127);
                    }
                    var_130 = wp::where(var_efc_is_sparse, var_114, var_123);
                    goto start_for_4;
                end_for_4:;
            }
            var_131 = wp::where(var_58, var_68, var_105);
            var_132 = wp::where(var_58, var_72, var_109);
            var_133 = wp::where(var_58, var_74, var_111);
            var_134 = wp::where(var_58, var_91, var_130);
        }
        if (!var_53) {
            // elif contact_exclude == 1:                                                         <L 3168>
            var_136 = (var_43 == var_135);
            if (var_136) {
                // contact_pos = contact_pos_in[conid]                                            <L 3169>
                var_137 = wp::address(var_contact_pos_in, var_1);
                var_139 = wp::load(var_137);
                var_138 = wp::copy(var_139);
                // contact_frame = contact_frame_in[conid]                                        <L 3170>
                var_140 = wp::address(var_contact_frame_in, var_1);
                var_142 = wp::load(var_140);
                var_141 = wp::copy(var_142);
                // normal = wp.vec3(contact_frame[0, 0], contact_frame[0, 1], contact_frame[0, 2])       <L 3171>
                var_145 = wp::extract(var_141, var_143, var_144);
                var_148 = wp::extract(var_141, var_146, var_147);
                var_151 = wp::extract(var_141, var_149, var_150);
                var_152 = wp::vec_t<3, wp::float32>(var_145, var_148, var_151);
                // efcid0 = contact_efc_address_in[conid][0]                                      <L 3174>
                var_154 = wp::address(var_contact_efc_address_in, var_1, var_153);
                var_156 = wp::load(var_154);
                var_155 = wp::copy(var_156);
                // if efc_is_sparse and efcid0 >= 0:                                              <L 3175>
                var_157 = var_efc_is_sparse;
                if (var_157) {
                    var_159 = (var_155 >= var_158);
                    var_157 = var_157 && var_159;
                }
                if (var_157) {
                    // if dofid >= efc_J_rownnz_in[worldid, efcid0]:                              <L 3177>
                    var_160 = wp::address(var_efc_J_rownnz_in, var_15, var_155);
                    var_162 = wp::load(var_160);
                    var_161 = (var_2 >= var_162);
                    if (var_161) {
                        // return                                                                 <L 3178>
                        continue;
                    }
                    // sparseid = efc_J_rowadr_in[worldid, efcid0] + dofid                        <L 3179>
                    var_163 = wp::address(var_efc_J_rowadr_in, var_15, var_155);
                    var_165 = wp::load(var_163);
                    var_164 = wp::add(var_165, var_2);
                    // colind = efc_J_colind_in[worldid, 0, sparseid]                             <L 3180>
                    var_167 = wp::address(var_efc_J_colind_in, var_15, var_166, var_164);
                    var_169 = wp::load(var_167);
                    var_168 = wp::copy(var_169);
                }
                if (!var_157) {
                    // colind = dofid                                                             <L 3183>
                    var_170 = wp::copy(var_2);
                }
                var_171 = wp::where(var_157, var_168, var_170);
                // jacp1, _ = support.jac_dof(                                                    <L 3185>
                // body_parentid, body_rootid, dof_bodyid, body_isdofancestor, subtree_com_in, cdof_in, contact_pos, b1, colind, worldid       <L 3186>
                jac_dof_0(var_body_parentid, var_body_rootid, var_dof_bodyid, var_body_isdofancestor, var_subtree_com_in, var_cdof_in, var_138, var_30, var_171, var_15, var_172, var_173);
                // jacp2, _ = support.jac_dof(                                                    <L 3188>
                // body_parentid, body_rootid, dof_bodyid, body_isdofancestor, subtree_com_in, cdof_in, contact_pos, b2, colind, worldid       <L 3189>
                jac_dof_0(var_body_parentid, var_body_rootid, var_dof_bodyid, var_body_isdofancestor, var_subtree_com_in, var_cdof_in, var_138, var_33, var_171, var_15, var_174, var_175);
                // jacdif = jacp2 - jacp1                                                         <L 3192>
                var_176 = wp::sub(var_174, var_172);
                // wp.atomic_add(actuator_moment_out[worldid], rowadr + colind, wp.dot(normal, jacdif))       <L 3195>
                var_177 = wp::view(var_actuator_moment_out, var_15);
                var_178 = wp::add(var_50, var_171);
                var_179 = wp::dot(var_152, var_176);
                var_180 = wp::atomic_add(var_177, var_178, var_179);
            }
        }
        var_181 = wp::where(var_53, var_65, var_155);
        var_182 = wp::where(var_53, var_134, var_171);
    }
}



extern "C" __global__ void _compute_body_inertial_frames_d3bdd81a_cuda_kernel_forward(
    wp::launch_bounds_t<2> dim,
    wp::array_t<wp::vec_t<3, wp::float32>> var_body_ipos,
    wp::array_t<wp::quat_t<wp::float32>> var_body_iquat,
    wp::array_t<wp::vec_t<3, wp::float32>> var_xpos_in,
    wp::array_t<wp::quat_t<wp::float32>> var_xquat_in,
    wp::array_t<wp::vec_t<3, wp::float32>> var_xipos_out,
    wp::array_t<wp::mat_t<3, 3, wp::float32>> var_ximat_out)
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
        wp::vec_t<3, wp::float32>* var_2;
        wp::vec_t<3, wp::float32> var_3;
        wp::vec_t<3, wp::float32> var_4;
        wp::quat_t<wp::float32>* var_5;
        wp::quat_t<wp::float32> var_6;
        wp::quat_t<wp::float32> var_7;
        wp::shape_t* var_8;
        const wp::int32 var_9 = 0;
        wp::int32 var_10;
        wp::shape_t var_11;
        wp::int32 var_12;
        wp::vec_t<3, wp::float32>* var_13;
        wp::vec_t<3, wp::float32> var_14;
        wp::vec_t<3, wp::float32> var_15;
        wp::vec_t<3, wp::float32> var_16;
        wp::shape_t* var_17;
        const wp::int32 var_18 = 0;
        wp::int32 var_19;
        wp::shape_t var_20;
        wp::int32 var_21;
        wp::quat_t<wp::float32>* var_22;
        wp::quat_t<wp::float32> var_23;
        wp::quat_t<wp::float32> var_24;
        wp::mat_t<3, 3, wp::float32> var_25;
        //---------
        // forward
        // def _compute_body_inertial_frames(                                                     <L 150>
        // worldid, bodyid = wp.tid()                                                             <L 161>
        builtin_tid2d(var_0, var_1);
        // xpos = xpos_in[worldid, bodyid]                                                        <L 162>
        var_2 = wp::address(var_xpos_in, var_0, var_1);
        var_4 = wp::load(var_2);
        var_3 = wp::copy(var_4);
        // xquat = xquat_in[worldid, bodyid]                                                      <L 163>
        var_5 = wp::address(var_xquat_in, var_0, var_1);
        var_7 = wp::load(var_5);
        var_6 = wp::copy(var_7);
        // xipos_out[worldid, bodyid] = xpos + math.rot_vec_quat(body_ipos[worldid % body_ipos.shape[0], bodyid], xquat)       <L 164>
        var_8 = &(var_body_ipos.shape);
        var_11 = wp::load(var_8);
        var_10 = wp::extract(var_11, var_9);
        var_12 = wp::mod(var_0, var_10);
        var_13 = wp::address(var_body_ipos, var_12, var_1);
        var_15 = wp::load(var_13);
        var_14 = rot_vec_quat_0(var_15, var_6);
        var_16 = wp::add(var_3, var_14);
        wp::array_store(var_xipos_out, var_0, var_1, var_16);
        // ximat_out[worldid, bodyid] = math.quat_to_mat(math.mul_quat(xquat, body_iquat[worldid % body_iquat.shape[0], bodyid]))       <L 165>
        var_17 = &(var_body_iquat.shape);
        var_20 = wp::load(var_17);
        var_19 = wp::extract(var_20, var_18);
        var_21 = wp::mod(var_0, var_19);
        var_22 = wp::address(var_body_iquat, var_21, var_1);
        var_24 = wp::load(var_22);
        var_23 = mul_quat_0(var_6, var_24);
        var_25 = quat_to_mat_0(var_23);
        wp::array_store(var_ximat_out, var_0, var_1, var_25);
    }
}



extern "C" __global__ void _spatial_site_tendon_d90d9688_cuda_kernel_forward(
    wp::launch_bounds_t<2> dim,
    wp::array_t<wp::int32> var_body_parentid,
    wp::array_t<wp::int32> var_body_rootid,
    wp::array_t<wp::int32> var_body_dofnum,
    wp::array_t<wp::int32> var_body_dofadr,
    wp::array_t<wp::int32> var_site_bodyid,
    wp::array_t<wp::int32> var_ten_J_rownnz,
    wp::array_t<wp::int32> var_ten_J_rowadr,
    wp::array_t<wp::int32> var_ten_J_colind,
    wp::array_t<wp::int32> var_wrap_objid,
    wp::array_t<wp::int32> var_tendon_site_pair_adr,
    wp::array_t<wp::int32> var_wrap_site_pair_adr,
    wp::array_t<wp::float32> var_wrap_pulley_scale,
    wp::array_t<wp::vec_t<3, wp::float32>> var_site_xpos_in,
    wp::array_t<wp::vec_t<3, wp::float32>> var_subtree_com_in,
    wp::array_t<wp::vec_t<6, wp::float32>> var_cdof_in,
    wp::array_t<wp::float32> var_ten_J_out,
    wp::array_t<wp::float32> var_ten_length_out)
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
        wp::int32* var_5;
        wp::int32 var_6;
        wp::int32 var_7;
        wp::float32* var_8;
        wp::float32 var_9;
        wp::float32 var_10;
        const wp::int32 var_11 = 0;
        wp::int32 var_12;
        wp::int32* var_13;
        wp::int32 var_14;
        wp::int32 var_15;
        const wp::int32 var_16 = 1;
        wp::int32 var_17;
        wp::int32* var_18;
        wp::int32 var_19;
        wp::int32 var_20;
        wp::vec_t<3, wp::float32>* var_21;
        wp::vec_t<3, wp::float32> var_22;
        wp::vec_t<3, wp::float32> var_23;
        wp::vec_t<3, wp::float32>* var_24;
        wp::vec_t<3, wp::float32> var_25;
        wp::vec_t<3, wp::float32> var_26;
        wp::vec_t<3, wp::float32> var_27;
        wp::vec_t<3, wp::float32> var_28;
        wp::float32 var_29;
        wp::array_t<wp::float32> var_30;
        wp::float32 var_31;
        wp::float32 var_32;
        const wp::float32 var_33 = 1e-15;
        bool var_34;
        const wp::float32 var_35 = 1.0;
        const wp::float32 var_36 = 0.0;
        const wp::float32 var_37 = 0.0;
        wp::vec_t<3, wp::float32> var_38;
        wp::vec_t<3, wp::float32> var_39;
        wp::int32* var_40;
        wp::int32 var_41;
        wp::int32 var_42;
        wp::int32* var_43;
        wp::int32 var_44;
        wp::int32 var_45;
        bool var_46;
        wp::int32* var_47;
        wp::int32 var_48;
        wp::int32 var_49;
        wp::int32* var_50;
        wp::int32 var_51;
        wp::int32 var_52;
        wp::int32* var_53;
        wp::int32 var_54;
        wp::vec_t<3, wp::float32>* var_55;
        wp::vec_t<3, wp::float32> var_56;
        wp::vec_t<3, wp::float32> var_57;
        wp::int32* var_58;
        wp::int32 var_59;
        wp::vec_t<3, wp::float32>* var_60;
        wp::vec_t<3, wp::float32> var_61;
        wp::vec_t<3, wp::float32> var_62;
        wp::float32 var_63;
        //---------
        // forward
        // def _spatial_site_tendon(                                                              <L 4072>
        // worldid, elementid = wp.tid()                                                          <L 4094>
        builtin_tid2d(var_0, var_1);
        // site_pair_adr = wrap_site_pair_adr[elementid]                                          <L 4097>
        var_2 = wp::address(var_wrap_site_pair_adr, var_1);
        var_4 = wp::load(var_2);
        var_3 = wp::copy(var_4);
        // tenid = tendon_site_pair_adr[elementid]                                                <L 4098>
        var_5 = wp::address(var_tendon_site_pair_adr, var_1);
        var_7 = wp::load(var_5);
        var_6 = wp::copy(var_7);
        // pulley_scale = wrap_pulley_scale[site_pair_adr]                                        <L 4101>
        var_8 = wp::address(var_wrap_pulley_scale, var_3);
        var_10 = wp::load(var_8);
        var_9 = wp::copy(var_10);
        // id0 = wrap_objid[site_pair_adr + 0]                                                    <L 4103>
        var_12 = wp::add(var_3, var_11);
        var_13 = wp::address(var_wrap_objid, var_12);
        var_15 = wp::load(var_13);
        var_14 = wp::copy(var_15);
        // id1 = wrap_objid[site_pair_adr + 1]                                                    <L 4104>
        var_17 = wp::add(var_3, var_16);
        var_18 = wp::address(var_wrap_objid, var_17);
        var_20 = wp::load(var_18);
        var_19 = wp::copy(var_20);
        // pnt0 = site_xpos_in[worldid, id0]                                                      <L 4106>
        var_21 = wp::address(var_site_xpos_in, var_0, var_14);
        var_23 = wp::load(var_21);
        var_22 = wp::copy(var_23);
        // pnt1 = site_xpos_in[worldid, id1]                                                      <L 4107>
        var_24 = wp::address(var_site_xpos_in, var_0, var_19);
        var_26 = wp::load(var_24);
        var_25 = wp::copy(var_26);
        // dif = pnt1 - pnt0                                                                      <L 4108>
        var_27 = wp::sub(var_25, var_22);
        // vec, length = math.normalize_with_norm(dif)                                            <L 4109>
        normalize_with_norm_0(var_27, var_28, var_29);
        // wp.atomic_add(ten_length_out[worldid], tenid, length * pulley_scale)                   <L 4110>
        var_30 = wp::view(var_ten_length_out, var_0);
        var_31 = wp::mul(var_29, var_9);
        var_32 = wp::atomic_add(var_30, var_6, var_31);
        // if length < MJ_MINVAL:                                                                 <L 4112>
        var_34 = (var_29 < var_33);
        if (var_34) {
            // vec = wp.vec3(1.0, 0.0, 0.0)                                                       <L 4113>
            var_38 = wp::vec_t<3, wp::float32>(var_35, var_36, var_37);
        }
        var_39 = wp::where(var_34, var_38, var_28);
        // body0 = site_bodyid[id0]                                                               <L 4115>
        var_40 = wp::address(var_site_bodyid, var_14);
        var_42 = wp::load(var_40);
        var_41 = wp::copy(var_42);
        // body1 = site_bodyid[id1]                                                               <L 4116>
        var_43 = wp::address(var_site_bodyid, var_19);
        var_45 = wp::load(var_43);
        var_44 = wp::copy(var_45);
        // if body0 != body1:                                                                     <L 4117>
        var_46 = (var_41 != var_44);
        if (var_46) {
            // rownnz = ten_J_rownnz[tenid]                                                       <L 4118>
            var_47 = wp::address(var_ten_J_rownnz, var_6);
            var_49 = wp::load(var_47);
            var_48 = wp::copy(var_49);
            // rowadr = ten_J_rowadr[tenid]                                                       <L 4119>
            var_50 = wp::address(var_ten_J_rowadr, var_6);
            var_52 = wp::load(var_50);
            var_51 = wp::copy(var_52);
            // offset0 = pnt0 - subtree_com_in[worldid, body_rootid[body0]]                       <L 4120>
            var_53 = wp::address(var_body_rootid, var_41);
            var_54 = wp::load(var_53);
            var_55 = wp::address(var_subtree_com_in, var_0, var_54);
            var_57 = wp::load(var_55);
            var_56 = wp::sub(var_22, var_57);
            // offset1 = pnt1 - subtree_com_in[worldid, body_rootid[body1]]                       <L 4121>
            var_58 = wp::address(var_body_rootid, var_44);
            var_59 = wp::load(var_58);
            var_60 = wp::address(var_subtree_com_in, var_0, var_59);
            var_62 = wp::load(var_60);
            var_61 = wp::sub(var_25, var_62);
            // _accumulate_jac_chain(                                                             <L 4122>
            // body_parentid,                                                                     <L 4123>
            // body_dofnum,                                                                       <L 4124>
            // body_dofadr,                                                                       <L 4125>
            // ten_J_colind,                                                                      <L 4126>
            // cdof_in,                                                                           <L 4127>
            // offset0,                                                                           <L 4128>
            // vec,                                                                               <L 4129>
            // body0,                                                                             <L 4130>
            // rowadr,                                                                            <L 4131>
            // rownnz,                                                                            <L 4132>
            // -pulley_scale,                                                                     <L 4133>
            var_63 = wp::neg(var_9);
            // worldid,                                                                           <L 4134>
            // ten_J_out,                                                                         <L 4135>
            _accumulate_jac_chain_0(var_body_parentid, var_body_dofnum, var_body_dofadr, var_ten_J_colind, var_cdof_in, var_56, var_39, var_41, var_51, var_48, var_63, var_0, var_ten_J_out);
            // _accumulate_jac_chain(                                                             <L 4137>
            // body_parentid,                                                                     <L 4138>
            // body_dofnum,                                                                       <L 4139>
            // body_dofadr,                                                                       <L 4140>
            // ten_J_colind,                                                                      <L 4141>
            // cdof_in,                                                                           <L 4142>
            // offset1,                                                                           <L 4143>
            // vec,                                                                               <L 4144>
            // body1,                                                                             <L 4145>
            // rowadr,                                                                            <L 4146>
            // rownnz,                                                                            <L 4147>
            // pulley_scale,                                                                      <L 4148>
            // worldid,                                                                           <L 4149>
            // ten_J_out,                                                                         <L 4150>
            _accumulate_jac_chain_0(var_body_parentid, var_body_dofnum, var_body_dofadr, var_ten_J_colind, var_cdof_in, var_61, var_39, var_44, var_51, var_48, var_9, var_0, var_ten_J_out);
        }
    }
}



extern "C" __global__ void _transmission_body_moment_scale_e0bec4b0_cuda_kernel_forward(
    wp::launch_bounds_t<3> dim,
    wp::array_t<wp::int32> var_actuator_trntype_body_adr,
    wp::array_t<wp::int32> var_moment_rowadr_in,
    wp::array_t<wp::int32> var_actuator_trntype_body_ncon_in,
    wp::array_t<wp::float32> var_actuator_moment_out)
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
        wp::int32 var_4;
        wp::int32 var_5;
        const wp::int32 var_6 = 0;
        bool var_7;
        wp::int32* var_8;
        wp::int32 var_9;
        wp::int32 var_10;
        wp::int32* var_11;
        wp::int32 var_12;
        wp::int32 var_13;
        wp::int32 var_14;
        wp::float32* var_15;
        wp::float32 var_16;
        wp::float32 var_17;
        wp::float32 var_18;
        wp::float32 var_19;
        //---------
        // forward
        // def _transmission_body_moment_scale(                                                   <L 3199>
        // worldid, trnbodyid, dofid = wp.tid()                                                   <L 3209>
        builtin_tid3d(var_0, var_1, var_2);
        // ncon = actuator_trntype_body_ncon_in[worldid, trnbodyid]                               <L 3211>
        var_3 = wp::address(var_actuator_trntype_body_ncon_in, var_0, var_1);
        var_5 = wp::load(var_3);
        var_4 = wp::copy(var_5);
        // if ncon > 0:                                                                           <L 3213>
        var_7 = (var_4 > var_6);
        if (var_7) {
            // actid = actuator_trntype_body_adr[trnbodyid]                                       <L 3214>
            var_8 = wp::address(var_actuator_trntype_body_adr, var_1);
            var_10 = wp::load(var_8);
            var_9 = wp::copy(var_10);
            // rowadr = moment_rowadr_in[worldid, actid]                                          <L 3215>
            var_11 = wp::address(var_moment_rowadr_in, var_0, var_9);
            var_13 = wp::load(var_11);
            var_12 = wp::copy(var_13);
            // actuator_moment_out[worldid, rowadr + dofid] /= -float(ncon)                       <L 3216>
            var_14 = wp::add(var_12, var_2);
            var_15 = wp::address(var_actuator_moment_out, var_0, var_14);
            var_16 = wp::float(var_4);
            var_17 = wp::neg(var_16);
            var_19 = wp::load(var_15);
            var_18 = wp::div(var_19, var_17);
            wp::array_store(var_actuator_moment_out, var_0, var_14, var_18);
        }
    }
}



extern "C" __global__ void _geom_local_to_global_e28b714c_cuda_kernel_forward(
    wp::launch_bounds_t<2> dim,
    wp::array_t<wp::int32> var_body_rootid,
    wp::array_t<wp::int32> var_body_weldid,
    wp::array_t<wp::int32> var_body_mocapid,
    wp::array_t<wp::int32> var_geom_bodyid,
    wp::array_t<wp::vec_t<3, wp::float32>> var_geom_pos,
    wp::array_t<wp::quat_t<wp::float32>> var_geom_quat,
    wp::array_t<wp::vec_t<3, wp::float32>> var_xpos_in,
    wp::array_t<wp::quat_t<wp::float32>> var_xquat_in,
    wp::array_t<wp::vec_t<3, wp::float32>> var_geom_xpos_out,
    wp::array_t<wp::mat_t<3, 3, wp::float32>> var_geom_xmat_out)
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
        bool var_5;
        wp::int32* var_6;
        const wp::int32 var_7 = 0;
        bool var_8;
        wp::int32 var_9;
        wp::int32* var_10;
        wp::int32 var_11;
        wp::int32* var_12;
        const wp::int32 var_13 = -1;
        bool var_14;
        wp::int32 var_15;
        wp::vec_t<3, wp::float32>* var_16;
        wp::vec_t<3, wp::float32> var_17;
        wp::vec_t<3, wp::float32> var_18;
        wp::quat_t<wp::float32>* var_19;
        wp::quat_t<wp::float32> var_20;
        wp::quat_t<wp::float32> var_21;
        wp::shape_t* var_22;
        const wp::int32 var_23 = 0;
        wp::int32 var_24;
        wp::shape_t var_25;
        wp::int32 var_26;
        wp::vec_t<3, wp::float32>* var_27;
        wp::vec_t<3, wp::float32> var_28;
        wp::vec_t<3, wp::float32> var_29;
        wp::vec_t<3, wp::float32> var_30;
        wp::shape_t* var_31;
        const wp::int32 var_32 = 0;
        wp::int32 var_33;
        wp::shape_t var_34;
        wp::int32 var_35;
        wp::quat_t<wp::float32>* var_36;
        wp::quat_t<wp::float32> var_37;
        wp::quat_t<wp::float32> var_38;
        wp::mat_t<3, 3, wp::float32> var_39;
        //---------
        // forward
        // def _geom_local_to_global(                                                             <L 180>
        // worldid, geomid = wp.tid()                                                             <L 195>
        builtin_tid2d(var_0, var_1);
        // bodyid = geom_bodyid[geomid]                                                           <L 196>
        var_2 = wp::address(var_geom_bodyid, var_1);
        var_4 = wp::load(var_2);
        var_3 = wp::copy(var_4);
        // if body_weldid[bodyid] == 0 and body_mocapid[body_rootid[bodyid]] == -1:               <L 198>
        var_6 = wp::address(var_body_weldid, var_3);
        var_9 = wp::load(var_6);
        var_8 = (var_9 == var_7);
        var_5 = var_8;
        if (var_5) {
            var_10 = wp::address(var_body_rootid, var_3);
            var_11 = wp::load(var_10);
            var_12 = wp::address(var_body_mocapid, var_11);
            var_15 = wp::load(var_12);
            var_14 = (var_15 == var_13);
            var_5 = var_5 && var_14;
        }
        if (var_5) {
            // return                                                                             <L 201>
            continue;
        }
        // xpos = xpos_in[worldid, bodyid]                                                        <L 203>
        var_16 = wp::address(var_xpos_in, var_0, var_3);
        var_18 = wp::load(var_16);
        var_17 = wp::copy(var_18);
        // xquat = xquat_in[worldid, bodyid]                                                      <L 204>
        var_19 = wp::address(var_xquat_in, var_0, var_3);
        var_21 = wp::load(var_19);
        var_20 = wp::copy(var_21);
        // geom_xpos_out[worldid, geomid] = xpos + math.rot_vec_quat(geom_pos[worldid % geom_pos.shape[0], geomid], xquat)       <L 205>
        var_22 = &(var_geom_pos.shape);
        var_25 = wp::load(var_22);
        var_24 = wp::extract(var_25, var_23);
        var_26 = wp::mod(var_0, var_24);
        var_27 = wp::address(var_geom_pos, var_26, var_1);
        var_29 = wp::load(var_27);
        var_28 = rot_vec_quat_0(var_29, var_20);
        var_30 = wp::add(var_17, var_28);
        wp::array_store(var_geom_xpos_out, var_0, var_1, var_30);
        // geom_xmat_out[worldid, geomid] = math.quat_to_mat(math.mul_quat(xquat, geom_quat[worldid % geom_quat.shape[0], geomid]))       <L 206>
        var_31 = &(var_geom_quat.shape);
        var_34 = wp::load(var_31);
        var_33 = wp::extract(var_34, var_32);
        var_35 = wp::mod(var_0, var_33);
        var_36 = wp::address(var_geom_quat, var_35, var_1);
        var_38 = wp::load(var_36);
        var_37 = mul_quat_0(var_20, var_38);
        var_39 = quat_to_mat_0(var_37);
        wp::array_store(var_geom_xmat_out, var_0, var_1, var_39);
    }
}



extern "C" __global__ void _cam_local_to_global_ed67ebfa_cuda_kernel_forward(
    wp::launch_bounds_t<2> dim,
    wp::array_t<wp::int32> var_cam_mode,
    wp::array_t<wp::int32> var_cam_bodyid,
    wp::array_t<wp::int32> var_cam_targetbodyid,
    wp::array_t<wp::vec_t<3, wp::float32>> var_cam_pos,
    wp::array_t<wp::quat_t<wp::float32>> var_cam_quat,
    wp::array_t<wp::vec_t<3, wp::float32>> var_cam_poscom0,
    wp::array_t<wp::vec_t<3, wp::float32>> var_cam_pos0,
    wp::array_t<wp::mat_t<3, 3, wp::float32>> var_cam_mat0,
    wp::array_t<wp::vec_t<3, wp::float32>> var_xpos_in,
    wp::array_t<wp::quat_t<wp::float32>> var_xquat_in,
    wp::array_t<wp::vec_t<3, wp::float32>> var_subtree_com_in,
    wp::array_t<wp::vec_t<3, wp::float32>> var_cam_xpos_out,
    wp::array_t<wp::mat_t<3, 3, wp::float32>> var_cam_xmat_out)
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
        wp::int32* var_13;
        const wp::int32 var_14 = 3;
        bool var_15;
        wp::int32 var_16;
        wp::int32* var_17;
        const wp::int32 var_18 = 4;
        bool var_19;
        wp::int32 var_20;
        bool var_21;
        wp::int32* var_22;
        const wp::int32 var_23 = 0;
        bool var_24;
        wp::int32 var_25;
        wp::int32* var_26;
        wp::int32 var_27;
        wp::int32 var_28;
        wp::vec_t<3, wp::float32>* var_29;
        wp::vec_t<3, wp::float32> var_30;
        wp::vec_t<3, wp::float32> var_31;
        wp::quat_t<wp::float32>* var_32;
        wp::quat_t<wp::float32> var_33;
        wp::quat_t<wp::float32> var_34;
        wp::vec_t<3, wp::float32>* var_35;
        wp::vec_t<3, wp::float32> var_36;
        wp::vec_t<3, wp::float32> var_37;
        wp::vec_t<3, wp::float32> var_38;
        wp::quat_t<wp::float32>* var_39;
        wp::quat_t<wp::float32> var_40;
        wp::quat_t<wp::float32> var_41;
        wp::mat_t<3, 3, wp::float32> var_42;
        wp::int32* var_43;
        const wp::int32 var_44 = 1;
        bool var_45;
        wp::int32 var_46;
        wp::shape_t* var_47;
        const wp::int32 var_48 = 0;
        wp::int32 var_49;
        wp::shape_t var_50;
        wp::int32 var_51;
        wp::mat_t<3, 3, wp::float32>* var_52;
        wp::mat_t<3, 3, wp::float32> var_53;
        wp::mat_t<3, 3, wp::float32> var_54;
        wp::int32* var_55;
        wp::int32 var_56;
        wp::vec_t<3, wp::float32>* var_57;
        wp::vec_t<3, wp::float32> var_58;
        wp::vec_t<3, wp::float32> var_59;
        wp::shape_t* var_60;
        const wp::int32 var_61 = 0;
        wp::int32 var_62;
        wp::shape_t var_63;
        wp::int32 var_64;
        wp::vec_t<3, wp::float32>* var_65;
        wp::vec_t<3, wp::float32> var_66;
        wp::vec_t<3, wp::float32> var_67;
        wp::int32* var_68;
        const wp::int32 var_69 = 2;
        bool var_70;
        wp::int32 var_71;
        wp::shape_t* var_72;
        const wp::int32 var_73 = 0;
        wp::int32 var_74;
        wp::shape_t var_75;
        wp::int32 var_76;
        wp::mat_t<3, 3, wp::float32>* var_77;
        wp::mat_t<3, 3, wp::float32> var_78;
        wp::mat_t<3, 3, wp::float32> var_79;
        wp::int32* var_80;
        wp::int32 var_81;
        wp::vec_t<3, wp::float32>* var_82;
        wp::shape_t* var_83;
        const wp::int32 var_84 = 0;
        wp::int32 var_85;
        wp::shape_t var_86;
        wp::int32 var_87;
        wp::vec_t<3, wp::float32>* var_88;
        wp::vec_t<3, wp::float32> var_89;
        wp::vec_t<3, wp::float32> var_90;
        wp::vec_t<3, wp::float32> var_91;
        bool var_92;
        wp::int32* var_93;
        const wp::int32 var_94 = 3;
        bool var_95;
        wp::int32 var_96;
        wp::int32* var_97;
        const wp::int32 var_98 = 4;
        bool var_99;
        wp::int32 var_100;
        wp::int32* var_101;
        wp::int32 var_102;
        wp::int32 var_103;
        wp::vec_t<3, wp::float32>* var_104;
        wp::vec_t<3, wp::float32> var_105;
        wp::vec_t<3, wp::float32> var_106;
        wp::quat_t<wp::float32>* var_107;
        wp::quat_t<wp::float32> var_108;
        wp::quat_t<wp::float32> var_109;
        wp::vec_t<3, wp::float32>* var_110;
        wp::vec_t<3, wp::float32> var_111;
        wp::vec_t<3, wp::float32> var_112;
        wp::vec_t<3, wp::float32> var_113;
        wp::int32* var_114;
        wp::int32 var_115;
        wp::vec_t<3, wp::float32>* var_116;
        wp::vec_t<3, wp::float32> var_117;
        wp::vec_t<3, wp::float32> var_118;
        wp::int32* var_119;
        const wp::int32 var_120 = 4;
        bool var_121;
        wp::int32 var_122;
        wp::int32* var_123;
        wp::int32 var_124;
        wp::vec_t<3, wp::float32>* var_125;
        wp::vec_t<3, wp::float32> var_126;
        wp::vec_t<3, wp::float32> var_127;
        wp::vec_t<3, wp::float32> var_128;
        wp::vec_t<3, wp::float32>* var_129;
        wp::vec_t<3, wp::float32> var_130;
        wp::vec_t<3, wp::float32> var_131;
        wp::vec_t<3, wp::float32> var_132;
        const wp::float32 var_133 = 0.0;
        const wp::float32 var_134 = 0.0;
        const wp::float32 var_135 = 1.0;
        wp::vec_t<3, wp::float32> var_136;
        wp::vec_t<3, wp::float32> var_137;
        wp::vec_t<3, wp::float32> var_138;
        wp::vec_t<3, wp::float32> var_139;
        wp::vec_t<3, wp::float32> var_140;
        const wp::int32 var_141 = 0;
        wp::float32 var_142;
        const wp::int32 var_143 = 0;
        wp::float32 var_144;
        const wp::int32 var_145 = 0;
        wp::float32 var_146;
        const wp::int32 var_147 = 1;
        wp::float32 var_148;
        const wp::int32 var_149 = 1;
        wp::float32 var_150;
        const wp::int32 var_151 = 1;
        wp::float32 var_152;
        const wp::int32 var_153 = 2;
        wp::float32 var_154;
        const wp::int32 var_155 = 2;
        wp::float32 var_156;
        const wp::int32 var_157 = 2;
        wp::float32 var_158;
        wp::mat_t<3, 3, wp::float32> var_159;
        wp::int32* var_160;
        wp::int32 var_161;
        wp::int32 var_162;
        wp::vec_t<3, wp::float32>* var_163;
        wp::vec_t<3, wp::float32> var_164;
        wp::vec_t<3, wp::float32> var_165;
        wp::quat_t<wp::float32>* var_166;
        wp::quat_t<wp::float32> var_167;
        wp::quat_t<wp::float32> var_168;
        wp::vec_t<3, wp::float32>* var_169;
        wp::vec_t<3, wp::float32> var_170;
        wp::vec_t<3, wp::float32> var_171;
        wp::vec_t<3, wp::float32> var_172;
        wp::quat_t<wp::float32>* var_173;
        wp::quat_t<wp::float32> var_174;
        wp::quat_t<wp::float32> var_175;
        wp::mat_t<3, 3, wp::float32> var_176;
        wp::int32 var_177;
        wp::vec_t<3, wp::float32> var_178;
        wp::quat_t<wp::float32> var_179;
        wp::int32 var_180;
        wp::vec_t<3, wp::float32> var_181;
        wp::quat_t<wp::float32> var_182;
        //---------
        // forward
        // def _cam_local_to_global(                                                              <L 829>
        // worldid, camid = wp.tid()                                                              <L 847>
        builtin_tid2d(var_0, var_1);
        // cam_pos_id = worldid % cam_pos.shape[0]                                                <L 848>
        var_2 = &(var_cam_pos.shape);
        var_5 = wp::load(var_2);
        var_4 = wp::extract(var_5, var_3);
        var_6 = wp::mod(var_0, var_4);
        // cam_quat_id = worldid % cam_quat.shape[0]                                              <L 849>
        var_7 = &(var_cam_quat.shape);
        var_10 = wp::load(var_7);
        var_9 = wp::extract(var_10, var_8);
        var_11 = wp::mod(var_0, var_9);
        // is_target_cam = (cam_mode[camid] == CamLightType.TARGETBODY) or (cam_mode[camid] == CamLightType.TARGETBODYCOM)       <L 850>
        var_13 = wp::address(var_cam_mode, var_1);
        var_16 = wp::load(var_13);
        var_15 = (var_16 == var_14);
        var_12 = var_15;
        if (!var_12) {
            var_17 = wp::address(var_cam_mode, var_1);
            var_20 = wp::load(var_17);
            var_19 = (var_20 == var_18);
            var_12 = var_12 || var_19;
        }
        // invalid_target = is_target_cam and (cam_targetbodyid[camid] < 0)                       <L 851>
        var_21 = var_12;
        if (var_21) {
            var_22 = wp::address(var_cam_targetbodyid, var_1);
            var_25 = wp::load(var_22);
            var_24 = (var_25 < var_23);
            var_21 = var_21 && var_24;
        }
        // if invalid_target:                                                                     <L 852>
        if (var_21) {
            // bodyid = cam_bodyid[camid]                                                         <L 853>
            var_26 = wp::address(var_cam_bodyid, var_1);
            var_28 = wp::load(var_26);
            var_27 = wp::copy(var_28);
            // xpos = xpos_in[worldid, bodyid]                                                    <L 854>
            var_29 = wp::address(var_xpos_in, var_0, var_27);
            var_31 = wp::load(var_29);
            var_30 = wp::copy(var_31);
            // xquat = xquat_in[worldid, bodyid]                                                  <L 855>
            var_32 = wp::address(var_xquat_in, var_0, var_27);
            var_34 = wp::load(var_32);
            var_33 = wp::copy(var_34);
            // cam_xpos_out[worldid, camid] = xpos + math.rot_vec_quat(cam_pos[cam_pos_id, camid], xquat)       <L 856>
            var_35 = wp::address(var_cam_pos, var_6, var_1);
            var_37 = wp::load(var_35);
            var_36 = rot_vec_quat_0(var_37, var_33);
            var_38 = wp::add(var_30, var_36);
            wp::array_store(var_cam_xpos_out, var_0, var_1, var_38);
            // cam_xmat_out[worldid, camid] = math.quat_to_mat(math.mul_quat(xquat, cam_quat[cam_quat_id, camid]))       <L 857>
            var_39 = wp::address(var_cam_quat, var_11, var_1);
            var_41 = wp::load(var_39);
            var_40 = mul_quat_0(var_33, var_41);
            var_42 = quat_to_mat_0(var_40);
            wp::array_store(var_cam_xmat_out, var_0, var_1, var_42);
        }
        if (!var_21) {
            // elif cam_mode[camid] == CamLightType.TRACK:                                        <L 858>
            var_43 = wp::address(var_cam_mode, var_1);
            var_46 = wp::load(var_43);
            var_45 = (var_46 == var_44);
            if (var_45) {
                // cam_xmat_out[worldid, camid] = cam_mat0[worldid % cam_mat0.shape[0], camid]       <L 859>
                var_47 = &(var_cam_mat0.shape);
                var_50 = wp::load(var_47);
                var_49 = wp::extract(var_50, var_48);
                var_51 = wp::mod(var_0, var_49);
                var_52 = wp::address(var_cam_mat0, var_51, var_1);
                var_54 = wp::load(var_52);
                var_53 = wp::copy(var_54);
                wp::array_store(var_cam_xmat_out, var_0, var_1, var_53);
                // body_xpos = xpos_in[worldid, cam_bodyid[camid]]                                <L 860>
                var_55 = wp::address(var_cam_bodyid, var_1);
                var_56 = wp::load(var_55);
                var_57 = wp::address(var_xpos_in, var_0, var_56);
                var_59 = wp::load(var_57);
                var_58 = wp::copy(var_59);
                // cam_xpos_out[worldid, camid] = body_xpos + cam_pos0[worldid % cam_pos0.shape[0], camid]       <L 861>
                var_60 = &(var_cam_pos0.shape);
                var_63 = wp::load(var_60);
                var_62 = wp::extract(var_63, var_61);
                var_64 = wp::mod(var_0, var_62);
                var_65 = wp::address(var_cam_pos0, var_64, var_1);
                var_67 = wp::load(var_65);
                var_66 = wp::add(var_58, var_67);
                wp::array_store(var_cam_xpos_out, var_0, var_1, var_66);
            }
            if (!var_45) {
                // elif cam_mode[camid] == CamLightType.TRACKCOM:                                 <L 862>
                var_68 = wp::address(var_cam_mode, var_1);
                var_71 = wp::load(var_68);
                var_70 = (var_71 == var_69);
                if (var_70) {
                    // cam_xmat_out[worldid, camid] = cam_mat0[worldid % cam_mat0.shape[0], camid]       <L 863>
                    var_72 = &(var_cam_mat0.shape);
                    var_75 = wp::load(var_72);
                    var_74 = wp::extract(var_75, var_73);
                    var_76 = wp::mod(var_0, var_74);
                    var_77 = wp::address(var_cam_mat0, var_76, var_1);
                    var_79 = wp::load(var_77);
                    var_78 = wp::copy(var_79);
                    wp::array_store(var_cam_xmat_out, var_0, var_1, var_78);
                    // cam_xpos_out[worldid, camid] = (                                           <L 864>
                    // subtree_com_in[worldid, cam_bodyid[camid]] + cam_poscom0[worldid % cam_poscom0.shape[0], camid]       <L 865>
                    var_80 = wp::address(var_cam_bodyid, var_1);
                    var_81 = wp::load(var_80);
                    var_82 = wp::address(var_subtree_com_in, var_0, var_81);
                    var_83 = &(var_cam_poscom0.shape);
                    var_86 = wp::load(var_83);
                    var_85 = wp::extract(var_86, var_84);
                    var_87 = wp::mod(var_0, var_85);
                    var_88 = wp::address(var_cam_poscom0, var_87, var_1);
                    var_90 = wp::load(var_82);
                    var_91 = wp::load(var_88);
                    var_89 = wp::add(var_90, var_91);
                    // cam_xpos_out[worldid, camid] = (                                           <L 864>
                    wp::array_store(var_cam_xpos_out, var_0, var_1, var_89);
                }
                if (!var_70) {
                    // elif cam_mode[camid] == CamLightType.TARGETBODY or cam_mode[camid] == CamLightType.TARGETBODYCOM:       <L 867>
                    var_93 = wp::address(var_cam_mode, var_1);
                    var_96 = wp::load(var_93);
                    var_95 = (var_96 == var_94);
                    var_92 = var_95;
                    if (!var_92) {
                        var_97 = wp::address(var_cam_mode, var_1);
                        var_100 = wp::load(var_97);
                        var_99 = (var_100 == var_98);
                        var_92 = var_92 || var_99;
                    }
                    if (var_92) {
                        // bodyid = cam_bodyid[camid]                                             <L 868>
                        var_101 = wp::address(var_cam_bodyid, var_1);
                        var_103 = wp::load(var_101);
                        var_102 = wp::copy(var_103);
                        // xpos = xpos_in[worldid, bodyid]                                        <L 869>
                        var_104 = wp::address(var_xpos_in, var_0, var_102);
                        var_106 = wp::load(var_104);
                        var_105 = wp::copy(var_106);
                        // xquat = xquat_in[worldid, bodyid]                                      <L 870>
                        var_107 = wp::address(var_xquat_in, var_0, var_102);
                        var_109 = wp::load(var_107);
                        var_108 = wp::copy(var_109);
                        // cam_xpos_out[worldid, camid] = xpos + math.rot_vec_quat(cam_pos[cam_pos_id, camid], xquat)       <L 871>
                        var_110 = wp::address(var_cam_pos, var_6, var_1);
                        var_112 = wp::load(var_110);
                        var_111 = rot_vec_quat_0(var_112, var_108);
                        var_113 = wp::add(var_105, var_111);
                        wp::array_store(var_cam_xpos_out, var_0, var_1, var_113);
                        // pos = xpos_in[worldid, cam_targetbodyid[camid]]                        <L 872>
                        var_114 = wp::address(var_cam_targetbodyid, var_1);
                        var_115 = wp::load(var_114);
                        var_116 = wp::address(var_xpos_in, var_0, var_115);
                        var_118 = wp::load(var_116);
                        var_117 = wp::copy(var_118);
                        // if cam_mode[camid] == CamLightType.TARGETBODYCOM:                      <L 873>
                        var_119 = wp::address(var_cam_mode, var_1);
                        var_122 = wp::load(var_119);
                        var_121 = (var_122 == var_120);
                        if (var_121) {
                            // pos = subtree_com_in[worldid, cam_targetbodyid[camid]]             <L 874>
                            var_123 = wp::address(var_cam_targetbodyid, var_1);
                            var_124 = wp::load(var_123);
                            var_125 = wp::address(var_subtree_com_in, var_0, var_124);
                            var_127 = wp::load(var_125);
                            var_126 = wp::copy(var_127);
                        }
                        var_128 = wp::where(var_121, var_126, var_117);
                        // mat_3 = wp.normalize(cam_xpos_out[worldid, camid] - pos)               <L 876>
                        var_129 = wp::address(var_cam_xpos_out, var_0, var_1);
                        var_131 = wp::load(var_129);
                        var_130 = wp::sub(var_131, var_128);
                        var_132 = wp::normalize(var_130);
                        // mat_1 = wp.normalize(wp.cross(wp.vec3(0.0, 0.0, 1.0), mat_3))          <L 878>
                        var_136 = wp::vec_t<3, wp::float32>(var_133, var_134, var_135);
                        var_137 = wp::cross(var_136, var_132);
                        var_138 = wp::normalize(var_137);
                        // mat_2 = wp.normalize(wp.cross(mat_3, mat_1))                           <L 879>
                        var_139 = wp::cross(var_132, var_138);
                        var_140 = wp::normalize(var_139);
                        // cam_xmat_out[worldid, camid] = wp.mat33(                               <L 881>
                        // mat_1[0], mat_2[0], mat_3[0],                                          <L 882>
                        var_142 = wp::extract(var_138, var_141);
                        var_144 = wp::extract(var_140, var_143);
                        var_146 = wp::extract(var_132, var_145);
                        // mat_1[1], mat_2[1], mat_3[1],                                          <L 883>
                        var_148 = wp::extract(var_138, var_147);
                        var_150 = wp::extract(var_140, var_149);
                        var_152 = wp::extract(var_132, var_151);
                        // mat_1[2], mat_2[2], mat_3[2]                                           <L 884>
                        var_154 = wp::extract(var_138, var_153);
                        var_156 = wp::extract(var_140, var_155);
                        var_158 = wp::extract(var_132, var_157);
                        var_159 = wp::mat_t<3, 3, wp::float32>(var_142, var_144, var_146, var_148, var_150, var_152, var_154, var_156, var_158);
                        // cam_xmat_out[worldid, camid] = wp.mat33(                               <L 881>
                        wp::array_store(var_cam_xmat_out, var_0, var_1, var_159);
                    }
                    if (!var_92) {
                        // bodyid = cam_bodyid[camid]                                             <L 888>
                        var_160 = wp::address(var_cam_bodyid, var_1);
                        var_162 = wp::load(var_160);
                        var_161 = wp::copy(var_162);
                        // xpos = xpos_in[worldid, bodyid]                                        <L 889>
                        var_163 = wp::address(var_xpos_in, var_0, var_161);
                        var_165 = wp::load(var_163);
                        var_164 = wp::copy(var_165);
                        // xquat = xquat_in[worldid, bodyid]                                      <L 890>
                        var_166 = wp::address(var_xquat_in, var_0, var_161);
                        var_168 = wp::load(var_166);
                        var_167 = wp::copy(var_168);
                        // cam_xpos_out[worldid, camid] = xpos + math.rot_vec_quat(cam_pos[cam_pos_id, camid], xquat)       <L 891>
                        var_169 = wp::address(var_cam_pos, var_6, var_1);
                        var_171 = wp::load(var_169);
                        var_170 = rot_vec_quat_0(var_171, var_167);
                        var_172 = wp::add(var_164, var_170);
                        wp::array_store(var_cam_xpos_out, var_0, var_1, var_172);
                        // cam_xmat_out[worldid, camid] = math.quat_to_mat(math.mul_quat(xquat, cam_quat[cam_quat_id, camid]))       <L 892>
                        var_173 = wp::address(var_cam_quat, var_11, var_1);
                        var_175 = wp::load(var_173);
                        var_174 = mul_quat_0(var_167, var_175);
                        var_176 = quat_to_mat_0(var_174);
                        wp::array_store(var_cam_xmat_out, var_0, var_1, var_176);
                    }
                    var_177 = wp::where(var_92, var_102, var_161);
                    var_178 = wp::where(var_92, var_105, var_164);
                    var_179 = wp::where(var_92, var_108, var_167);
                }
            }
        }
        var_180 = wp::where(var_21, var_27, var_177);
        var_181 = wp::where(var_21, var_30, var_178);
        var_182 = wp::where(var_21, var_33, var_179);
    }
}



extern "C" __global__ void _cfrc_ext_f67cbe30_cuda_kernel_forward(
    wp::launch_bounds_t<2> dim,
    wp::array_t<wp::int32> var_body_rootid,
    wp::array_t<wp::vec_t<6, wp::float32>> var_xfrc_applied_in,
    wp::array_t<wp::vec_t<3, wp::float32>> var_xipos_in,
    wp::array_t<wp::vec_t<3, wp::float32>> var_subtree_com_in,
    wp::array_t<wp::vec_t<6, wp::float32>> var_cfrc_ext_out)
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
        const wp::int32 var_2 = 0;
        bool var_3;
        const wp::float32 var_4 = 0.0;
        const wp::float32 var_5 = 0.0;
        const wp::float32 var_6 = 0.0;
        const wp::float32 var_7 = 0.0;
        const wp::float32 var_8 = 0.0;
        const wp::float32 var_9 = 0.0;
        wp::vec_t<6, wp::float32> var_10;
        const wp::int32 var_11 = 0;
        wp::vec_t<6, wp::float32>* var_12;
        wp::vec_t<6, wp::float32> var_13;
        wp::vec_t<6, wp::float32> var_14;
        wp::int32* var_15;
        wp::int32 var_16;
        wp::vec_t<3, wp::float32>* var_17;
        wp::vec_t<3, wp::float32> var_18;
        wp::vec_t<3, wp::float32> var_19;
        wp::vec_t<3, wp::float32>* var_20;
        wp::vec_t<3, wp::float32> var_21;
        wp::vec_t<3, wp::float32> var_22;
        wp::vec_t<3, wp::float32> var_23;
        wp::vec_t<6, wp::float32> var_24;
        //---------
        // forward
        // def _cfrc_ext(                                                                         <L 1489>
        // worldid, bodyid = wp.tid()                                                             <L 1499>
        builtin_tid2d(var_0, var_1);
        // if bodyid == 0:                                                                        <L 1500>
        var_3 = (var_1 == var_2);
        if (var_3) {
            // cfrc_ext_out[worldid, 0] = wp.spatial_vector(0.0, 0.0, 0.0, 0.0, 0.0, 0.0)         <L 1501>
            var_10 = wp::vec_t<6, wp::float32>({var_4, var_5, var_6, var_7, var_8, var_9});
            wp::array_store(var_cfrc_ext_out, var_0, var_11, var_10);
        }
        if (!var_3) {
            // xfrc_applied = xfrc_applied_in[worldid, bodyid]                                    <L 1503>
            var_12 = wp::address(var_xfrc_applied_in, var_0, var_1);
            var_14 = wp::load(var_12);
            var_13 = wp::copy(var_14);
            // subtree_com = subtree_com_in[worldid, body_rootid[bodyid]]                         <L 1504>
            var_15 = wp::address(var_body_rootid, var_1);
            var_16 = wp::load(var_15);
            var_17 = wp::address(var_subtree_com_in, var_0, var_16);
            var_19 = wp::load(var_17);
            var_18 = wp::copy(var_19);
            // xipos = xipos_in[worldid, bodyid]                                                  <L 1505>
            var_20 = wp::address(var_xipos_in, var_0, var_1);
            var_22 = wp::load(var_20);
            var_21 = wp::copy(var_22);
            // cfrc_ext_out[worldid, bodyid] = support.transform_force(xfrc_applied, subtree_com - xipos)       <L 1506>
            var_23 = wp::sub(var_18, var_21);
            var_24 = transform_force_1(var_13, var_23);
            wp::array_store(var_cfrc_ext_out, var_0, var_1, var_24);
        }
    }
}



extern "C" __global__ void _cfrc_ext_tendon_actuator_f699e449_cuda_kernel_forward(
    wp::launch_bounds_t<2> dim,
    wp::array_t<wp::int32> var_actuator_trntype,
    wp::array_t<wp::vec_t<2, wp::int32>> var_actuator_trnid,
    wp::array_t<wp::vec_t<6, wp::float32>> var_actuator_gear,
    wp::array_t<wp::float32> var_actuator_force_in,
    wp::array_t<wp::float32> var_ten_frc_out)
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
        const wp::int32 var_3 = 3;
        bool var_4;
        wp::int32 var_5;
        wp::float32* var_6;
        wp::float32 var_7;
        wp::float32 var_8;
        const wp::float32 var_9 = 0.0;
        bool var_10;
        wp::vec_t<2, wp::int32>* var_11;
        const wp::int32 var_12 = 0;
        wp::int32 var_13;
        wp::vec_t<2, wp::int32> var_14;
        wp::shape_t* var_15;
        const wp::int32 var_16 = 0;
        wp::int32 var_17;
        wp::shape_t var_18;
        wp::int32 var_19;
        wp::vec_t<6, wp::float32>* var_20;
        const wp::int32 var_21 = 0;
        wp::float32 var_22;
        wp::vec_t<6, wp::float32> var_23;
        wp::array_t<wp::float32> var_24;
        wp::float32 var_25;
        wp::float32 var_26;
        //---------
        // forward
        // def _cfrc_ext_tendon_actuator(                                                         <L 1791>
        // worldid, actid = wp.tid()                                                              <L 1801>
        builtin_tid2d(var_0, var_1);
        // if actuator_trntype[actid] != TrnType.TENDON:                                          <L 1802>
        var_2 = wp::address(var_actuator_trntype, var_1);
        var_5 = wp::load(var_2);
        var_4 = (var_5 != var_3);
        if (var_4) {
            // return                                                                             <L 1803>
            continue;
        }
        // force = actuator_force_in[worldid, actid]                                              <L 1805>
        var_6 = wp::address(var_actuator_force_in, var_0, var_1);
        var_8 = wp::load(var_6);
        var_7 = wp::copy(var_8);
        // if force == 0.0:                                                                       <L 1806>
        var_10 = (var_7 == var_9);
        if (var_10) {
            // return                                                                             <L 1807>
            continue;
        }
        // tenid = actuator_trnid[actid][0]                                                       <L 1809>
        var_11 = wp::address(var_actuator_trnid, var_1);
        var_14 = wp::load(var_11);
        var_13 = wp::extract(var_14, var_12);
        // gear = actuator_gear[worldid % actuator_gear.shape[0], actid][0]                       <L 1810>
        var_15 = &(var_actuator_gear.shape);
        var_18 = wp::load(var_15);
        var_17 = wp::extract(var_18, var_16);
        var_19 = wp::mod(var_0, var_17);
        var_20 = wp::address(var_actuator_gear, var_19, var_1);
        var_23 = wp::load(var_20);
        var_22 = wp::extract(var_23, var_21);
        // wp.atomic_add(ten_frc_out[worldid], tenid, force * gear)                               <L 1811>
        var_24 = wp::view(var_ten_frc_out, var_0);
        var_25 = wp::mul(var_7, var_22);
        var_26 = wp::atomic_add(var_24, var_13, var_25);
    }
}



extern "C" __global__ void _cacc_branch_ffe1d6c4_cuda_kernel_forward(
    wp::launch_bounds_t<2> dim,
    wp::array_t<wp::int32> var_body_parentid,
    wp::array_t<wp::int32> var_body_dofnum,
    wp::array_t<wp::int32> var_body_dofadr,
    wp::array_t<wp::int32> var_body_branches,
    wp::array_t<wp::int32> var_body_branch_start,
    wp::array_t<wp::float32> var_qvel_in,
    wp::array_t<wp::float32> var_qacc_in,
    wp::array_t<wp::vec_t<6, wp::float32>> var_cdof_in,
    wp::array_t<wp::vec_t<6, wp::float32>> var_cdof_dot_in,
    bool var_flg_acc,
    wp::array_t<wp::vec_t<6, wp::float32>> var_cacc_out)
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
        const wp::int32 var_5 = 1;
        wp::int32 var_6;
        wp::int32* var_7;
        wp::int32 var_8;
        wp::int32 var_9;
        wp::int32* var_10;
        wp::int32 var_11;
        wp::int32 var_12;
        wp::int32* var_13;
        wp::int32 var_14;
        wp::int32 var_15;
        wp::vec_t<6, wp::float32>* var_16;
        wp::vec_t<6, wp::float32> var_17;
        wp::vec_t<6, wp::float32> var_18;
        wp::range_t var_19;
        wp::int32 var_20;
        wp::int32* var_21;
        wp::int32 var_22;
        wp::int32 var_23;
        wp::int32* var_24;
        wp::int32 var_25;
        wp::int32 var_26;
        wp::int32* var_27;
        wp::int32 var_28;
        wp::int32 var_29;
        wp::range_t var_30;
        wp::int32 var_31;
        wp::int32 var_32;
        wp::vec_t<6, wp::float32>* var_33;
        wp::int32 var_34;
        wp::float32* var_35;
        wp::vec_t<6, wp::float32> var_36;
        wp::vec_t<6, wp::float32> var_37;
        wp::float32 var_38;
        wp::vec_t<6, wp::float32> var_39;
        wp::int32 var_40;
        wp::vec_t<6, wp::float32>* var_41;
        wp::int32 var_42;
        wp::float32* var_43;
        wp::vec_t<6, wp::float32> var_44;
        wp::vec_t<6, wp::float32> var_45;
        wp::float32 var_46;
        wp::vec_t<6, wp::float32> var_47;
        wp::vec_t<6, wp::float32> var_48;
        //---------
        // forward
        // def _cacc_branch(                                                                      <L 1342>
        // worldid, branchid = wp.tid()                                                           <L 1359>
        builtin_tid2d(var_0, var_1);
        // start = body_branch_start[branchid]                                                    <L 1361>
        var_2 = wp::address(var_body_branch_start, var_1);
        var_4 = wp::load(var_2);
        var_3 = wp::copy(var_4);
        // end = body_branch_start[branchid + 1]                                                  <L 1362>
        var_6 = wp::add(var_1, var_5);
        var_7 = wp::address(var_body_branch_start, var_6);
        var_9 = wp::load(var_7);
        var_8 = wp::copy(var_9);
        // bodyid = body_branches[start]                                                          <L 1364>
        var_10 = wp::address(var_body_branches, var_3);
        var_12 = wp::load(var_10);
        var_11 = wp::copy(var_12);
        // pid = body_parentid[bodyid]                                                            <L 1365>
        var_13 = wp::address(var_body_parentid, var_11);
        var_15 = wp::load(var_13);
        var_14 = wp::copy(var_15);
        // local_cacc = cacc_out[worldid, pid]                                                    <L 1366>
        var_16 = wp::address(var_cacc_out, var_0, var_14);
        var_18 = wp::load(var_16);
        var_17 = wp::copy(var_18);
        // for i in range(start, end):                                                            <L 1367>
        var_19 = wp::range(var_3, var_8);
        start_for_0:;
            if (iter_cmp(var_19) == 0) goto end_for_0;
            var_20 = wp::iter_next(var_19);
            // bodyid = body_branches[i]                                                          <L 1368>
            var_21 = wp::address(var_body_branches, var_20);
            var_23 = wp::load(var_21);
            var_22 = wp::copy(var_23);
            // dofnum = body_dofnum[bodyid]                                                       <L 1369>
            var_24 = wp::address(var_body_dofnum, var_22);
            var_26 = wp::load(var_24);
            var_25 = wp::copy(var_26);
            // dofadr = body_dofadr[bodyid]                                                       <L 1370>
            var_27 = wp::address(var_body_dofadr, var_22);
            var_29 = wp::load(var_27);
            var_28 = wp::copy(var_29);
            // for j in range(dofnum):                                                            <L 1371>
            var_30 = wp::range(var_25);
            start_for_2:;
                if (iter_cmp(var_30) == 0) goto end_for_2;
                var_31 = wp::iter_next(var_30);
                // local_cacc += cdof_dot_in[worldid, dofadr + j] * qvel_in[worldid, dofadr + j]       <L 1372>
                var_32 = wp::add(var_28, var_31);
                var_33 = wp::address(var_cdof_dot_in, var_0, var_32);
                var_34 = wp::add(var_28, var_31);
                var_35 = wp::address(var_qvel_in, var_0, var_34);
                var_37 = wp::load(var_33);
                var_38 = wp::load(var_35);
                var_36 = wp::mul(var_37, var_38);
                var_39 = wp::add(var_17, var_36);
                // if flg_acc:                                                                    <L 1373>
                if (var_flg_acc) {
                    // local_cacc += cdof_in[worldid, dofadr + j] * qacc_in[worldid, dofadr + j]       <L 1374>
                    var_40 = wp::add(var_28, var_31);
                    var_41 = wp::address(var_cdof_in, var_0, var_40);
                    var_42 = wp::add(var_28, var_31);
                    var_43 = wp::address(var_qacc_in, var_0, var_42);
                    var_45 = wp::load(var_41);
                    var_46 = wp::load(var_43);
                    var_44 = wp::mul(var_45, var_46);
                    var_47 = wp::add(var_39, var_44);
                }
                var_48 = wp::where(var_flg_acc, var_47, var_39);
                wp::assign(var_17, var_48);
                goto start_for_2;
            end_for_2:;
            // cacc_out[worldid, bodyid] = local_cacc                                             <L 1375>
            wp::array_store(var_cacc_out, var_0, var_22, var_17);
            wp::assign(var_11, var_22);
            goto start_for_0;
        end_for_0:;
    }
}

