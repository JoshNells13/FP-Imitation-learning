# Humanoid Whole-Body Motion Imitation Learning — Booster K1 (22-DoF)

Proyek ini berisi pipeline *end-to-end* **Whole-Body Motion Imitation Learning** untuk robot humanoid **Booster K1 (22-DoF)** menggunakan **Python + MuJoCo + MJLab + PyTorch** (tanpa ROS 2, tanpa Isaac Sim, dan tanpa Isaac Lab), serta terintegrasi langsung dengan **Website Monitoring & Control Platform** pada repository terpisah: **[FP-Website-Monitoring-Control-imitation-learning](https://github.com/JoshNells13/FP-Website-Monitoring-Control-imitation-learning)** (folder lokal [`../monitoring-control-imitation`](../monitoring-control-imitation)).

Tiga modul utama yang digabungkan di dalam proyek ini:
1. **[GMR (`YanjieZe/GMR`)](https://github.com/YanjieZe/GMR)** — *General Motion Retargeting* berbasis *two-stage differential inverse kinematics* (`Mink` + `DAQP`) untuk memetakan gerakan manusia (`.bvh` Xsens / LAFAN1) ke sendi robot **Booster K1**.
2. **[BeyondMimic (`HybridRobotics/whole_body_tracking`)](https://github.com/HybridRobotics/whole_body_tracking)** — Metodologi *whole-body tracking* (*anchor-yaw-aligned relative keybody tracking*, *asymmetric actor-critic observations*, *6 exponential tracking rewards*, *adaptive failure-bin curriculum sampling*, dan *task-space height terminations*).
3. **[MJLab + MuJoCo-Warp (`mujocolab/mjlab`)](https://github.com/mujocolab/mjlab)** — Simulator fisika MuJoCo berbasis GPU (`mujoco-warp`) dan algoritma *Proximal Policy Optimization* (**PPO**) lewat `rsl-rl-lib`.

---

## Spesifikasi Model Robot & Motion Reference

| Komponen | Detail |
| :--- | :--- |
| **Model Robot** | **Booster K1** (`booster_k1`, `22-DoF`: `2` sendi kepala + `8` sendi lengan + `12` sendi kaki) |
| **Root / Anchor Body** | `Trunk` (tinggi berdiri $\approx 0.53\text{ m}$, skala retargeting `0.56`) |
| **Tracked Keybodies** | `14` link tubuh (`Trunk`, `Head_2`, bahu, lengan atas, tangan, pinggul, betis, dan telapak kaki) |
| **Dimensi Observasi & Aksi** | Actor: `125 dim` \| Critic (Privileged): `251 dim` \| Action: `22 dim` |
| **Motion Reference** | `data/motions/k1_kick.npz` (*Martial-Arts Front Kick*, `149` frame @ `50 FPS`, durasi `2.98 detik`) |

---

## Tutorial Lengkap Mulai Dari 0: Dari Asal Aset, Konversi Data, Training, hingga Evaluasi

Bagian ini mengajarkan **dari nol (0)** bagaimana seluruh aset robot dan gerakan manusia disiapkan, dikonversi tahap demi tahap, dilatih dengan PPO, hingga dievaluasi.

```text
[TAHAP 0: ASET DASAR]
  ├── Model 3D Robot Booster K1 : third_party/GMR/assets/booster_k1/k1_mjlab.xml + 23 Mesh .STL
  └── Mocap Manusia Mentah      : third_party/GMR/assets/xsens_bvh_test/251021_04_boxing_120Hz_cm_3DsMax.bvh
          │
          ▼  (scripts/generate_kick_bvh.py)
[TAHAP 1: FILE BVH MANUSIA]
  └── data/bvh/human_kick_xsens.bvh (Kerangka Manusia 1.78m, 23 Sendi, 360 Frame @ 120 Hz)
          │
          ▼  (scripts/retarget_bvh_gmr.py — Two-Stage Mink + DAQP IK & Ground Z-Alignment)
[TAHAP 2: RETARGETED ROBOT MOTION (30 FPS)]
  ├── data/retargeted/k1_kick.pkl (Dictionary Python: root_pos, root_rot xyzw, dof_pos)
  ├── data/retargeted/k1_kick.csv (Matriks 90 Frame x 29 Kolom: [x,y,z, qx,qy,qz,qw, j0..j21])
  └── outputs/videos/gmr_retargeted_k1_kick.mp4
          │
          ▼  (scripts/convert_csv_to_npz.py — Interpolasi Slerp 50 FPS + MuJoCo Forward Kinematics)
[TAHAP 3: MJLAB MOTION REFERENCE (50 FPS)]
  ├── data/motions/k1_kick.npz (149 Frame @ 50 FPS: joint_pos, joint_vel, body_pos_w, body_quat_w, body_lin/ang_vel_w)
  └── outputs/videos/k1_kick_reference.mp4
          │
          ▼  (scripts/train_ppo.py — 2.048 GPU Parallel Envs, BeyondMimic PPO)
[TAHAP 4: TRAINED PPO POLICY]
  ├── logs/rsl_rl/k1_tracking/<timestamp>_beyondmimic_k1_kick/ (Log TensorBoard & model_*.pt)
  └── outputs/checkpoints/k1_kick_policy.pt & k1_kick_policy.onnx
          │
          ▼  (scripts/evaluate_and_play.py + create_comparison_video.py)
[TAHAP 5: EVALUASI KUANTITATIF & VIDEO PERBANDINGAN]
  ├── outputs/evaluation/k1_metrics.json (MAE, RMSE, R-MPKPE, MPKPE, Success Rate 256 Episode)
  ├── outputs/evaluation/k1_joint_tracking_comparison.png & ppo_training_curves.png
  └── outputs/videos/k1_trained_policy_tracking.mp4 & k1_side_by_side_comparison.mp4
```

---

### Tahap 0: Dari Mana Asal Aset Robot & Aset Gerakan Manusia?

#### 1. Asal Aset Robot **Booster K1 (22-DoF)**
Seluruh geometri dan kinematika robot berasal dari folder [`third_party/GMR/assets/booster_k1/`](./third_party/GMR/assets/booster_k1/):
- **`k1_mjlab.xml`**: File MuJoCo XML yang mendefinisikan struktur pohon kinematik 23 link tubuh mulai dari `Trunk` (root), `Head_1`–`Head_2`, `Left_Arm_1`–`left_hand_link`, `Right_Arm_1`–`right_hand_link`, `Left_Hip_Pitch`–`left_foot_link`, hingga `Right_Hip_Pitch`–`right_foot_link`, lengkap dengan batas sudut 22 sendi, inersia massa, geometri *collision*, dan sensor IMU.
- **`meshes/*.STL`**: 23 file mesh 3D (`.STL`) untuk visualisasi tiap bagian tubuh robot.
- **`third_party/mjlab/src/mjlab/asset_zoo/robots/booster_k1/k1_constants.py`**: Mendefinisikan parameter motor aktuator PD (*stiffness* $K_p$, *damping* $K_d$, *armature*, dan *effort limit*) serta pose berdiri awal (`K1_KNEES_BENT_KEYFRAME`).

#### 2. Asal Aset Gerakan Manusia (`.bvh` Motion Capture)
Data gerakan manusia berasal dari file **`.bvh` (*Biovision Hierarchy*)**:
- **Sumber Xsens (`120 Hz`, satuan cm)**: Berasal dari `third_party/GMR/assets/xsens_bvh_test/251021_04_boxing_120Hz_cm_3DsMax.bvh` (rekaman *motion capture* manusia tinggi $1.78\text{ m}$ dengan 72 kanal rotasi/posisi Euler).
- **Generator Gerakan Kick (`scripts/generate_kick_bvh.py`)**: Mengambil postur dasar *fighting guard* dari file Xsens di atas, lalu membentuk trajektori *Martial-Arts Front Kick* (`data/bvh/human_kick_xsens.bvh`, `360` frame @ `120 Hz`) menggunakan fungsi *minimum-jerk* ($10u^3 - 15u^4 + 6u^5$) yang mengatur perpindahan berat badan ke kaki tumpu kiri, angkatan paha/lutut kanan, tendangan lurus ke depan, serta ayunan tangan penyeimbang.
- **Sumber Eksternal (Custom BVH)**: Anda dapat menaruh file `.bvh` apa saja yang berformat **Xsens** atau **LAFAN1** ke dalam folder `data/bvh/`.

---

### Tahap 1: Membuat File `.bvh` Gerakan Manusia (`data/bvh/`)

Buka **PowerShell** dan masuk ke direktori `human-imitation-learning`:
```powershell
cd C:\Users\ASUS\Documents\fp-ichiro\human-imitation-learning
```

Jalankan generator `.bvh` tendangan manusia:
```powershell
.\.venv\Scripts\python.exe scripts\generate_kick_bvh.py
```
- **Hasil**: File `data/bvh/human_kick_xsens.bvh` (`360` frame @ `120 Hz`, durasi `3.0 detik`).
- *(Opsional)* Jika ingin membuat sampel format LAFAN1:
  ```powershell
  .\.venv\Scripts\python.exe scripts\generate_lafan1_sample_bvh.py
  ```

---

### Tahap 2: Retargeting `.bvh` Manusia $\rightarrow$ `.pkl` & `.csv` Robot Booster K1 via GMR (`scripts/retarget_bvh_gmr.py`)

Karena ukuran tulang dan struktur sendi manusia di file `.bvh` berbeda jauh dengan robot **Booster K1**, kita gunakan **GMR (*General Motion Retargeting*)** untuk memetakan gerakan manusia ke 22 sendi robot:

```powershell
.\.venv\Scripts\python.exe scripts\retarget_bvh_gmr.py `
  --robot booster_k1 `
  --bvh-file data\bvh\human_kick_xsens.bvh `
  --reset-to-zero `
  --save-pkl data\retargeted\k1_kick.pkl `
  --save-csv data\retargeted\k1_kick.csv `
  --save-video outputs\videos\gmr_retargeted_k1_kick.mp4
```

#### Detail Proses di Dalam `retarget_bvh_gmr.py`:
1. **Pembacaan & Downsampling**: Membaca `human_kick_xsens.bvh` (`120 Hz`) dan men-downsample menjadi `30 FPS` (`90` frame).
2. **Tabel Pemetaan IK (`bvh_xsens_to_k1.json`)**: Menskalakan tinggi kerangka manusia dengan faktor `0.56` dan menjalankan solver *Differential Inverse Kinematics* (`Mink` + `DAQP`) dua tahap (`ik_match_table1` & `ik_match_table2`) agar posisi & orientasi `Trunk`, `Head_2`, tangan, paha, betis, dan telapak kaki robot mengikuti gerakan manusia.
3. **Koreksi Kontak Lantai (`adjust_ground_contact`)**: Menggeser tinggi $z$ root supaya persentil ke-5 terendah dari telapak kaki berada tepat di atas permukaan lantai ($z = 0.024\text{ m}$).
4. **Penyimpanan 29 Kolom CSV (`data/retargeted/k1_kick.csv`)**:
   Setiap baris mewakili 1 frame pada `30 FPS` dengan **29 kolom**:
   - Kolom `0..2`: Posisi root `Trunk` $(x, y, z)$ dalam meter.
   - Kolom `3..6`: Kuaternion orientasi root `Trunk` dalam urutan **`[qx, qy, qz, qw]`**.
   - Kolom `7..28`: Sudut **22 sendi Booster K1** (`AAHead_yaw` s/d `Right_Ankle_Roll`) dalam radian.

---

### Tahap 3: Konversi `.csv` (30 FPS) $\rightarrow$ Format Referensi MJLab `.npz` (50 FPS) (`scripts/convert_csv_to_npz.py`)

Simulator kontrol PPO di MJLab bekerja pada frekuensi `50 Hz` (physics timestep `0.005s` $\times$ decimation `4` = `0.02s`) dan membutuhkan posisi serta kecepatan 3D dari seluruh 23 bagian tubuh robot untuk menghitung *reward* pelacakan *keybody*. Jalankan:

```powershell
.\.venv\Scripts\python.exe scripts\convert_csv_to_npz.py `
  --robot booster_k1 `
  --input-csv data\retargeted\k1_kick.csv `
  --output-npz data\motions\k1_kick.npz `
  --output-video outputs\videos\k1_kick_reference.mp4
```

#### Detail Proses di Dalam `convert_csv_to_npz.py`:
1. **Interpolasi `30 FPS` $\rightarrow$ `50 FPS`**: Menginterpolasi `90` frame menjadi `149` frame (`2.98 detik`) menggunakan interpolasi linear pada posisi/sudut sendi dan **Slerp** pada rotasi root, sekaligus menghitung turunan kecepatan (`joint_vel` dan kecepatan root).
2. **Forward Kinematics di MuJoCo-Warp**: Menjalankan `sim.forward()` pada setiap frame di dalam scene `booster_k1_flat_tracking_env_cfg` untuk mengekstrak posisi dunia `body_pos_w` `(149, 23, 3)`, orientasi `body_quat_w` `(149, 23, 4)` (`wxyz`), kecepatan linear `body_lin_vel_w` `(149, 23, 3)`, dan kecepatan sudut `body_ang_vel_w` `(149, 23, 3)`.
3. **Hasil Akhir**: File **`data/motions/k1_kick.npz`** dan video referensi **`outputs/videos/k1_kick_reference.mp4`**.

---

### Tahap 4: Menjalankan Simulasi 3D Interaktif Sebelum / Sesudah Training (`scripts/play_interactive.py`)

#### A. Cek Gerakan Referensi `.npz` Tanpa Training (`--reference-only`)
- **Jendela Desktop MuJoCo Native (GLFW)**:
  ```powershell
  .\.venv\Scripts\python.exe scripts\play_interactive.py `
    --robot booster_k1 `
    --motion-file data\motions\k1_kick.npz `
    --reference-only `
    --viewer native
  ```
- **Browser Web Viser (`http://localhost:8080`)**:
  ```powershell
  .\.venv\Scripts\python.exe scripts\play_interactive.py `
    --robot booster_k1 `
    --motion-file data\motions\k1_kick.npz `
    --reference-only `
    --viewer viser
  ```

#### B. Uji Coba Policy PPO Hasil Training (`.pt`)
```powershell
.\.venv\Scripts\python.exe scripts\play_interactive.py `
  --robot booster_k1 `
  --motion-file data\motions\k1_kick.npz `
  --checkpoint-file outputs\checkpoints\k1_kick_policy.pt `
  --viewer native
```

---

### Tahap 5: Melatih Policy PPO BeyondMimic dari Nol & Finetuning (`scripts/train_ppo.py`)

#### A. Training PPO dari Nol (Scratch)
```powershell
.\.venv\Scripts\python.exe scripts\train_ppo.py `
  --robot booster_k1 `
  --motion-file data\motions\k1_kick.npz `
  --num-envs 2048 `
  --max-iterations 500 `
  --save-interval 100 `
  --run-name beyondmimic_k1_kick `
  --output-checkpoint outputs\checkpoints\k1_kick_policy.pt
```
- Model otomatis tersimpan setiap `100` iterasi di `logs/rsl_rl/k1_tracking/<timestamp>_beyondmimic_k1_kick/`.
- Begitu selesai, checkpoint terakhir langsung disalin ke `outputs/checkpoints/k1_kick_policy.pt` beserta versi ONNX-nya di `outputs/checkpoints/k1_kick_policy.onnx`.

#### B. Melanjutkan Training / Finetuning dari Checkpoint (`--resume-from`)
Jika gerakan robot masih goyang atau ingin menambah iterasi latihan tanpa mengulang dari nol:
```powershell
.\.venv\Scripts\python.exe scripts\train_ppo.py `
  --robot booster_k1 `
  --motion-file data\motions\k1_kick.npz `
  --resume-from outputs\checkpoints\k1_kick_policy.pt `
  --num-envs 2048 `
  --max-iterations 500 `
  --save-interval 100 `
  --run-name beyondmimic_k1_kick_finetune `
  --output-checkpoint outputs\checkpoints\k1_kick_policy.pt
```

#### C. Tuning Parameter Fisika & Reward (Jika Ingin Modifikasi Lebih Lanjut)
1. **Robot Mudah Jatuh Saat Kaki Kanan Diangkat (`Success Rate` Rendah)**:
   - Buka [`third_party/mjlab/src/mjlab/tasks/tracking/config/k1/env_cfgs.py`](./third_party/mjlab/src/mjlab/tasks/tracking/config/k1/env_cfgs.py) atau [`tracking_env_cfg.py`](./third_party/mjlab/src/mjlab/tasks/tracking/tracking_env_cfg.py).
   - Longgarkan batas toleransi jatuh pada `terminations["anchor_pos"]` atau `terminations["ee_body_pos"]` (misalnya `threshold=0.25` menjadi `0.28`).
2. **Tendangan Kurang Tinggi (`MAE` atau `R-MPKPE` Besar)**:
   - Di [`tracking_env_cfg.py`](./third_party/mjlab/src/mjlab/tasks/tracking/tracking_env_cfg.py), naikkan bobot reward `motion_body_pos` atau `motion_body_ori` (`weight=1.5` $\rightarrow$ `2.0`).
3. **VRAM GPU Penuh (*Out of Memory*)**:
   - Turunkan `--num-envs` dari `2048` ke `1024` atau `512`.

---

### Tahap 6: Evaluasi Metrik & Rekam Video Hasil Training (`scripts/evaluate_and_play.py`)

Jalankan evaluasi lengkap beserta perekaman video `.mp4`:
```powershell
.\.venv\Scripts\python.exe scripts\evaluate_and_play.py `
  --robot booster_k1 `
  --checkpoint-file outputs\checkpoints\k1_kick_policy.pt `
  --motion-file data\motions\k1_kick.npz `
  --output-video outputs\videos\k1_trained_policy_tracking.mp4 `
  --output-metrics outputs\evaluation\k1_metrics.json `
  --output-plot outputs\evaluation\k1_joint_tracking_comparison.png
```

Buat grafik kurva konvergensi TensorBoard dan video perbandingan *side-by-side*:
```powershell
.\.venv\Scripts\python.exe scripts\plot_training_curves.py
.\.venv\Scripts\python.exe scripts\create_comparison_video.py
```

---

### Tahap 7: Menjalankan Website Monitoring & Control Platform

- **Repository GitHub Website Monitoring & Control**: https://github.com/JoshNells13/FP-Website-Monitoring-Control-imitation-learning

Seluruh hasil dari Tahap 0 s/d Tahap 6 di atas otomatis terbaca secara *live* (tanpa database dan tanpa data dummy) oleh aplikasi Website Monitoring & Control pada repository **[FP-Website-Monitoring-Control-imitation-learning](https://github.com/JoshNells13/FP-Website-Monitoring-Control-imitation-learning)** (folder lokal [`../monitoring-control-imitation`](../monitoring-control-imitation)):

```powershell
cd C:\Users\ASUS\Documents\fp-ichiro\monitoring-control-imitation
.\start_platform.ps1
```
- **Backend Flask + Socket.IO**: `http://localhost:5000`
- **Frontend Next.js Bento UI**: `http://localhost:3000`

---

## Kesesuaian dengan Paper Resmi BeyondMimic & Bukti Implementasi di Kode

- **Paper BeyondMimic (arXiv:2508.08241)**: https://arxiv.org/abs/2508.08241 (*"BeyondMimic: From Motion Tracking to Versatile Humanoid Control via Guided Diffusion"*, Qiayuan Liao et al. — Hybrid Robotics / UC Berkeley & Stanford).
- **Paper GMR (arXiv:2510.02252)**: https://arxiv.org/abs/2510.02252 (*"GMR: General Motion Retargeting"*, Yanjie Ze et al.).

| Komponen di Paper BeyondMimic | Rumus / Mekanisme di Paper | Bukti File & Baris Kode di Proyek |
| :--- | :--- | :--- |
| **1. Anchor-Yaw-Aligned Relative Keybody Pose** | Menyejajarkan target posisi & rotasi 14 bagian tubuh terhadap posisi $(x, y)$ dan arah hadap *yaw* dari anchor (`Trunk`) sementara tinggi $z$ serta *roll/pitch* tetap mengikuti referensi: $\Delta q = \text{yaw}(q_{\text{anchor}}^{\text{robot}} \otimes (q_{\text{anchor}}^{\text{ref}})^{-1})$. | [`commands.py`](./third_party/mjlab/src/mjlab/tasks/tracking/mdp/commands.py) (`update_relative_body_poses`) |
| **2. Asymmetric Actor-Critic Observations** | **Actor (`125 dim`)**: `command (44)` + `motion_anchor_pos_b (3)` + `motion_anchor_ori_b (6)` + `base_lin_vel (3)` + `base_ang_vel (3)` + `joint_pos (22)` + `joint_vel (22)` + `actions (22)`.<br>**Critic (`251 dim`)**: Ditambah posisi 3D (`42 dim`) & orientasi 6D (`84 dim`) seluruh 14 bagian tubuh. | [`observations.py`](./third_party/mjlab/src/mjlab/tasks/tracking/mdp/observations.py) & [`tracking_env_cfg.py`](./third_party/mjlab/src/mjlab/tasks/tracking/tracking_env_cfg.py) |
| **3. 6 Exponential Tracking Rewards + 3 Penalti Fisika** | Fungsi reward eksponensial $\exp(-\|\text{error}\|^2 / \sigma^2)$:<br>• `motion_global_root_pos` ($\sigma=0.3$, bobot `0.5`)<br>• `motion_global_root_ori` ($\sigma=0.4$, bobot `0.5`)<br>• `motion_body_pos` ($\sigma=0.3$, bobot `1.5`)<br>• `motion_body_ori` ($\sigma=0.4$, bobot `1.5`)<br>• `motion_body_lin_vel` ($\sigma=1.0$, bobot `1.0`)<br>• `motion_body_ang_vel` ($\sigma=3.14$, bobot `1.0`)<br>• Penalti: `action_rate_l2 (-0.1)`, `joint_limit (-5.0)`, `self_collisions (-10.0)`. | [`rewards.py`](./third_party/mjlab/src/mjlab/tasks/tracking/mdp/rewards.py) & [`tracking_env_cfg.py`](./third_party/mjlab/src/mjlab/tasks/tracking/tracking_env_cfg.py) |
| **4. Adaptive Failure-Bin Curriculum Sampling & RSI** | Membagi durasi gerakan ke dalam *bins*, menghitung *exponential moving average* (EMA) kegagalan per *bin*, menghaluskannya dengan *1D causal convolution kernel*, lalu melakukan *multinomial sampling* + *Reference State Initialization* dengan gangguan acak. | [`commands.py`](./third_party/mjlab/src/mjlab/tasks/tracking/mdp/commands.py) (`_adaptive_sampling` & `_resample_command`) |
| **5. Task-Space Early Termination** | Menghentikan episode ketika robot jatuh: proyeksi gravitasi `bad_anchor_ori > 0.8`, simpangan tinggi anchor `bad_anchor_pos_z_only > 0.25 m`, atau simpangan tinggi tangan/kaki `bad_motion_body_pos_z_only > 0.25 m`. | [`terminations.py`](./third_party/mjlab/src/mjlab/tasks/tracking/mdp/terminations.py) & [`env_cfgs.py`](./third_party/mjlab/src/mjlab/tasks/tracking/config/k1/env_cfgs.py) |
| **6. Konfigurasi Booster K1 & Arsitektur PPO Resmi** | Mapping `Trunk` anchor, `14` *tracked keybodies*, sensor kontak `self_collision`, serta jaringan Actor-Critic MLP `[512, 256, 128]` ELU dengan *empirical observation normalization*. | [`env_cfgs.py`](./third_party/mjlab/src/mjlab/tasks/tracking/config/k1/env_cfgs.py) & [`rl_cfg.py`](./third_party/mjlab/src/mjlab/tasks/tracking/config/k1/rl_cfg.py) |
| **7. Two-Stage Differential IK Retargeting (GMR)** | Memetakan gerakan `.bvh` manusia ke 22 sendi Booster K1 lewat optimasi kuadratik dua tahap (`ik_match_table1` & `ik_match_table2`) menggunakan `Mink` + `DAQP`. | [`motion_retarget.py`](./third_party/GMR/general_motion_retargeting/motion_retarget.py) & [`bvh_xsens_to_k1.json`](./third_party/GMR/general_motion_retargeting/ik_configs/bvh_xsens_to_k1.json) |
