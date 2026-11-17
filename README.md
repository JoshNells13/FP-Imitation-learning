# Humanoid Whole-Body Motion Imitation Learning — Booster K1 (22-DoF)

Proyek ini berisi pipeline *end-to-end* **Whole-Body Motion Imitation Learning** untuk robot humanoid **Booster K1 (22-DoF)** menggunakan **Python + MuJoCo + MJLab + PyTorch** (tanpa ROS 2, tanpa Isaac Sim, dan tanpa Isaac Lab).

Tiga modul utama yang digabungkan di dalam proyek ini:
1. **[GMR (`YanjieZe/GMR`)](https://github.com/YanjieZe/GMR)** — *General Motion Retargeting* berbasis *two-stage differential inverse kinematics* (`Mink` + `DAQP`) untuk memetakan gerakan manusia (`.bvh` Xsens / LAFAN1) ke sendi robot **Booster K1**.
2. **[BeyondMimic (`HybridRobotics/whole_body_tracking`)](https://github.com/HybridRobotics/whole_body_tracking)** — Metodologi *whole-body tracking* (*anchor-yaw-aligned relative keybody tracking*, *asymmetric actor-critic observations*, *6 exponential tracking rewards*, *adaptive failure-bin curriculum sampling*, dan *task-space height terminations*).
3. **[MJLab + MuJoCo-Warp (`mujocolab/mjlab`)](https://github.com/mujocolab/mjlab)** — Simulator fisika MuJoCo berbasis GPU (`mujoco-warp`) dan algoritma *Proximal Policy Optimization* (**PPO**) lewat `rsl-rl-lib`.

---

## Spesifikasi Model Robot & Motion Reference

| Komponen | Detail |
| :--- | :--- |
| **Model Robot** | **Booster K1** (`booster_k1`, `22-DoF`: `2` sendi kepala + `8` sendi lengan + `12` sendi kaki) |
| **Root / Anchor Body** | `Trunk` (tinggi berdiri $\approx 0.50\text{ m}$, skala retargeting `0.56`) |
| **Tracked Keybodies** | `14` link tubuh (`Trunk`, `Head_2`, bahu, lengan atas, tangan, pinggul, betis, dan telapak kaki) |
| **Dimensi Observasi & Aksi** | Actor: `125 dim` \| Critic (Privileged): `251 dim` \| Action: `22 dim` |
| **Motion Reference** | `data/motions/k1_kick.npz` (*Martial-Arts Front Kick*, `149` frame @ `50 FPS`, durasi `3.0 detik`) |

---

## Kesesuaian dengan Paper Resmi BeyondMimic & Bukti Implementasi di Kode

Proyek ini mengadopsi langsung metodologi *Whole-Body Motion Tracking* dari paper resmi **BeyondMimic** dan **GMR**:
- **Paper BeyondMimic (arXiv:2508.08241)**: https://arxiv.org/abs/2508.08241 (*"BeyondMimic: From Motion Tracking to Versatile Humanoid Control via Guided Diffusion"*, Qiayuan Liao, Takara E. Truong, Xiaoyu Huang, Yuman Gao, Guy Tevet, Koushil Sreenath, C. Karen Liu — Hybrid Robotics / UC Berkeley & Stanford).
- **Website Resmi BeyondMimic**: https://beyondmimic.github.io/
- **Repositori Resmi BeyondMimic**: https://github.com/HybridRobotics/whole_body_tracking (di baris 14–17 [`third_party/whole_body_tracking/README.md`](file:///c:/Users/ASUS/Documents/human-imitation-learning/third_party/whole_body_tracking/README.md#L14-L17) tertulis resmi dari penulis paper bahwa implementasi BeyondMimic untuk MuJoCo dilanjutkan di dalam `mjlab/tasks/tracking`).
- **Paper GMR (arXiv:2510.02252)**: https://arxiv.org/abs/2510.02252 (*"GMR: General Motion Retargeting"*, Yanjie Ze et al.).

### Tabel Bukti Implementasi Paper BeyondMimic di Kode Proyek:

| Komponen di Paper BeyondMimic | Rumus / Mekanisme di Paper | Bukti File & Baris Kode di Proyek |
| :--- | :--- | :--- |
| **1. Anchor-Yaw-Aligned Relative Keybody Pose** | Menyejajarkan target posisi & rotasi 14 bagian tubuh terhadap posisi $(x, y)$ dan arah hadap *yaw* dari anchor (`Trunk`) sementara tinggi $z$ serta *roll/pitch* tetap mengikuti referensi: $\Delta q = \text{yaw}(q_{\text{anchor}}^{\text{robot}} \otimes (q_{\text{anchor}}^{\text{ref}})^{-1})$. | [`commands.py#L379-L408`](file:///c:/Users/ASUS/Documents/human-imitation-learning/third_party/mjlab/src/mjlab/tasks/tracking/mdp/commands.py#L379-L408) (`update_relative_body_poses`) |
| **2. Asymmetric Actor-Critic Observations** | **Actor (`125 dim`)**: `command (44)` + `motion_anchor_pos_b (3)` + `motion_anchor_ori_b (6)` + `base_lin_vel (3)` + `base_ang_vel (3)` + `joint_pos (22)` + `joint_vel (22)` + `actions (22)`.<br>**Critic (`251 dim`)**: Ditambah posisi 3D (`42 dim`) & orientasi 6D (`84 dim`) seluruh 14 bagian tubuh. | [`observations.py#L18-L69`](file:///c:/Users/ASUS/Documents/human-imitation-learning/third_party/mjlab/src/mjlab/tasks/tracking/mdp/observations.py#L18-L69) & [`tracking_env_cfg.py#L60-L157`](file:///c:/Users/ASUS/Documents/human-imitation-learning/third_party/mjlab/src/mjlab/tasks/tracking/tracking_env_cfg.py#L60-L157) |
| **3. 6 Exponential Tracking Rewards + 3 Penalti Fisika** | Fungsi reward eksponensial $\exp(-\|\text{error}\|^2 / \sigma^2)$:<br>• `motion_global_root_pos` ($\sigma=0.3$, bobot `0.5`)<br>• `motion_global_root_ori` ($\sigma=0.4$, bobot `0.5`)<br>• `motion_body_pos` ($\sigma=0.3$, bobot `1.0`)<br>• `motion_body_ori` ($\sigma=0.4$, bobot `1.0`)<br>• `motion_body_lin_vel` ($\sigma=1.0$, bobot `1.0`)<br>• `motion_body_ang_vel` ($\sigma=3.14$, bobot `1.0`)<br>• Penalti: `action_rate_l2 (-0.1)`, `joint_limit (-10.0)`, `self_collisions (-10.0)`. | [`rewards.py#L26-L135`](file:///c:/Users/ASUS/Documents/human-imitation-learning/third_party/mjlab/src/mjlab/tasks/tracking/mdp/rewards.py#L26-L135) & [`tracking_env_cfg.py#L196-L263`](file:///c:/Users/ASUS/Documents/human-imitation-learning/third_party/mjlab/src/mjlab/tasks/tracking/tracking_env_cfg.py#L196-L263) |
| **4. Adaptive Failure-Bin Curriculum Sampling & RSI** | Membagi durasi gerakan ke dalam *bins*, menghitung *exponential moving average* (EMA) kegagalan per *bin*, menghaluskannya dengan *1D causal convolution kernel*, lalu melakukan *multinomial sampling* + *Reference State Initialization* dengan gangguan acak (*pose/velocity perturbations*). | [`commands.py#L250-L377`](file:///c:/Users/ASUS/Documents/human-imitation-learning/third_party/mjlab/src/mjlab/tasks/tracking/mdp/commands.py#L250-L377) (`_adaptive_sampling` & `_resample_command`) |
| **5. Task-Space Early Termination** | Menghentikan episode ketika robot jatuh: proyeksi gravitasi `bad_anchor_ori > 0.8`, simpangan tinggi anchor `bad_anchor_pos_z_only > 0.22 m`, atau simpangan tinggi tangan/kaki `bad_motion_body_pos_z_only > 0.22 m`. | [`terminations.py#L18-L86`](file:///c:/Users/ASUS/Documents/human-imitation-learning/third_party/mjlab/src/mjlab/tasks/tracking/mdp/terminations.py#L18-L86) & [`env_cfgs.py#L51-L65`](file:///c:/Users/ASUS/Documents/human-imitation-learning/third_party/mjlab/src/mjlab/tasks/tracking/config/k1/env_cfgs.py#L51-L65) |
| **6. Konfigurasi Booster K1 & Arsitektur PPO Resmi** | Mapping `Trunk` anchor, `14` *tracked keybodies*, sensor kontak `self_collision`, serta jaringan Actor-Critic MLP `[512, 256, 128]` ELU dengan *empirical observation normalization*, `clip_param=0.2`, `entropy_coef=0.005`, `desired_kl=0.01`. | [`env_cfgs.py#L18-L90`](file:///c:/Users/ASUS/Documents/human-imitation-learning/third_party/mjlab/src/mjlab/tasks/tracking/config/k1/env_cfgs.py#L18-L90) & [`rl_cfg.py#L10-L39`](file:///c:/Users/ASUS/Documents/human-imitation-learning/third_party/mjlab/src/mjlab/tasks/tracking/config/k1/rl_cfg.py#L10-L39) |
| **7. Two-Stage Differential IK Retargeting (GMR)** | Memetakan gerakan `.bvh` manusia ke 22 sendi Booster K1 lewat optimasi kuadratik dua tahap (`ik_match_table1` & `ik_match_table2`) menggunakan `Mink` + `DAQP`. | [`motion_retarget.py#L183-L249`](file:///c:/Users/ASUS/Documents/human-imitation-learning/third_party/GMR/general_motion_retargeting/motion_retarget.py#L183-L249) & [`bvh_xsens_to_k1.json`](file:///c:/Users/ASUS/Documents/human-imitation-learning/third_party/GMR/general_motion_retargeting/ik_configs/bvh_xsens_to_k1.json) |

---

## Struktur Folder & Perbedaan Folder Data

```text
human-imitation-learning/
├── data/
│   ├── bvh/
│   │   ├── human_kick_xsens.bvh                # Data mentah gerakan MANUSIA (.bvh @ 120 Hz)
│   │   └── lafan1_sample.bvh                   # Sampel BVH format LAFAN1
│   ├── retargeted/
│   │   ├── k1_kick.pkl                         # Hasil retargeting sudut sendi Booster K1 (.pkl)
│   │   └── k1_kick.csv                         # Hasil retargeting 29 kolom (.csv @ 30 FPS)
│   └── motions/
│       └── k1_kick.npz                         # Data referensi LENGKAP 50 FPS siap simulasi & training
├── outputs/
│   ├── checkpoints/
│   │   └── k1_kick_policy.pt / .onnx           # Lokasi penyimpanan policy PPO hasil training
│   ├── evaluation/
│   │   ├── k1_metrics.json                     # Metrik evaluasi (MAE, RMSE, MPKPE, Success Rate)
│   │   ├── k1_joint_tracking_comparison.png    # Grafik perbandingan sudut sendi Referensi vs Simulasi
│   │   └── ppo_training_curves.png             # Grafik kurva konvergensi reward & error PPO
│   └── videos/
│       ├── gmr_retargeted_k1_kick.mp4          # Video preview retargeting GMR Booster K1 (Kick)
│       └── k1_kick_reference.mp4               # Video motion reference MJLab 50 FPS Booster K1 (Kick)
└── scripts/
    ├── generate_kick_bvh.py                    # Generator gerakan BVH tendangan (Kick) manusia
    ├── retarget_bvh_gmr.py                     # Pipeline retargeting GMR (.bvh -> .pkl, .csv, .mp4)
    ├── convert_csv_to_npz.py                   # Konverter CSV ke format referensi MJLab 50 FPS (.npz)
    ├── train_ppo.py                            # Skrip utama training PPO BeyondMimic di MJLab
    ├── evaluate_and_play.py                    # Skrip evaluasi metrik lengkap & perekaman video (.mp4)
    ├── plot_training_curves.py                 # Ekstraktor log TensorBoard ke grafik kurva training
    ├── create_comparison_video.py              # Pembuat video perbandingan Reference vs Policy
    └── play_interactive.py                     # Viewer simulasi 3D interaktif (Native MuJoCo / Web Viser)
```

### Bedanya Folder `data/bvh/`, `data/retargeted/`, dan `data/motions/`:
- **`data/bvh/`**: Berisi file *Motion Capture* (`.bvh`) dengan ukuran dan struktur **tulang manusia** (`1.78 m`, 23 sendi manusia). Belum bisa langsung dipakai di robot.
- **`data/retargeted/`**: Berisi file `.csv` dan `.pkl` hasil pemetaan GMR ke **22 sudut sendi Booster K1** (`30 FPS`), tetapi belum memiliki data kecepatan dan posisi 3D tiap bagian tubuh.
- **`data/motions/`**: Berisi file `.npz` (`50 FPS`) yang sudah dihitung *Forward Kinematics*-nya di MuJoCo (menyimpan sudut sendi, kecepatan sendi, posisi 3D, orientasi, dan kecepatan seluruh 23 bagian tubuh robot). **File `.npz` di folder inilah yang langsung dipakai untuk simulasi dan training PPO.**

---

## 1. Persiapan Awal

Semua library (`Python 3.12`, `torch` + CUDA, `mujoco`, `mujoco-warp`, `rsl-rl-lib`, `mink`, `daqp`, `tensorboard`) sudah terpasang di dalam virtual environment `.venv`.

Buka terminal **PowerShell** dan masuk ke direktori proyek:
```powershell
cd c:\Users\ASUS\Documents\human-imitation-learning
```

---

## 2. Cara Menjalankan Simulasi 3D Interaktif

Simulasi 3D bisa langsung dijalankan lewat [`scripts/play_interactive.py`](file:///c:/Users/ASUS/Documents/human-imitation-learning/scripts/play_interactive.py), baik untuk mengecek gerakan referensi (*reference motion*) sebelum training maupun untuk menguji hasil policy PPO setelah training selesai.

### A. Simulasi Motion Reference (Tanpa Training)
Tambahkan flag `--reference-only` untuk memutar gerakan referensi **Kick** secara kinematik pada **Booster K1**:

- **Tampilan Desktop MuJoCo Native (GLFW)**:
  ```powershell
  .\.venv\Scripts\python.exe scripts\play_interactive.py `
    --robot booster_k1 `
    --motion-file data\motions\k1_kick.npz `
    --reference-only `
    --viewer native
  ```

- **Tampilan Browser Web Viser (`http://localhost:8080`)**:
  ```powershell
  .\.venv\Scripts\python.exe scripts\play_interactive.py `
    --robot booster_k1 `
    --motion-file data\motions\k1_kick.npz `
    --reference-only `
    --viewer viser
  ```

### B. Simulasi Policy PPO Hasil Training
Setelah proses training menghasilkan file checkpoint `.pt`, jalankan perintah berikut tanpa `--reference-only`:

```powershell
.\.venv\Scripts\python.exe scripts\play_interactive.py `
  --robot booster_k1 `
  --motion-file data\motions\k1_kick.npz `
  --checkpoint-file outputs\checkpoints\k1_kick_policy.pt `
  --viewer native
```

---

## 3. Cara Menyiapkan Motion Reference (Pipeline Data)

File referensi `data/motions/k1_kick.npz` sebenarnya sudah siap pakai. Namun kalau ingin membuat ulang atau memproses file `.bvh` baru ke **Booster K1**, urutannya sebagai berikut:

### Langkah 1: Generate File BVH Gerakan Kick Manusia
Menghasilkan file `data/bvh/human_kick_xsens.bvh` (`360` frame @ `120 Hz`):
```powershell
.\.venv\Scripts\python.exe scripts\generate_kick_bvh.py
```

### Langkah 2: Retargeting BVH ke Booster K1 via GMR
Memetakan gerakan manusia ke 22 sendi Booster K1 sekaligus merender video preview:
```powershell
.\.venv\Scripts\python.exe scripts\retarget_bvh_gmr.py `
  --robot booster_k1 `
  --bvh-file data\bvh\human_kick_xsens.bvh `
  --reset-to-zero `
  --save-pkl data\retargeted\k1_kick.pkl `
  --save-csv data\retargeted\k1_kick.csv `
  --save-video outputs\videos\gmr_retargeted_k1_kick.mp4
```

### Langkah 3: Konversi CSV ke Format `.npz` 50 FPS MJLab
Menginterpolasi gerakan ke frekuensi kontrol 50 Hz dan menghitung *Forward Kinematics* semua link tubuh di MuJoCo-Warp:
```powershell
.\.venv\Scripts\python.exe scripts\convert_csv_to_npz.py `
  --robot booster_k1 `
  --input-csv data\retargeted\k1_kick.csv `
  --output-npz data\motions\k1_kick.npz `
  --output-video outputs\videos\k1_kick_reference.mp4
```

---

## 4. Cara Menjalankan Training PPO dari Awal

Jalankan [`scripts/train_ppo.py`](file:///c:/Users/ASUS/Documents/human-imitation-learning/scripts/train_ppo.py) untuk melatih jaringan *Actor-Critic* PPO supaya robot **Booster K1** bisa mengeksekusi gerakan tendangan secara seimbang di simulasi fisika MuJoCo-Warp:

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
- Selama proses berjalan, model otomatis tersimpan setiap `100` iterasi di dalam folder `logs/rsl_rl/k1_tracking/<timestamp>_beyondmimic_k1_kick/`.
- Begitu selesai, checkpoint terakhir langsung disalin ke `outputs/checkpoints/k1_kick_policy.pt` beserta versi ONNX-nya di `outputs/checkpoints/k1_kick_policy.onnx`.

### Memantau Grafik Training via TensorBoard
Buka terminal PowerShell baru, lalu jalankan:
```powershell
.\.venv\Scripts\tensorboard.exe --logdir logs\rsl_rl\k1_tracking
```
Akses `http://localhost:6006` di browser untuk memantau kenaikan `Train/mean_reward`, durasi bertahan berdiri `Train/mean_episode_length`, dan penurunan error `Metrics/motion/...`.

---

## 5. Cara Training Lagi Kalau Hasilnya Masih Belum Sesuai

Kalau setelah dievaluasi gerakan robot masih goyang, sering jatuh sebelum tendangan selesai, atau angkatan kakinya kurang tinggi, proses training tidak perlu diulang dari nol. Berikut beberapa cara untuk melatih ulang atau melanjutkan training:

### Cara 1: Lanjutkan Training dari Checkpoint Terakhir (`--resume-from`)
Kalau grafik reward masih naik tapi jumlah iterasinya habis, gunakan flag `--resume-from` supaya bobot jaringan, *normalizer*, dan optimizer `AdamW` diteruskan dari checkpoint terakhir:

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
Perintah di atas akan menambah **500 iterasi lagi** di atas model sebelumnya dan memperbarui file `outputs/checkpoints/k1_kick_policy.pt`.

Kalau ingin melanjutkan dari titik iterasi tertentu di folder log (misalnya `model_300.pt` terasa lebih stabil daripada `model_500.pt`), tinggal arahkan `--resume-from` ke file tersebut:
```powershell
.\.venv\Scripts\python.exe scripts\train_ppo.py `
  --robot booster_k1 `
  --motion-file data\motions\k1_kick.npz `
  --resume-from logs\rsl_rl\k1_tracking\<nama_folder_run>\model_300.pt `
  --max-iterations 500 `
  --output-checkpoint outputs\checkpoints\k1_kick_policy.pt
```

### Cara 2: Tambah Jumlah Iterasi (`--max-iterations`)
Gerakan bertumpu satu kaki seperti *kick* butuh koordinasi keseimbangan ekstra. Menaikkan `--max-iterations` ke `1000` atau `1500` biasanya membuat tumpuan kaki kiri jauh lebih kokoh:
```powershell
.\.venv\Scripts\python.exe scripts\train_ppo.py `
  --robot booster_k1 `
  --motion-file data\motions\k1_kick.npz `
  --num-envs 2048 `
  --max-iterations 1000 `
  --save-interval 100 `
  --output-checkpoint outputs\checkpoints\k1_kick_policy.pt
```

### Cara 3: Tuning Parameter Fisika & Reward
Kalau sudah ditambah iterasinya tapi gerakannya masih kurang pas, beberapa parameter di bawah ini bisa disesuaikan sebelum menjalankan `train_ppo.py` lagi:

1. **Robot Mudah Jatuh Saat Kaki Kanan Diangkat (`Success Rate` Rendah)**:
   - Buka [`third_party/mjlab/src/mjlab/tasks/tracking/config/k1/env_cfgs.py`](file:///c:/Users/ASUS/Documents/human-imitation-learning/third_party/mjlab/src/mjlab/tasks/tracking/config/k1/env_cfgs.py).
   - Longgarkan batas toleransi jatuh pada `terminations["anchor_pos"]` atau `terminations["ee_body_pos"]` (misalnya ubah `threshold=0.22` jadi `0.28`), atau kurangi rentang gangguan dorongan pada `events["push_robot"]`.

2. **Tendangan Kurang Tinggi / Kurang Mirip Referensi (`MAE` atau `R-MPKPE` Besar)**:
   - Buka [`third_party/mjlab/src/mjlab/tasks/tracking/tracking_env_cfg.py`](file:///c:/Users/ASUS/Documents/human-imitation-learning/third_party/mjlab/src/mjlab/tasks/tracking/tracking_env_cfg.py).
   - Naikkan bobot reward `motion_body_pos` atau `motion_body_ori` (dari `1.0` ke `1.5`), atau kecilkan penalti `action_rate_l2` (dari `-0.1` ke `-0.05`) supaya gerakan sendi lebih responsif.

3. **Sendi Terlihat Lemas atau Bergetar (*Oscillation*)**:
   - Buka [`third_party/mjlab/src/mjlab/asset_zoo/robots/booster_k1/k1_constants.py`](file:///c:/Users/ASUS/Documents/human-imitation-learning/third_party/mjlab/src/mjlab/asset_zoo/robots/booster_k1/k1_constants.py).
   - Naikkan nilai `stiffness` ($K_p$) pada pinggul/lutut kalau kaki tumpuan terasa kurang kuat, atau naikkan `damping` ($K_d$) kalau sendi terlihat bergetar.

4. **VRAM GPU Penuh (*Out of Memory*)**:
   - Turunkan jumlah environment paralel `--num-envs` dari `2048` ke `1024` atau `512`:
     ```powershell
     .\.venv\Scripts\python.exe scripts\train_ppo.py --robot booster_k1 --num-envs 1024 --max-iterations 500
     ```

---

## 6. Cara Evaluasi Metrik & Rekam Video Hasil Training

Setelah training selesai, jalankan [`scripts/evaluate_and_play.py`](file:///c:/Users/ASUS/Documents/human-imitation-learning/scripts/evaluate_and_play.py) untuk menghitung akurasi tracking sekaligus merekam video `.mp4`:

```powershell
.\.venv\Scripts\python.exe scripts\evaluate_and_play.py `
  --robot booster_k1 `
  --checkpoint-file outputs\checkpoints\k1_kick_policy.pt `
  --motion-file data\motions\k1_kick.npz `
  --output-video outputs\videos\k1_trained_policy_tracking.mp4 `
  --output-metrics outputs\evaluation\k1_metrics.json `
  --output-plot outputs\evaluation\k1_joint_tracking_comparison.png
```

Untuk membuat grafik kurva training dari TensorBoard dan video perbandingan *side-by-side*:
```powershell
.\.venv\Scripts\python.exe scripts\plot_training_curves.py
.\.venv\Scripts\python.exe scripts\create_comparison_video.py
```

### Patokan Metrik Hasil Training yang Bagus
Cek tabel di terminal atau isi file `outputs/evaluation/k1_metrics.json`:
- **Motion Completion Success Rate**: Target **$\ge 85\%$** (robot tetap berdiri sampai klip berakhir).
- **Joint Tracking MAE**: Target **$< 0.20\text{ rad}$ ($\approx 11^\circ$)**.
- **Anchor-Relative Keybody Error (R-MPKPE)**: Target **$< 70\text{ mm}$**.
- **Torso Anchor Orientation Error**: Target **$< 0.12\text{ rad}$ ($\approx 7^\circ$)**.

Kalau angka di atas belum tercapai, tinggal jalankan lagi langkah di **Bagian 5 (`--resume-from`)** sampai gerakannya stabil.
