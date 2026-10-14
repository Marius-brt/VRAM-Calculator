"""
Writes src/lib/data/gpus.json from the table below.

Sources: vendor spec sheets (memory, bandwidth, TDP). fp16 = dense FP16/BF16 tensor throughput with
FP32 accumulate, which is what inference frameworks use (GeForce cards run this at half their
FP16-accumulate peak). Cards without matrix units (AMD before RDNA 3, GTX 16, Apple) list their
shader FP16 rate, and so do Turing GeForce cards, which matches the 2080 Ti's measured llama.cpp
prompt speed better than its tensor peak. Pascal lists its FP32 rate, as its FP16 runs at 1/64 speed.
Laptop GPUs list their highest-power spec. fp8 / fp4 = the card has native FP8 / FP4 tensor cores.
price = rough on-demand rental in USD per GPU-hour on community and hyperscaler clouds; None when it
isn't commonly rented.

The GPU picker lists each vendor's cards in table order, so keep it newest generation first and,
within a generation, data center, then workstation, then consumer, highest tier first.
Edit the table and re-run: python3 scripts/gpus.py
"""
import json, os

# key, label, vendor, type, category, memory GB, bandwidth GB/s, fp16 TFLOPS, fp8, fp4, TDP W, $/h
GPUS = [
  # NVIDIA Blackwell
  ("b300_288", "B300 (288GB)", "nvidia", "discrete", "datacenter", 288, 8000, 2250, 1, 1, 1100, 7.5),
  ("b200_180", "B200 (180GB)", "nvidia", "discrete", "datacenter", 180, 8000, 2250, 1, 1, 1000, 5.5),
  ("rtx_6000_blackwell_96", "RTX PRO 6000 Blackwell (96GB)", "nvidia", "discrete", "workstation", 96, 1792, 500, 1, 1, 600, 1.7),
  ("rtx_6000_blackwell_maxq_96", "RTX PRO 6000 Blackwell Max-Q (96GB)", "nvidia", "discrete", "workstation", 96, 1792, 440, 1, 1, 300, None),
  ("rtx_5000_blackwell_48", "RTX PRO 5000 Blackwell (48GB)", "nvidia", "discrete", "workstation", 48, 1344, 280, 1, 1, 300, None),
  ("rtx_4500_blackwell_32", "RTX PRO 4500 Blackwell (32GB)", "nvidia", "discrete", "workstation", 32, 896, 200, 1, 1, 200, None),
  ("rtx_4000_blackwell_24", "RTX PRO 4000 Blackwell (24GB)", "nvidia", "discrete", "workstation", 24, 672, 150, 1, 1, 140, None),
  ("gb10_128", "DGX Spark GB10 (128GB)", "nvidia", "apu", "workstation", 128, 273, 100, 1, 1, 140, None),
  ("5090_32", "RTX 5090 (32GB)", "nvidia", "discrete", "consumer", 32, 1792, 209, 1, 1, 575, 0.7),
  ("5080_16", "RTX 5080 (16GB)", "nvidia", "discrete", "consumer", 16, 960, 113, 1, 1, 360, 0.35),
  ("5070ti_16", "RTX 5070 Ti (16GB)", "nvidia", "discrete", "consumer", 16, 896, 88, 1, 1, 300, 0.25),
  ("5070_12", "RTX 5070 (12GB)", "nvidia", "discrete", "consumer", 12, 672, 62, 1, 1, 250, 0.2),
  ("5060ti_16", "RTX 5060 Ti (16GB)", "nvidia", "discrete", "consumer", 16, 448, 47, 1, 1, 180, 0.15),
  ("5060ti_8", "RTX 5060 Ti (8GB)", "nvidia", "discrete", "consumer", 8, 448, 47, 1, 1, 180, None),
  ("5060_8", "RTX 5060 (8GB)", "nvidia", "discrete", "consumer", 8, 448, 38, 1, 1, 145, None),
  ("5050_8", "RTX 5050 (8GB)", "nvidia", "discrete", "consumer", 8, 320, 26, 1, 1, 130, None),
  ("5090_laptop_24", "RTX 5090 Laptop (24GB)", "nvidia", "discrete", "consumer", 24, 896, 114, 1, 1, 150, None),
  ("5080_laptop_16", "RTX 5080 Laptop (16GB)", "nvidia", "discrete", "consumer", 16, 896, 83, 1, 1, 150, None),
  ("5070ti_laptop_12", "RTX 5070 Ti Laptop (12GB)", "nvidia", "discrete", "consumer", 12, 672, 62, 1, 1, 115, None),
  # NVIDIA Hopper
  ("h200_141", "H200 SXM (141GB)", "nvidia", "discrete", "datacenter", 141, 4800, 989, 1, 0, 700, 3.6),
  ("h200nvl_141", "H200 NVL (141GB)", "nvidia", "discrete", "datacenter", 141, 4800, 836, 1, 0, 600, 3.3),
  ("gh200_144", "GH200 (144GB)", "nvidia", "discrete", "datacenter", 144, 4900, 989, 1, 0, 1000, 3.2),
  ("h100_80", "H100 SXM (80GB)", "nvidia", "discrete", "datacenter", 80, 3350, 989, 1, 0, 700, 2.4),
  ("h100nvl_94", "H100 NVL (94GB)", "nvidia", "discrete", "datacenter", 94, 3900, 835, 1, 0, 400, 2.4),
  ("h100pcie_80", "H100 PCIe (80GB)", "nvidia", "discrete", "datacenter", 80, 2000, 756, 1, 0, 350, 2.0),
  ("h800_80", "H800 SXM (80GB)", "nvidia", "discrete", "datacenter", 80, 3350, 989, 1, 0, 700, None),
  ("h20_96", "H20 (96GB)", "nvidia", "discrete", "datacenter", 96, 4000, 148, 1, 0, 400, None),
  # NVIDIA Ada Lovelace
  ("l40s_48", "L40S (48GB)", "nvidia", "discrete", "datacenter", 48, 864, 362, 1, 0, 350, 0.9),
  ("l40_48", "L40 (48GB)", "nvidia", "discrete", "datacenter", 48, 864, 181, 1, 0, 300, 0.8),
  ("l4_24", "L4 (24GB)", "nvidia", "discrete", "datacenter", 24, 300, 121, 1, 0, 72, 0.45),
  ("rtx_6000_ada_48", "RTX 6000 Ada (48GB)", "nvidia", "discrete", "workstation", 48, 960, 364, 1, 0, 300, 1.0),
  ("rtx_5000_ada_32", "RTX 5000 Ada (32GB)", "nvidia", "discrete", "workstation", 32, 576, 261, 1, 0, 250, None),
  ("rtx_4500_ada_24", "RTX 4500 Ada (24GB)", "nvidia", "discrete", "workstation", 24, 432, 159, 1, 0, 210, None),
  ("rtx_4000_ada_20", "RTX 4000 Ada (20GB)", "nvidia", "discrete", "workstation", 20, 360, 107, 1, 0, 130, 0.25),
  ("rtx_4000_sff_ada_20", "RTX 4000 SFF Ada (20GB)", "nvidia", "discrete", "workstation", 20, 280, 77, 1, 0, 70, None),
  ("rtx_2000_ada_16", "RTX 2000 Ada (16GB)", "nvidia", "discrete", "workstation", 16, 224, 48, 1, 0, 70, None),
  ("4090_24", "RTX 4090 (24GB)", "nvidia", "discrete", "consumer", 24, 1008, 165, 1, 0, 450, 0.4),
  ("4080super_16", "RTX 4080 SUPER (16GB)", "nvidia", "discrete", "consumer", 16, 736, 104, 1, 0, 320, 0.25),
  ("4080_16", "RTX 4080 (16GB)", "nvidia", "discrete", "consumer", 16, 717, 97, 1, 0, 320, 0.25),
  ("4070tisuper_16", "RTX 4070 Ti SUPER (16GB)", "nvidia", "discrete", "consumer", 16, 672, 88, 1, 0, 285, 0.2),
  ("4070ti_12", "RTX 4070 Ti (12GB)", "nvidia", "discrete", "consumer", 12, 504, 80, 1, 0, 285, 0.15),
  ("4070super_12", "RTX 4070 SUPER (12GB)", "nvidia", "discrete", "consumer", 12, 504, 71, 1, 0, 220, 0.15),
  ("4070_12", "RTX 4070 (12GB)", "nvidia", "discrete", "consumer", 12, 504, 58, 1, 0, 200, 0.13),
  ("4060ti_16", "RTX 4060 Ti (16GB)", "nvidia", "discrete", "consumer", 16, 288, 44, 1, 0, 165, 0.12),
  ("4060ti_8", "RTX 4060 Ti (8GB)", "nvidia", "discrete", "consumer", 8, 288, 44, 1, 0, 160, None),
  ("4060_8", "RTX 4060 (8GB)", "nvidia", "discrete", "consumer", 8, 272, 30, 1, 0, 115, None),
  ("4090_laptop_16", "RTX 4090 Laptop (16GB)", "nvidia", "discrete", "consumer", 16, 576, 86, 1, 0, 150, None),
  ("4080_laptop_12", "RTX 4080 Laptop (12GB)", "nvidia", "discrete", "consumer", 12, 432, 68, 1, 0, 150, None),
  ("4070_laptop_8", "RTX 4070 Laptop (8GB)", "nvidia", "discrete", "consumer", 8, 256, 40, 1, 0, 115, None),
  ("4060_laptop_8", "RTX 4060 Laptop (8GB)", "nvidia", "discrete", "consumer", 8, 256, 29, 1, 0, 115, None),
  # NVIDIA Ampere
  ("a100_80", "A100 SXM (80GB)", "nvidia", "discrete", "datacenter", 80, 2039, 312, 0, 0, 400, 1.5),
  ("a100pcie_80", "A100 PCIe (80GB)", "nvidia", "discrete", "datacenter", 80, 1935, 312, 0, 0, 300, 1.3),
  ("a100_40", "A100 (40GB)", "nvidia", "discrete", "datacenter", 40, 1555, 312, 0, 0, 400, 1.1),
  ("a800_80", "A800 (80GB)", "nvidia", "discrete", "datacenter", 80, 2039, 312, 0, 0, 400, None),
  ("a40_48", "A40 (48GB)", "nvidia", "discrete", "datacenter", 48, 696, 150, 0, 0, 300, 0.45),
  ("a30_24", "A30 (24GB)", "nvidia", "discrete", "datacenter", 24, 933, 165, 0, 0, 165, 0.5),
  ("a10_24", "A10 (24GB)", "nvidia", "discrete", "datacenter", 24, 600, 125, 0, 0, 150, 0.6),
  ("a10g_24", "A10G (24GB)", "nvidia", "discrete", "datacenter", 24, 600, 70, 0, 0, 300, 1.0),
  ("rtx_a6000_48", "RTX A6000 (48GB)", "nvidia", "discrete", "workstation", 48, 768, 155, 0, 0, 300, 0.5),
  ("rtx_a5000_24", "RTX A5000 (24GB)", "nvidia", "discrete", "workstation", 24, 768, 111, 0, 0, 230, 0.3),
  ("rtx_a4500_20", "RTX A4500 (20GB)", "nvidia", "discrete", "workstation", 20, 640, 95, 0, 0, 200, None),
  ("rtx_a4000_16", "RTX A4000 (16GB)", "nvidia", "discrete", "workstation", 16, 448, 77, 0, 0, 140, 0.2),
  ("rtx_a2000_12", "RTX A2000 (12GB)", "nvidia", "discrete", "workstation", 12, 288, 32, 0, 0, 70, None),
  ("3090ti_24", "RTX 3090 Ti (24GB)", "nvidia", "discrete", "consumer", 24, 1008, 80, 0, 0, 450, 0.25),
  ("3090_24", "RTX 3090 (24GB)", "nvidia", "discrete", "consumer", 24, 936, 71, 0, 0, 350, 0.2),
  ("3080ti_12", "RTX 3080 Ti (12GB)", "nvidia", "discrete", "consumer", 12, 912, 68, 0, 0, 350, 0.15),
  ("3080_12", "RTX 3080 (12GB)", "nvidia", "discrete", "consumer", 12, 912, 61, 0, 0, 350, 0.13),
  ("3080_10", "RTX 3080 (10GB)", "nvidia", "discrete", "consumer", 10, 760, 60, 0, 0, 320, 0.12),
  ("3070ti_8", "RTX 3070 Ti (8GB)", "nvidia", "discrete", "consumer", 8, 608, 43, 0, 0, 290, None),
  ("3070_8", "RTX 3070 (8GB)", "nvidia", "discrete", "consumer", 8, 448, 41, 0, 0, 220, None),
  ("3060ti_8", "RTX 3060 Ti (8GB)", "nvidia", "discrete", "consumer", 8, 448, 32, 0, 0, 200, None),
  ("3060_12", "RTX 3060 (12GB)", "nvidia", "discrete", "consumer", 12, 360, 25, 0, 0, 170, 0.08),
  ("3060_8", "RTX 3060 (8GB)", "nvidia", "discrete", "consumer", 8, 240, 25, 0, 0, 170, None),
  ("3050_8", "RTX 3050 (8GB)", "nvidia", "discrete", "consumer", 8, 224, 18, 0, 0, 130, None),
  ("3080ti_laptop_16", "RTX 3080 Ti Laptop (16GB)", "nvidia", "discrete", "consumer", 16, 512, 47, 0, 0, 150, None),
  # NVIDIA Turing
  ("t4_16", "T4 (16GB)", "nvidia", "discrete", "datacenter", 16, 320, 65, 0, 0, 70, 0.3),
  ("titan_rtx_24", "TITAN RTX (24GB)", "nvidia", "discrete", "consumer", 24, 672, 33, 0, 0, 280, None),
  ("2080ti_11", "RTX 2080 Ti (11GB)", "nvidia", "discrete", "consumer", 11, 616, 27, 0, 0, 250, None),
  ("2080super_8", "RTX 2080 SUPER (8GB)", "nvidia", "discrete", "consumer", 8, 496, 22, 0, 0, 250, None),
  ("2080_8", "RTX 2080 (8GB)", "nvidia", "discrete", "consumer", 8, 448, 20, 0, 0, 215, None),
  ("2070super_8", "RTX 2070 SUPER (8GB)", "nvidia", "discrete", "consumer", 8, 448, 18, 0, 0, 215, None),
  ("2070_8", "RTX 2070 (8GB)", "nvidia", "discrete", "consumer", 8, 448, 15, 0, 0, 175, None),
  ("2060super_8", "RTX 2060 SUPER (8GB)", "nvidia", "discrete", "consumer", 8, 448, 14, 0, 0, 175, None),
  ("2060_12", "RTX 2060 (12GB)", "nvidia", "discrete", "consumer", 12, 336, 14, 0, 0, 184, None),
  ("2060_6", "RTX 2060 (6GB)", "nvidia", "discrete", "consumer", 6, 336, 13, 0, 0, 160, None),
  ("1660ti_6", "GTX 1660 Ti (6GB)", "nvidia", "discrete", "consumer", 6, 288, 11, 0, 0, 120, None),
  ("1660super_6", "GTX 1660 SUPER (6GB)", "nvidia", "discrete", "consumer", 6, 336, 10, 0, 0, 125, None),
  ("1660_6", "GTX 1660 (6GB)", "nvidia", "discrete", "consumer", 6, 192, 10, 0, 0, 120, None),
  # NVIDIA Volta
  ("v100_32", "V100 (32GB)", "nvidia", "discrete", "datacenter", 32, 900, 125, 0, 0, 300, 0.4),
  ("v100_16", "V100 (16GB)", "nvidia", "discrete", "datacenter", 16, 900, 125, 0, 0, 300, 0.3),
  # NVIDIA Pascal
  ("titan_xp_12", "TITAN Xp (12GB)", "nvidia", "discrete", "consumer", 12, 548, 12.1, 0, 0, 250, None),
  ("1080ti_11", "GTX 1080 Ti (11GB)", "nvidia", "discrete", "consumer", 11, 484, 11.3, 0, 0, 250, None),
  ("1080_8", "GTX 1080 (8GB)", "nvidia", "discrete", "consumer", 8, 320, 8.9, 0, 0, 180, None),
  ("1070ti_8", "GTX 1070 Ti (8GB)", "nvidia", "discrete", "consumer", 8, 256, 8.2, 0, 0, 180, None),
  ("1070_8", "GTX 1070 (8GB)", "nvidia", "discrete", "consumer", 8, 256, 6.5, 0, 0, 150, None),
  ("1060_6", "GTX 1060 (6GB)", "nvidia", "discrete", "consumer", 6, 192, 4.4, 0, 0, 120, None),
  # AMD CDNA 4, RDNA 4, Strix Halo
  ("mi355x_288", "Instinct MI355X (288GB)", "amd", "discrete", "datacenter", 288, 8000, 2500, 1, 1, 1400, 4.5),
  ("mi350x_288", "Instinct MI350X (288GB)", "amd", "discrete", "datacenter", 288, 8000, 2300, 1, 1, 1000, 4.0),
  ("radeon_ai_pro_r9700_32", "Radeon AI PRO R9700 (32GB)", "amd", "discrete", "workstation", 32, 640, 96, 1, 0, 300, None),
  ("rx_9070xt_16", "RX 9070 XT (16GB)", "amd", "discrete", "consumer", 16, 640, 97, 1, 0, 304, None),
  ("rx_9070_16", "RX 9070 (16GB)", "amd", "discrete", "consumer", 16, 640, 72, 1, 0, 220, None),
  ("rx_9070gre_12", "RX 9070 GRE (12GB)", "amd", "discrete", "consumer", 12, 432, 69, 1, 0, 220, None),
  ("rx_9060xt_16", "RX 9060 XT (16GB)", "amd", "discrete", "consumer", 16, 320, 51, 1, 0, 160, None),
  ("rx_9060xt_8", "RX 9060 XT (8GB)", "amd", "discrete", "consumer", 8, 320, 51, 1, 0, 150, None),
  ("ryzen_ai_max_395_128", "Ryzen AI Max+ 395 (96GB of 128GB)", "amd", "apu", "consumer", 96, 256, 59, 0, 0, 120, None),
  ("ryzen_ai_max_395_64", "Ryzen AI Max+ 395 (48GB of 64GB)", "amd", "apu", "consumer", 48, 256, 59, 0, 0, 120, None),
  # AMD CDNA 3
  ("mi325x_256", "Instinct MI325X (256GB)", "amd", "discrete", "datacenter", 256, 6000, 1307, 1, 0, 1000, 2.8),
  ("mi300x_192", "Instinct MI300X (192GB)", "amd", "discrete", "datacenter", 192, 5300, 1307, 1, 0, 750, 2.2),
  ("mi300a_128", "Instinct MI300A (128GB)", "amd", "apu", "datacenter", 128, 5300, 981, 1, 0, 550, None),
  # AMD RDNA 3
  ("radeon_pro_w7900_48", "Radeon PRO W7900 (48GB)", "amd", "discrete", "workstation", 48, 864, 122, 0, 0, 295, None),
  ("radeon_pro_w7800_32", "Radeon PRO W7800 (32GB)", "amd", "discrete", "workstation", 32, 576, 90, 0, 0, 260, None),
  ("rx_7900xtx_24", "RX 7900 XTX (24GB)", "amd", "discrete", "consumer", 24, 960, 123, 0, 0, 355, None),
  ("rx_7900xt_20", "RX 7900 XT (20GB)", "amd", "discrete", "consumer", 20, 800, 103, 0, 0, 315, None),
  ("rx_7900gre_16", "RX 7900 GRE (16GB)", "amd", "discrete", "consumer", 16, 576, 92, 0, 0, 260, None),
  ("rx_7800xt_16", "RX 7800 XT (16GB)", "amd", "discrete", "consumer", 16, 624, 74, 0, 0, 263, None),
  ("rx_7700xt_12", "RX 7700 XT (12GB)", "amd", "discrete", "consumer", 12, 432, 70, 0, 0, 245, None),
  ("rx_7600xt_16", "RX 7600 XT (16GB)", "amd", "discrete", "consumer", 16, 288, 45, 0, 0, 190, None),
  ("rx_7600_8", "RX 7600 (8GB)", "amd", "discrete", "consumer", 8, 288, 43, 0, 0, 165, None),
  # AMD CDNA 2
  ("mi250x_128", "Instinct MI250X (128GB)", "amd", "discrete", "datacenter", 128, 3277, 383, 0, 0, 560, None),
  ("mi250_128", "Instinct MI250 (128GB)", "amd", "discrete", "datacenter", 128, 3277, 362, 0, 0, 500, None),
  ("mi210_64", "Instinct MI210 (64GB)", "amd", "discrete", "datacenter", 64, 1638, 181, 0, 0, 300, None),
  # AMD RDNA 2
  ("radeon_pro_w6800_32", "Radeon PRO W6800 (32GB)", "amd", "discrete", "workstation", 32, 512, 36, 0, 0, 250, None),
  ("rx_6950xt_16", "RX 6950 XT (16GB)", "amd", "discrete", "consumer", 16, 576, 47, 0, 0, 335, None),
  ("rx_6900xt_16", "RX 6900 XT (16GB)", "amd", "discrete", "consumer", 16, 512, 46, 0, 0, 300, None),
  ("rx_6800xt_16", "RX 6800 XT (16GB)", "amd", "discrete", "consumer", 16, 512, 41, 0, 0, 300, None),
  ("rx_6800_16", "RX 6800 (16GB)", "amd", "discrete", "consumer", 16, 512, 32, 0, 0, 250, None),
  ("rx_6750xt_12", "RX 6750 XT (12GB)", "amd", "discrete", "consumer", 12, 432, 27, 0, 0, 250, None),
  ("rx_6700xt_12", "RX 6700 XT (12GB)", "amd", "discrete", "consumer", 12, 384, 26, 0, 0, 230, None),
  ("rx_6700_10", "RX 6700 (10GB)", "amd", "discrete", "consumer", 10, 320, 23, 0, 0, 175, None),
  ("rx_6650xt_8", "RX 6650 XT (8GB)", "amd", "discrete", "consumer", 8, 280, 22, 0, 0, 180, None),
  ("rx_6600xt_8", "RX 6600 XT (8GB)", "amd", "discrete", "consumer", 8, 256, 21, 0, 0, 160, None),
  ("rx_6600_8", "RX 6600 (8GB)", "amd", "discrete", "consumer", 8, 224, 18, 0, 0, 132, None),
  # AMD CDNA
  ("mi100_32", "Instinct MI100 (32GB)", "amd", "discrete", "datacenter", 32, 1229, 185, 0, 0, 300, None),
  # AMD RDNA
  ("rx_5700xt_8", "RX 5700 XT (8GB)", "amd", "discrete", "consumer", 8, 448, 20, 0, 0, 225, None),
  ("rx_5700_8", "RX 5700 (8GB)", "amd", "discrete", "consumer", 8, 448, 16, 0, 0, 180, None),
  # AMD Vega and Polaris
  ("radeon_vii_16", "Radeon VII (16GB)", "amd", "discrete", "consumer", 16, 1024, 27, 0, 0, 300, None),
  ("rx_vega64_8", "RX Vega 64 (8GB)", "amd", "discrete", "consumer", 8, 484, 25, 0, 0, 295, None),
  ("rx_vega56_8", "RX Vega 56 (8GB)", "amd", "discrete", "consumer", 8, 410, 21, 0, 0, 210, None),
  ("rx_580_8", "RX 580 (8GB)", "amd", "discrete", "consumer", 8, 256, 6.2, 0, 0, 185, None),
  # Intel
  ("gaudi3_128", "Gaudi 3 (128GB)", "intel", "discrete", "datacenter", 128, 3700, 1835, 1, 0, 900, None),
  ("arc_pro_b60_24", "Arc Pro B60 (24GB)", "intel", "discrete", "workstation", 24, 456, 98, 0, 0, 200, None),
  ("arc_pro_b50_16", "Arc Pro B50 (16GB)", "intel", "discrete", "workstation", 16, 224, 85, 0, 0, 70, None),
  ("arc_b580_12", "Arc B580 (12GB)", "intel", "discrete", "consumer", 12, 456, 117, 0, 0, 190, None),
  ("arc_b570_10", "Arc B570 (10GB)", "intel", "discrete", "consumer", 10, 380, 92, 0, 0, 150, None),
  ("arc_a770_16", "Arc A770 (16GB)", "intel", "discrete", "consumer", 16, 560, 138, 0, 0, 225, None),
  ("arc_a770_8", "Arc A770 (8GB)", "intel", "discrete", "consumer", 8, 512, 138, 0, 0, 225, None),
  ("arc_a750_8", "Arc A750 (8GB)", "intel", "discrete", "consumer", 8, 512, 118, 0, 0, 225, None),
  ("arc_a580_8", "Arc A580 (8GB)", "intel", "discrete", "consumer", 8, 512, 84, 0, 0, 185, None),
  ("arc_a380_6", "Arc A380 (6GB)", "intel", "discrete", "consumer", 6, 186, 33, 0, 0, 75, None),
]

# Apple, newest chip first: one entry per memory size, largest first.
# chip, sizes, bandwidth GB/s, GPU TFLOPS (FP16 runs at FP32 rate), package power W
APPLE = [
  ("M5 Max", [128, 64, 48], 614, 60.0, 70),
  ("M5 Max", [36], 460, 48.0, 60),
  ("M5 Pro", [64, 48, 24], 307, 30.0, 40),
  ("M5", [32, 24, 16], 153, 15.0, 25),
  ("M4 Max", [128, 64, 48], 546, 18.4, 70),
  ("M4 Max", [36], 410, 14.7, 60),
  ("M4 Pro", [64, 48, 24], 273, 9.2, 40),
  ("M4", [32, 24, 16], 120, 4.3, 25),
  ("M3 Ultra", [512, 256], 819, 32.8, 160),
  ("M3 Ultra", [96], 819, 24.6, 140),
  ("M3 Max", [128, 64, 48], 400, 16.4, 70),
  ("M3 Max", [36], 300, 12.3, 60),
  ("M3 Pro", [36, 18], 150, 7.4, 40),
  ("M3", [24, 16, 8], 100, 4.1, 22),
  ("M2 Ultra", [192, 128, 64], 800, 27.2, 120),
  ("M2 Max", [96, 64, 32], 400, 13.6, 60),
  ("M2 Pro", [32, 16], 200, 6.8, 40),
  ("M2", [24, 16, 8], 100, 3.6, 20),
  ("M1 Ultra", [128, 64], 800, 20.8, 120),
  ("M1 Max", [64, 32], 400, 10.4, 60),
  ("M1 Pro", [32, 16], 200, 5.2, 30),
  ("M1", [16, 8], 68, 2.6, 20),
]

out = []
for key, label, vendor, typ, cat, mem, bw, tf, fp8, fp4, tdp, price in GPUS:
    out.append(dict(key=key, label=label, vendor=vendor, type=typ, category=cat, memory=mem, bandwidth_gbs=bw,
                    fp16_tflops=tf, fp8=bool(fp8), fp4=bool(fp4), tdp_watts=tdp, hourly_price_usd=price))
for chip, sizes, bw, tf, tdp in APPLE:
    for size in sizes:
        out.append(dict(key=f"{chip.lower().replace(' ', '_')}_{size}", label=f"{chip} ({size}GB)", vendor="apple",
                        type="apple_silicon", category="consumer", memory=size, bandwidth_gbs=bw, fp16_tflops=tf,
                        fp8=False, fp4=False, tdp_watts=tdp, hourly_price_usd=None))

keys = [g["key"] for g in out]
assert len(keys) == len(set(keys)), "duplicate keys"
# Vendors in the picker's order; the stable sort keeps the table's newest-first order within each.
VENDORS = ["nvidia", "amd", "apple", "intel"]
path = os.path.join(os.path.dirname(__file__), "..", "src", "lib", "data", "gpus.json")
json.dump(sorted(out, key=lambda g: VENDORS.index(g["vendor"])), open(path, "w"), indent=1)
print(len(out), "GPUs written to", os.path.normpath(path))
