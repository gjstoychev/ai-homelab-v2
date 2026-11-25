# Hardware

Two nodes on the same LGA1700 platform (Z690 chipset, 13th/14th gen Intel). `formula` runs every experiment so far. `hero` is being rebuilt and gets two GPUs of its own, so each node can run tensor parallelism (TP=2) on its own and the two together can be compared across the network.

## formula — GPU compute node

| Component | Model |
|---|---|
| Motherboard | ASUS ROG Maximus Z690 Formula (BIOS 4101) |
| CPU | Intel Core i7-14700 (20 cores, 28 threads) |
| RAM | 64 GB (2×32 GB) Corsair Vengeance RGB DDR5-6000 CL40 (CMH64GX5M2D6000Z40) |
| GPU 0 | MSI GeForce RTX 5060 Ti 16 GB — PCIe 5.0 x8, CPU lanes (port 00:01.0) |
| GPU 1 | MSI GeForce RTX 5060 Ti 16 GB — PCIe 5.0 x8, CPU lanes (port 00:01.1) |
| System disk | Samsung PM9A1 512 GB NVMe M.2 |
| PSU | Fractal Design Ion+ 760W Platinum |
| Cooling | NZXT Kraken 280 RGB White |
| Case | NZXT H9 Flow (2023) |

Both GPUs share the CPU's PCIe 5.0 x16 lanes, split x8/x8, so the two slots are equal. There is no NVLink on this card and the driver reports no peer-to-peer support between the cards (`nvidia-smi topo -p2p r`: "chipset not supported"; `nvidia-smi topo -m`: `PHB`), so all GPU-to-GPU traffic goes through host memory. Idle links train down to 2.5 GT/s for power saving, so check `LnkSta` under load.

Firmware state: Intel's default CPU power limits (PL1 65 W, PL2 219 W), Resizable BAR on, console on the CPU's iGPU (UHD 770), both GPUs free for compute.

Network: one wired port (`lan0`, Marvell AQtion 10GbE, driver `atlantic`). A direct link to `hero` needs a second NIC or a switch.

## hero — second node (being rebuilt)

| Component | Model |
|---|---|
| Motherboard | ASUS ROG Maximus Z690 Hero |
| CPU | Intel Core i5-13400 (with iGPU, UHD 730) |
| RAM | 32 GB (2×16 GB) Corsair Vengeance RGB White DDR5-6000 CL36 (CMH32GX5M2E6000C36W) |
| GPUs | Two, to be filled in when installed |
| System disk | Samsung PM9A1 512 GB NVMe M.2 |
| PSU | Seasonic G12-GM 650W 80 PLUS Gold |
| Cooling | NZXT Kraken X63 RGB (RL-KRX63-R1) |
| Case | NZXT H510 Elite |

To be filled in when it is built: GPUs, BIOS version, slot wiring and link widths, P2P support, NICs, firmware settings. The PSU (650 W) should be checked against the two GPUs' power limits before the second card goes in.

## Topologies to compare

| | One node, TP=2 | Two nodes |
|---|---|---|
| Layout | 2 GPUs in `formula` (or in `hero`) | 2 GPUs in each node |
| GPU-to-GPU interconnect | PCIe through host memory, no NVLink, no P2P | PCIe inside a node, Ethernet between the nodes |
| Typical split | Tensor parallelism or layer split within one process | TP=2 inside each node and pipeline parallelism across them (e.g. vLLM with Ray, llama.cpp RPC), or one replica per node |

Tensor parallelism exchanges activations on every layer and needs a fast link; pipeline parallelism only passes activations at the stage boundary and tolerates a slower network at the price of pipeline bubbles.

## Name history

Older write-ups and the board-swap experiment use the old host names.

| Now | Before | Change |
|---|---|---|
| `formula` | `z790` | 2026-11: NZXT N7 Z790 replaced by ASUS ROG Maximus Z690 Formula; GPU1 moved from a chipset Gen4 x4 slot to CPU lanes, Gen5 x8. Measured before and after: `experiments/2026-11-boardswap/`. |
| `hero` | `b760` | 2026-11: ASRock B760 Pro RS + i5-14400F replaced by ASUS ROG Maximus Z690 Hero + i5-13400. |
