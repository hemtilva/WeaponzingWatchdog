# Running the GPU experiments

This branch (`gpu-vectorized`) adds a vectorised simulator and DQN that run thousands of
markets at once on an NVIDIA GPU. It needs no C++ build: everything is Python + PyTorch.

## 1. Setup (Windows, once)

1. Install **Python 3.11** from python.org (tick "Add python.exe to PATH") and **Git**.
2. Install or update the **NVIDIA driver** (GeForce Experience or nvidia.com).
3. In a terminal (PowerShell):
   ```powershell
   git clone https://github.com/hemtilva/WeaponzingWatchdog.git
   cd WeaponzingWatchdog
   git checkout gpu-vectorized
   python -m venv .venv
   .venv\Scripts\activate
   pip install -r requirements.txt
   ```
4. Install the **CUDA build of PyTorch**: open https://pytorch.org/get-started/locally/,
   choose Stable / Windows / Pip / Python / the newest CUDA version listed, and run the
   command it shows. Then check:
   ```powershell
   python -c "import torch; print(torch.cuda.is_available(), torch.cuda.get_device_name(0))"
   ```
   It must print `True` and the GPU name.

## 2. Check that everything is correct (~1 minute)

```powershell
python -m pytest -q tests/test_vec_market.py tests/test_vec_dqn.py
```
All tests must pass. `test_vec_market.py` proves the GPU simulator reproduces the validated
CPU engine exactly.

## 3. Measure speed and estimated times (~2 minutes)

```powershell
python scripts/gpu/run_queue.py --benchmark
```
Send the printed numbers back: they decide how much of the queue is worth running.

## 4. Run the experiments

```powershell
python scripts/gpu/run_queue.py              # everything, in priority order
python scripts/gpu/run_queue.py --only Q1,Q2 # or selected items
```
- Results are saved after every evaluation in `results/gpu/<tag>.csv`; trained networks in
  `results/gpu/checkpoints/`. **Stopping and restarting continues where it left off.**
- Start with `--only Q1` (a few minutes): it must reproduce the known result (framing under
  the mean rule, none under the median rule or the cap).
- Keep the laptop **plugged in, on a hard surface, with sleep disabled** (Settings > System >
  Power > Screen and sleep: Never). It will get hot; that is expected and safe.

## 5. Send the results back

```powershell
git add results/gpu
git commit -m "GPU results: <which items>"
git push origin gpu-vectorized
```

## Queue

| Item | Question |
|---|---|
| Q1 | Sanity: one learner vs Nash rivals reproduces the CPU result |
| Q2 | All three sellers are DQN learners: does framing emerge? mean vs median |
| Q3 | Two learners among 8 Nash sellers (N = 10) |
| Q4 | Four learners among 6 Nash sellers (N = 10) |
| Q5 | N = 20, one learner: theory says it cannot frame alone |
| Q6 | N = 20, two learners: framing needs both together |
| Q7 | All five sellers learning (N = 5) |
| Q8 | Defences (EMA, mean + escrow, cap + escrow) against all-DQN markets |

Each run is evaluated at 25%, 50% and 100% of its training budget, so the results show
whether learning had converged.
