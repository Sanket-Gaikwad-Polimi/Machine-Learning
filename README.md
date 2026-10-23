# Anomaly Detection on MVTec AD Screw

Machine Learning for Mechanical Systems, Politecnico di Milano, 2026–27, Project 3.

This repository holds the project code **and the dataset** (`screw/`). Follow the steps below once to get set up on your laptop (Windows).

## 1. Get the project files (code + dataset)

- Accept the GitHub invitation from Sanket (check your email).
- Install Git: https://git-scm.com (default options).
- In VS Code: `Ctrl+Shift+P` → **Git: Clone** → paste the repository link `https://github.com/Sanket-Gaikwad-Polimi/Machine-Learning` → choose a folder → **Open**.

The download is about 250 MB because it includes the dataset, so it can take a few minutes.
After cloning, the images are at `screw\train\good\000.png`, and so on. Nothing else to download.

## 2. Install Python 3.12.10 (everyone uses this exact version)

- Download: https://www.python.org/ftp/python/3.12.10/python-3.12.10-amd64.exe
- In the installer, leave **"Add python.exe to PATH" unticked**, then click **Install Now**.
- Check in a terminal: `py -0` should list 3.12.

## 3. Create the project environment

Open the project in VS Code (**File → Open Folder**), then open a terminal (**Terminal → New Terminal**) and run:

```powershell
py -3.12 -m venv .venv
.venv\Scripts\Activate.ps1
python --version
```

It must print `Python 3.12.10`, and the terminal prompt must start with `(.venv)`.

If you get *"running scripts is disabled on this system"*, run the line below, type `Y`, and activate again:

```powershell
Set-ExecutionPolicy -Scope CurrentUser RemoteSigned
```

## 4. Install the packages (same versions for everyone)

Run `nvidia-smi` first.

**If it shows an NVIDIA GPU:**

```powershell
pip install torch==2.14.1 torchvision==0.29.1 --index-url https://download.pytorch.org/whl/cu126
```

**If the command is not recognised (no NVIDIA GPU):**

```powershell
pip install torch==2.14.1 torchvision==0.29.1 --index-url https://download.pytorch.org/whl/cpu
```

**Then everyone runs:**

```powershell
pip install numpy==2.5.2 scipy==1.18.1 scikit-learn==1.9.1 matplotlib==3.11.2 pandas==3.1.0 pillow==12.3.0 tqdm==4.70.1 jupyter==1.1.1 ipykernel==7.4.0
```

## 5. Test

```powershell
python check_env.py
```

If it ends with `Pretrained backbone loaded OK`, you are ready.

## Every time you work

Check that the terminal prompt starts with `(.venv)`. If it doesn't, run:

```powershell
.venv\Scripts\Activate.ps1
```

## Dataset licence

`screw/` is the screw category of the MVTec AD dataset, licensed CC BY-NC-SA 4.0 (non-commercial use, attribution required; see `screw/license.txt`).
Citation: Bergmann, Fauser, Sattlegger, Steger, "MVTec AD – A Comprehensive Real-World Dataset for Unsupervised Anomaly Detection", CVPR 2019.
