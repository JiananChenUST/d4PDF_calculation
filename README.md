# Near-Storm RH and Saturation Deficit Calculation (d4PDF)

## Overview

This project contains Python scripts to calculate area-averaged **Relative Humidity (RH)** and **Saturation Deficit (SD)** within a **500 km** radius around tropical cyclone centers using d4PDF data (Nature Run and +4K warming scenarios).

The scripts use 6-hourly specific humidity (`Q`) and temperature (`T`) on a 1.25° grid at 12 pressure levels.

---

## Required Data

### 1. Tropical Cyclone Track Files

**Location**: [`Tracks/`](https://drive.google.com/drive/folders/1KlvbMoYtHmCUNRnz75iLH-oskhVnAQGy?usp=sharing) *(Please click the link to get tracks file stored in goolge drive)*

Please place all track files in a folder named **`Tracks`** in the same directory as the Python script.

**For +4K Warming Scenarios (HFB_4K_*)**:

- Ensemble members **101–115** for the following models:
  - `MRI_HFB_4K_MR_m101.nc` … `MRI_HFB_4K_MR_m115.nc`
  - `MRI_HFB_4K_MI_m101.nc` … `MRI_HFB_4K_MI_m115.nc`
  - `MRI_HFB_4K_MP_m101.nc` … `MRI_HFB_4K_MP_m115.nc`
  - `MRI_HFB_4K_CC_m101.nc` … `MRI_HFB_4K_CC_m115.nc`
  - `MRI_HFB_4K_HA_m101.nc` … `MRI_HFB_4K_HA_m115.nc`
  - `MRI_HFB_4K_GF_m101.nc` … `MRI_HFB_4K_GF_m115.nc`

**For Nature Run (NAT)**:

- Ensemble members **1–15** for the following models:
  - `xytrackk319b_HPB_NAT_m001_1951-2010.txt` to `xytrackk319b_HPB_NAT_m015_1951-2010.txt`

### 2. 6-Hourly Model Output (`.dat` files)

**Required fields**:

- Specific humidity (`Q`)
- Air temperature (`T`)

**Specification**:

- Grid: 1.25° × 1.25° (288 lon × 97 lat)
- Pressure levels: 1000, 925, 850, 700, 600, 500, 400, 300, 250, 200, 150, 100 hPa
- Time resolution: 6-hourly
- Format: Big-endian 32-bit float (`>f4`)

---

## How to Run

### Nature Run (NAT)

1. Open `NAT.py`,  `4K_MI.py`, `4K_MR.py`,`4K_MP.py`, `4K_CC.py`, `4K_GF.py`, `4K_HA.py`
2. Update only these four lines with your correct paths:
   ```python
   TRACK_PATH_TEMPLATE = "..."
   Q_PATH_TEMPLATE     = "..."
   T_PATH_TEMPLATE     = "..."
   OUTPUT_DIR          = "..."
   ```
