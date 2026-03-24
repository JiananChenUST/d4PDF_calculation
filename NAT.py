# =============================================================================
# =============================================================================
#
# PURPOSE:
# This script processes d4PDF Nature Run (NAT) ensemble data to compute area-averaged
# Relative Humidity (RH) and Saturation Deficit (SD) within a specified radius
# around tropical cyclone (TC) centers at multiple pressure levels.
#
# All file paths are fully independent and easy to adapt to different systems.
#
# AUTHOR: Your Name
# LAST UPDATED: March 2026
# COMPATIBILITY: Python 3.8+
# DEPENDENCIES: numpy, pandas, xarray
#
# =============================================================================
# ============================ IMPORTANT USER CONFIGURATION ============================
# =============================================================================
#
# IMPORTANT NOTES FOR USERS WITH DIFFERENT DATA LOCATIONS:
#
# 1. OUTPUT_DIR
#    → Change this to the directory where you want the output CSV files and the log file to be saved.
#
# 2. TRACK_PATH_TEMPLATE
#    → I have provided a TRACK dataset. 

#
# 3. Q_PATH_TEMPLATE and T_PATH_TEMPLATE
#    → These are the most critical paths to modify if your data is stored elsewhere.
#    → They define the exact location of the specific humidity (Q) and temperature (T) binary files.
#    → The templates include three placeholders:
#         {ensemble_id:03d}   → ensemble member number (padded with zeros to 3 digits)
#         {year}              → four-digit year
#         {month:02d}         → two-digit month (01–12)
#
# How to adapt for your system:
#    - Replace the base directories (e.g., '/home/jchenfw/d4PDF/Data/') with your own paths.
#    - Adjust any folder names or file naming patterns to match your data organization.
#    - Do NOT remove or rename the {ensemble_id:03d}, {year}, or {month:02d} placeholders.
#
#
# Only the four lines in the "USER CONFIGURATION" section below normally need to be changed.
#
# =============================================================================

# ================== ONLY CHANGE THESE FOUR Paths ==================

# Output directory for CSV files and log
OUTPUT_DIR = '/home/jchenfw/d4PDF/NAT_HKU'# Please replace an absolute path, not relative path, I tried relative path, it does not work. Thanks alot!

# Track file path template 
# Please replace an absolute path, not relative path, I tried relative path, it does not work. Thanks a lot!
TRACK_PATH_TEMPLATE = (
    '/home/jchenfw/d4PDF/Tracks/' 
    'xytrackk319b_HPB_NAT_m{ensemble_id:03d}_1951-2010.txt' 
)

# Specific humidity (Q) file path template
Q_PATH_TEMPLATE = (
    '/home/jchenfw/d4PDF/Data/'
    'd4PDF_GCM/HPB_NAT/m{ensemble_id:03d}/atm_snp_6hr_1.25deg_HPB_NAT_m{ensemble_id:03d}_'
    '{year}{month:02d}_Q_subset.dat'
)

# Temperature (T) file path template 
T_PATH_TEMPLATE = (
    '/home/jchenfw/d4PDF/Data/'      
    'd4PDF_GCM/HPB_NAT/m{ensemble_id:03d}/atm_snp_6hr_1.25deg_HPB_NAT_m{ensemble_id:03d}_'
    '{year}{month:02d}_T_subset.dat'
)


# ==================================================================

START_YEAR = 1951
END_YEAR = 2010
RADIUS_KM = 500 # Radius (km) around each storm center for spatial averaging
ensemble_id_start = 1
ensemble_id_end = 15
# =============================================================================
# PATH CONFIGURATION COMPLETE
# Only modify the four lines above if your data is stored in a different location
# or uses a different naming convention.
# =============================================================================

import numpy as np
import pandas as pd
from datetime import datetime, timedelta
import os
import logging
import xarray as xr   # Note: xarray is imported but not used in this version;
                      # it can be safely removed if desired.

# Create output directory if it does not exist
os.makedirs(OUTPUT_DIR, exist_ok=True)

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(os.path.join(OUTPUT_DIR, "production.log")),
        logging.StreamHandler()
    ]
)

# ============================= GRID DEFINITION =============================
LON = np.arange(0, 360, 1.25)
LAT = np.linspace(-60, 60, 97)
LEV = np.array([1000, 925, 850, 700, 600, 500, 400, 300, 250, 200, 150, 100])
lon_grid, lat_grid = np.meshgrid(LON, LAT)

# ============================= HELPER FUNCTIONS =============================
def days_in_month(year: int, month: int) -> int:
    """Return the number of days in a given month, correctly handling leap years."""
    if month == 2:
        return 29 if (year % 4 == 0 and (year % 100 != 0 or year % 400 == 0)) else 28
    return [31, 28, 31, 30, 31, 30, 31, 31, 30, 31, 30, 31][month - 1]

def distance_mask(storm_lat: float, storm_lon: float) -> np.ndarray:
    """Returns boolean mask of grid points within RADIUS_KM of the storm center."""
    R = 6371.0
    storm_lat_rad = np.deg2rad(storm_lat)
    storm_lon_rad = np.deg2rad(storm_lon)
    lat_rad = np.deg2rad(lat_grid)
    lon_rad = np.deg2rad(lon_grid)
    dlat = lat_rad - storm_lat_rad
    dlon = lon_rad - storm_lon_rad
    a = np.sin(dlat / 2)**2 + np.cos(lat_rad) * np.cos(storm_lat_rad) * np.sin(dlon / 2)**2
    c = 2 * np.arcsin(np.sqrt(a))
    return R * c <= RADIUS_KM

def sat_vp(T: np.ndarray) -> np.ndarray:
    """Magnus formula for saturation vapor pressure (hPa)."""
    return 6.112 * np.exp(17.67 * (T - 273.15) / (T - 273.15 + 243.5))

def sat_q(es: np.ndarray, p: float) -> np.ndarray:
    """Saturation specific humidity (kg/kg)."""
    rs = 0.622 * es / (p - es)
    return rs / (1 + rs)

# ============================= DATA LOADING =============================
def load_month_field(q_path: str, t_path: str, year: int, month: int):
    """Load one month of Q and T data. Returns (q_array, t_array) or (None, None)."""
    nt = days_in_month(year, month) * 4
    expected_size = nt * 12 * 97 * 288
    try:
        q_raw = np.fromfile(q_path, dtype='>f4')
        t_raw = np.fromfile(t_path, dtype='>f4')
        if len(q_raw) != expected_size or len(t_raw) != expected_size:
            logging.warning(f"Size mismatch for {os.path.basename(q_path)}: "
                            f"expected {expected_size}, got Q={len(q_raw)}, T={len(t_raw)}")
            return None, None
        q = q_raw.reshape(nt, 12, 97, 288)
        t = t_raw.reshape(nt, 12, 97, 288)
        logging.debug(f"Loaded {os.path.basename(q_path)} → shape {q.shape}")
        return q, t
    except Exception as e:
        logging.error(f"Failed to load:\n Q: {q_path}\n T: {t_path}\nError: {e}")
        return None, None

# ============================= MAIN PROCESSING =============================
def process_ensemble(ensemble_id: int):
    """Process a single ensemble member."""
    ens_str = f"m{ensemble_id:03d}"
    # Build track file path
    track_file = TRACK_PATH_TEMPLATE.format(ensemble_id=ensemble_id)
    if not os.path.exists(track_file):
        logging.warning(f"Track file not found: {track_file}")
        return

    # Load tracks from fixed-width text file
    tracks = []
    with open(track_file, 'r') as f:
        for _ in range(3):  # skip header
            f.readline()
        for line in f:
            try:
                y = int(line[14:21])
                if not (START_YEAR <= y <= END_YEAR):
                    continue
                m = int(line[21:24])
                tracks.append({
                    'year': y,
                    'month': m,
                    'ISO_TIME': datetime(y, m, int(line[24:27]), int(line[27:30])),
                    'LAT': float(line[40:48]),
                    'LON': float(line[32:40]),
                    'PRES': float(line[64:72]),
                    'VMAX': float(line[56:64]),
                    'SID': f"{int(line[0:7]):04d}"
                })
            except Exception:
                continue

    df_all = pd.DataFrame(tracks)
    logging.info(f"Ensemble {ens_str}: {len(df_all)} TC points in the requested period")

    for year in range(START_YEAR, END_YEAR + 1):
        df_year = df_all[df_all['year'] == year]
        if df_year.empty:
            continue

        results = []
        total_points = len(df_year)

        logging.info(f"Processing {total_points} TC points for {ens_str} year {year}")

        for idx, row in enumerate(df_year.itertuples(index=False), start=1):
            # Optional simple progress feedback
            if idx % 50 == 0 or idx == total_points:
                logging.info(f"  → Processed {idx:4d}/{total_points} points for {ens_str} {year}")

            # Build Q and T file paths
            q_path = Q_PATH_TEMPLATE.format(
                ensemble_id=ensemble_id,
                year=year,
                month=row.month
            )
            t_path = T_PATH_TEMPLATE.format(
                ensemble_id=ensemble_id,
                year=year,
                month=row.month
            )

            q_month, t_month = load_month_field(q_path, t_path, year, row.month)
            if q_month is None:
                continue

            # Find time index
            start_time = datetime(year, row.month, 1, 0)
            times = [start_time + timedelta(hours=6 * i) for i in range(len(q_month))]
            try:
                t_idx = times.index(row.ISO_TIME)
            except ValueError:
                logging.warning(f"Time {row.ISO_TIME} not found in monthly data")
                continue

            mask = distance_mask(row.LAT, row.LON)

            row_data = {
                'SID': row.SID,
                'ISO_TIME': row.ISO_TIME,
                'LAT': row.LAT,
                'LON': row.LON,
                'PRES': row.PRES,
                'VMAX': row.VMAX
            }

            for lev_idx, p in enumerate(LEV):
                q_lev = q_month[t_idx, lev_idx]
                t_lev = t_month[t_idx, lev_idx]
                valid = mask & np.isfinite(q_lev) & np.isfinite(t_lev)
                if not valid.any():
                    row_data[f'RH_{p}hPa'] = np.nan
                    row_data[f'SD_{p}hPa'] = np.nan
                    continue

                es = sat_vp(t_lev[valid])
                qs = sat_q(es, p)
                rh = q_lev[valid] / qs
                sd = qs - q_lev[valid]

                row_data[f'RH_{p}hPa'] = np.nanmean(rh)
                row_data[f'SD_{p}hPa'] = np.nanmean(sd)

            results.append(row_data)

        if results:
            df_out = pd.DataFrame(results)
            out_file = os.path.join(OUTPUT_DIR, f'NAT_{year}_EN_{ensemble_id:03d}.csv')
            df_out.to_csv(out_file, index=False)
            logging.info(f"Saved {len(df_out)} rows for {ens_str} {year} → {out_file}")

# ============================= MAIN EXECUTION =============================
if __name__ == '__main__':
    logging.info("=== Starting near-storm RH & SD production run ===")

    # Automatically detect available ensemble members
    ens_list = []
    for i in range(ensemble_id_start, ensemble_id_end + 1):
        track_path = TRACK_PATH_TEMPLATE.format(ensemble_id=i)
        if os.path.exists(track_path):
            ens_list.append(i)

    logging.info(f"Found {len(ens_list)} ensemble members: {ens_list}")

    for ens_id in ens_list:
        try:
            process_ensemble(ens_id)
        except Exception as e:
            logging.error(f"Ensemble m{ens_id:03d} failed: {e}", exc_info=True)

    logging.info("=== FULL PRODUCTION COMPLETE! ===")
    print("\nAll processing finished.")
    print(f"Output directory: {OUTPUT_DIR}")
    print("Please check the production.log file for detailed information.")