# =============================================================================
# full_production_near_storm_tcwv_multi_rad_NAT.py
# =============================================================================
#
# PURPOSE:
# Process d4PDF NAT simulations to compute:
#   - area-averaged TCWV at radii =  100, ..., 500 km
#   - area-averaged TCWV in the 600–800 km annulus
# using the validated calculate_full_tcwv function.
#
# AUTHOR: Adapted for Nature Geoscience (Jianan Chen)
# LAST UPDATED: April 2026
# =============================================================================

import numpy as np
import pandas as pd
from datetime import datetime, timedelta
import os
import logging
from tqdm import tqdm

# ============================ USER CONFIGURATION ============================
OUTPUT_DIR = './NAT_HKU_TCWV'   

TRACK_PATH_TEMPLATE = (
    './Tracks/'
    'xytrackk319b_HPB_NAT_m{ensemble_id:03d}_1951-2010.txt'
)

# <<< PLEASE UPDATE WITH YOUR ACTUAL NAT Q BINARY PATH >>>
Q_PATH_TEMPLATE = (
    #'nc.x/'
    #'HPB_NAT_m{ensemble_id:03d}/d_monit_a_nc/{year}/'
    #'q.nc'
)

START_YEAR = 1951
END_YEAR   = 2010
ensemble_id_start = 1
ensemble_id_end   = 6      # change to 6 (or higher) when ready

RADII_KM = np.arange(100, 501, 50)   # 100, 150, ..., 500 km



os.makedirs(OUTPUT_DIR, exist_ok=True)

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(os.path.join(OUTPUT_DIR, "production_NAT_TCWV_multi_rad.log")),
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
    if month == 2:
        return 29 if (year % 4 == 0 and (year % 100 != 0 or year % 400 == 0)) else 28
    return [31,28,31,30,31,30,31,31,30,31,30,31][month-1]

def distance_mask(storm_lat: float, storm_lon: float, radius_km: float) -> np.ndarray:
    R = 6371.0
    storm_lat_rad = np.deg2rad(storm_lat)
    storm_lon_rad = np.deg2rad(storm_lon)
    lat_rad = np.deg2rad(lat_grid)
    lon_rad = np.deg2rad(lon_grid)
    dlat = lat_rad - storm_lat_rad
    dlon = lon_rad - storm_lon_rad
    a = np.sin(dlat/2)**2 + np.cos(lat_rad)*np.cos(storm_lat_rad)*np.sin(dlon/2)**2
    c = 2 * np.arcsin(np.sqrt(a))
    return R * c <= radius_km

def calculate_full_tcwv(q_month, t_idx, surface_pres_hpa):
    """Validated full-grid TCWV (kg m⁻²) from track-central PRES to model top"""
    g = 9.80665
    tcwv = np.zeros((97, 288), dtype=float)
    prev_p = float(surface_pres_hpa)
    for i, p_lev in enumerate(LEV):
        if p_lev > prev_p:
            continue
        q_lev = q_month[t_idx, i]
        valid = np.isfinite(q_lev) & (q_lev >= 0)
        if valid.any():
            dp_pa = (prev_p - p_lev) * 100.0
            tcwv[valid] += q_lev[valid] * dp_pa
        prev_p = p_lev
    return tcwv / g

# ============================= TRACK LOADING (NAT text file) =============================
def load_cyclone_tracks_per_year(track_file: str, year: int) -> pd.DataFrame:
    if not os.path.exists(track_file):
        logging.warning(f"Track file not found: {track_file}")
        return pd.DataFrame()
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
                    'SID': f"{int(line[0:7]):04d}",
                    'ISO_TIME': datetime(y, m, int(line[24:27]), int(line[27:30])),
                    'LAT': float(line[40:48]),
                    'LON': float(line[32:40]),
                    'PRES': float(line[64:72]),
                    'VMAX': float(line[56:64])
                })
            except Exception:
                continue
    df = pd.DataFrame(tracks)
    if not df.empty:
        df = df.sort_values('ISO_TIME').reset_index(drop=True)
    logging.info(f"Loaded {len(df)} valid TC points for year {year} from {os.path.basename(track_file)}")
    return df

# ============================= DATA LOADING (binary Q) =============================
def load_month_q(q_path: str, year: int, month: int):
    nt = days_in_month(year, month) * 4
    expected_size = nt * 12 * 97 * 288
    try:
        q_raw = np.fromfile(q_path, dtype='>f4')
        if len(q_raw) != expected_size:
            logging.warning(f"Size mismatch for {q_path}: expected {expected_size}, got {len(q_raw)}")
            return None
        q = q_raw.reshape(nt, 12, 97, 288)
        return q
    except Exception as e:
        logging.error(f"Failed to load Q: {q_path}\nError: {e}")
        return None

# ============================= MAIN PROCESSING =============================
def process_ensemble(ensemble_id: int):
    ens_str = f"m{ensemble_id:03d}"
    track_file = TRACK_PATH_TEMPLATE.format(ensemble_id=ensemble_id)
    if not os.path.exists(track_file):
        logging.warning(f"Track file not found: {track_file}")
        return

    for year in range(START_YEAR, END_YEAR + 1):
        df_year = load_cyclone_tracks_per_year(track_file, year)
        if df_year.empty:
            continue

        results = []
        pbar = tqdm(df_year.itertuples(index=False), total=len(df_year),
                    desc=f"{ens_str} {year}", leave=False)

        for row in pbar:
            month = row.ISO_TIME.month
            q_path = Q_PATH_TEMPLATE.format(ensemble_id=ensemble_id, year=year, month=month)
            q_month = load_month_q(q_path, year, month)
            if q_month is None:
                continue

            start_time = datetime(year, month, 1, 0)
            times = [start_time + timedelta(hours=6 * i) for i in range(len(q_month))]
            try:
                t_idx = times.index(row.ISO_TIME)
            except ValueError:
                continue

            row_data = {
                'SID': row.SID,
                'ISO_TIME': row.ISO_TIME,
                'LAT': row.LAT,
                'LON': row.LON,
                'PRES': row.PRES,
                'VMAX': row.VMAX
            }

            # Compute TCWV
            surface_p = row.PRES if not np.isnan(row.PRES) else 1013.25
            tcwv_full = calculate_full_tcwv(q_month, t_idx, surface_p)

            # Multi-radius TCWV (100–500 km)
            for r in RADII_KM:
                mask = distance_mask(row.LAT, row.LON, r)
                valid = mask & np.isfinite(tcwv_full)
                mean_tcwv = np.nanmean(tcwv_full[valid]) if valid.any() else np.nan
                row_data[f'TCWV_{r:03d}km'] = float(mean_tcwv)

            # 600–800 km annulus
            mask_800 = distance_mask(row.LAT, row.LON, 800)
            mask_600 = distance_mask(row.LAT, row.LON, 600)
            annulus_mask = mask_800 & ~mask_600 & np.isfinite(tcwv_full)
            mean_annulus = np.nanmean(tcwv_full[annulus_mask]) if annulus_mask.any() else np.nan
            row_data['TCWV_annulus_600_800km'] = float(mean_annulus)

            results.append(row_data)

        if results:
            df_out = pd.DataFrame(results)
            out_file = os.path.join(OUTPUT_DIR, f'NAT_TCWV_multi_rad_{year}_EN_{ensemble_id:03d}.csv')
            df_out.to_csv(out_file, index=False)
            logging.info(f"Saved {len(df_out)} rows with multi-radius + annulus TCWV for {ens_str} {year} → {out_file}")

# ============================= MAIN EXECUTION =============================
if __name__ == '__main__':
    logging.info("=== Starting NAT TCWV multi-radius + 600-800 km annulus production ===")
    ens_list = []
    for i in range(ensemble_id_start, ensemble_id_end + 1):
        track_path = TRACK_PATH_TEMPLATE.format(ensemble_id=i)
        if os.path.exists(track_path):
            ens_list.append(i)
    logging.info(f"Found {len(ens_list)} NAT ensembles: {ens_list}")

    for ens_id in ens_list:
        try:
            process_ensemble(ens_id)
        except Exception as e:
            logging.error(f"Ensemble m{ens_id:03d} failed: {e}", exc_info=True)

    logging.info("=== NAT PRODUCTION COMPLETE ===")
    print("\nAll processing finished.")
    print(f"Output directory: {OUTPUT_DIR}")
    print("Check production_NAT_TCWV_multi_rad.log for details.")