# =============================================================================
# full_production_near_storm_rh_sd_4K_FINAL_PATH.py
# =============================================================================
#
# PURPOSE:
# Process d4PDF +4K warming scenario (HFB_4K_GF) to compute area-averaged
# Relative Humidity (RH) and Saturation Deficit (SD) within 500km of
# tropical cyclone centers at multiple pressure levels.
#
# Track files must be placed in ./Tracks (GFI_HFB_4K_GF_mXXX.nc)
# Q and T paths are configured via templates below.
#
# AUTHOR: Jianan 
# LAST UPDATED: March 2026
# COMPATIBILITY: Python 3.8+
# DEPENDENCIES: numpy, pandas, xarray
#
# =============================================================================
# ============================ USER CONFIGURATION ============================
# =============================================================================


# =============================================================================

# only need to change the following four DIRs

# Output directory for CSV files and log file
OUTPUT_DIR = '/home/jchenfw/d4PDF/4K_GF_HKU'   # Please replace an absolute path, not relative path, I tried relative path, it does not work. Thanks!

# Path template for tropical cyclone track files
TRACK_PATH_TEMPLATE = '/home/jchenfw/d4PDF/Tracks/GFI_HFB_4K_GF_m{ensemble_id:03d}.nc'  # Please replace an absolute path, not relative path, I tried relative path, it does not work. Thanks!

# Path templates for specific humidity (Q) and temperature (T) binary files
Q_PATH_TEMPLATE = (
    '/home/jchenfw/d4PDF/Data/'
    'd4PDF_GCM/HFB_4K_GF/m{ensemble_id:03d}/atm_snp_6hr_1.25deg_HFB_4K_GF_'   # please replace this with actually path, thanks a lot!
    'm{ensemble_id:03d}_{year}{month:02d}_Q_subset.dat'
)

T_PATH_TEMPLATE = (
    '/home/jchenfw/d4PDF/Data/'                                              # please replace this with actually path, thanks a lot!
    'd4PDF_GCM/HFB_4K_GF/m{ensemble_id:03d}/atm_snp_6hr_1.25deg_HFB_4K_GF_' 
    'm{ensemble_id:03d}_{year}{month:02d}_T_subset.dat'
)


# =============================================================================

# Radius (km) around each storm center for spatial averaging
START_YEAR = 2051
END_YEAR = 2110
RADIUS_KM = 500
ensemble_id_start = 101
ensemble_id_end = 115


import numpy as np
import pandas as pd
from datetime import datetime, timedelta
import os
import logging
import xarray as xr

# Create output directory
os.makedirs(OUTPUT_DIR, exist_ok=True)

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(os.path.join(OUTPUT_DIR, "production_4K.log")),
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
    R = 6371.0  # Earth radius in km
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

# ============================= TRACK LOADING =============================
def load_cyclone_tracks_per_year(track_file: str, year: int) -> pd.DataFrame:
    if not os.path.exists(track_file):
        logging.warning(f"Track file not found: {track_file}")
        return pd.DataFrame()
    try:
        with xr.open_dataset(track_file) as ds:
            time_arr = ds['track_time'].values
            lat_arr = ds['track_lat'].values
            lon_arr = ds['track_lon'].values
            pres_arr = ds['track_pres'].values
            wind_arr = ds['track_wind'].values

            records = []
            n_storm, n_time = time_arr.shape
            for i in range(n_storm):
                for j in range(n_time):
                    if np.ma.is_masked(time_arr[i, j]) or np.isnan(time_arr[i, j]):
                        continue
                    t_val = pd.Timestamp(time_arr[i, j])
                    if t_val.year != year:
                        continue
                    if np.isnan(lat_arr[i, j]) or np.isnan(lon_arr[i, j]):
                        continue
                    records.append({
                        'SID': f"{i+1:04d}",
                        'ISO_TIME': t_val,
                        'LAT': float(lat_arr[i, j]),
                        'LON': float(lon_arr[i, j]),
                        'PRES': float(pres_arr[i, j]) if not np.isnan(pres_arr[i, j]) else np.nan,
                        'VMAX': float(wind_arr[i, j]) if not np.isnan(wind_arr[i, j]) else np.nan
                    })

            df = pd.DataFrame(records)
            if not df.empty:
                df['ISO_TIME'] = pd.to_datetime(df['ISO_TIME'])
                df = df.sort_values(['SID', 'ISO_TIME']).reset_index(drop=True)

            logging.info(f"Loaded {len(df)} valid TC points for year {year} from {os.path.basename(track_file)}")
            return df
    except Exception as e:
        logging.error(f"Failed to load tracks from {track_file}: {e}")
        return pd.DataFrame()

# ============================= DATA LOADING =============================
def load_month_field(q_path: str, t_path: str, year: int, month: int):
    """Load one month of Q and T binary data."""
    nt = days_in_month(year, month) * 4
    expected_size = nt * 12 * 97 * 288
    try:
        q_raw = np.fromfile(q_path, dtype='>f4')
        t_raw = np.fromfile(t_path, dtype='>f4')
        if len(q_raw) != expected_size or len(t_raw) != expected_size:
            logging.warning(f"Size mismatch for {q_path}: expected {expected_size}, "
                            f"got Q={len(q_raw)}, T={len(t_raw)}")
            return None, None
        q = q_raw.reshape(nt, 12, 97, 288)
        t = t_raw.reshape(nt, 12, 97, 288)
        return q, t
    except Exception as e:
        logging.error(f"Failed to load:\n Q: {q_path}\n T: {t_path}\nError: {e}")
        return None, None

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
        total_points = len(df_year)

        logging.info(f"Processing {total_points} TC points for {ens_str} year {year}")

        for idx, row in enumerate(df_year.itertuples(index=False), start=1):
            # Optional: print simple progress every 50 points (or remove if not needed)
            if idx % 50 == 0 or idx == total_points:
                logging.info(f"  → Processed {idx:4d}/{total_points} points for {ens_str} {year}")

            month = row.ISO_TIME.month
            q_path = Q_PATH_TEMPLATE.format(ensemble_id=ensemble_id, year=year, month=month)
            t_path = T_PATH_TEMPLATE.format(ensemble_id=ensemble_id, year=year, month=month)

            q_month, t_month = load_month_field(q_path, t_path, year, month)
            if q_month is None:
                continue

            # Find exact 6-hourly time index
            start_time = datetime(year, month, 1, 0)
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
            out_file = os.path.join(OUTPUT_DIR, f'4K_GF_{year}_EN_{ensemble_id:03d}.csv')
            df_out.to_csv(out_file, index=False)
            logging.info(f"Saved {len(df_out)} rows for {ens_str} {year} → {out_file}")

# ============================= MAIN EXECUTION =============================
if __name__ == '__main__':
    logging.info("=== Starting +4K near-storm RH & SD production run ===")

    # Auto-detect available GF ensembles
    ens_list = []
    for i in range(ensemble_id_start, ensemble_id_end + 1):
        track_path = TRACK_PATH_TEMPLATE.format(ensemble_id=i)
        if os.path.exists(track_path):
            ens_list.append(i)

    logging.info(f"Found {len(ens_list)} +4K GF ensembles: {ens_list}")

    for ens_id in ens_list:
        try:
            process_ensemble(ens_id)
        except Exception as e:
            logging.error(f"Ensemble m{ens_id:03d} failed: {e}", exc_info=True)

    logging.info("=== 4K PRODUCTION COMPLETE! ===")
    print("\nAll processing finished.")
    print(f"Output directory: {OUTPUT_DIR}")
    print("Please check the production_4K.log file for detailed information.")