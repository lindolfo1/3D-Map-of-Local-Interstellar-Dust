from astroquery.gaia import Gaia
import astropy.units as u 
from astropy.coordinates import SkyCoord
import pandas as pd 
import numpy as np
import time
from tqdm import tqdm
import os


def import_data():
    print("fetching data...")
    job = Gaia.launch_job_async(
        "select ra, dec, parallax, parallax_error, "
        "phot_g_mean_mag, phot_bp_mean_mag, phot_rp_mean_mag "
        "from gaiadr3.gaia_source "
        "where parallax_over_error > 5 "
        "and phot_bp_mean_mag is not null and phot_rp_mean_mag is not null "
    )
    print("job COMPLETED; fetching results...")
    r = job.get_results()
    df = r.to_pandas()
    df.to_csv("01-gaia_results.csv", index=False)


def import_data_chunked(outfile="01-gaia_results_full.csv",
                        l_step=30, b_step=30):
    l_chunks = list(range(0, 360, l_step))
    b_chunks = list(range(-90, 90, b_step))
    total_chunks = len(l_chunks) * len(b_chunks)
    
    print(f"querying {total_chunks} sky chunks "
          f"({l_step}° × {b_step}° each)\n")
    
    chunks = []
    total_rows = 0
    capped_chunks = []
    start_time = time.time()
    
    with tqdm(total=total_chunks, desc="chunks", unit="chunk") as pbar:
        for l_start in l_chunks:
            for b_start in b_chunks:
                l_end = l_start + l_step
                b_end = b_start + b_step
                chunk_start = time.time()
                
                query = f"""
                    SELECT source_id, ra, dec, l, b, parallax, parallax_error,
                           phot_g_mean_mag, phot_bp_mean_mag, phot_rp_mean_mag
                    FROM gaiadr3.gaia_source
                    WHERE parallax_over_error > 5
                      AND parallax > 0.5
                      AND phot_bp_mean_mag IS NOT NULL
                      AND phot_rp_mean_mag IS NOT NULL
                      AND l >= {l_start} AND l < {l_end}
                      AND b >= {b_start} AND b < {b_end}
                """
                
                try:
                    job = Gaia.launch_job_async(query)
                    df_chunk = job.get_results().to_pandas()
                    n = len(df_chunk)
                    elapsed = time.time() - chunk_start
                    
                    chunks.append(df_chunk)
                    total_rows += n
                    
                    capped = n >= 3_000_000
                    if capped:
                        capped_chunks.append((l_start, b_start))
                    
                    pbar.set_postfix({
                        "last": f"l={l_start},b={b_start:+d}",
                        "rows": f"{n:,}",
                        "time": f"{elapsed:.0f}s",
                        "total": f"{total_rows:,}",
                        "capped": len(capped_chunks),
                    })
                    
                except Exception as e:
                    pbar.write(f"  ERROR at l={l_start}, b={b_start}: {e}")
                    with open("import_failures.log", "a") as f:
                        f.write(f"{l_start},{b_start},{e}\n")
                
                pbar.update(1)
    
    total_elapsed = time.time() - start_time
    
    print(f"\nconcatenating {len(chunks)} chunks...")
    df = pd.concat(chunks, ignore_index=True)
    
    print(f"deduplicating (boundary overlaps)...")
    before = len(df)
    df = df.drop_duplicates(subset="source_id")
    print(f"  dropped {before - len(df):,} duplicates")
    
    print(f"\nsaving to {outfile}...")
    df.to_csv(outfile, index=False)
    
    print(f"\n{'='*50}")
    print(f"DONE in {total_elapsed/60:.1f} minutes")
    print(f"total stars: {len(df):,}")
    print(f"sky coverage: l=[0,360), b=[-90,90)")
    if capped_chunks:
        print(f"\n{len(capped_chunks)} chunks hit the 3M row cap:")
        for l, b in capped_chunks:
            print(f"  l={l}, b={b:+d}  — consider subdividing")
    print(f"{'='*50}")


def clean_data():
    print("cleaning data")
    df = pd.read_csv("01-gaia_results_full.csv")
    print(f"{len(df):,} stars")
    print(f"ra: {df['ra'].min():.1f} to {df['ra'].max():.1f}")
    print(f"dec: {df['dec'].min():.1f} to {df['dec'].max():.1f}")
    df["bp_rp"] = df["phot_bp_mean_mag"] - df["phot_rp_mean_mag"]
    df["distance"] = 1000 / df["parallax"]
    df.to_parquet("02-cleaned_results_full.parquet")


def generate_abs_mag():
    df = pd.read_parquet("02-cleaned_results_full.parquet")
    df["abs_mag"] = df["phot_g_mean_mag"] - 2.5 * np.log10((df["distance"] / 10) ** 2)
    df.to_parquet("03-abs_mag_full.parquet")


def import_failed_chunks(failed_regions, outfile="01-gaia_results_full.csv",
                         l_step=30, b_step=30, max_retries=3, retry_delay=10):
    """Re-query specific chunks that failed in the main import.
    
    failed_regions: list of (l_start, b_start) tuples
    """
    print(f"retrying {len(failed_regions)} failed chunks\n")
    
    chunks = []
    
    with tqdm(total=len(failed_regions), desc="retries", unit="chunk") as pbar:
        for l_start, b_start in failed_regions:
            l_end = l_start + l_step
            b_end = b_start + b_step
            
            query = f"""
                SELECT source_id, ra, dec, l, b, parallax, parallax_error,
                       phot_g_mean_mag, phot_bp_mean_mag, phot_rp_mean_mag
                FROM gaiadr3.gaia_source
                WHERE parallax_over_error > 5 AND parallax > 2
                  AND phot_bp_mean_mag IS NOT NULL
                  AND phot_rp_mean_mag IS NOT NULL
                  AND l >= {l_start} AND l < {l_end}
                  AND b >= {b_start} AND b < {b_end}
            """
            
            for attempt in range(1, max_retries + 1):
                try:
                    job = Gaia.launch_job_async(query)
                    df_chunk = job.get_results().to_pandas()
                    chunks.append(df_chunk)
                    pbar.set_postfix({
                        "last": f"l={l_start},b={b_start:+d}",
                        "rows": f"{len(df_chunk):,}",
                        "attempt": attempt,
                    })
                    break
                except Exception as e:
                    pbar.write(f"  attempt {attempt}/{max_retries} at "
                              f"l={l_start},b={b_start}: {e}")
                    if attempt < max_retries:
                        time.sleep(retry_delay)
                    else:
                        pbar.write(f"  GIVING UP on l={l_start},b={b_start}")
            
            pbar.update(1)
    
    if not chunks:
        print("no chunks recovered")
        return
    
    new_data = pd.concat(chunks, ignore_index=True)
    print(f"\nrecovered {len(new_data):,} stars")
    
    if os.path.exists(outfile):
        print(f"appending to existing {outfile}")
        existing = pd.read_csv(outfile)
        combined = pd.concat([existing, new_data], ignore_index=True)
        before = len(combined)
        combined = combined.drop_duplicates(subset="source_id")
        print(f"  dropped {before - len(combined):,} duplicates after merge")
        combined.to_csv(outfile, index=False)
        print(f"saved {len(combined):,} total stars to {outfile}")
    else:
        new_data.to_csv(outfile, index=False)
        print(f"saved to {outfile}")

if __name__ == "__main__":
    # import_data_chunked()
    import_failed_chunks([(0, -30), (30, -30)])
    clean_data()
    generate_abs_mag()