import pandas as pd 
import numpy as np
from scipy.spatial import cKDTree
from astropy.coordinates import SkyCoord
import astropy.units as u
from tqdm import tqdm
import matplotlib.pyplot as plt 
import os

STARS_CACHE = "05-stars_lb.parquet"
FIELD_CACHE = "06-dust_field.npz"


def build_kd_tree(stars):
    l_rad = np.deg2rad(stars["l"])
    b_rad = np.deg2rad(stars["b"])
    stars["x"] = stars["d"] * np.cos(b_rad) * np.cos(l_rad)
    stars["y"] = stars["d"] * np.cos(b_rad) * np.sin(l_rad)
    stars["z"] = stars["d"] * np.sin(b_rad)

    positions = stars[["x", "y", "z"]].to_numpy()
    
    print("building kd-tree...")
    tree = cKDTree(positions)
    print("finished building tree.")
    return tree


def dust_field(stars, tree, sigma_r=30, theta_res=0.2, d_res=25, min_stars=3):
    print(f"DEBUG: {len(stars):,} stars in tree")
    print(f"DEBUG: stars sky range: l={stars['l'].min():.0f}-{stars['l'].max():.0f}, "
          f"b={stars['b'].min():.0f}-{stars['b'].max():.0f}")
    
    cutoff = 3 * sigma_r

    x_arr = stars["x"].to_numpy()
    y_arr = stars["y"].to_numpy()
    z_arr = stars["z"].to_numpy()
    E_arr = stars["E"].to_numpy()
    inv_2sig2 = 1.0 / (2 * sigma_r ** 2)

    l_min, l_max = stars["l"].min(), stars["l"].max()
    b_min, b_max = stars["b"].min(), stars["b"].max()
    l_grid = np.arange(l_min, l_max, theta_res)
    b_grid = np.arange(b_min, b_max, theta_res)
    d_grid = np.arange(50, 500, d_res)

    E_cumulative = np.full((len(l_grid), len(b_grid), len(d_grid)), np.nan)
    total = len(l_grid) * len(b_grid) * len(d_grid)

    with tqdm(total=total, desc="building field") as pbar:
        for i, lq in enumerate(l_grid):
            cos_lq, sin_lq = np.cos(np.deg2rad(lq)), np.sin(np.deg2rad(lq))
            for j, bq in enumerate(b_grid):
                cos_bq, sin_bq = np.cos(np.deg2rad(bq)), np.sin(np.deg2rad(bq))
                for k, dq in enumerate(d_grid):
                    xq = dq * cos_bq * cos_lq
                    yq = dq * cos_bq * sin_lq
                    zq = dq * sin_bq
                    idx = tree.query_ball_point([xq, yq, zq], r=cutoff)
                    if len(idx) < min_stars:
                        pbar.update(1)
                        continue
                    idx = np.asarray(idx)
                    dx = x_arr[idx] - xq
                    dy = y_arr[idx] - yq
                    dz = z_arr[idx] - zq
                    weights = np.exp(-(dx*dx + dy*dy + dz*dz) * inv_2sig2)
                    sum_w = weights.sum()
                    if sum_w > min_stars:
                        E_cumulative[i, j, k] = (weights * E_arr[idx]).sum() / sum_w
                    pbar.update(1)

    density = np.zeros_like(E_cumulative)
    density[:, :, 1:-1] = (E_cumulative[:, :, 2:] - E_cumulative[:, :, :-2]) / (2 * d_res)
    density = np.maximum(density, 0)
    return E_cumulative, density, (l_grid, b_grid, d_grid)


def plot_dust_projection(E_cumulative, density, grids):
    l_grid, b_grid, d_grid = grids
    
    total_dust = np.nansum(density, axis=2) * (d_grid[1] - d_grid[0])
    
    L, B = np.meshgrid(l_grid, b_grid, indexing="ij")
    
    L_flat = L.ravel()
    B_flat = B.ravel()
    dust_flat = total_dust.ravel()
    
    valid = ~np.isnan(dust_flat) & (dust_flat > 0)
    L_flat = L_flat[valid]
    B_flat = B_flat[valid]
    dust_flat = dust_flat[valid]
    
    dust_norm = dust_flat / np.percentile(dust_flat, 99)
    dust_norm = np.clip(dust_norm, 0, 1)
    
    colors = np.zeros((len(dust_norm), 4))
    colors[:, 0] = 0.55
    colors[:, 1] = 0.27
    colors[:, 2] = 0.07
    colors[:, 3] = dust_norm
    
    plt.figure(figsize=(10, 8))
    plt.scatter(L_flat, B_flat, c=colors, s=20, marker="s")
    plt.xlabel("galactic longitude (°)")
    plt.ylabel("galactic latitude (°)")
    plt.gca().set_aspect("equal")
    plt.savefig("dust_projection.png", dpi=300, bbox_inches="tight")
    plt.close()


def diagnostic_plots(E_cumulative, density, grids, outfile="diagnostics.png"):
    l_grid, b_grid, d_grid = grids

    print("=" * 50)
    print("Diagnostic summary")
    print("=" * 50)
    n_nan = np.isnan(E_cumulative).sum()
    print(f"Grid shape (l, b, d):       {E_cumulative.shape}")
    print(f"Total cells:                {E_cumulative.size:,}")
    print(f"NaN cells:                  {n_nan:,} ({n_nan/E_cumulative.size*100:.1f}%)")
    print(f"Cumulative E range:         {np.nanmin(E_cumulative):.3f} to {np.nanmax(E_cumulative):.3f} mag")
    print(f"Cumulative E median:        {np.nanmedian(E_cumulative):.3f} mag")
    print(f"Density range:              {np.nanmin(density):.5f} to {np.nanmax(density):.5f} mag/pc")
    print(f"Density median:             {np.nanmedian(density):.5f} mag/pc")
    print(f"Density 99th percentile:    {np.nanpercentile(density, 99):.5f} mag/pc")
    print()

    fig, axes = plt.subplots(2, 3, figsize=(18, 10))

    i_c = len(l_grid) // 2
    j_c = len(b_grid) // 2
    l_c, b_c = l_grid[i_c], b_grid[j_c]

    ax = axes[0, 0]
    ax.plot(d_grid, E_cumulative[i_c, j_c, :], 'o-', color='saddlebrown', markersize=4)
    ax.set_xlabel("distance (pc)")
    ax.set_ylabel("cumulative color excess (mag)")
    ax.set_title(f"cumulative · sightline (l={l_c:.1f}°, b={b_c:.1f}°)")
    ax.grid(alpha=0.3)

    ax = axes[0, 1]
    ax.plot(d_grid, density[i_c, j_c, :], 'o-', color='saddlebrown', markersize=4)
    ax.set_xlabel("distance (pc)")
    ax.set_ylabel("dust density (mag/pc)")
    ax.set_title(f"density · sightline (l={l_c:.1f}°, b={b_c:.1f}°)")
    ax.grid(alpha=0.3)

    ax = axes[0, 2]
    coverage = (~np.isnan(E_cumulative)).sum(axis=2)
    im = ax.imshow(coverage.T, origin='lower',
                   extent=[l_grid[0], l_grid[-1], b_grid[0], b_grid[-1]],
                   cmap='viridis', aspect='auto', interpolation='nearest')
    ax.set_xlabel("l (°)")
    ax.set_ylabel("b (°)")
    ax.set_title("coverage: valid distance bins per sightline")
    plt.colorbar(im, ax=ax, label="count")

    target_distances = [150, 300, 450]
    vmax = np.nanpercentile(density, 99)

    for idx, target_d in enumerate(target_distances):
        ax = axes[1, idx]
        k = np.argmin(np.abs(d_grid - target_d))
        actual_d = d_grid[k]
        im = ax.imshow(density[:, :, k].T, origin='lower',
                       extent=[l_grid[0], l_grid[-1], b_grid[0], b_grid[-1]],
                       cmap='copper', aspect='auto', interpolation='nearest',
                       vmin=0, vmax=vmax)
        ax.set_xlabel("l (°)")
        ax.set_ylabel("b (°)")
        ax.set_title(f"density slice · d ≈ {actual_d:.0f} pc")
        plt.colorbar(im, ax=ax, label="mag/pc")

    plt.tight_layout()
    plt.savefig(outfile, dpi=200, bbox_inches="tight")
    plt.close()
    print(f"saved: {outfile}")


def load_or_build_stars(input_path, cache_path=STARS_CACHE):
    if os.path.exists(cache_path):
        print(f"loading cached stars from {cache_path}")
        return pd.read_parquet(cache_path)
    
    print(f"building stars table from {input_path}")
    df = pd.read_parquet(input_path)
    df = df.dropna(subset=["color_excess"])
    
    coords = SkyCoord(ra=df["ra"].to_numpy() * u.deg,
                      dec=df["dec"].to_numpy() * u.deg,
                      frame="icrs")
    galactic = coords.galactic
    df["l"] = galactic.l.deg
    df["b"] = galactic.b.deg
    
    stars = df[["l", "b", "distance", "color_excess"]].rename(
        columns={"distance": "d", "color_excess": "E"}
    )
    stars.to_parquet(cache_path)
    print(f"saved: {cache_path}")
    return stars


def load_or_build_field(stars, cache_path=FIELD_CACHE, **kwargs):
    if os.path.exists(cache_path):
        print(f"loading cached field from {cache_path}")
        data = np.load(cache_path)
        return (data["E_cumulative"],
                data["density"],
                (data["l_grid"], data["b_grid"], data["d_grid"]))
    
    print("computing dust field (this is the slow part)")
    tree = build_kd_tree(stars)
    E_cumulative, density, (l_grid, b_grid, d_grid) = dust_field(stars, tree, **kwargs)
    
    np.savez(cache_path,
             E_cumulative=E_cumulative,
             density=density,
             l_grid=l_grid,
             b_grid=b_grid,
             d_grid=d_grid)
    print(f"saved: {cache_path}")
    return E_cumulative, density, (l_grid, b_grid, d_grid)


if __name__ == "__main__":
    stars = load_or_build_stars("04-color_excess_full.parquet")
    E_cumulative, density, grids = load_or_build_field(stars, sigma_r=17, theta_res=2, d_res=25, min_stars=3)
    plot_dust_projection(E_cumulative, density, grids)
    diagnostic_plots(E_cumulative, density, grids)