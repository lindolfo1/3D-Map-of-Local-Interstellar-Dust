import pandas as pd 
import numpy as np
import matplotlib.pyplot as plt 

def hr_scatterplot(df, coefficients, x_data, y_data,
                   outfile="hr_diagram_full.png", max_distance=None):
    if max_distance is not None:
        df = df[df["distance"] <= max_distance]

    vmin = np.percentile(df["distance"], 5)
    vmax = np.percentile(df["distance"], 95)
    plt.figure(figsize=(10, 8))
    plt.scatter(df["bp_rp"], df["abs_mag"], c=df["distance"],
                cmap="viridis", s=1, vmin=vmin, vmax=vmax)
    plt.scatter(x_data, y_data, color="red", zorder=5,
                edgecolor="black", linewidth=0.5)

    poly_func = np.poly1d(coefficients)
    y_smooth = np.linspace(min(y_data), max(y_data), 100)
    x_smooth = poly_func(y_smooth)
    plt.plot(x_smooth, y_smooth, color="red")

    plt.xlabel("BP−RP")
    plt.ylabel("abs_mag")
    plt.gca().invert_yaxis()
    plt.savefig(outfile, dpi=300, bbox_inches="tight")
    plt.close()


def wd_clean_mask(df):
    mask = df["abs_mag"] <= 5*df["bp_rp"] + 8
    return df[mask]


def giant_clean_mask(df):
    is_giant = (df["abs_mag"] < 2.5) & (df["bp_rp"] > 1.0)
    return df[~is_giant]


def blue_line(df, bin_width=0.5, min_mag=1.5, max_mag=8, degree=3):
    edges = np.arange(min_mag, max_mag + bin_width, bin_width)
    df["mag_bin"] = pd.cut(df["abs_mag"], edges)
    edge_colors = df.groupby("mag_bin", observed=True)["bp_rp"].quantile(0.1)
    print(df["mag_bin"].value_counts())
    centers = edge_colors.index.categories.mid

    coefficients = np.polyfit(centers, edge_colors.values, degree)
    return [coefficients, edge_colors.values, centers]


def calculate_dust(df, coefficients, min_mag, max_mag):
    star_mask = (df["abs_mag"] >= min_mag) & (df["abs_mag"] <= max_mag)
    probe_stars = df[star_mask]
    poly_func = np.poly1d(coefficients)
    intrinsic_color = poly_func(probe_stars["abs_mag"])
    df["color_excess"] = probe_stars["bp_rp"] - intrinsic_color
    return df


def create_histogram(df):
    star_mask = df["color_excess"].notna()
    probe_stars = df[star_mask]
    counts, bins = np.histogram(probe_stars["color_excess"], 200)

    peak_bin_index = counts.argmax()
    peak_excess = (bins[peak_bin_index] + bins[peak_bin_index + 1]) / 2
    print(f"peak at color_excess = {bins[peak_bin_index]:.4f}, subtracting baseline")
    df["color_excess"] = df["color_excess"] - peak_excess

    plt.figure(figsize=(10, 8))
    plt.stairs(counts, bins)
    plt.xlabel("color excess")
    plt.ylabel("count")
    plt.savefig("stars_color_excess_hist.png", dpi=300, bbox_inches="tight")
    plt.close()

    return df
    

if __name__ == "__main__":
    df = pd.read_parquet("03-abs_mag_full.parquet")
    df = wd_clean_mask(df)
    df = giant_clean_mask(df)
    min_mag = -0.5
    max_mag = 8
    degree = 6
    [coefficients, x_data, y_data] = blue_line(df, min_mag=min_mag, max_mag=max_mag, degree=degree)
    df = calculate_dust(df, coefficients, min_mag=min_mag, max_mag=max_mag)

    df = create_histogram(df)
    hr_scatterplot(df, coefficients, x_data, y_data, outfile="hr_diagram.png")
    
    df = df.drop(columns=["mag_bin"])
    df.to_parquet("04-color_excess_full.parquet")