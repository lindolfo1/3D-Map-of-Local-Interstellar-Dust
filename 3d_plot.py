import os
import numpy as np
import pandas as pd
import plotly.graph_objects as go
from scipy.interpolate import RegularGridInterpolator

FIELD_CACHE = "06-dust_field.npz"
STARS_CACHE = "05-stars_lb.parquet"
STAR_SAMPLE_SIZE = 80000


def load_field(cache_path):
    if not os.path.exists(cache_path):
        raise FileNotFoundError(f"{cache_path} not found")
    data = np.load(cache_path)
    return {
        "density": data["density"],
        "l_grid": data["l_grid"],
        "b_grid": data["b_grid"],
        "d_grid": data["d_grid"],
    }


def load_stars(cache_path, field, sample_size=STAR_SAMPLE_SIZE):
    if not os.path.exists(cache_path):
        raise FileNotFoundError(f"{cache_path} not found")
    stars = pd.read_parquet(cache_path)
    stars = stars[
        (stars["l"] >= field["l_grid"].min()) & (stars["l"] <= field["l_grid"].max()) &
        (stars["b"] >= field["b_grid"].min()) & (stars["b"] <= field["b_grid"].max()) &
        (stars["d"] >= field["d_grid"].min()) & (stars["d"] <= field["d_grid"].max())
    ]
    if len(stars) > sample_size:
        stars = stars.sample(n=sample_size, random_state=42)

    l_rad = np.deg2rad(stars["l"].to_numpy())
    b_rad = np.deg2rad(stars["b"].to_numpy())
    d = stars["d"].to_numpy()

    brightness = (50.0 / d) ** 0.5
    brightness = np.clip(brightness, 0.15, 1.0)

    return {
        "x": d * np.cos(b_rad) * np.cos(l_rad),
        "y": d * np.cos(b_rad) * np.sin(l_rad),
        "z": d * np.sin(b_rad),
        "brightness": brightness,
    }


def resample_to_cartesian(field, voxels=50):
    density = field["density"]
    l_grid = field["l_grid"]
    b_grid = field["b_grid"]
    d_grid = field["d_grid"]

    density_safe = np.nan_to_num(density, nan=0.0)
    interp = RegularGridInterpolator(
        (l_grid, b_grid, d_grid), density_safe,
        bounds_error=False, fill_value=0.0, method="linear",
    )

    d_max = d_grid.max()
    rng = np.linspace(-d_max, d_max, voxels)
    X, Y, Z = np.meshgrid(rng, rng, rng, indexing="ij")

    R = np.sqrt(X**2 + Y**2 + Z**2)
    with np.errstate(invalid="ignore", divide="ignore"):
        B = np.degrees(np.arcsin(np.where(R > 0, Z / R, 0.0)))
        L = np.degrees(np.arctan2(Y, X)) % 360

    points = np.stack([L.ravel(), B.ravel(), R.ravel()], axis=-1)
    density_cart = interp(points).reshape(X.shape)

    in_wedge = (
        (R >= d_grid.min()) & (R <= d_grid.max()) &
        (B >= b_grid.min()) & (B <= b_grid.max())
    )
    density_cart = np.where(in_wedge, density_cart, 0.0)
    return X, Y, Z, density_cart


def add_polar_grid(fig, d_max=500, l_step=30, d_arcs=(100, 200, 300, 400),
                   label_size=14, distance_label_size=12,
                   show_angle_labels=True, show_distance_labels=True):
    line_color = "rgba(120, 220, 240, 0.25)"
    arc_color = "rgba(120, 220, 240, 0.4)"
    label_color = "rgba(180, 230, 245, 0.5)"
    inner_arc_color = "rgba(160, 240, 250, 0.55)"

    for l_deg in range(0, 360, l_step):
        l_rad = np.deg2rad(l_deg)
        x_end = d_max * np.cos(l_rad)
        y_end = d_max * np.sin(l_rad)
        fig.add_trace(go.Scatter3d(
            x=[0, x_end], y=[0, y_end], z=[0, 0],
            mode="lines",
            line=dict(color=line_color, width=1),
            showlegend=False, hoverinfo="skip",
        ))
        if show_angle_labels:
            x_lbl = (d_max + 35) * np.cos(l_rad)
            y_lbl = (d_max + 35) * np.sin(l_rad)
            fig.add_trace(go.Scatter3d(
                x=[x_lbl], y=[y_lbl], z=[0],
                mode="text",
                text=[f"{l_deg}°"],
                textfont=dict(color=label_color, size=label_size,
                              family="Inter, sans-serif"),
                showlegend=False, hoverinfo="skip",
            ))

    theta = np.linspace(0, 2 * np.pi, 200)
    for i, d in enumerate(d_arcs):
        color = inner_arc_color if i < len(d_arcs) // 2 else arc_color
        fig.add_trace(go.Scatter3d(
            x=d * np.cos(theta), y=d * np.sin(theta), z=np.zeros_like(theta),
            mode="lines",
            line=dict(color=color, width=1.5),
            showlegend=False, hoverinfo="skip",
        ))
        if show_distance_labels:
            fig.add_trace(go.Scatter3d(
                x=[d * np.cos(np.deg2rad(60))],
                y=[d * np.sin(np.deg2rad(60))],
                z=[0],
                mode="text",
                text=[f"{d}pc"],
                textfont=dict(color=label_color, size=distance_label_size,
                              family="Inter, sans-serif"),
                showlegend=False, hoverinfo="skip",
            ))

    fig.add_trace(go.Scatter3d(
        x=[0], y=[0], z=[0], mode="markers",
        marker=dict(size=14, color="rgba(255, 230, 150, 0.25)"),
        showlegend=False, hoverinfo="skip",
    ))
    fig.add_trace(go.Scatter3d(
        x=[0], y=[0], z=[0], mode="markers",
        marker=dict(size=4, color="rgba(255, 255, 240, 1.0)"),
        showlegend=False, hoverinfo="skip",
    ))


def build_figure(field, stars,
                 voxels=50, surface_count=20, opacity=0.07,
                 star_marker_size=1.2, star_opacity=0.7,
                 label_size=14, distance_label_size=12,
                 title_size=16, base_font_size=11,
                 show_angle_labels=True, show_distance_labels=True):
    X, Y, Z, density_cart = resample_to_cartesian(field, voxels=voxels)

    valid = density_cart[density_cart > 0]
    cmin = np.percentile(valid, 50)
    cmax = np.percentile(valid, 99)

    ember = [
        [0.00, "rgba(20, 0, 0, 0.0)"],
        [0.10, "rgba(50, 10, 5, 0.2)"],
        [0.20, "rgba(95, 25, 12, 0.4)"],
        [0.30, "rgba(140, 45, 18, 0.55)"],
        [0.40, "rgba(185, 75, 25, 0.65)"],
        [0.50, "rgba(220, 105, 35, 0.72)"],
        [0.60, "rgba(240, 135, 45, 0.78)"],
        [0.70, "rgba(250, 165, 65, 0.85)"],
        [0.80, "rgba(255, 195, 95, 0.9)"],
        [0.90, "rgba(255, 220, 140, 0.95)"],
        [1.00, "rgba(255, 245, 200, 1.0)"],
    ]

    fig = go.Figure(data=go.Volume(
        x=X.flatten(), y=Y.flatten(), z=Z.flatten(),
        value=density_cart.flatten(),
        isomin=cmin, isomax=cmax,
        opacity=opacity,
        surface_count=surface_count,
        colorscale=ember,
        showscale=False,
        caps=dict(x_show=False, y_show=False, z_show=False),
    ))

    fig.add_trace(go.Scatter3d(
        x=stars["x"], y=stars["y"], z=stars["z"],
        mode="markers",
        marker=dict(
            size=star_marker_size,
            color=stars["brightness"],
            colorscale=[[0, "rgba(255, 250, 230, 0.0)"], [1, "rgba(255, 250, 230, 1.0)"]],
            opacity=1.0,
            showscale=False,
        ),
        name="stars",
        showlegend=False,
        visible="legendonly",
        hoverinfo="skip",
    ))

    add_polar_grid(fig, d_max=int(field["d_grid"].max()),
                   label_size=label_size,
                   distance_label_size=distance_label_size,
                   show_angle_labels=show_angle_labels,
                   show_distance_labels=show_distance_labels)

    invisible_axis = dict(
        showbackground=False, showgrid=False, showline=False,
        showticklabels=False, zeroline=False, title="",
    )

    fig.update_layout(
        title=dict(
            text="INTERSTELLAR DUST",
            x=0.5, xanchor="center", y=0.95,
            font=dict(family="Inter, system-ui, sans-serif", size=title_size,
                      color="rgba(230, 240, 250, 0.85)"),
        ),
        scene=dict(
            xaxis=invisible_axis, yaxis=invisible_axis, zaxis=invisible_axis,
            bgcolor="rgb(0, 0, 0)",
            aspectmode="data",
            camera=dict(
                eye=dict(x=1.2, y=-1.2, z=0.6),
                up=dict(x=0, y=0, z=1),
            ),
        ),
        autosize=True,
        paper_bgcolor="rgb(0, 0, 0)",
        font=dict(color="rgba(220, 230, 245, 0.7)",
                  family="Inter, sans-serif",
                  size=base_font_size),
        showlegend=False,
        margin=dict(l=0, r=0, t=40, b=0),
    )
    return fig


def volume_visualize(field, stars, outfile="dust_volume.html", **kwargs):
    fig = build_figure(field, stars, **kwargs)
    fig.update_layout(
        updatemenus=[
            dict(
                type="buttons", direction="right",
                x=0.5, y=0.04, xanchor="center", yanchor="bottom",
                showactive=False,
                bgcolor="rgba(20, 25, 40, 0.7)",
                bordercolor="rgba(100, 180, 220, 0.3)", borderwidth=1,
                font=dict(color="rgba(220, 230, 245, 0.85)",
                          family="Inter, sans-serif", size=11),
                pad=dict(l=10, r=10, t=6, b=6),
                buttons=[
                    dict(label="show stars", method="restyle",
                         args=[{"visible": True}, [1]]),
                    dict(label="hide stars", method="restyle",
                         args=[{"visible": "legendonly"}, [1]]),
                ],
            ),
        ],
    )
    fig.write_html(
        outfile,
        config={
            "responsive": True,
            "displayModeBar": True,
            "displaylogo": False,
            "modeBarButtonsToRemove": [
                "zoom3d", "pan3d", "orbitRotation", "tableRotation",
                "resetCameraDefault3d", "resetCameraLastSave3d",
                "hoverClosest3d",
            ],
            "toImageButtonOptions": {
                "format": "png",
                "filename": "interstellar_dust",
                "width": 2400,
                "height": 1600,
                "scale": 2,
            },
        },
        include_plotlyjs="cdn",
        full_html=True,
    )
    print(f"saved: {outfile}")


def export_angle_series(field, stars, outdir="renders",
                        width=4000, height=2500, scale=2,
                        show_stars=False, **build_kwargs):
    os.makedirs(outdir, exist_ok=True)

    angles = {
        "top":         dict(camera=dict(eye=dict(x=0.01, y=0.01, z=2.5), up=dict(x=0, y=1, z=0)), width=2500, height=2500),
        "front":       dict(camera=dict(eye=dict(x=2.2, y=0, z=0.3), up=dict(x=0, y=0, z=1)), width=3840, height=2160),
        "side":        dict(camera=dict(eye=dict(x=0, y=2.2, z=0.3), up=dict(x=0, y=0, z=1)), width=3840, height=2160),
        "isometric":   dict(camera=dict(eye=dict(x=1.4, y=-1.4, z=1.0), up=dict(x=0, y=0, z=1)), width=3000, height=2400),
        "low-angle":   dict(camera=dict(eye=dict(x=1.6, y=-1.6, z=0.3), up=dict(x=0, y=0, z=1)), width=3840, height=2160),
        "high-angle":  dict(camera=dict(eye=dict(x=1.0, y=-1.0, z=1.8), up=dict(x=0, y=0, z=1)), width=3000, height=2400),
        "edge-on":     dict(camera=dict(eye=dict(x=0, y=2.5, z=0.05), up=dict(x=0, y=0, z=1)), width=4000, height=1500),
        "rear":        dict(camera=dict(eye=dict(x=-2.2, y=0, z=0.3), up=dict(x=0, y=0, z=1)), width=3840, height=2160),
    }

    fig = build_figure(field, stars,
                       label_size=22, distance_label_size=20,
                       title_size=32, base_font_size=18,
                       **build_kwargs)
    if show_stars:
        fig.data[1].visible = True

    for name, cfg in angles.items():
        fig.update_layout(scene_camera=cfg["camera"])
        outfile = os.path.join(outdir, f"dust_{name}.png")
        fig.write_image(outfile, width=cfg["width"], height=cfg["height"], scale=scale)
        print(f"  saved: {outfile}")

    print(f"\nrendered {len(angles)} angles to {outdir}/")


def export_rotation_gif(field, stars, outfile="dust_rotation.gif",
                        frames=120, fps=20,
                        width=1200, height=800,
                        elevation=0.0, radius=1.3,
                        show_stars=False, **build_kwargs):
    from PIL import Image

    frame_dir = "_gif_frames"
    os.makedirs(frame_dir, exist_ok=True)

    fig = build_figure(field, stars,
                       label_size=20, distance_label_size=18,
                       title_size=24, base_font_size=14,
                       show_angle_labels=False,
                       show_distance_labels=False,
                       **build_kwargs)
    if show_stars:
        fig.data[1].visible = True

    print(f"rendering {frames} frames...")
    frame_paths = []
    for i in range(frames):
        angle = 2 * np.pi * i / frames
        camera = dict(
            eye=dict(
                x=radius * np.cos(angle),
                y=radius * np.sin(angle),
                z=elevation,
            ),
            up=dict(x=0, y=0, z=1),
            center=dict(x=0, y=0, z=0),
        )
        fig.update_layout(scene_camera=camera)

        frame_path = os.path.join(frame_dir, f"frame_{i:03d}.png")
        fig.write_image(frame_path, width=width, height=height)
        frame_paths.append(frame_path)
        print(f"  frame {i+1}/{frames}", end="\r")

    print(f"\nassembling GIF...")
    images = [Image.open(p) for p in frame_paths]
    images[0].save(
        outfile,
        save_all=True,
        append_images=images[1:],
        duration=int(1000 / fps),
        loop=0,
        optimize=True,
    )
    print(f"saved: {outfile}")

    for p in frame_paths:
        os.remove(p)
    os.rmdir(frame_dir)
    print(f"cleaned up temp frames")


if __name__ == "__main__":
    field = load_field(FIELD_CACHE)
    stars = load_stars(STARS_CACHE, field)
    volume_visualize(field, stars)
    # export_rotation_gif(
    #     field, stars,
    #     outfile="dust_rotation.gif",
    #     frames=120,
    #     fps=20,
    #     radius=1.1,
    #     elevation=0.0,
    #     surface_count=40,
    # )
    # export_angle_series(field, stars, outdir="renders", show_stars=False)