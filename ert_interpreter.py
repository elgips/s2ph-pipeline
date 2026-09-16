import matplotlib.pyplot as plt
from matplotlib.widgets import TextBox, Button
import matplotlib.image as mpimg
import tkinter as tk
from tkinter import ttk
from tkinter.filedialog import askopenfilename
import numpy as np
import pandas as pd
import os
import json

# -----------------------------
# PARAMETERS
# -----------------------------
W_METER = 50.0
D_METER = 20.0
MEAS_LENGTH = 50
TYPE_FIELDS = {
    "circle": {"x", "y", "radius", "resistivity"},
    "layer":  {"y", "resistivity"}
}

# GLOBALS
btn_detect = None
btn_export = None
seg_circles = []   # list of dicts: {x,y,r}
seg_layers  = []   # list of dicts: {y}
last_layer = None
last_blob  = None

current_img = None
current_img_path = None
user_width = W_METER
user_depth = D_METER
user_scheme = MEAS_LENGTH
user_rhoa_max = 1.0
user_rhoa_min = 0.0
user_scale = 0  # 0 = linear, 1 = log
entities_df = pd.DataFrame({
    "type":   pd.Series(dtype="string"),
    "x":      pd.Series(dtype="float"),
    "y":      pd.Series(dtype="float"),
    "radius": pd.Series(dtype="float"),
    "resistivity":  pd.Series(dtype="float"),
    "number": pd.Series(dtype="string"),  # NEW
    "is_background": pd.Series(dtype="bool"),  # marks the auto-computed background row
})

# -----------------------------
# ROI / DOI (region-of-investigation) HELPERS
# -----------------------------
# NOTE: seg_params is defined further below, but since these are only
# ever called from GUI callbacks (i.e. after the whole module has been
# loaded), referencing it here at call-time is safe.
DOI_DEFAULTS = {"extra_depth": 0.1, "top_exclude_m": 0.2}


def compute_doi_mask(h, w, width_m, depth_m, scheme_m, extra_depth=0.1, top_exclude_m=0.2):
    """Boolean mask (h,w) of the region of investigation (ROI), following the
    same dome/'banana' sensitivity boundary used for segmentation: deepest in
    the middle of the array, tapering to zero at the outer electrodes."""
    mask = np.zeros((h, w), dtype=bool)
    if width_m <= 0 or depth_m <= 0 or scheme_m <= 0:
        return mask

    elec_px = scheme_m * w / width_m
    half = elec_px / 2.0
    if half <= 0:
        return mask

    top_exclude_px = int(np.clip(top_exclude_m * h / depth_m, 0, h))
    max_depth_px = (scheme_m / 5.0 * (1 + extra_depth)) * h / depth_m
    x_center = w / 2.0

    for x in range(w):
        xi = (x - x_center) / half
        if abs(xi) > 1:
            continue
        depth = max_depth_px * np.sqrt(1 - xi ** 2)
        y1 = int(np.clip(depth, 0, h))
        if y1 > top_exclude_px:
            mask[top_exclude_px:y1, x] = True
    return mask


def get_current_doi_mask():
    """ROI mask for the currently loaded image, using the live width/depth/
    scheme settings. Returns None if no image is loaded."""
    if current_img is None:
        return None
    h, w = current_img.shape[:2]
    extra_depth = seg_params.get("extra_depth", DOI_DEFAULTS["extra_depth"]) if "seg_params" in globals() else DOI_DEFAULTS["extra_depth"]
    top_exclude_m = seg_params.get("top_exclude_m", DOI_DEFAULTS["top_exclude_m"]) if "seg_params" in globals() else DOI_DEFAULTS["top_exclude_m"]
    return compute_doi_mask(h, w, user_width, user_depth, user_scheme, extra_depth, top_exclude_m)


shape_artists = {}  # key = df index, value = patch/line
# Temp previews
temp_circle = {}  # {'center': (x, y)}
temp_layer_y = None

# Hide tk window
root = tk.Tk()
root.withdraw()

# -----------------------------
# FIGURE
# -----------------------------
#fig, (left_ax, right_ax) = plt.subplots(1, 2, figsize=(10, 5),
#gridspec_kw={'width_ratios': [1, 2]})
fig = plt.figure(figsize=(12, 6))
gs = fig.add_gridspec(6, 6, wspace=0.3, hspace=0.3)

#fig.subplots_adjust(left=0.05, right=0.95, wspace=0.3, bottom=0.1)

# right_ax.set_aspect('equal', adjustable='box')
# right_ax.set_title("Image")
# right_ax.set_xticks([])
# right_ax.set_yticks([])
# left_ax.axis('off')

# Image: top-right, spans multiple rows and columns
ax_image = fig.add_subplot(gs[0:5, 2:6])
ax_image.set_aspect('equal', adjustable='box')
ax_image.set_title("Image")
ax_image.set_xticks([])
ax_image.set_yticks([])

# Text boxes: top-left
ax_text = fig.add_subplot(gs[0:2, 0:2])
ax_text.axis('off')

# Table: bottom-left
ax_table = fig.add_subplot(gs[2:5, 0:2])
ax_table.axis('off')

# Buttons: bottom row full width
ax_buttons = fig.add_subplot(gs[5, :])
ax_buttons.axis('off')


def save_project():
    filename = tk.filedialog.asksaveasfilename(
        defaultextension=".json",
        filetypes=[("JSON files", "*.json"), ("All files", "*.*")]
    )
    if not filename:
        return
    
    # Store the path only if an image exists
    img_path_to_save = current_img_path if current_img_path is not None else None

    project_data = {
        "image_path": img_path_to_save,  # can be None
        "width": user_width,
        "depth": user_depth,
        "scheme": user_scheme,
        "rhoa_min": user_rhoa_min,
        "rhoa_max": user_rhoa_max,
        "scale": user_scale,
        "entities": entities_df.to_dict(orient="records")
    }
    
    with open(filename, "w") as f:
        json.dump(project_data, f, indent=2)
    
    print(f"Project saved to {filename}")

def load_project():
    global current_img, current_img_path, entities_df
    global user_width, user_depth, user_scheme, user_rhoa_min, user_rhoa_max, user_scale
    
    filename = tk.filedialog.askopenfilename(
        title="Open project",
        filetypes=[("JSON files", "*.json"), ("All files", "*.*")]
    )
    if not filename:
        return
    
    with open(filename, "r") as f:
        project_data = json.load(f)
    
    image_path = project_data.get("image_path", None)

    if image_path is not None and os.path.exists(image_path):
        current_img = mpimg.imread(image_path)
        current_img_path = image_path
        ax_image.imshow(current_img, origin='upper', aspect='equal')
        ax_image.set_xticks([])
        ax_image.set_yticks([])
    else:
        if image_path is None:
            print("No image path stored in this project.")
        else:
            print(f"Image file not found: {image_path}")
        current_img = None
        current_img_path = None
        ax_image.cla()
        ax_image.set_xticks([])
        ax_image.set_yticks([])
    
    # Load other parameters
    user_width = project_data.get("width", user_width)
    user_depth = project_data.get("depth", user_depth)
    user_scheme = project_data.get("scheme", user_scheme)
    user_rhoa_min = project_data.get("rhoa_min", user_rhoa_min)
    user_rhoa_max = project_data.get("rhoa_max", user_rhoa_max)
    user_scale = project_data.get("scale", user_scale)
    
    # Load entities
    entities_df = pd.DataFrame(project_data.get("entities", []))
    if "is_background" not in entities_df.columns:
        entities_df["is_background"] = False
    else:
        entities_df["is_background"] = entities_df["is_background"].fillna(False)
    
    update_entity_numbers()
    redraw_shapes()
    sync_textboxes_from_state()
    print(f"Project loaded from {filename}")


def _scale_to_rhoa(val):
    """Map a raw (grayscale) pixel value to the user-defined rhoa scale."""
    if current_img is None or val is None or (isinstance(val, float) and np.isnan(val)):
        return np.nan
    if seg_params.get("fixed_scale", True):
        # Fixed-range render: grey maps analytically to rho, independent of
        # what this particular image happens to contain. Must agree with
        # normalize_field() or the readout and the segmentation disagree.
        v = float(val)
        if v > 1.0:
            v = v / 255.0
        v = float(np.clip(v, 0.0, 1.0))
        lo = np.log10(seg_params["rho_plot_min"])
        hi = np.log10(seg_params["rho_plot_max"])
        return float(10.0 ** (hi - v * (hi - lo)))
    img_min, img_max = current_img.min(), current_img.max()
    val_stretched = (val - img_min) / (img_max - img_min)
    val_stretched = np.clip(val_stretched, 0, 1)
    if user_scale == 0:
        return user_rhoa_min + (1 - val_stretched) * (user_rhoa_max - user_rhoa_min)
    else:
        eps = 1e-12
        return np.exp(np.log(user_rhoa_min + eps) + (1 - val_stretched) *
                       (np.log(user_rhoa_max + eps) - np.log(user_rhoa_min + eps)))


def compute_resistivity(row):
        if current_img is None:
            return 0.0
        doi = get_current_doi_mask()
        if row["type"] == "circle":
            # extract pixels inside circle
            y_idx, x_idx = np.ogrid[:current_img.shape[0], :current_img.shape[1]]
            col = (row["x"]/user_width + 0.5) * current_img.shape[1]
            row_pix = -(row["y"]/user_depth) * current_img.shape[0]
            mask = (x_idx - col)**2 + (y_idx - row_pix)**2 <= (row["radius"]*current_img.shape[1]/user_width)**2
            if doi is not None:
                mask = mask & doi
            pixels = current_img[mask]
            if pixels.size == 0:
                return np.nan
            val = pixels.mean()
        elif row["type"] == "layer":
            # Layer bounds: top = shallower layer, bottom = this layer
            layers_above = entities_df[
                (entities_df["type"] == "layer") &
                (entities_df["y"] > row["y"])
                ]

            if not layers_above.empty:
                y_top = -(layers_above["y"].min() / user_depth) * current_img.shape[0]
            else:
                y_top = 0  # top of image

            y_bottom = -(row["y"] / user_depth) * current_img.shape[0]

            # Clip to valid indices
            y_top = int(np.clip(y_top, 0, current_img.shape[0]))
            y_bottom = int(np.clip(y_bottom, 0, current_img.shape[0]))

            if y_bottom <= y_top:
                return np.nan

            pixels = current_img[y_top:y_bottom, :]

            # -----------------------------
            # EXCLUDE CIRCLES (MASK ONLY)
            # -----------------------------
            h, w = pixels.shape[:2]
            exclude_mask = np.zeros((h, w), dtype=bool)

            for _, circ in entities_df[entities_df["type"] == "circle"].iterrows():
                col = (circ["x"] / user_width + 0.5) * current_img.shape[1]
                row_c = -(circ["y"] / user_depth) * current_img.shape[0]
                radius = circ["radius"] * current_img.shape[1] / user_width
                yy, xx = np.mgrid[y_top:y_bottom, 0:pixels.shape[1]]
                circ_mask = (xx - col) ** 2 + (yy - row_c) ** 2 <= radius ** 2

                exclude_mask |= circ_mask

            gray = pixels[..., :3].mean(axis=2)   # RGB → scalar
            include_mask = ~exclude_mask
            if doi is not None:
                include_mask &= doi[y_top:y_bottom, :]
            if not include_mask.any():
                return np.nan
            val = np.nanmean(gray[include_mask])


        return _scale_to_rhoa(val)
def update_entity_numbers():
    """Update entity numbers: Circle 1, Circle 2, Layer 1, ... The auto-computed
    Background row (is_background=True) keeps its fixed label and isn't counted."""
    circle_count = 1
    layer_count = 1
    numbers = []
    for idx, row in entities_df.iterrows():
        if row.get("is_background") is True:
            numbers.append("Background")
        elif row["type"] == "circle":
            numbers.append(f"Circle {circle_count}")
            circle_count += 1
        elif row["type"] == "layer":
            numbers.append(f"Layer {layer_count}")
            layer_count += 1
        else:
            numbers.append("")
    entities_df["number"] = numbers
def on_mouse_move_preview(event):
    global temp_circle, temp_layer_y

    if current_img is None or event.inaxes != ax_image:
        return
    xpix, ypix = event.xdata, event.ydata
    if xpix is None or ypix is None:
        return

    h, w = current_img.shape[:2]
    x_m = (xpix / w - 0.5) * user_width
    y_m = -(ypix / h) * user_depth

    redraw_shapes()  # clear previous previews

    if draw_mode == 'circle' and 'center' in temp_circle:
        # Draw radius preview
        x0, y0 = temp_circle['center']
        radius = np.sqrt((x_m - x0)**2 + (y_m - y0)**2)
        x_pix = (x0/ user_width + 0.5) * w
        y_pix = -y0 / user_depth * h
        radius_pix = radius / user_width * w
        circ = plt.Circle((x_pix, y_pix), radius_pix, edgecolor='orange', linestyle='--', facecolor='none')
        ax_image.add_patch(circ)
        ax_image.text(x_pix, y_pix, f"R={radius:.2f}", color='orange', fontsize=8, ha='center', va='bottom')

    elif draw_mode == 'layer':
        y_pix = -y_m / user_depth * h
        ax_image.hlines(y_pix, 0, w-1, colors='orange', linestyles='--')
        ax_image.text(w-1, y_pix, f"Y={y_m:.2f}", color='orange', fontsize=8, ha='right', va='bottom')

    fig.canvas.draw_idle()
    
# Palette for layer shadow fills (cycled if more layers than colors)
LAYER_FILL_COLORS = plt.get_cmap("tab10").colors
BACKGROUND_FILL_COLOR = (0.5, 0.5, 0.5)  # neutral gray for the un-interfaced background
ZONE_FILL_ALPHA = 0.35


def _draw_zone(ax, mask, color, label=None, label_color=None, alpha=ZONE_FILL_ALPHA):
    """Fill `mask` (h,w bool) with a transparent color and trace its border
    (border naturally follows any ROI clipping already baked into mask)."""
    if mask is None or not mask.any():
        return
    h, w = mask.shape

    rgba = np.zeros((h, w, 4))
    rgba[..., 0] = color[0]
    rgba[..., 1] = color[1]
    rgba[..., 2] = color[2]
    rgba[..., 3] = np.where(mask, alpha, 0.0)
    ax.imshow(rgba, origin='upper', zorder=2)
    ax.contour(mask.astype(float), levels=[0.5], colors=[color],
               linewidths=1.4, zorder=3)

    if label:
        rows_with_zone = np.where(mask.any(axis=1))[0]
        cols_with_zone = np.where(mask.any(axis=0))[0]
        if rows_with_zone.size and cols_with_zone.size:
            y_txt = rows_with_zone.min() + 4
            x_txt = cols_with_zone.max() - 4
            ax.text(x_txt, y_txt, label, color=label_color or color,
                    fontsize=8, ha='right', va='top', zorder=4)


def redraw_shapes():
    """Draw all shapes on the image and update resistivity values."""
    if current_img is None:
        return
    sync_background_row()   # keep the auto-computed Background row current
    update_entity_numbers()

    ax_image.cla()
    ax_image.imshow(current_img, origin='upper', aspect='equal')
    ax_image.set_xticks([])
    ax_image.set_yticks([])

    h, w = current_img.shape[:2]
    doi = get_current_doi_mask()

    # --- Layers: draw as transparent colorful shadows, clipped to the ROI ---
    # (Background row is excluded here; it's drawn separately below.)
    # Sort shallow -> deep (y near 0 first, most negative/deepest last)
    real_layers = entities_df[
        (entities_df["type"] == "layer") & (entities_df.get("is_background") != True)
    ]
    layer_rows = real_layers.sort_values("y", ascending=False)

    prev_bottom_pix = 0  # top of image; grows downward as we move through layers
    for i, (idx, row) in enumerate(layer_rows.iterrows()):
        y_bottom_pix = int(np.clip(-row["y"] / user_depth * h, 0, h))
        y_top_pix = int(np.clip(prev_bottom_pix, 0, h))

        band = np.zeros((h, w), dtype=bool)
        if y_bottom_pix > y_top_pix:
            band[y_top_pix:y_bottom_pix, :] = True
        if doi is not None:
            band &= doi

        color = LAYER_FILL_COLORS[i % len(LAYER_FILL_COLORS)]
        _draw_zone(ax_image, band, color, label=entities_df.loc[idx, "number"], label_color='blue')

        prev_bottom_pix = y_bottom_pix
        entities_df.loc[idx, "resistivity"] = compute_resistivity(row)

    # --- Background: below the lowest layer, or the whole ROI if no layers exist ---
    bg_band = np.zeros((h, w), dtype=bool)
    bg_band[int(np.clip(prev_bottom_pix, 0, h)):h, :] = True
    if doi is not None:
        bg_band &= doi
    _draw_zone(ax_image, bg_band, BACKGROUND_FILL_COLOR, label="Background", label_color='black')

    # --- Circles: drawn last so they always sit on top of the layer shading ---
    for idx, row in entities_df[entities_df["type"] == "circle"].iterrows():
        x_pix = (row["x"]/user_width + 0.5) * w
        y_pix = -row["y"]/user_depth * h
        radius_pix = row["radius"]/user_width * w
        circle = plt.Circle((x_pix, y_pix), radius_pix, edgecolor='red', facecolor='none',
                             linewidth=1.8, zorder=5)
        ax_image.add_patch(circle)
        ax_image.text(x_pix, y_pix, entities_df.loc[idx, "number"], color='red',
                       fontsize=8, ha='center', va='center', zorder=6)
        entities_df.loc[idx, "resistivity"] = compute_resistivity(row)

    fig.canvas.draw_idle()
    
# -----------------------------
# SAFE CALLBACKS
# -----------------------------
def safe_float(text, default):
    try:
        return float(text)
    except ValueError:
        print(f"Invalid input '{text}', using previous value: {default}")
        return default

def safe_int(text, default):
    try:
        return int(text)
    except ValueError:
        print(f"Invalid input '{text}', using previous value: {default}")
        return default
def sync_textboxes_from_state():
    text1.set_val(str(user_width))
    text2.set_val(str(user_depth))
    text3.set_val(str(user_scheme))
    text4.set_val(str(user_rhoa_min))
    text5.set_val(str(user_rhoa_max))
    text6.set_val(str(user_scale))

def submit1(text):
    global user_width
    user_width = safe_float(text, user_width)
def submit2(text):
    global user_depth
    user_depth = safe_float(text, user_depth)
def submit3(text):
    global user_scheme
    user_scheme = safe_float(text, user_scheme)
def submit4(text):
    global user_rhoa_min
    user_rhoa_min = safe_float(text, user_rhoa_min)
def submit5(text):
    global user_rhoa_max
    user_rhoa_max = safe_float(text, user_rhoa_max)
def submit6(text):
    global user_scale
    user_scale = safe_int(text, user_scale)
    user_scale = 0 if user_scale != 1 else 1  # only 0 or 1

# -----------------------------
# TEXTBOXES
# -----------------------------
text1 = TextBox(plt.axes([0.15, 0.7, 0.1, 0.05]), 'Width [m]:', initial=str(user_width))
text1.on_submit(submit1)
text2 = TextBox(plt.axes([0.15, 0.65, 0.1, 0.05]), 'Depth [m]:', initial=str(user_depth))
text2.on_submit(submit2)
text3 = TextBox(plt.axes([0.15, 0.6, 0.1, 0.05]), 'Scheme width [m]:', initial=str(user_scheme))
text3.on_submit(submit3)
text4 = TextBox(plt.axes([0.15, 0.55, 0.1, 0.05]), 'min resistivity [rhoa]:', initial=str(user_rhoa_min))
text4.on_submit(submit4)
text5 = TextBox(plt.axes([0.15, 0.5, 0.1, 0.05]), 'max resistivity [rhoa]:', initial=str(user_rhoa_max))
text5.on_submit(submit5)
text6 = TextBox(plt.axes([0.15, 0.45, 0.1, 0.05]), 'scale log/lin[1 or 0]:', initial=str(user_scale))
text6.on_submit(submit6)

# -----------------------------
# DYNAMIC CURSOR LABELS
# -----------------------------
x_text = ax_text.text(-0.35, 0.7, "X [m]: 0.00", transform=ax_text.transAxes, fontsize=10)
y_text = ax_text.text(-0.35, 0.6, "Y [m]: 0.00", transform=ax_text.transAxes, fontsize=10)
rhoa_text = ax_text.text(-0.35, 0.5, "Rhoa: 0.000", transform=ax_text.transAxes, fontsize=10)

# -----------------------------
# LOAD / EXIT BUTTONS
# -------------
def new_project(event=None):
    global current_img, current_img_path, entities_df
    global user_width, user_depth, user_scheme, user_rhoa_min, user_rhoa_max, user_scale

    # Ask to save current project first
    if current_img is not None or not entities_df.empty:
        answer = tk.messagebox.askyesnocancel(
            "Save Current Project?",
            "Do you want to save the current project before creating a new one?"
        )
        if answer is True:
            save_project()  # save current
        elif answer is None:  # Cancel
            return

    # Reset globals
    current_img = None
    current_img_path = None
    entities_df = pd.DataFrame({
        "type": pd.Series(dtype="string"),
        "x": pd.Series(dtype="float"),
        "y": pd.Series(dtype="float"),
        "radius": pd.Series(dtype="float"),
        "resistivity": pd.Series(dtype="float"),
        "number": pd.Series(dtype="string"),
        "is_background": pd.Series(dtype="bool"),
    })

    # Reset parameters to defaults
    user_width = W_METER
    user_depth = D_METER
    user_scheme = MEAS_LENGTH
    user_rhoa_min = 1.0
    user_rhoa_max = 1.0
    user_scale = 0

    # Clear axes
    ax_image.cla()
    ax_image.set_title("Image")
    ax_image.set_xticks([])
    ax_image.set_yticks([])

    redraw_shapes()
    print("New project initialized.")


def load_image(event):
    global current_img, current_img_path
    folder = os.getcwd()
    filename = askopenfilename(
        initialdir=folder,
        title="Select an image",
        filetypes=[("PNG files", "*.png"), ("All files", "*.*")]
    )
    if filename:
        current_img = mpimg.imread(filename)
        current_img_path = filename  # <-- set the global path here
        ax_image.clear()
        ax_image.imshow(current_img, origin='upper', aspect='equal')
        ax_image.set_xticks([])
        ax_image.set_yticks([])
        plt.draw()
        redraw_shapes()

def close(event):
    plt.close(fig)

load_button = Button(plt.axes([0.6, 0.02, 0.15, 0.06]), 'Load Image')
load_button.on_clicked(load_image)

exit_button = Button(plt.axes([0.8, 0.02, 0.15, 0.06]), 'Exit')
exit_button.on_clicked(close)
def open_table(event):
    open_entities_table()

table_button = Button(
    plt.axes([0.05, 0.02, 0.2, 0.06]),
    "Entities Table"
)
table_button.on_clicked(open_table)

# Save Project Button
save_proj_button = Button(plt.axes([0.05, 0.1, 0.2, 0.06]), 'Save Project')
save_proj_button.on_clicked(lambda event: save_project())

# Load Project Button
load_proj_button = Button(plt.axes([0.25, 0.1, 0.2, 0.06]), 'Load Project')
load_proj_button.on_clicked(lambda event: load_project())

#New project Button
new_proj_button = Button(plt.axes([0.45, 0.1, 0.2, 0.06]), 'New Project')
new_proj_button.on_clicked(new_project)

seg_button = Button(plt.axes([0.65, 0.1, 0.25, 0.06]), "Open Segmentation")
def on_open_segmentation(event):
    print("Opening segmentation window")
    open_segmentation_window()

seg_button.on_clicked(on_open_segmentation)

# -----------------------------
# MOUSE MOVE CALLBACK
# -----------------------------
def on_mouse_move(event):
    if current_img is None or event.inaxes != ax_image:
        return

    xpix = event.xdata
    ypix = event.ydata
    if xpix is None or ypix is None:
        return

    row = int(np.clip(ypix + 0.5, 0, current_img.shape[0]-1))
    col = int(np.clip(xpix + 0.5, 0, current_img.shape[1]-1))

    # Metric coordinates
    x_m = (col / current_img.shape[1] - 0.5) * user_width
    y_m = -(row / current_img.shape[0]) * user_depth

    # Rhoa value as scalar
    pixel = current_img[row, col]
    val = float(pixel.mean()) if isinstance(pixel, np.ndarray) else float(pixel)
    # Normalize to image min/max for full brightness range
    img_min = current_img.min()
    img_max = current_img.max()
    val_stretched = (val - img_min) / (img_max - img_min)  # now 0->darkest, 1->brightest
    val_stretched = np.clip(val_stretched, 0, 1)  # safety
    # Scale according to user-defined min, max, and scale type
    min_val = user_rhoa_min
    max_val = user_rhoa_max
    if user_scale == 0:  # linear
        scaled_rhoa = min_val+ (1-val_stretched)* (max_val - min_val)

    else:  # log
        eps = 1e-12
        # normalize val in log space to [0,1] first

        scaled_rhoa =np.exp(np.log(min_val + eps)+(1 - val_stretched) * (np.log(max_val + eps)-np.log(min_val + eps)))
        #print(val_stretched,scaled_rhoa)

    #scaled_rhoa = np.clip(scaled_rhoa, 0, 1)

    # Update labels
    x_text.set_text(f"X [m]: {x_m:.2f}")
    y_text.set_text(f"Y [m]: {y_m:.2f}")
    rhoa_text.set_text(f"Rhoa: {val:.3f} (scaled: {scaled_rhoa:.3f})")

    fig.canvas.draw_idle()
def open_entities_table():
    global entities_df

    sync_background_row()  # make sure Background is present/current before showing the table

    table_win = tk.Toplevel(root)
    table_win.title("Entities Table")
    table_win.geometry("600x350")

    columns = list(entities_df.columns)

    # 2. Frames
    table_frame = tk.Frame(table_win)
    table_frame.pack(fill="both", expand=True)

    btn_frame = tk.Frame(table_win)
    btn_frame.pack(fill="x", pady=5)

    tree = ttk.Treeview(
        table_frame,
        columns=columns,
        show="headings",
        selectmode="browse"
    )
    
    for col in columns:
        tree.heading(col, text=col)
        tree.column(col, width=100, anchor="center")

    tree.pack(fill="both", expand=True)

    # -----------------------------
    # Helper functions
    # -----------------------------
    def refresh_table():
        tree.delete(*tree.get_children())
        for _, row in entities_df.iterrows():
            item = tree.insert("", "end", values=list(row))
            style_row(item, row)

    def add_row():
        global entities_df
        new_row = {
            "type": "circle",
            "x": 0.0,
            "y": 0.0,
            "radius": 1.0,
            "resistivity": 1.0,
            "is_background": False,
        }
        entities_df = pd.concat([entities_df, pd.DataFrame([new_row])], ignore_index=True)
        redraw_shapes()
        refresh_table()

    def on_double_click(event):
        region = tree.identify("region", event.x, event.y)
        if region != "cell":
            return

        row_id = tree.identify_row(event.y)
        col_id = tree.identify_column(event.x)

        if not row_id or not col_id:
            return

        col_index = int(col_id[1:]) - 1
        col_name = columns[col_index]
        row_index = tree.index(row_id)

        current_type = entities_df.loc[row_index, "type"]

        # --- BLOCK editing the auto-computed Background row entirely ---
        if entities_df.loc[row_index].get("is_background") is True:
            return

        # --- BLOCK editing irrelevant fields ---
        if col_name == "resistivity":
            return

        if col_name != "type" and col_name not in TYPE_FIELDS.get(current_type, set()):
            return


        x, y, w, h = tree.bbox(row_id, col_id)

        # ==================================================
        # TYPE COLUMN → DROPDOWN
        # ==================================================
        if col_name == "type":
            combo = ttk.Combobox(
                table_win,
                values=["circle", "layer"],
                state="readonly"
            )
            combo.place(x=x, y=y + tree.winfo_y(), width=w, height=h)
            combo.set(current_type)
            combo.focus_set()

            def save_type(event=None):
                new_type = combo.get()
                entities_df.loc[row_index, "type"] = new_type
                combo.destroy()  # destroy first
                refresh_table()  # then refresh

            combo.bind("<<ComboboxSelected>>", save_type)
            combo.bind("<Return>", save_type)
            combo.bind("<Escape>", lambda e: combo.destroy())
            return

        # ==================================================
        # OTHER COLUMNS → ENTRY
        # ==================================================
        entry = tk.Entry(table_win)
        entry.place(x=x, y=y + tree.winfo_y(), width=w, height=h)
        entry.insert(0, tree.item(row_id, "values")[col_index])
        entry.focus()

        def save_edit(event=None):
            try:
                val = float(entry.get())
                entities_df.loc[row_index, col_name] = val
                redraw_shapes()
                refresh_table()
            except ValueError:
                pass
            entry.destroy()

        entry.bind("<Return>", save_edit)
        entry.bind("<FocusOut>", save_edit)

    tree.bind("<Double-1>", on_double_click)

    def delete_row():
        global entities_df
        selected = tree.selection()
        if not selected:
            return
        idx = tree.index(selected[0])
        entities_df = entities_df.drop(entities_df.index[idx]).reset_index(drop=True)
        redraw_shapes()  # recompute Background/layer values and re-render
        refresh_table()

    def update_selected():
        global entities_df
        selected = tree.selection()
        if not selected:
            return
        idx = tree.index(selected[0])
        entities_df.loc[idx, "resistivity"] += 1.0
        refresh_table()

    def style_row(item_id, row_data):
        typ = row_data["type"]
        invalid = False
        values = []

        for col in columns:
            if col == "type":
                values.append(row_data[col])
                continue

            if col not in TYPE_FIELDS.get(typ, set()):
                values.append("")
                continue

            val = row_data[col]
            values.append(val)

            if col == "y" and not (isinstance(val, (int, float)) and val < 0):
                invalid = True
            if col == "radius" and typ == "circle" and not (isinstance(val, (int, float)) and val > 0):
                invalid = True

        tree.item(item_id, values=values)

        tag = "invalid" if invalid else "valid"
        tree.tag_configure("invalid", background="#ffcccc")
        tree.tag_configure("valid", background="white")
        tree.item(item_id, tags=(tag,))

    # -----------------------------
    # Buttons
    # -----------------------------
    tk.Button(btn_frame, text="Add Row", command=add_row).pack(side="left", padx=5)
    tk.Button(btn_frame, text="Update Row", command=update_selected).pack(side="left", padx=5)
    tk.Button(btn_frame, text="Delete Row", command=delete_row).pack(side="left", padx=5)

    # Export CSV button
    def export_csv():
        filename = tk.filedialog.asksaveasfilename(
            defaultextension=".csv",
            filetypes=[("CSV files", "*.csv"), ("All files", "*.*")]
        )
        if filename:
            df_to_save = entities_df.copy()

            for idx, row in df_to_save.iterrows():
                typ = row["type"]
                for col in ["x", "y", "radius", "resistivity"]:
                    if col not in TYPE_FIELDS.get(typ, set()):
                        df_to_save.at[idx, col] = np.nan  # <-- use NaN instead of ""
            df_to_save.to_csv(filename, index=False)
            print(f"Table exported to {filename}")

    tk.Button(btn_frame, text="Export CSV", command=export_csv).pack(side="left", padx=5)

    refresh_table()

fig.canvas.mpl_connect('motion_notify_event', on_mouse_move)

# -----------------------------
# DRAW MODE SETUP
# -----------------------------
draw_mode = None  # None, 'circle', 'layer'
temp_circle = {}  # for storing click info for circle

def set_draw_mode_circle(event=None):
    global draw_mode, temp_circle
    draw_mode = 'circle'
    temp_circle = {}
    print("Draw mode: Circle")

def set_draw_mode_layer(event=None):
    global draw_mode
    draw_mode = 'layer'
    print("Draw mode: Layer")

# Buttons
draw_circle_button = Button(plt.axes([0.25, 0.02, 0.15, 0.06]), 'Draw Circle')
draw_circle_button.on_clicked(set_draw_mode_circle)

draw_layer_button = Button(plt.axes([0.45, 0.02, 0.15, 0.06]), 'Draw Layer')
draw_layer_button.on_clicked(set_draw_mode_layer)

# -----------------------------
# MOUSE CLICK CALLBACK FOR DRAWING
# -----------------------------
def on_mouse_click(event):
    global draw_mode, temp_circle

    if current_img is None or event.inaxes != ax_image:
        return
    xpix, ypix = event.xdata, event.ydata
    if xpix is None or ypix is None:
        return

    # Metric coordinates
    x_m = (xpix / current_img.shape[1] - 0.5) * user_width
    y_m = -(ypix / current_img.shape[0]) * user_depth

    if draw_mode == 'circle':
        if 'center' not in temp_circle:
            temp_circle['center'] = (x_m, y_m)
            print(f"Circle center set at ({x_m:.2f}, {y_m:.2f})")
        else:
            # Second click → radius
            x0, y0 = temp_circle['center']
            radius = np.sqrt((x_m - x0)**2 + (y_m - y0)**2)
            new_row = {"type": "circle", "x": x0, "y": y0, "radius": radius, "resistivity": 1.0}
            entities_df.loc[len(entities_df)] = new_row
            temp_circle = {}  # reset
            redraw_shapes()
            print(f"Circle drawn at ({x0:.2f}, {y0:.2f}) with radius {radius:.2f}")

    elif draw_mode == 'layer':
        # Single click → y-coordinate of layer
        new_row = {"type": "layer", "x": 0.0, "y": y_m, "radius": 0.0, "resistivity": 1.0}
        entities_df.loc[len(entities_df)] = new_row
        redraw_shapes()
        print(f"Layer drawn at y = {y_m:.2f}")

    else:
        return  # not in draw mode

fig.canvas.mpl_connect('button_press_event', on_mouse_click)
fig.canvas.mpl_connect('motion_notify_event', on_mouse_move_preview)

#plt.show()
# ===============================================
# SEGMENTATION WINDOW (NEW)
# ===============================================
import skimage.filters as filters
#import skimage.morphology as morphology
from skimage.morphology import footprint_rectangle
from skimage import measure
from skimage import morphology, filters

# GLOBALS FOR SEGMENTATION
seg_img_gray = None
doi_mask = None
seg_params = {
    "num_levels": 3,
    "min_object_size": 180,
    "layer_min_area": 350,
    "layer_min_height": 100,
    "blob_min_size": 400,
    "blob_max_aspect": 3.0,
    "width_ratio_thresh": 0.75,
    "continuity_thresh": 0.5,
    "top_exclude_m": 0.2,
    # Threshold offsets are in DECADES of log10(rho), not per-image fractions.
    # Valid only for fixed-range renders (see rho_plot_min/max below): the same
    # offset then means the same physical contrast in every scene, and is
    # directly comparable to the saturation constant c0 = 0.378 decade.
    "bright_threshold":0.05,
    "dark_threshold":0.05,
    "extra_depth":0.1,
    "rho_plot_min": 10.0,     # cMin used when the raster was exported
    "rho_plot_max": 1000.0,   # cMax used when the raster was exported
    "fixed_scale": True,      # False -> legacy per-image min-max normalisation
    # Discard 'layer' components flush against the ROI top. Expressed in
    # METRES (tau = a/2, the nominal vertical resolution) so it is
    # resolution-independent, matching layer_floor_close_frac below, which is
    # a fraction of local DOI depth. The former pixel form silently changed
    # meaning whenever the raster resolution changed.
    "layer_top_rim_margin_m": 1.25,
    "layer_floor_close_frac": 0.17,  # min fraction of local DOI depth a band must clear the floor by, to count its bottom edge as a real interface
    "circle_min_doi_extent_frac": 0.65,  # discard circle detections where local DOI depth extent is below this fraction of the max (poorly-resolved edge zone)
}

def open_segmentation_window():
    global seg_img_gray, doi_mask, seg_params

    # 1. Load current image
    if current_img is None:
        print("Load an image first!")
        return
    seg_img_gray = np.mean(current_img[..., :3], axis=2)  # RGB → grayscale

    # 2. DOI mask (shared with the main ROI display so both stay in sync)
    doi_mask = get_current_doi_mask()

    # 3. Create figure
    fig_seg = plt.figure("ERT Segmentation", figsize=(12,6))
    gs = fig_seg.add_gridspec(2, 2, width_ratios=[1,2], wspace=0.3, hspace=0.3)

    ax_params = fig_seg.add_subplot(gs[:,0])
    ax_params.axis('off')
    ax_result = fig_seg.add_subplot(gs[:,1])
    ax_result.axis('off')
    global btn_detect
    btn_detect_ax = fig_seg.add_axes([0.05, 0.05, 0.35, 0.07])
    btn_detect = Button(btn_detect_ax, "Detect Circles & Layers")
    btn_detect.on_clicked(
        lambda event: on_detect(event, ax_result, fig_seg)
    )

    btn_detect_ax.set_zorder(10)
    # --- Close button ---
    # ax_close = fig_seg.add_axes([0.05, 0.22, 0.35, 0.07])
    # btn_close = Button(ax_close, "Close Segmentation")

    # def _close(event):
        # plt.close(fig_seg)

    # btn_close.on_clicked(_close)
    # ax_close.set_zorder(10)
# )



    # # 4. Parameter boxes
    # param_boxes = {}
    # y_pos = 0.9
    # for key, val in seg_params.items():
        # axbox = plt.axes([0.2, y_pos, 0.2, 0.05])
        # tb = TextBox(axbox, f"{key}", initial=str(val))
        # tb.on_submit(lambda text, k=key: update_seg_param(k, text, ax_result, fig_seg))
        # param_boxes[key] = tb
        # y_pos -= 0.08

    # 5. Run initial segmentation
    run_segmentation(ax_result, fig_seg)
    fig_seg.canvas.draw_idle()
    fig_seg.show()
    



def update_seg_param(key, text, ax_result, fig_seg):
    global seg_params
    try:
        if key in ["num_levels"]:
            seg_params[key] = int(text)
        else:
            seg_params[key] = float(text)
    except ValueError:
        print(f"Invalid input for {key}, using previous value: {seg_params[key]}")
    run_segmentation(ax_result, fig_seg)



def normalize_field(img_gray):
    """Map a greyscale render to the field the segmentation thresholds.

    fixed_scale=True (default): the image is assumed to be a fixed-range
    render with cMin/cMax = rho_plot_min/rho_plot_max and cMap='binary'
    (high value -> black), so grey maps analytically to resistivity. Returns
    log10(rho_plot_max / rho), i.e. DECADES below the plot maximum. Bright
    still reads high, so bright_rows / dark_rows keep their existing meaning,
    but differences in the returned field are now exactly differences in
    log10(rho) -- which makes the threshold offsets physical contrasts,
    comparable between scenes and to c0.

    fixed_scale=False: legacy per-image min-max normalisation. Retained only
    for images whose colour limits are unknown. Offsets are then per-image
    fractions and are NOT comparable between scenes: the same numeric offset
    means a different physical contrast in every image, because the
    denominator is set by whatever that image happens to contain (including
    pixels outside the model domain).
    """
    g = np.asarray(img_gray, float)
    if g.max() > 1.0:                      # 8-bit input
        g = g / 255.0
    if not seg_params.get("fixed_scale", True):
        return (g - g.min()) / (g.max() - g.min() + 1e-12)
    span = np.log10(seg_params["rho_plot_max"] / seg_params["rho_plot_min"])
    return np.clip(g, 0.0, 1.0) * span


def run_segmentation(ax_result, fig_seg):
    global current_img, doi_mask, seg_params

    if current_img is None:
        return
    # --- Grayscale & map to decades (see normalize_field) ---
    img_gray = np.mean(current_img[..., :3], axis=2)
    img_n = normalize_field(img_gray)

    # --- Process: layers first (full DOI width), then blobs (local contrast) ---
    layer, blob = process_mask(img_n, doi_mask)
    global last_layer, last_blob
    last_layer = layer
    last_blob  = blob

    # --- Display with DOI mask ---
    masked_img = img_n.copy()
    masked_img[~doi_mask] = np.nan  # blank outside DOI
    # --- Display ---
    ax_result.cla()
    ax_result.imshow(masked_img, origin='upper', cmap='gray', vmin=0, vmax=1)
    ax_result.set_title("Segmentation (DOI only)")
    ax_result.set_xticks([])
    ax_result.set_yticks([])

    # Overlay DOI area (transparent blue)
    ax_result.imshow(np.where(doi_mask, 0.1, np.nan), origin='upper', cmap='Blues', alpha=0.2)

    # Overlay layers (blue)
    ax_result.contour(layer, colors='blue', linewidths=1.0)

    # Overlay blobs (red)
    ax_result.contour(blob, colors='red', linewidths=1.0)

    fig_seg.canvas.draw_idle()

def compute_row_profile(img_n, doi_mask):
    """Median value across the DOI-valid width, for each row: a single
    depth-wise 'layer signal' that stays robust to blobs (which only ever
    occupy a minority of a row's width, so the median discounts them)."""
    h, w = img_n.shape
    profile = np.full(h, np.nan)
    for y in range(h):
        xs = np.where(doi_mask[y])[0]
        if xs.size:
            profile[y] = np.median(img_n[y, xs])
    return profile

def process_mask(img_n, doi_mask):
    """Two-stage detection, in this order:
    (1) LAYERS -- found by looking across the *entire* DOI width at each
        depth. A genuine interface shifts the typical value across the
        whole array at that depth, not just part of it, so this is checked
        on a per-row median (robust to blobs) rather than per-pixel.
    (2) BLOBS -- found only afterward, as localized contrast against the
        layer profile each pixel actually sits in (a residual, not a
        single global absolute threshold), so an anomaly is judged
        relative to its own surrounding layer rather than the whole image.

    NOTE: blob candidates are NOT excluded from rows already classified as
    'layer'. A circle embedded inside a layer (e.g. an anomaly sitting well
    within a bounded middle stratum) has its strongest residual signal deep
    inside that layer's own depth range, not at its edge -- blanket-
    excluding layer rows from blob candidacy would throw that signal away
    along with the genuine layer bulk (which reads as ~0 residual anyway,
    since the layer's own rows are exactly what the baseline is built from).
    """
    h, w = doi_mask.shape
    profile = compute_row_profile(img_n, doi_mask)
    valid = profile[~np.isnan(profile)]
    if valid.size < 10:
        return np.zeros((h, w), dtype=bool), np.zeros((h, w), dtype=bool)

    # A row-median profile with no vertical contrast means no LAYER structure.
    # It does NOT mean no structure: the row median is deliberately
    # insensitive to compact anomalies, which occupy a minority of each row,
    # so a scene with anomalies over a homogeneous background has a flat
    # profile BY DESIGN. Suppress the layer branch only -- the blob branch
    # below is unaffected. Meaningful only on a fixed scale, where the spread
    # is an absolute contrast in decades.
    layer = np.zeros((h, w), dtype=bool)
    degenerate_profile = False
    if seg_params.get("fixed_scale", True):
        floor = min(seg_params["bright_threshold"],
                    seg_params["dark_threshold"])
        degenerate_profile = float(np.ptp(valid)) < floor

    if not degenerate_profile:
        t_low, t_high = filters.threshold_multiotsu(valid, classes=3)

        bright_rows = profile > (t_high + seg_params["bright_threshold"])
        dark_rows = profile < (t_low - seg_params["dark_threshold"])
        layer_rows = bright_rows | dark_rows

        for y in range(h):
            if layer_rows[y]:
                layer[y, doi_mask[y]] = True
        layer = morphology.remove_small_objects(
            layer, min_size=seg_params["layer_min_area"] - 1)

    # --- Blobs: local contrast against the row's own baseline ---
    baseline = np.zeros((h, w))
    for y in range(h):
        if not np.isnan(profile[y]):
            baseline[y, :] = profile[y]
    residual = img_n - baseline
    # The layer branch thresholds the row-median profile (vertical structure);
    # this thresholds the residual against that profile (lateral structure).
    # Different quantities with different amplitudes, so they may take
    # independent offsets. Falling back to max(bright, dark) preserves the
    # previous behaviour when 'blob_threshold' is unset.
    contrast_thresh = seg_params.get(
        "blob_threshold",
        max(seg_params["bright_threshold"], seg_params["dark_threshold"]))
    blob_candidate = (np.abs(residual) > contrast_thresh) & doi_mask
    blob = filter_blobs_by_geometry(blob_candidate)

    return layer, blob

def filter_blobs_by_geometry(mask):
    labels = measure.label(mask)
    out = np.zeros_like(mask)
    for p in measure.regionprops(labels):
        minr, minc, maxr, maxc = p.bbox
        hb = maxr-minr+1
        wb = maxc-minc+1
        if p.area >= seg_params["blob_min_size"] and (wb/hb) <= seg_params["blob_max_aspect"]:
            out[p.coords[:,0], p.coords[:,1]] = True
    return out
def on_detect(event, ax_result, fig_seg):
    print("DETECT BUTTON CLICKED")
    global seg_circles, seg_layers

    if last_layer is None or last_blob is None:
        print("Run segmentation first")
        return

    seg_circles, seg_layers = extract_circles_and_layers(
        last_layer, last_blob
    )

    draw_detected_entities(ax_result)
    show_export_button(fig_seg)
    fig_seg.canvas.draw_idle()


def extract_circles_and_layers(layer_mask, blob_mask):
    circles = []
    layers  = []

    # --- BLOBS → CIRCLES ---
    # The DOI narrows sharply toward the edges of the array; a genuinely
    # flat interface still gets smoothed into an apparently curved one
    # there (same dome-shaped resolution pattern as the DOI boundary
    # itself), which can look like a small compact anomaly even with no
    # real circle present. Discard candidates centered in that
    # poorly-resolved zone.
    h_doi, w_doi = doi_mask.shape if doi_mask is not None else (0, 0)
    col_extent = None
    if doi_mask is not None:
        col_extent = np.zeros(w_doi)
        for c in range(w_doi):
            rows_c = np.where(doi_mask[:, c])[0]
            if rows_c.size:
                col_extent[c] = rows_c.max() - rows_c.min()
    max_extent = col_extent.max() if col_extent is not None and col_extent.size else 0
    min_extent_frac = seg_params.get("circle_min_doi_extent_frac", 0.65)

    labels = measure.label(blob_mask)
    for p in measure.regionprops(labels):
        y0, x0 = p.centroid
        if col_extent is not None and max_extent > 0:
            col = int(np.clip(round(x0), 0, w_doi - 1))
            if col_extent[col] < min_extent_frac * max_extent:
                continue  # poorly-resolved edge zone, not a trustworthy detection
        r = np.sqrt(p.area / np.pi)
        circles.append({
            "x": x0,
            "y": y0,
            "r": r
        })

    # --- LAYERS: each detected band can represent ONE or TWO interfaces ---
    # A genuine interface always has some background above it. A component
    # flush against the very top of the ROI is instead the naturally
    # brightest sliver of the background 'world' zone crossing the bright
    # threshold (this shows up whenever any real feature exists elsewhere in
    # the image) -- not a real boundary, so it gets dropped here.
    #
    # A band's TOP edge is always a real interface (world/layer or
    # layer/layer boundary). Its BOTTOM edge is a *second*, separate
    # interface only if the band actually closes within the resolvable
    # region -- if the band's lower edge simply runs into the
    # depth-of-investigation floor, that isn't a boundary, it's just where
    # resolution runs out (this is what distinguishes a real 1-interface
    # case, like a half-space below a single boundary, from a real
    # 2-interface case, like a bounded middle layer).
    top_roi_row = 0
    if doi_mask is not None:
        rows_with_roi = np.where(doi_mask.any(axis=1))[0]
        if rows_with_roi.size:
            top_roi_row = rows_with_roi.min()
    tau_m = seg_params.get("layer_top_rim_margin_m", 1.25)
    top_margin = (int(round(tau_m * h_doi / user_depth))
                  if (user_depth and h_doi) else 3)
    close_frac_thresh = seg_params.get("layer_floor_close_frac", 0.17)

    labels = measure.label(layer_mask)
    for p in measure.regionprops(labels):
        if p.bbox[0] <= top_roi_row + top_margin:
            continue  # world-rim artifact, not a real interface

        coords = p.coords
        cols = np.unique(coords[:, 1])
        col_lo, col_hi = {}, {}
        for c in cols:
            rows_c = coords[coords[:, 1] == c, 0]
            col_lo[c] = rows_c.min()
            col_hi[c] = rows_c.max()

        # Resolution-independent test: how far (as a fraction of the local
        # DOI depth) does the band's bottom edge sit above the local floor?
        # A small/negative median fraction means it runs right into the
        # floor (not a real interface); a comfortably positive fraction
        # means it closes with room to spare (a real second interface).
        closes_within_doi = True
        if doi_mask is not None and cols.size:
            fracs = []
            for c in cols:
                doi_rows_c = np.where(doi_mask[:, c])[0]
                if doi_rows_c.size == 0:
                    continue
                local_floor = doi_rows_c.max()
                depth_extent = max(local_floor - top_roi_row, 1)
                fracs.append((local_floor - col_hi[c]) / depth_extent)
            if fracs and np.median(fracs) < close_frac_thresh:
                closes_within_doi = False

        # Top edge: always a real interface. Position = per-column centroid
        # of the top edge (tracks the CV shape's centerline, not skewed by
        # columns where the band happens to be locally thicker).
        y_top = float(np.mean(list(col_lo.values())))
        layers.append({"y": y_top, "cols": cols, "col_lo": col_lo, "col_hi": col_hi})

        # Bottom edge: a second, separate interface, only if the band closes.
        if closes_within_doi:
            y_bottom = float(np.mean(list(col_hi.values())))
            layers.append({"y": y_bottom, "cols": cols, "col_lo": col_lo, "col_hi": col_hi})

    return circles, layers

def draw_detected_entities(ax):
    # Circles
    for i, c in enumerate(seg_circles, 1):
        circ = plt.Circle((c["x"], c["y"]), c["r"],
                          edgecolor="lime", facecolor="none", lw=2)
        ax.add_patch(circ)
        ax.text(c["x"], c["y"], f"C{i}", color="lime", fontsize=9)

    # Layers: shade each detected band once (its top and bottom edges, if
    # both are real interfaces, share the same underlying CV shape), then
    # draw a boundary line for every real interface -- one per band if it
    # runs off the bottom of the ROI, two if it closes within it.
    shaded_bands = set()
    for i, l in enumerate(seg_layers, 1):
        cols = l.get("cols")
        if cols is not None and len(cols) and id(cols) not in shaded_bands:
            shaded_bands.add(id(cols))
            cols_sorted = np.sort(cols)
            lo = [l["col_lo"][c] for c in cols_sorted]
            hi = [l["col_hi"][c] for c in cols_sorted]
            ax.fill_between(cols_sorted, lo, hi, color="magenta", alpha=0.25, linewidth=0)
        ax.axhline(l["y"], color="magenta", lw=2)
        ax.text(5, l["y"], f"L{i}", color="magenta", fontsize=9)

def show_export_button(fig_seg):
    global btn_export

    btn_exp_ax = fig_seg.add_axes([0.05, 0.14, 0.35, 0.07])
    btn_export = Button(btn_exp_ax, "Export to Master Table")

    def _export(event):
        export_to_entities_table()

    btn_export.on_clicked(_export)
    btn_exp_ax.set_zorder(10)

def compute_background_resistivity():
    global entities_df, current_img
    if current_img is None:
        return np.nan

    h, w = current_img.shape[:2]
    doi = get_current_doi_mask()

    # Find lowest REAL layer (most negative y). The Background row itself is
    # also type=="layer" once persisted, so it must be excluded here or it
    # would end up bounding itself.
    layers = entities_df[
        (entities_df["type"] == "layer") & (entities_df.get("is_background") != True)
    ]
    if layers.empty:
        y_top_pix = 0  # no interfaces at all -> background spans the whole ROI
    else:
        y_low = layers["y"].min()  # deepest real layer (negative value)
        y_top_pix = int(np.clip(-y_low / user_depth * h, 0, h))

    if h <= y_top_pix:
        return np.nan

    band = np.zeros((h, w), dtype=bool)
    band[y_top_pix:h, :] = True
    if doi is not None:
        band &= doi

    # Exclude circles from the averaging, same as compute_resistivity does for layers
    y_idx, x_idx = np.ogrid[:h, :w]
    for _, circ in entities_df[entities_df["type"] == "circle"].iterrows():
        col = (circ["x"] / user_width + 0.5) * w
        row_c = -(circ["y"] / user_depth) * h
        radius = circ["radius"] * w / user_width
        circ_mask = (x_idx - col) ** 2 + (y_idx - row_c) ** 2 <= radius ** 2
        band &= ~circ_mask

    if not band.any():
        return np.nan

    gray = current_img[..., :3].mean(axis=2)
    val = np.nanmean(gray[band])
    return _scale_to_rhoa(val)

def sync_background_row():
    """Keep exactly one 'Background' row in entities_df, always up to date,
    so it shows up in the Entities Table like any other row."""
    global entities_df
    if current_img is None:
        return

    rhoa_bg = compute_background_resistivity()
    bg_mask = entities_df.get("is_background") == True
    bg_idx = entities_df.index[bg_mask] if len(entities_df) else []

    if len(bg_idx) == 0:
        new_row = {
            "type": "layer",
            "x": 0.0,
            "y": -(user_depth),
            "radius": 0.0,
            "resistivity": rhoa_bg,
            "number": "Background",
            "is_background": True,
        }
        entities_df.loc[len(entities_df)] = new_row
    else:
        idx0 = bg_idx[0]
        entities_df.loc[idx0, "y"] = -(user_depth)
        entities_df.loc[idx0, "resistivity"] = rhoa_bg
        entities_df.loc[idx0, "is_background"] = True
        if len(bg_idx) > 1:  # guard against stray duplicates
            entities_df.drop(index=bg_idx[1:], inplace=True)
            entities_df.reset_index(drop=True, inplace=True)

def add_background_row():
    global entities_df
    rhoa_bg = compute_background_resistivity()
    bg_row = {
        "type": "layer",
        "x": 0.0,
        "y": -(user_depth),  # bottom
        "radius": 0.0,
        "resistivity": rhoa_bg,
        "number": "Background"
    }
    return pd.DataFrame([bg_row])

def export_to_entities_table():
    global entities_df

    def next_number(prefix):
        nums = [
            int(s.split()[-1])
            for s in entities_df["number"].dropna()
            if isinstance(s, str) and s.startswith(prefix)
        ]
        return max(nums)+1 if nums else 1

    # --- Circles ---
    n = next_number("Circle")
    for c in seg_circles:
        entities_df.loc[len(entities_df)] = {
            "type": "circle",
            "x":  (c["x"]/current_img.shape[1])*W_METER - W_METER/2,
            "y": -(c["y"]/current_img.shape[0])*D_METER,
            "radius": (c["r"]/current_img.shape[1])*W_METER,
            "resistivity": np.nan,
            "number": f"Circle {n}"
        }
        n += 1

    # --- Layers ---
    n = next_number("Layer")
    for l in seg_layers:
        entities_df.loc[len(entities_df)] = {
            "type": "layer",
            "x": 0.0,
            "y": -(l["y"]/current_img.shape[0])*D_METER,
            "radius": 0.0,
            "resistivity": np.nan,
            "number": f"Layer {n}"
        }
        n += 1

    redraw_shapes()

plt.show()



