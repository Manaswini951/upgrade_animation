import io
import math
import cv2
import numpy as np
import streamlit as st
from PIL import Image

st.set_page_config(
    page_title="10-in-1 Character Animator",
    page_icon="🎨",
    layout="wide",
)

st.title("🎨 Advanced Hand-Drawn Character Animator")
st.write(
    "Upload a character drawing. Colored markers are automatically rigged into "
    "hierarchical body components (Head, Torso, Arms, Legs) to generate coordinated, "
    "multi-part animations like Walking and Laughing."
)

# ============================================================
# ACCURATE HSV COLOR BOUNDS WITH RED/VIOLET SEPARATION
# ============================================================

COLOR_RANGES = {
    "Red": [
        (np.array([0, 70, 50]), np.array([8, 255, 255])),
        (np.array([172, 70, 50]), np.array([180, 255, 255])),
    ],
    "Orange": [(np.array([9, 70, 50]), np.array([20, 255, 255]))],
    "Yellow": [(np.array([21, 70, 50]), np.array([34, 255, 255]))],
    "Green": [(np.array([35, 50, 40]), np.array([84, 255, 255]))],
    "Cyan / Light Blue": [(np.array([85, 50, 40]), np.array([105, 255, 255]))],
    "Blue": [(np.array([106, 50, 40]), np.array([130, 255, 255]))],
    "Purple / Violet": [(np.array([131, 50, 40]), np.array([155, 255, 255]))],
    "Pink / Magenta": [(np.array([156, 50, 40]), np.array([171, 255, 255]))],
}


def extract_accurate_colors(image):
    """
    Scans image using defined HSV ranges to eliminate false overlaps
    between adjacent spectrums (e.g., Red vs. Magenta/Violet).
    """
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    total_pixels = image.shape[0] * image.shape[1]
    detected = []

    for name, ranges in COLOR_RANGES.items():
        combined_mask = np.zeros(hsv.shape[:2], dtype=np.uint8)
        for lower, upper in ranges:
            mask = cv2.inRange(hsv, lower, upper)
            combined_mask = cv2.bitwise_or(combined_mask, mask)

        pixel_count = cv2.countNonZero(combined_mask)
        coverage = (pixel_count / total_pixels) * 100

        # Ignore tiny speckles (< 0.08% of total area)
        if coverage > 0.08:
            detected.append({
                "label": f"{name} ({coverage:.2f}% area)",
                "name": name,
                "mask": combined_mask,
            })

    return detected


def detect_parts_from_masks(chosen_colors, image_shape):
    h, w = image_shape[:2]
    combined = np.zeros((h, w), dtype=np.uint8)

    for color_info in chosen_colors:
        combined = cv2.bitwise_or(combined, color_info["mask"])

    clean_mask = cv2.morphologyEx(combined, cv2.MORPH_CLOSE, np.ones((5, 5), np.uint8))
    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(clean_mask, connectivity=8)

    parts = []
    min_area = max(40, int(h * w * 0.0001))

    for i in range(1, num_labels):
        area = int(stats[i, cv2.CC_STAT_AREA])
        if area < min_area:
            continue

        comp = (labels == i).astype(np.uint8) * 255
        ys, xs = np.where(comp > 0)
        if len(xs) < 10:
            continue

        base_y = float(np.max(ys))
        base_x = float(np.mean(xs[ys == int(base_y)]))

        tip_y = float(np.min(ys))
        tip_x = float(np.mean(xs[ys == int(tip_y)]))

        length = max(10.0, float(math.hypot(tip_x - base_x, tip_y - base_y)))
        expanded = cv2.dilate(comp, np.ones((5, 5), np.uint8), iterations=1)

        parts.append({
            "mask": expanded > 0,
            "base": (base_x, base_y),
            "tip": (tip_x, tip_y),
            "center": (float(np.mean(xs)), float(np.mean(ys))),
            "length": length,
            "area": area,
        })

    parts.sort(key=lambda p: p["area"], reverse=True)
    return parts


def classify_body_parts(parts, image_shape):
    h, _ = image_shape[:2]
    rigged = {"head": [], "torso": [], "arms": [], "legs": []}

    for p in parts:
        _, cy = p["center"]
        rel_y = cy / h

        if rel_y < 0.28:
            rigged["head"].append(p)
        elif 0.28 <= rel_y < 0.62:
            rigged["torso"].append(p)
            rigged["arms"].append(p)
        else:
            rigged["legs"].append(p)

    return rigged


def animate_frame_elastic(original, parts, motion_type, frame_index, total_frames, intensity):
    h, w = original.shape[:2]
    grid_x, grid_y = np.meshgrid(np.arange(w, dtype=np.float32), np.arange(h, dtype=np.float32))
    map_x = grid_x.copy()
    map_y = grid_y.copy()

    phase = 2.0 * math.pi * float(frame_index) / float(total_frames)
    sine = math.sin(phase)
    cosine = math.cos(phase)

    rigged = classify_body_parts(parts, (h, w))

    if motion_type == "Advanced Walk":
        stride_shift = (float(frame_index) / total_frames) * (w * 0.12)
        vertical_bob = abs(math.sin(phase * 2.0)) * (intensity * 0.4)

        map_x -= stride_shift
        map_y += vertical_bob

        for idx, leg in enumerate(rigged["legs"]):
            mask = leg["mask"]
            bx, by = leg["base"]
            leg_phase = phase if idx % 2 == 0 else phase + math.pi

            dist = np.sqrt((grid_x - bx) ** 2 + (grid_y - by) ** 2)
            weight = np.power(np.clip(dist / leg["length"], 0.0, 1.0), 1.2)

            dx = math.sin(leg_phase) * (intensity * 1.5) * weight
            dy = np.maximum(0.0, -math.cos(leg_phase)) * (intensity * 0.8) * weight
            map_x[mask] -= dx[mask]
            map_y[mask] -= dy[mask]

        for idx, arm in enumerate(rigged["arms"]):
            mask = arm["mask"]
            bx, by = arm["base"]
            arm_phase = phase + math.pi if idx % 2 == 0 else phase

            dist = np.sqrt((grid_x - bx) ** 2 + (grid_y - by) ** 2)
            weight = np.power(np.clip(dist / arm["length"], 0.0, 1.0), 1.0)

            dx = math.sin(arm_phase) * (intensity * 1.2) * weight
            map_x[mask] -= dx[mask]

    elif motion_type == "Belly Laugh":
        fast_tremor = math.sin(phase * 6.0) * (intensity * 0.5)
        slow_nod = math.sin(phase) * (intensity * 0.3)

        for torso in rigged["torso"]:
            mask = torso["mask"]
            cx, cy = torso["center"]

            scale = 1.0 + (fast_tremor * 0.02)
            dx = (grid_x - cx) * (scale - 1.0)
            dy = (grid_y - cy) * (scale - 1.0) + fast_tremor

            map_x[mask] -= dx[mask]
            map_y[mask] -= dy[mask]

        for head in rigged["head"]:
            mask = head["mask"]
            bx, by = head["base"]

            dist = np.sqrt((grid_x - bx) ** 2 + (grid_y - by) ** 2)
            weight = np.clip(dist / max(10.0, head["length"]), 0.0, 1.0)

            dy = (slow_nod - abs(fast_tremor * 0.5)) * weight
            dx = fast_tremor * 0.3 * weight

            map_x[mask] -= dx[mask]
            map_y[mask] -= dy[mask]

    else:
        for idx, part in enumerate(parts):
            mask = part["mask"]
            bx, by = part["base"]
            cx, cy = part["center"]
            length = part["length"]

            dist = np.sqrt((grid_x - bx) ** 2 + (grid_y - by) ** 2)
            norm_dist = np.clip(dist / length, 0.0, 1.0)
            weight = np.power(norm_dist, 1.5)
            side = 1.0 if idx % 2 == 0 else -1.0

            if motion_type == "Natural Sway":
                dx = sine * intensity * side * weight
                map_x[mask] -= dx[mask]

            elif motion_type == "Playful Bounce":
                dy = sine * (intensity * 0.8) * weight
                dx = -sine * (intensity * 0.25) * side * weight
                map_x[mask] -= dx[mask]
                map_y[mask] -= dy[mask]

            elif motion_type == "Dynamic Wave":
                dx = sine * (intensity * 1.4) * side * weight
                dy = (1.0 - abs(sine)) * (intensity * 0.5) * weight
                map_x[mask] -= dx[mask]
                map_y[mask] -= dy[mask]

            elif motion_type == "Rapid Twitch":
                snappy_sine = math.sin(phase * 3.0)
                dx = snappy_sine * (intensity * 0.6) * side * (weight ** 2)
                map_x[mask] -= dx[mask]

            elif motion_type == "Breathing Pulse":
                scale_delta = sine * (intensity / 100.0) * 0.6
                dx = (grid_x - cx) * scale_delta * weight
                dy = (grid_y - cy) * scale_delta * weight
                map_x[mask] -= dx[mask]
                map_y[mask] -= dy[mask]

            elif motion_type == "Curved Smile":
                norm_x = (grid_x - cx) / max(10.0, length * 0.5)
                dy = sine * (intensity * 0.6) * np.square(norm_x) * weight
                map_y[mask] -= dy[mask]

            elif motion_type == "Curious Tilt":
                dx = sine * (intensity * 0.8) * weight
                dy = cosine * (intensity * 0.4) * weight
                map_x[mask] -= dx[mask]
                map_y[mask] -= dy[mask]

            elif motion_type == "Jitter / Shiver":
                jitter_x = math.sin(phase * 5.0) * (intensity * 0.35) * weight
                jitter_y = math.cos(phase * 4.0) * (intensity * 0.25) * weight
                map_x[mask] -= jitter_x[mask]
                map_y[mask] -= jitter_y[mask]

            elif motion_type == "Wind Flutter":
                wave_travel = np.sin(phase * 2.0 - norm_dist * 4.0)
                dx = wave_travel * (intensity * 0.9) * side * np.power(norm_dist, 1.2)
                map_x[mask] -= dx[mask]

    warped = cv2.remap(original, map_x, map_y, interpolation=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT)
    return warped


def build_gif(frames, duration=50):
    buf = io.BytesIO()
    prepared = [f.convert("P", palette=Image.ADAPTIVE) for f in frames]
    prepared[0].save(buf, format="GIF", save_all=True, append_images=prepared[1:], duration=duration, loop=0)
    return buf.getvalue()


ALL_MOTIONS = [
    "Advanced Walk",
    "Belly Laugh",
    "Natural Sway",
    "Playful Bounce",
    "Dynamic Wave",
    "Rapid Twitch",
    "Breathing Pulse",
    "Curved Smile",
    "Curious Tilt",
    "Jitter / Shiver",
    "Wind Flutter",
]

# ============================================================
# STREAMLIT INTERFACE
# ============================================================

uploaded_file = st.file_uploader("Upload character drawing", type=["jpg", "jpeg", "png", "webp"])

if uploaded_file is not None:
    try:
        file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
        img = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)

        if img is None:
            st.error("Invalid image format.")
            st.stop()

        h, w = img.shape[:2]
        if max(h, w) > 1000:
            scale = 1000.0 / max(h, w)
            img = cv2.resize(img, (int(w * scale), int(h * scale)), interpolation=cv2.INTER_AREA)

        detected_colors = extract_accurate_colors(img)

        st.sidebar.header("Controls")
        chosen_colors = []
        if detected_colors:
            color_dict = {c["label"]: c for c in detected_colors}
            selected_labels = st.sidebar.multiselect(
                "Marker Colors to Animate",
                list(color_dict.keys()),
                default=[list(color_dict.keys())[0]],
            )
            chosen_colors = [color_dict[lbl] for lbl in selected_labels]
        else:
            st.sidebar.warning("No distinct marker colors detected.")

        intensity = float(st.sidebar.slider("Motion Strength", 4, 30, 12))

        parts = detect_parts_from_masks(chosen_colors, img.shape) if chosen_colors else []
        classified = classify_body_parts(parts, img.shape) if parts else {}

        col1, col2 = st.columns(2)
        with col1:
            st.subheader("Original Image")
            st.image(cv2.cvtColor(img, cv2.COLOR_BGR2RGB), use_container_width=True)

        with col2:
            st.subheader(f"Rigged Skeleton ({len(parts)} regions detected)")
            preview = img.copy()

            colors_map = {
                "head": (255, 0, 0),
                "torso": (0, 255, 255),
                "arms": (0, 255, 0),
                "legs": (0, 0, 255),
            }

            for group, group_parts in classified.items():
                for p in group_parts:
                    cv2.circle(preview, (int(p["base"][0]), int(p["base"][1])), 6, colors_map[group], -1)
                    cv2.circle(preview, (int(p["tip"][0]), int(p["tip"][1])), 4, (255, 255, 255), -1)
                    cv2.line(
                        preview,
                        (int(p["base"][0]), int(p["base"][1])),
                        (int(p["tip"][0]), int(p["tip"][1])),
                        colors_map[group],
                        2,
                    )

            st.image(cv2.cvtColor(preview, cv2.COLOR_BGR2RGB), use_container_width=True)
            st.caption("🔵 Head | 🟡 Torso | 🟢 Arms | 🔴 Legs")

        st.markdown("---")

        btn_col1, btn_col2 = st.columns([1, 1])
        with btn_col1:
            generate_all = st.button("✨ Generate All Motions", type="primary", disabled=(len(parts) == 0))
        with btn_col2:
            single_motion = st.selectbox("Select Single Motion:", ALL_MOTIONS)
            generate_single = st.button(f"Render '{single_motion}'", disabled=(len(parts) == 0))

        # Render All
        if generate_all:
            progress = st.progress(0, text="Rendering animations...")
            results = []

            for idx, motion_name in enumerate(ALL_MOTIONS):
                frames = []
                for i in range(16):
                    warped = animate_frame_elastic(img, parts, motion_name, i, 16, intensity)
                    rgb_frame = np.ascontiguousarray(cv2.cvtColor(warped, cv2.COLOR_BGR2RGB))
                    frames.append(Image.fromarray(rgb_frame))
                gif_bytes = build_gif(frames, duration=50)
                results.append((motion_name, gif_bytes))
                progress.progress((idx + 1) / len(ALL_MOTIONS))

            progress.empty()
            st.success("All motions generated!")

            for row_start in range(0, len(results), 3):
                row_cols = st.columns(3)
                for col_idx in range(3):
                    item_idx = row_start + col_idx
                    if item_idx < len(results):
                        name, gif_data = results[item_idx]
                        with row_cols[col_idx]:
                            st.markdown(f"#### {item_idx + 1}. {name}")
                            st.image(gif_data, use_container_width=True)
                            st.download_button(
                                f"Download {name}",
                                data=gif_data,
                                file_name=f"{name.lower().replace(' ', '_')}.gif",
                                mime="image/gif",
                                key=f"dl_{item_idx}",
                            )

        # Render Single
        elif generate_single:
            with st.spinner(f"Rendering {single_motion}..."):
                frames = []
                for i in range(16):
                    warped = animate_frame_elastic(img, parts, single_motion, i, 16, intensity)
                    rgb_frame = np.ascontiguousarray(cv2.cvtColor(warped, cv2.COLOR_BGR2RGB))
                    frames.append(Image.fromarray(rgb_frame))
                gif_bytes = build_gif(frames, duration=50)

            st.subheader(f"Result: {single_motion}")
            st.image(gif_bytes, width=500)
            st.download_button(
                f"Download {single_motion} GIF",
                data=gif_bytes,
                file_name=f"{single_motion.lower().replace(' ', '_')}.gif",
                mime="image/gif",
            )

    except Exception as e:
        st.error(f"Error encountered: {e}")
