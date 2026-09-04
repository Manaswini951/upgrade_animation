import base64
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


def get_color_name(rgb):
    r, g, b = int(rgb[0]), int(rgb[1]), int(rgb[2])
    pixel = np.uint8([[[b, g, r]]])
    hsv = cv2.cvtColor(pixel, cv2.COLOR_BGR2HSV)[0][0]
    hue, sat = int(hsv[0]), int(hsv[1])

    if sat < 35:
        return "Neutral"
    if hue < 8 or hue >= 172:
        return "Red"
    elif 8 <= hue < 22:
        return "Orange"
    elif 22 <= hue < 35:
        return "Yellow"
    elif 35 <= hue < 85:
        return "Green"
    elif 85 <= hue < 130:
        return "Blue"
    elif 130 <= hue < 155:
        return "Purple"
    elif 155 <= hue < 172:
        return "Pink"
    return "Accent"


def extract_dominant_colors(image, num_clusters=5):
    try:
        small = cv2.resize(image, (120, 120), interpolation=cv2.INTER_AREA)
        gray = cv2.cvtColor(small, cv2.COLOR_BGR2GRAY)
        valid = (gray > 45) & (gray < 235)
        pixels = small[valid].reshape(-1, 3).astype(np.float32)

        if len(pixels) < 50:
            return []

        k = min(num_clusters, max(1, len(pixels) // 20))
        criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 15, 1.0)
        _, labels, centers = cv2.kmeans(pixels, k, None, criteria, 3, cv2.KMEANS_PP_CENTERS)

        counts = np.bincount(labels.flatten(), minlength=k)
        total = max(1, int(np.sum(counts)))
        detected = []

        for idx, center in enumerate(centers):
            b, g, r = int(center[0]), int(center[1]), int(center[2])
            coverage = float((counts[idx] / total) * 100)
            if coverage < 1.5:
                continue

            hsv = cv2.cvtColor(np.uint8([[[b, g, r]]]), cv2.COLOR_BGR2HSV)[0][0]
            if int(hsv[1]) < 30:
                continue

            hex_code = f"#{r:02x}{g:02x}{b:02x}"
            detected.append({
                "label": f"{get_color_name((r, g, b))} ({hex_code}) {coverage:.1f}%",
                "hsv": (int(hsv[0]), int(hsv[1]), int(hsv[2])),
                "hex": hex_code,
            })

        return detected
    except Exception:
        return []


def detect_parts_safe(image, chosen_colors):
    h, w = image.shape[:2]
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    combined = np.zeros((h, w), dtype=np.uint8)

    for color in chosen_colors:
        h_val, s_val, v_val = color["hsv"]
        lower = np.array([max(0, h_val - 16), max(25, s_val - 70), max(25, v_val - 70)], dtype=np.uint8)
        upper = np.array([min(179, h_val + 16), 255, 255], dtype=np.uint8)
        mask = cv2.inRange(hsv, lower, upper)
        combined = cv2.bitwise_or(combined, mask)

    clean_mask = cv2.morphologyEx(combined, cv2.MORPH_CLOSE, np.ones((5, 5), np.uint8))
    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(clean_mask, connectivity=8)

    parts = []
    min_area = max(60, int(h * w * 0.0001))

    for i in range(1, num_labels):
        area = int(stats[i, cv2.CC_STAT_AREA])
        if area < min_area:
            continue

        comp = (labels == i).astype(np.uint8) * 255
        ys, xs = np.where(comp > 0)
        if len(xs) < 15:
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
    """Sorts detected components into body regions based on spatial layout."""
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

    # ============================================================
    # 1. ADVANCED WALK CYCLE (Translational + Phase-Opposed Limbs)
    # ============================================================
    if motion_type == "Advanced Walk":
        # Global Horizontal Shift & Vertical Hip Bobbing
        stride_shift = (float(frame_index) / total_frames) * (w * 0.12)
        vertical_bob = abs(math.sin(phase * 2.0)) * (intensity * 0.4)

        map_x -= stride_shift
        map_y += vertical_bob

        # Alternating Legs
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

        # Arm Opposition (Counter-swinging)
        for idx, arm in enumerate(rigged["arms"]):
            mask = arm["mask"]
            bx, by = arm["base"]
            arm_phase = phase + math.pi if idx % 2 == 0 else phase

            dist = np.sqrt((grid_x - bx) ** 2 + (grid_y - by) ** 2)
            weight = np.power(np.clip(dist / arm["length"], 0.0, 1.0), 1.0)

            dx = math.sin(arm_phase) * (intensity * 1.2) * weight
            map_x[mask] -= dx[mask]

    # ============================================================
    # 2. BELLY LAUGH (Multi-Frequency Tremor + Chest Expansion)
    # ============================================================
    elif motion_type == "Belly Laugh":
        fast_tremor = math.sin(phase * 6.0) * (intensity * 0.5)
        slow_nod = math.sin(phase) * (intensity * 0.3)

        # Torso / Chest Vibration
        for torso in rigged["torso"]:
            mask = torso["mask"]
            cx, cy = torso["center"]

            scale = 1.0 + (fast_tremor * 0.02)
            dx = (grid_x - cx) * (scale - 1.0)
            dy = (grid_y - cy) * (scale - 1.0) + fast_tremor

            map_x[mask] -= dx[mask]
            map_y[mask] -= dy[mask]

        # Head Pitching Backwards & Nodding
        for head in rigged["head"]:
            mask = head["mask"]
            bx, by = head["base"]

            dist = np.sqrt((grid_x - bx) ** 2 + (grid_y - by) ** 2)
            weight = np.clip(dist / max(10.0, head["length"]), 0.0, 1.0)

            dy = (slow_nod - abs(fast_tremor * 0.5)) * weight
            dx = fast_tremor * 0.3 * weight

            map_x[mask] -= dx[mask]
            map_y[mask] -= dy[mask]

    # Standard Deformations
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
# STREAMLIT UI
# ============================================================

uploaded_file = st.file_uploader("Upload hand-drawn character", type=["jpg", "jpeg", "png", "webp"])

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

        detected_colors = extract_dominant_colors(img)

        st.sidebar.header("Controls")
        chosen_colors = []
        if detected_colors:
            color_dict = {c["label"]: c for c in detected_colors}
            selected_labels = st.sidebar.multiselect(
                "Colors to Animate",
                list(color_dict.keys()),
                default=[list(color_dict.keys())[0]],
            )
            chosen_colors = [color_dict[lbl] for lbl in selected_labels]
        else:
            st.sidebar.info("No distinct marker colors detected.")

        intensity = float(st.sidebar.slider("Motion Strength", 4, 30, 12))

        parts = detect_parts_safe(img, chosen_colors) if chosen_colors else []
        classified = classify_body_parts(parts, img.shape) if parts else {}

        col1, col2 = st.columns(2)
        with col1:
            st.subheader("Original Image")
            st.image(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))

        with col2:
            st.subheader(f"Rigged Skeleton ({len(parts)} regions detected)")
            preview = img.copy()

            # Color-coded skeletal visualization
            colors = {
                "head": (255, 0, 0),    # Blue
                "torso": (0, 255, 255), # Yellow
                "arms": (0, 255, 0),    # Green
                "legs": (0, 0, 255),    # Red
            }

            for group, group_parts in classified.items():
                for p in group_parts:
                    cv2.circle(preview, (int(p["base"][0]), int(p["base"][1])), 6, colors[group], -1)
                    cv2.circle(preview, (int(p["tip"][0]), int(p["tip"][1])), 4, (255, 255, 255), -1)
                    cv2.line(
                        preview,
                        (int(p["base"][0]), int(p["base"][1])),
                        (int(p["tip"][0]), int(p["tip"][1])),
                        colors[group],
                        2,
                    )

            st.image(cv2.cvtColor(preview, cv2.COLOR_BGR2RGB))
            st.caption("🔵 Head | 🟡 Torso | 🟢 Arms | 🔴 Legs")

        st.markdown("---")

        btn_col1, btn_col2 = st.columns([1, 1])
        with btn_col1:
            generate_all = st.button("✨ Generate All Motions", type="primary", disabled=(len(parts) == 0))
        with btn_col2:
            single_motion = st.selectbox("Select Motion to Render:", ALL_MOTIONS)
            generate_single = st.button(f"Render '{single_motion}'", disabled=(len(parts) == 0))

        # Render All
        if generate_all:
            progress = st.progress(0, text="Processing multi-region kinematics...")
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
            st.success("All motions rendered successfully!")

            for row_start in range(0, len(results), 3):
                row_cols = st.columns(3)
                for col_idx in range(3):
                    item_idx = row_start + col_idx
                    if item_idx < len(results):
                        name, data = results[item_idx]
                        with row_cols[col_idx]:
                            st.markdown(f"#### {item_idx + 1}. {name}")
                            b64 = base64.b64encode(data).decode("utf-8")
                            st.markdown(
                                f'<img src="data:image/gif;base64,{b64}" style="width: 100%; border-radius: 8px; margin-bottom: 8px;">',
                                unsafe_allow_html=True,
                            )
                            st.download_button(
                                f"Download {name}",
                                data=data,
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
            b64 = base64.b64encode(gif_bytes).decode("utf-8")
            st.markdown(
                f'<img src="data:image/gif;base64,{b64}" style="max-width: 600px; border-radius: 8px; margin-bottom: 8px;">',
                unsafe_allow_html=True,
            )
            st.download_button(
                f"Download {single_motion} GIF",
                data=gif_bytes,
                file_name=f"{single_motion.lower().replace(' ', '_')}.gif",
                mime="image/gif",
            )

    except Exception as e:
        st.error(f"Error encountered: {e}")
