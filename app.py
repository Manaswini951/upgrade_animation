import io
import math
import cv2
import numpy as np
import streamlit as st
from PIL import Image

st.set_page_config(
    page_title="Custom Color-Motion Character Animator",
    page_icon="🎨",
    layout="wide",
)

st.title("🎨 Custom Motion-per-Color Character Animator")
st.write(
    "Upload a character drawing. Assign custom motion behaviors to each detected color "
    "marker in your drawing (e.g., Red = Walk, Yellow = Laugh, Blue = Wave)."
)

# ============================================================
# ACCURATE HSV COLOR BOUNDS
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

ALL_MOTIONS = [
    "None (Static)",
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


def extract_accurate_colors(image):
    """Scans image using defined HSV ranges to detect color regions."""
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

        if coverage > 0.08:  # Ignore tiny speckles
            detected.append({
                "name": name,
                "coverage": coverage,
                "mask": combined_mask,
            })

    return detected


def detect_parts_per_color(detected_colors):
    """Extracts connected components for each distinct color region."""
    color_parts_map = {}

    for color_info in detected_colors:
        color_name = color_info["name"]
        mask = color_info["mask"]

        clean_mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, np.ones((5, 5), np.uint8))
        num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(clean_mask, connectivity=8)

        parts = []
        min_area = 40

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

        if parts:
            color_parts_map[color_name] = parts

    return color_parts_map


def classify_parts_by_position(parts, image_shape):
    """Helper to classify parts spatially for motions that require spatial hierarchy."""
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


def apply_single_motion(map_x, map_y, grid_x, grid_y, parts, motion_type, frame_index, total_frames, intensity, image_shape):
    """Applies displacement deformations for a specific motion type onto given part masks."""
    if motion_type == "None (Static)" or not parts:
        return

    phase = 2.0 * math.pi * float(frame_index) / float(total_frames)
    sine = math.sin(phase)
    cosine = math.cos(phase)
    h, w = image_shape[:2]

    # Special handling for hierarchical multi-part motions
    if motion_type in ["Advanced Walk", "Belly Laugh"]:
        rigged = classify_parts_by_position(parts, image_shape)

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

    # Standard per-part deformations
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


def animate_frame_custom(original, color_parts_map, color_motion_assignments, frame_index, total_frames, intensity):
    """Combines deformations across all custom color-motion mappings."""
    h, w = original.shape[:2]
    grid_x, grid_y = np.meshgrid(np.arange(w, dtype=np.float32), np.arange(h, dtype=np.float32))
    map_x = grid_x.copy()
    map_y = grid_y.copy()

    for color_name, parts in color_parts_map.items():
        motion = color_motion_assignments.get(color_name, "None (Static)")
        apply_single_motion(map_x, map_y, grid_x, grid_y, parts, motion, frame_index, total_frames, intensity, (h, w))

    warped = cv2.remap(original, map_x, map_y, interpolation=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT)
    return warped


def build_gif(frames, duration=50):
    buf = io.BytesIO()
    prepared = [f.convert("P", palette=Image.ADAPTIVE) for f in frames]
    prepared[0].save(buf, format="GIF", save_all=True, append_images=prepared[1:], duration=duration, loop=0)
    return buf.getvalue()


# ============================================================
# STREAMLIT UI
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
        color_parts_map = detect_parts_per_color(detected_colors)

        st.sidebar.header("⚙️ Motion Assignment Dashboard")
        color_motion_assignments = {}

        if color_parts_map:
            st.sidebar.write("Assign a specific motion to each detected color part:")
            
            # Default preset motions for intuitive auto-assignment
            presets = ["Advanced Walk", "Belly Laugh", "Dynamic Wave", "Playful Bounce", "Natural Sway"]
            
            for idx, (color_name, parts) in enumerate(color_parts_map.items()):
                default_motion = presets[idx % len(presets)]
                selected_motion = st.sidebar.selectbox(
                    f"Color: {color_name} ({len(parts)} regions)",
                    ALL_MOTIONS,
                    index=ALL_MOTIONS.index(default_motion),
                    key=f"motion_{color_name}",
                )
                color_motion_assignments[color_name] = selected_motion
        else:
            st.sidebar.warning("No distinct colored markers detected.")

        intensity = float(st.sidebar.slider("Motion Strength", 4, 30, 12))

        col1, col2 = st.columns(2)
        with col1:
            st.subheader("Original Drawing")
            st.image(cv2.cvtColor(img, cv2.COLOR_BGR2RGB), use_container_width=True)

        with col2:
            st.subheader("Detected Color Regions")
            preview = img.copy()

            for color_name, parts in color_parts_map.items():
                for p in parts:
                    cv2.circle(preview, (int(p["base"][0]), int(p["base"][1])), 6, (0, 255, 0), -1)
                    cv2.circle(preview, (int(p["tip"][0]), int(p["tip"][1])), 4, (0, 0, 255), -1)
                    cv2.line(preview, (int(p["base"][0]), int(p["base"][1])), (int(p["tip"][0]), int(p["tip"][1])), (255, 255, 0), 2)

            st.image(cv2.cvtColor(preview, cv2.COLOR_BGR2RGB), use_container_width=True)

        st.markdown("---")

        render_btn = st.button("✨ Render Custom Composition", type="primary", disabled=(len(color_parts_map) == 0))

        if render_btn:
            with st.spinner("Calculating composite motion vectors across all assigned colors..."):
                frames = []
                for i in range(16):
                    warped = animate_frame_custom(img, color_parts_map, color_motion_assignments, i, 16, intensity)
                    rgb_frame = np.ascontiguousarray(cv2.cvtColor(warped, cv2.COLOR_BGR2RGB))
                    frames.append(Image.fromarray(rgb_frame))
                gif_bytes = build_gif(frames, duration=50)

            st.subheader("🎬 Custom Composite Animation Result")
            st.image(gif_bytes, width=500)
            
            # Show active motion mapping summary
            summary_str = " | ".join([f"**{c}**: {m}" for c, m in color_motion_assignments.items()])
            st.caption(f"Active Mappings: {summary_str}")

            st.download_button(
                "Download Animation GIF",
                data=gif_bytes,
                file_name="custom_character_animation.gif",
                mime="image/gif",
            )

    except Exception as e:
        st.error(f"Error encountered: {e}")
