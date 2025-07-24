import numpy as np
import cv2
from PIL import Image
from skimage.metrics import structural_similarity as ssim


# Funkcje do zadan
def draw_line(X, Y, bufferr, color):
    x0, x1 = X
    y0, y1 = Y
    dx = abs(x1 - x0)
    dy = abs(y1 - y0)
    sx = 1 if x0 < x1 else -1
    sy = 1 if y0 < y1 else -1
    err = dx - dy
    while True:
        if 0 <= x0 < bufferr.shape[1] and 0 <= y0 < bufferr.shape[0]:
            bufferr[y0, x0] = color
        if x0 == x1 and y0 == y1:
            break
        e2 = 2 * err
        if e2 > -dy:
            err -= dy
            x0 += sx
        if e2 < dx:
            err += dx
            y0 += sy


def flood_fill(x, y, bufferr, new_color, original_color=None):
    if x < 0 or x >= bufferr.shape[1] or y < 0 or y >= bufferr.shape[0]:
        return
    if original_color is None:
        original_color = tuple(bufferr[y, x])
    if tuple(bufferr[y, x]) != original_color or np.array_equal(bufferr[y, x], new_color):
        return
    stack = [(x, y)]
    while stack:
        cx, cy = stack.pop()
        if cx < 0 or cx >= bufferr.shape[1] or cy < 0 or cy >= bufferr.shape[0]:
            continue
        if not np.array_equal(bufferr[cy, cx], original_color):
            continue
        bufferr[cy, cx] = new_color
        stack.extend([(cx + 1, cy), (cx - 1, cy), (cx, cy + 1), (cx, cy - 1)])


def draw_circle(center, radius, bufferr, color):
    cx, cy = center
    for y in range(cy - radius, cy + radius + 1):
        for x in range(cx - radius, cx + radius + 1):
            if (x - cx) * (x - cx) + (y - cy) * (y - cy) <= radius * radius:
                if 0 <= x < bufferr.shape[1] and 0 <= y < bufferr.shape[0]:
                    bufferr[y, x] = color


def draw_triangle(X, Y, bufferr, color):
    for i in range(3):
        x = [X[i], X[(i + 1) % 3]]
        y = [Y[i], Y[(i + 1) % 3]]
        draw_line(x, y, bufferr, color)

    min_y, max_y = min(Y), max(Y)
    for scan_y in range(min_y + 1, max_y):
        intersections = []
        for i in range(3):
            y1, y2 = Y[i], Y[(i + 1) % 3]
            x1, x2 = X[i], X[(i + 1) % 3]

            if y1 != y2 and min(y1, y2) < scan_y <= max(y1, y2):
                x_intersect = x1 + (scan_y - y1) * (x2 - x1) / (y2 - y1)
                intersections.append(int(x_intersect))
        intersections.sort()
        for i in range(0, len(intersections), 2):
            if i + 1 < len(intersections):
                x_start, x_end = intersections[i], intersections[i + 1]
                if x_start != x_end:  # Only draw line if there's a span
                    draw_line([x_start, x_end], [scan_y, scan_y], bufferr, color)


def draw_rectangle(X, Y, bufferr, color):
    x1, x2 = sorted(X)
    y1, y2 = sorted(Y)
    draw_line([x1, x2], [y1, y1], bufferr, color)  # Top edge
    draw_line([x2, x2], [y1, y2], bufferr, color)  # Right edge
    draw_line([x2, x1], [y2, y2], bufferr, color)  # Bottom edge
    draw_line([x1, x1], [y2, y1], bufferr, color)  # Left edge
    for y in range(y1 + 1, y2):
        draw_line([x1, x2], [y, y], bufferr, color)


def draw_polygon(X, Y, bufferr, color):
    n = len(X)
    if n < 3:
        return
    for i in range(n):
        x = [X[i], X[(i + 1) % n]]
        y = [Y[i], Y[(i + 1) % n]]
        draw_line(x, y, bufferr, color)
    min_y, max_y = min(Y), max(Y)
    for scan_y in range(min_y + 1, max_y):
        intersections = []
        for i in range(n):
            y1, y2 = Y[i], Y[(i + 1) % n]
            x1, x2 = X[i], X[(i + 1) % n]

            if y1 != y2 and min(y1, y2) < scan_y <= max(y1, y2):
                x_intersect = x1 + (scan_y - y1) * (x2 - x1) / (y2 - y1)
                intersections.append(int(x_intersect))
        intersections.sort()
        for i in range(0, len(intersections), 2):
            if i + 1 < len(intersections):
                x_start, x_end = intersections[i], intersections[i + 1]
                if x_start != x_end:  # Only draw line if there's a span
                    draw_line([x_start, x_end], [scan_y, scan_y], bufferr, color)


def draw(image_shape, image_data):
    height, width, layers = image_shape
    background_color = tuple(image_data["Canvas"]["background_color"])
    buffer = np.full((height, width, layers), background_color, dtype=np.uint8)

    shapes = sorted(image_data["Shapes"], key=lambda s: s["Z_layer"])

    for shape in shapes:
        color = tuple(shape.get("color", [255, 255, 255]))
        typ = shape["type"]

        if typ == "line":
            p1 = shape["p1"]
            p2 = shape["p2"]
            draw_line([p1[0], p2[0]], [p1[1], p2[1]], buffer, color)

        elif typ == "circle":
            center = shape["center"]
            radius = shape["radius"]
            draw_circle(center, radius, buffer, color)

        elif typ == "rectangle":
            p1 = shape["p1"]
            p2 = shape["p2"]
            draw_rectangle([p1[0], p2[0]], [p1[1], p2[1]], buffer, color)

        elif typ == "triangle":
            p1 = shape["p1"]
            p2 = shape["p2"]
            p3 = shape["p3"]
            draw_triangle([p1[0], p2[0], p3[0]], [p1[1], p2[1], p3[1]], buffer, color)

        elif typ == "freeform":
            points = shape["points"]
            X = [pt[0] for pt in points]
            Y = [pt[1] for pt in points]
            draw_polygon(X, Y, buffer, color)
    return buffer


def save_image_as_png(image_array, filename):
    img = Image.fromarray(image_array.astype('uint8'), 'RGB')
    img.save(filename)


# Updated example with all required elements
Example = {
    "Canvas": {
        "background_color": [223,223,223]
    },
    "Shapes": [
        # 1. Duży zielony okrąg (zasłonięty przez trójkąt)
        {"type": "circle", "center": [120, 120], "radius": 60, "color": [138, 120, 53], "Z_layer": 0},
        {"type": "triangle", "p1": [80, 100], "p2": [160, 90], "p3": [120, 160], "color": [255,0,0], "Z_layer": 2},

        # 2. Prostokąt z dwoma kwadratami w środku
        {"type": "rectangle", "p1": [200, 50], "p2": [320, 150], "color": [0, 0, 255], "Z_layer": 0},
        {"type": "rectangle", "p1": [220, 70], "p2": [250, 100], "color": [255, 0, 0], "Z_layer": 1},
        {"type": "rectangle", "p1": [270, 70], "p2": [300, 100], "color": [0, 255, 255], "Z_layer": 1},

        # 3. Wielokąt L
        {"type": "freeform", "points": [[50, 200], [70, 200], [70, 270], [110, 270], [110, 290], [50, 290]], "color": [155, 200, 255], "Z_layer": 1},

        # 4. Żółty okrąg wysunięty w połowie za brązowego prostokąta,
        {"type": "circle", "center": [280, 250], "radius": 20, "color": [255, 255, 0], "Z_layer": 0},
        {"type": "rectangle", "p1": [250, 250], "p2": [330, 310], "color": [139, 69, 19], "Z_layer": 2},


        # 5. Czarny trójkąt przykrywa część okręgu, drugi trójkąt pod spodem
        {"type": "triangle", "p1": [80, 300], "p2": [200, 300], "p3": [200, 360], "color": [0,0,0], "Z_layer": 0},
        {"type": "circle", "center": [200, 300], "radius": 40, "color": [0, 255, 0], "Z_layer": 1},
        {"type": "triangle", "p1": [210, 300], "p2": [170, 320], "p3": [270, 330], "color": [128, 128, 128], "Z_layer": 2},

    ]
}
# Save image as JPG
def save_image_as_jpg(image_array, filename):
    img = Image.fromarray(image_array.astype('uint8'), 'RGB')
    img.save(filename, "JPEG")

# MSE calculation
def mse(image1, image2):
    return np.mean((image1 - image2) ** 2)

def scale_shape(shape, original_size, target_size):
    scale_x = target_size[0] / original_size[0]
    scale_y = target_size[1] / original_size[1]

    if shape["type"] == "circle":
        return {
            "type": "circle",
            "center": [int(shape["center"][0] * scale_x), int(shape["center"][1] * scale_y)],
            "radius": int(shape["radius"] * min(scale_x, scale_y)),
            "color": shape["color"]
        }
    elif shape["type"] == "rectangle":
        return {
            "type": "rectangle",
            "p1": [int(shape["p1"][0] * scale_x), int(shape["p1"][1] * scale_y)],
            "p2": [int(shape["p2"][0] * scale_x), int(shape["p2"][1] * scale_y)],
            "color": shape["color"]
        }
    elif shape["type"] == "triangle":
        return {
            "type": "triangle",
            "p1": [int(shape["p1"][0] * scale_x), int(shape["p1"][1] * scale_y)],
            "p2": [int(shape["p2"][0] * scale_x), int(shape["p2"][1] * scale_y)],
            "p3": [int(shape["p3"][0] * scale_x), int(shape["p3"][1] * scale_y)],
            "color": shape["color"]
        }
    elif shape["type"] == "freeform":
        return {
            "type": "freeform",
            "points": [[int(pt[0] * scale_x), int(pt[1] * scale_y)] for pt in shape["points"]],
            "color": shape["color"]
        }
# Render and scale images
original_size = (400, 400)
sizes = [(400, 400), (300, 300), (500, 500), (600, 600), (200, 200)]
common_size = (400, 400)
images = []

for size in sizes:
    # Scale shapes to fit the current canvas size
    scaled_shapes = [scale_shape(shape, original_size, size) for shape in Example["Shapes"]]

    # Render the canvas
    # Render the canvas
    buffer = np.full((size[1], size[0], 3), Example["Canvas"]["background_color"], dtype=np.uint8)
    for shape in scaled_shapes:
        color = tuple(shape.get("color", [255, 255, 255]))  # Default to white if color is None
        if shape["type"] == "circle":
            cv2.circle(buffer, tuple(shape["center"]), shape["radius"], color, -1)
        elif shape["type"] == "triangle":
            pts = np.array([shape["p1"], shape["p2"], shape["p3"]], np.int32)
            cv2.fillPoly(buffer, [pts], color)
        elif shape["type"] == "rectangle":
            cv2.rectangle(buffer, tuple(shape["p1"]), tuple(shape["p2"]), color, -1)
        elif shape["type"] == "freeform":
            pts = np.array(shape["points"], np.int32)
            cv2.fillPoly(buffer, [pts], color)

    # Scale the canvas to the common size
    scaled_image = cv2.resize(buffer, common_size)
    images.append(scaled_image)

# Save scaled images as JPG
for i, img in enumerate(images):
    save_image_as_jpg(img, f"image_{i + 1}.jpg")

# Compare images using MSE and SSIM
for i in range(len(images)):
    for j in range(i + 1, len(images)):
        mse_value = mse(images[i], images[j])
        ssim_value = ssim(images[i], images[j], multichannel=True, win_size=3)
        print(f"Comparison between Image {i + 1} and Image {j + 1}: MSE={mse_value:.2f}, SSIM={ssim_value:.2f}")