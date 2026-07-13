import time
import numpy as np
import zed_grabber_cpp as zcpp
import cv2
import os
import open3d as o3d  

# Create directory for your GraspNet test data
save_dir = "my_zed_data"
os.makedirs(save_dir, exist_ok=True)

grabber = zcpp.ZedGrabber(
    views=["LEFT_BGR", "DEPTH_MEASURE"],
    camera_fps=60,
    enable_tracking=False,  # Tracking not needed since we aren't fusing frames
    depth_mode="NEURAL"
)
grabber.start()
time.sleep(0.5)  

intr_dict = grabber.get_intrinsics()
fx, fy = intr_dict['fx'], intr_dict['fy']
cx, cy = intr_dict['cx'], intr_dict['cy']

print("\n--- COPY THESE INTRINSICS FOR GRASPNET ---")
print(f"fx: {fx}, fy: {fy}")
print(f"cx: {cx}, cy: {cy}")
print(f"Intrinsic Matrix: np.array([[{fx}, 0.0, {cx}], [0.0, {fy}, {cy}], [0.0, 0.0, 1.0]])\n")

# 3. Define Crop boundaries & Thresholds (Assuming C++ wrapper returns meters)
X_MIN, X_MAX = 400, 880  
Y_MIN, Y_MAX = 100, 620
MIN_DEPTH_M = 0.20   # 20cm in meters
MAX_DEPTH_M = 1.00   # 100cm in meters

print("Showing live preview. Press 'q' to stop or wait for capture...")

# Run a brief loop to ensure we grab a fresh, stable frame
for i in range(30):
    _, rgb = grabber.get_latest("LEFT_BGR")
    _, depth_raw = grabber.get_latest("DEPTH_MEASURE")

    if rgb is None or depth_raw is None:
        continue

    # Clean up NaN / Inf values
    depth_float = np.nan_to_num(depth_raw, nan=0.0, posinf=0.0, neginf=0.0)

    # Apply Spatial ROI Crop
    cropped_depth = np.zeros_like(depth_float)
    cropped_depth[Y_MIN:Y_MAX, X_MIN:X_MAX] = depth_float[Y_MIN:Y_MAX, X_MIN:X_MAX]
    
    # Filter distance thresholds
    cropped_depth[(cropped_depth < MIN_DEPTH_M) | (cropped_depth > MAX_DEPTH_M)] = 0.0

    # --- LIVE VISUALIZATION ---
    # Normalize depth map to 0-255 so OpenCV can render it cleanly
    depth_visual = cv2.normalize(cropped_depth, None, 0, 255, cv2.NORM_MINMAX, dtype=cv2.CV_8U)
    depth_colored = cv2.applyColorMap(depth_visual, cv2.COLORMAP_JET)
    
    cv2.imshow("ZED Cropped Depth (Filtered)", depth_colored)
    cv2.imshow("ZED Live RGB", rgb[:, :, :3])
    
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# 4. Save Final Frames for GraspNet (Converting units to Millimeters)
# Convert the final frame's depth from meters to uint16 millimeters
depth_uint16 = np.round(cropped_depth * 1000.0).astype(np.uint16)

# Strip alpha channel if present to keep it standard BGR
color_img = rgb[:, :, :3] if rgb.shape[-1] == 4 else rgb

# Write outputs to disk
cv2.imwrite(os.path.join(save_dir, "color.png"), color_img)
cv2.imwrite(os.path.join(save_dir, "depth.png"), depth_uint16)
print(f"\nSaved GraspNet frame data to: '{save_dir}/'")

# 5. Create and save the workspace mask
mask = np.ones(color_img.shape[:2], dtype=np.uint8) * 255
cv2.imwrite(os.path.join(save_dir, "workspace_mask.png"), mask)
print("Saved workspace_mask.png!")

# --- 6. ADDED: POINT CLOUD GENERATION & FILTERING ---
print("\nGenerating 3D Point Cloud...")
height, width = color_img.shape[:2]

# Convert BGR to RGB because Open3D expects standard RGB ordering
rgb_rgb = cv2.cvtColor(color_img, cv2.COLOR_BGR2RGB)

# Convert arrays to Open3D Image objects (Using float32 meters for depth)
o3d_color = o3d.geometry.Image(rgb_rgb)
o3d_depth = o3d.geometry.Image(cropped_depth.astype(np.float32))

# Construct an RGBD Image 
rgbd_image = o3d.geometry.RGBDImage.create_from_color_and_depth(
    o3d_color, 
    o3d_depth, 
    depth_scale=1.0,           # Keep it at 1.0 since data is already in meters
    depth_trunc=MAX_DEPTH_M,   # Clip any accidental values beyond our threshold
    convert_rgb_to_intensity=False
)

# Define Camera Intrinsics using the exact parameters from the ZED SDK
intrinsics = o3d.camera.PinholeCameraIntrinsic(width, height, fx, fy, cx, cy)

# Project to 3D Point Cloud
pcd = o3d.geometry.PointCloud.create_from_rgbd_image(rgbd_image, intrinsics)

# --- ANTI-EDGE-OBSESSION FILTERING ---

# 1. Voxel Downsample: Clean up overlapping or overly dense areas
pcd = pcd.voxel_down_sample(voxel_size=0.004) # 4mm grid

# 2. Statistical Outlier Removal (SOR): Eliminate the floating edge spray
pcd, inlier_indices = pcd.remove_statistical_outlier(nb_neighbors=25, std_ratio=1.5)

# 3. RANSAC Table Plane Removal (NEW)
print("Running RANSAC plane segmentation to remove table...")
# distance_threshold: Max distance a point can be from the estimated plane to be considered an inlier (10mm here)
# ransac_n: Number of sampled points to estimate a plane
# num_iterations: How many times to randomly sample
plane_model, inliers = pcd.segment_plane(distance_threshold=0.010,
                                         ransac_n=3,
                                         num_iterations=200)

[a, b, c, d] = plane_model
print(f"Detected Table Plane Equation: {a:.2f}x + {b:.2f}y + {c:.2f}z + {d:.2f} = 0")

# Invert the selection to keep EVERYTHING EXCEPT the table plane
pcd = pcd.select_by_index(inliers, invert=True)

# 4. Optional Post-RANSAC Clean: Remove tiny floating point artifacts left over from the table slice
pcd, _ = pcd.remove_statistical_outlier(nb_neighbors=10, std_ratio=2.0)

# Save the final processed cloud to disk
cloud_path = os.path.join(save_dir, "workspace_cloud.ply")
o3d.io.write_point_cloud(cloud_path, pcd)
print(f"Success! Table removed and cloud saved to: '{cloud_path}' ({len(pcd.points)} points)")
# --- CLEANUP ---
grabber.stop()
cv2.destroyAllWindows()