import bpy, sys, mathutils

scene = bpy.data.scenes["Scene"]

# Set render resolution.
scene.render.resolution_x = int(sys.argv[-8])
scene.render.resolution_y = int(sys.argv[-7])

# Create a new camera.
new_camera = bpy.data.cameras.new("DCR Camera")
new_camera_obj = bpy.data.objects.new("DCR Camera", new_camera)
bpy.context.scene.camera = new_camera_obj

# Set camera FOV.
fov = 50.0
pi = 3.14159265
new_camera_obj.data.angle = fov * (pi / 180.0)

# Set camera rotation.
x = float(sys.argv[-6])
y = float(sys.argv[-5])
z = float(sys.argv[-4])
new_camera_obj.rotation_mode = "XYZ"
new_camera_obj.rotation_euler = (x, y, z)

# Set camera translation.
tx = float(sys.argv[-3])
ty = float(sys.argv[-2])
tz = float(sys.argv[-1])
new_camera_obj.location.x = tx
new_camera_obj.location.y = ty
new_camera_obj.location.z = tz

bpy.context.scene.render.filepath = sys.argv[-9]
bpy.ops.render.render(write_still=True)

# This is copied directly from https://blender.stackexchange.com/questions/38009/3x4-camera-matrix-from-blender-camera.
def get_calibration_matrix_K_from_blender(camd):
    f_in_mm = camd.lens
    scene = bpy.context.scene
    resolution_x_in_px = scene.render.resolution_x
    resolution_y_in_px = scene.render.resolution_y
    scale = scene.render.resolution_percentage / 100
    sensor_width_in_mm = camd.sensor_width
    sensor_height_in_mm = camd.sensor_height
    pixel_aspect_ratio = scene.render.pixel_aspect_x / scene.render.pixel_aspect_y
    if (camd.sensor_fit == 'VERTICAL'):
        # the sensor height is fixed (sensor fit is horizontal), 
        # the sensor width is effectively changed with the pixel aspect ratio
        s_u = resolution_x_in_px * scale / sensor_width_in_mm / pixel_aspect_ratio 
        s_v = resolution_y_in_px * scale / sensor_height_in_mm
    else: # 'HORIZONTAL' and 'AUTO'
        # the sensor width is fixed (sensor fit is horizontal), 
        # the sensor height is effectively changed with the pixel aspect ratio
        pixel_aspect_ratio = scene.render.pixel_aspect_x / scene.render.pixel_aspect_y
        s_u = resolution_x_in_px * scale / sensor_width_in_mm
        s_v = resolution_y_in_px * scale * pixel_aspect_ratio / sensor_height_in_mm
    

    # Parameters of intrinsic calibration matrix K
    alpha_u = f_in_mm * s_u
    alpha_v = f_in_mm * s_v
    u_0 = resolution_x_in_px * scale / 2
    v_0 = resolution_y_in_px * scale / 2
    skew = 0 # only use rectangular pixels

    K = mathutils.Matrix(
        ((alpha_u, skew,    u_0),
        (    0  , alpha_v, v_0),
        (    0  , 0,        1 )))
    return K

K = get_calibration_matrix_K_from_blender(new_camera)

print("K MATRIX INCOMING:")
for i in range(3):
    for j in range(3):
        print(f"{K[i][j]} ")

