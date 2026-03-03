
import style3dsim as sim

import warp as wp

import newton
import newton.examples
import newton.usd

from pxr import Usd,UsdGeom
from style3d.style3d_mini import style3d_mini
from pathlib import Path
from newton import Mesh
import numpy as np
from newton.solvers import SolverNotifyFlags

def _load_mesh_usd(usd_path,root_path) :

    usd_stage = Usd.Stage.Open(usd_path)
    prim = usd_stage.GetPrimAtPath(root_path)
    mesh = newton. usd. get_mesh(prim, load_uvs=True)
    indices = mesh.indices
    #points = mesh.vertices[:, [2, 0, 1]]  # y-up to z-up
    points = mesh.vertices  # y-up to z-up

    uv = None if mesh.uvs is None else mesh.uvs * 1e-3

    return indices, points, uv

def _load_scene_usd(usd_file_path) :
    builder = newton. ModelBuilder()
    builder. add_usd( usd_file_path, collapse_fixed_joints = False, enable_self_collisions = False )
    builder. add_ground_plane()

    # List to collect all prims with PhysicsClothAPI
    cloth_prims = []
    usd_stage = Usd.Stage.Open(usd_file_path)
    # Iterate over all prims in the stage
    for prim in usd_stage.Traverse():
        api_schemas = prim.GetMetadata("apiSchemas")
        if api_schemas is not None and "PhysicsClothAPI" in api_schemas.explicitItems:
            cloth_prims.append(prim)

    for prim in cloth_prims:
        print(f"-----------Prim with PhysicsClothAPI: {prim.GetPath()}")
        t, x, u = _load_mesh_usd( usd_file_path, prim.GetPath() )

        xformable = UsdGeom.Xformable(prim)
        # This returns the full transformation matrix (translation, rotation, scale)
        transform_matrix = xformable.GetLocalTransformation()
        # Get the translation component of the transformation matrix
        translation = transform_matrix.ExtractTranslation()

        builder. add_cloth_mesh(
            vertices = x,
            indices = t,
            rot = wp.quat_identity(),
            pos = wp.vec3( translation[0], translation[1], translation[2] ),
            vel = wp.vec3(0.0, 0.0, 0.0),
            density = 0.2,
            scale = 1
        )

    return builder

class Example:

    def __init__(self, viewer, args=None):
        self.fps = 100
        self.frame_dt = 1.0 / self.fps
        self.sim_time = 0.0
        self.sim_substeps = 1
        self.sim_dt = self.frame_dt / self.sim_substeps

        self.viewer = viewer
        self.args = args

        builder = _load_scene_usd ('push_cloth_zjrx/lefthand.usda')
        #builder = _load_scene_usd ('push_cloth_zjrx/allegro_left_hand_with_cube.usda')

        ## set joint targets and joint drive gains
        #for i in range(builder.joint_dof_count):
        #    builder.joint_target_ke[i] = 15
        #    builder.joint_target_kd[i] = 5
        #    builder.joint_target_pos[i] = 0.0

        # finalize model
        self. model = builder. finalize()

        self. state_0 = self. model. state()
        self. state_1 = self. model. state()
        self. control = self. model. control()

        self. solver =  style3d_mini. SolverStyle3dMini(self.model, njmax = 500 )

        # Create collision pipeline from command-line args (default: CollisionPipelineUnified with EXPLICIT)
        self. collision_pipeline = newton.examples. create_collision_pipeline(self.model, self.args)
        self. contacts = self. model.collide(self.state_0, collision_pipeline = self.collision_pipeline)

        self. viewer. set_model(self.model)

        self. capture()

        self.sim_frame=0
        self.palm_pos = np.array([0,0,0])

    def capture(self):
        if wp. get_device().is_cuda and False:
            with wp. ScopedCapture() as capture:
                self. simulate()
            self. graph = capture.graph
        else:
            self. graph = None

    def simulate(self):

        ###control
        drop_rate = 0.001
        advance_rate = 0.005
        hand_z_min = 0.3
        fi = self.sim_frame
        x = np.clip(advance_rate * fi, 0, 1.2)
        y = 0.5
        z = np.clip(0.3 - drop_rate * float(fi), hand_z_min, 1)

        joint_idx = 0

        joint_X_p_host = self.model.joint_X_p.numpy()
        #current_xform = joint_X_p_host[joint_idx]

        #if hasattr(current_xform, 'p'):
        #    pos = current_xform.p
        #    current_pos = np.array([float(pos[0]), float(pos[1]), float(pos[2])], dtype=np.float32)
        #    current_rot = current_xform.q
        #else:
        #    # 如果无法访问，使用初始位置和旋转
        #    initial_pos = np.array([x, y, z])
        #    current_pos = initial_pos.copy()
        #    current_rot = np.array([0.5, 0.5, 0.5, 0.5])

        #pos = current_xform.p
        #current_pos = np.array([float(pos[0]), float(pos[1]), float(pos[2])], dtype=np.float32)
        #current_rot = current_xform.q

        new_pos = np.array([x, y, z])
        #new_xform = wp.transform(new_pos, current_rot)
        joint_X_p_host[joint_idx,0:3] = new_pos
        joint_X_p_host[joint_idx,3:7] = np.array([0.5, 0.5, 0.5, 0.5]) # rotation

        self.model.joint_X_p.assign(joint_X_p_host)

        ##
        self.solver.notify_model_changed(SolverNotifyFlags.JOINT_PROPERTIES)

        for _ in range(self.sim_substeps):
            self. state_0.clear_forces()

            # apply forces to the model
            self. viewer.apply_forces(self.state_0)

            self. contacts = self.model.collide(self.state_0, collision_pipeline = self.collision_pipeline)
            self. solver. step(self.state_0, self.state_1, self.control, self.contacts, self.sim_dt)

            # swap states
            self.state_0, self.state_1 = self.state_1, self.state_0
        self.sim_frame += 1

    def step(self):
        if self.graph:
            wp.capture_launch(self.graph)
        else:
            self.simulate()

        self.sim_time += self.frame_dt

    def render(self):
        self.viewer.begin_frame(self.sim_time)
        self.viewer.log_state(self.state_0)
        self.viewer.log_contacts(self.contacts, self.state_0)
        self.viewer.end_frame()

if __name__ == "__main__":

    # Parse arguments and initialize viewer
    viewer, args = newton.examples.init()

    # Create viewer and run
    example = Example(viewer, args)

    newton.examples. run(example, args)
