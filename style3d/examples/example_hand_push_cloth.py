
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


def _quat_mul(q1, q2):
    w1, x1, y1, z1 = q1[3],q1[0],q1[1],q1[2]
    w2, x2, y2, z2 = q2[3],q2[0],q2[1],q2[2]

    return np.array([
        w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2,
        w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2,
        w1 * y2 - x1 * z2 + y1 * w2 + z1 * x2,
        w1 * z2 + x1 * y2 - y1 * x2 + z1 * w2
    ])


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

        ## set joint targets and joint drive gains
        for i in range(builder.joint_dof_count):
            builder.joint_target_ke[i] = 1
            builder.joint_target_kd[i] = 0.5
            builder.joint_target_pos[i] = 0.08

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

        self.sim_frame = 0


    def _control_hand_move(self):
        drop_rate = 0.002
        advance_rate = 0.002
        hand_z_min = 0.247
        fi = self.sim_frame
        x = np.clip(advance_rate * fi, 0, 1.2)
        y = 0.5
        z = np.clip(0.3 - drop_rate * float(fi), hand_z_min, 1)

        joint_idx = 0
        joint_X_p_host = self.model.joint_X_p.numpy()
        new_pos = np.array([x, y, z])
        joint_X_p_host[joint_idx,0:3] = new_pos
        #joint_X_p_host[joint_idx,3:7] = np.array([0.5, 0.5, 0.5, 0.5]) # rotation
        q0 = np.array([0.5, 0.5, 0.5, 0.5])
        q1 = np.array([ 0, -np.sin(np.pi/8), 0, np.cos(np.pi/8)])
        qnew = _quat_mul(q0,q1)
        #qnew = q0
        joint_X_p_host[joint_idx,3:7] = qnew # rotation
        self.model.joint_X_p.assign(joint_X_p_host)

        ##
        self.solver.notify_model_changed(SolverNotifyFlags.JOINT_PROPERTIES)

    def capture(self):
        if wp. get_device().is_cuda and False:
            with wp. ScopedCapture() as capture:
                self. simulate()
            self. graph = capture.graph
        else:
            self. graph = None

    def simulate(self):

        ###control
        self._control_hand_move()

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
