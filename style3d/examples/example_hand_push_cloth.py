
import style3dsim as sim

import warp as wp

import newton
import newton.examples
import newton.usd

from pxr import Usd



def build_demo_pendulum():
    builder = newton.ModelBuilder()

    # builder.add_usd(
    #    newton.examples.get_asset("cartpole.usda"),
    #    collapse_fixed_joints=False,
    #    enable_self_collisions=False,
    # )

    builder.add_ground_plane()

    # common geometry settings
    cuboid_hx = 0.1
    cuboid_hy = 0.1
    cuboid_hz = 0.75
    upper_hz = 0.25 * cuboid_hz

    # layout positions (y-rows)
    rows = [-3.0, 0.0, 3.0]
    drop_z = 2.0

    # -----------------------------
    # REVOLUTE (hinge) joint demo
    # -----------------------------
    y = rows[0]

    a_rev = builder.add_link(xform=wp.transform(p=wp.vec3(0.0, y, drop_z + upper_hz), q=wp.quat_identity()))
    b_rev = builder.add_link(
        xform=wp.transform(
            p=wp.vec3(0.0, y, drop_z - cuboid_hz), q=wp.quat_from_axis_angle(wp.vec3(1.0, 0.0, 0.0), 0.15)
        ),
        key="b_rev",
    )
    builder.add_shape_box(a_rev, hx=cuboid_hx, hy=cuboid_hy, hz=upper_hz)
    builder.add_shape_box(b_rev, hx=cuboid_hx, hy=cuboid_hy, hz=cuboid_hz)

    j_fixed_rev = builder.add_joint_fixed(
        parent=-1,
        child=a_rev,
        parent_xform=wp.transform(p=wp.vec3(0.0, y, drop_z + upper_hz), q=wp.quat_identity()),
        child_xform=wp.transform(p=wp.vec3(0.0, 0.0, 0.0), q=wp.quat_identity()),
        key="fixed_revolute_anchor",
    )

    j_revolute = builder.add_joint_revolute(
        parent=a_rev,
        child=b_rev,
        axis=wp.vec3(1.0, 0.0, 0.0),
        parent_xform=wp.transform(p=wp.vec3(0.0, 0.0, -upper_hz), q=wp.quat_identity()),
        child_xform=wp.transform(p=wp.vec3(0.0, 0.0, +cuboid_hz), q=wp.quat_identity()),
        key="revolute_a_b",
    )

    # Create articulation from joints
    builder.add_articulation([j_fixed_rev, j_revolute], key="revolute_articulation")

    # set initial joint angle
    builder.joint_q[-1] = wp.pi * 0.5

    return builder



def build_demo_shapes() :
    builder = newton.ModelBuilder()

    # add ground plane
    builder.add_ground_plane()

    # z height to drop shapes from
    drop_z = 2.0

    # SPHERE
    body_sphere = builder.add_body(xform=wp.transform(p=wp.vec3(0.0, -2.0, drop_z), q=wp.quat_identity()), key="sphere")
    builder.add_shape_sphere(body_sphere, radius=0.5)

    # MESH (bunny)
    usd_stage = Usd.Stage.Open(newton.examples.get_asset("bunny.usd"))
    demo_mesh = newton.usd.get_mesh(usd_stage.GetPrimAtPath("/root/bunny"))

    #usd_stage = Usd.Stage.Open('./push_cloth/table.usda')
    #demo_mesh = newton.usd.get_mesh(usd_stage.GetPrimAtPath("/table_001/table_001"))

    body_mesh = builder.add_body(xform=wp.transform(p=wp.vec3(0.0, 4.0, drop_z - 0.5), q=wp.quat(0.5, 0.5, 0.5, 0.5)), key="mesh")
    builder.add_shape_mesh(body_mesh, mesh=demo_mesh)

    # Color rigid bodies for VBD solver
    builder.color()

    return builder


def build_demo_usd() :
    builder = newton.ModelBuilder()
    builder.add_usd(
        #'push_cloth/aaa.usda',
        'push_cloth/humanoid.usda',
        collapse_fixed_joints=False,
        enable_self_collisions=False,
    )
    builder.add_ground_plane()
    return builder

class Example:

    def __init__(self, viewer, args=None):
        self.fps = 100
        self.frame_dt = 1.0 / self.fps
        self.sim_time = 0.0
        self.sim_substeps = 10
        self.sim_dt = self.frame_dt / self.sim_substeps

        self.viewer = viewer
        self.args = args

        #builder = build_demo_pendulum()
        #builder = build_demo_shapes()
        builder = build_demo_usd()

        # finalize model
        self.model = builder.finalize()


        self.state_0 = self.model.state()
        self.state_1 = self.model.state()
        self.control = self.model.control()

        self.solver = newton.solvers.SolverMuJoCo(self.model,njmax=500)

        #self.solver = newton.solvers.SolverXPBD(self.model)
        ## not required for MuJoCo, but required for other solvers
        #newton.eval_fk(self.model, self.model.joint_q, self.model.joint_qd, self.state_0)

        # Create collision pipeline from command-line args (default: CollisionPipelineUnified with EXPLICIT)
        self.collision_pipeline = newton.examples.create_collision_pipeline(self.model, self.args)
        self.contacts = self.model.collide(self.state_0, collision_pipeline = self.collision_pipeline)

        self.viewer.set_model(self.model)

        self.capture()

    def capture(self):
        if wp.get_device().is_cuda:
            with wp.ScopedCapture() as capture:
                self.simulate()
            self.graph = capture.graph
        else:
            self.graph = None

    def simulate(self):
        for _ in range(self.sim_substeps):
            self.state_0.clear_forces()

            # apply forces to the model
            self.viewer.apply_forces(self.state_0)

            self.contacts = self.model.collide(self.state_0, collision_pipeline = self.collision_pipeline)
            self.solver.step(self.state_0, self.state_1, self.control, self.contacts, self.sim_dt)

            # swap states
            self.state_0, self.state_1 = self.state_1, self.state_0

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
