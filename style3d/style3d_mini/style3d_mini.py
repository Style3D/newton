
from typing_extensions import override
import newton
from newton import Contacts, Control, State
import warp as wp
from pxr import Usd,UsdGeom
import style3dsim as sim
import numpy as np
import json
import os
from pathlib import Path

def _log_callback(file_name: str, func_name: str, line: int, level: sim.LogLevel, message: str):
    if level == sim.LogLevel.INFO:
        print("[info]: ", message)
    elif level == sim.LogLevel.ERROR:
        print("[error]: ", message)
    elif level == sim.LogLevel.WARNING:
        print("[warning]: ", message)
    elif level == sim.LogLevel.DEBUG:
        print("[debug]: ", message)

def _log_in_simulation(**kwargs):

    name = ''

    if not sim.is_login():
        login_file = None
        if 'login_file' in kwargs:
            login_file = kwargs['login_file']

        if login_file and os.path.exists(login_file):
            with open(login_file,'r') as f:
                login=json.load(f)
                name = login['name']
                pass_word = login['pass_word']
        else:
            name = input('Enter your name : ')
            pass_word = input('Enter your password : ')

        sim.login(name, pass_word, True, None)

    if sim.is_login():
        print(f'login successful {name}')
    else:
        print('login failed')

def _get_a_sim_world():

    password_dir = Path(__file__).parent.resolve()
    _log_in_simulation( login_file= password_dir / '..' / 'simulation_login.json' )

    sim.set_log_callback(_log_callback)

    world = sim.World()
    world_attrib = sim.WorldAttrib()
    world_attrib.enable_gpu = True
    world_attrib.gravity = sim.Vec3f(0, 0, -9.8)
    world_attrib.ground_direction = sim.Vec3f(0., 0., 1.)
    world_attrib.ground_height = 2e-1
    world_attrib.ground_static_friction = 1.0
    world_attrib.ground_dynamic_friction = 0.9
    world_attrib.time_step = 1e-3
    world_attrib.enable_rigid_self_collision = False
    world_attrib.enable_collision_force_map_rigidbody_piece = True
    world_attrib.enable_plastic_bending = True
    world_attrib.enable_volume_conserve = True
    world.set_attrib(world_attrib)

    print( f'time step {world_attrib.time_step}' )
    print( f'gravity {world_attrib.gravity.x}, {world_attrib.gravity.y}, {world_attrib.gravity.z} ' )

    return world


def _to_sim_transform(trans,scale_model_2_sim):
    translation = sim.Vec3f(trans[0]*scale_model_2_sim, trans[1]*scale_model_2_sim, trans[2]*scale_model_2_sim)
    rotation = sim.Quat(trans[3], trans[4], trans[5], trans[6])
    scaling = sim.Vec3f(1.0, 1.0, 1.0)
    #return translation, rotation, scaling
    return sim.Transform(translation, rotation, scaling)


class SolverStyle3dMini(newton.solvers.SolverBase):

    def __init__(self, model: newton.Model, **kwargs):

        if 'njmax' in kwargs:
            njmax = kwargs['njmax']
        else:
            njmax = 100

        if 'scale' in kwargs:
            self.scale_model_2_sim = kwargs['scale']
        else:            
            self.scale_model_2_sim = 1.0    

        self.rigid_solver = newton.solvers.SolverMuJoCo(model, njmax = njmax)

        self.model = model

        self.world = _get_a_sim_world()

        ### add_cloth_to_simulation
        self._add_cloth_to_simulation(self.scale_model_2_sim)

        ### add_rigid_body to simulation
        self._add_rigid_body_to_simulation(self.scale_model_2_sim)


    def _add_cloth_to_simulation(self, scale):
        # TODO: handle multiple cloth
        x, t = self. _extract_cloth_mesh()

        if len(t) > 0:

            self.cloth = sim. Cloth(t, x * scale, np.array([], dtype=float), False)

            cloth_attrib = sim. ClothAttrib()

            cloth_attrib.bend_stiff = sim.Vec3f(1e-7,1e-7,1e-7)    

            self.cloth.set_attrib(cloth_attrib)

            self.cloth.attach(self.world)

        else:
            self.cloth = None


    def _add_rigid_body_to_simulation(self, scale):
        shape_geo_src = self.model.shape_source
        shape_geo_type = self.model.shape_type.numpy()
        shape_geo_scale = self.model.shape_scale.numpy()
        #shape_geo_thickness = self.model.shape_thickness.numpy()
        shape_geo_is_solid = self.model.shape_is_solid.numpy()
        shape_transform = self.model.shape_transform.numpy()
        shape_transform_q = self.model.body_q.numpy()
        shape_flags = self.model.shape_flags.numpy()

        shape_2_body_index = self.model.shape_body.numpy() # 

        self.sim_rigid_bodies = [] # simulation rigid bodies
        self.sim_2_body_index = []

        for si, mi in enumerate(shape_2_body_index):

            flag_str = []
            if shape_flags[si] & newton.ShapeFlags.COLLIDE_SHAPES:
                flag_str.append('COLLIDE_SHAPES')
            if shape_flags[si] & newton.ShapeFlags.VISIBLE:
                flag_str.append('VISIBLE')
            if shape_flags[si] & newton.ShapeFlags.COLLIDE_PARTICLES:
                flag_str.append('COLLIDE_PARTICLES')
            if shape_flags[si] & newton.ShapeFlags.SITE:
                flag_str.append('SITE')
            if shape_flags[si] & newton.ShapeFlags.HYDROELASTIC:
                flag_str.append('HYDROELASTIC')

            print(f"shape {si} rigid {mi}, flags: {flag_str}")

            if 'COLLIDE_PARTICLES' not in flag_str:
                continue


            shape_source_i = shape_geo_src[si]
            shape_type_i = shape_geo_type[si]
            transform_i = shape_transform[si]

            trans = transform_i
            #translation, rotation, scaling = to_sim_transform(trans)
            #transform = sim.Transform(translation, rotation, scaling)
            transform = _to_sim_transform(trans,self.scale_model_2_sim)

            shape_type_str =''
            if shape_type_i == newton.GeoType.MESH:
                mesh = sim.Mesh(shape_source_i.indices, shape_source_i.vertices * scale)
                rigid_body = sim.RigidBody(mesh, transform)
                shape_type_str = 'MESH' 
            elif shape_type_i == newton.GeoType.SPHERE:
                sphereSize = sim.SphereSize()
                sphereSize.radius = shape_geo_scale[si] * scale 
                rigid_body = sim.RigidBody(sphereSize, transform)
                shape_type_str = 'SPHERE' 
            elif shape_type_i == newton.GeoType.BOX:

                # TODO: get geo size some where
                # s = geo_size[geom_id]
                #shape_geo_scale[si]
                s = shape_geo_scale[si]

                boxSize = sim.BoxSize()
                boxSize.length_x = 2 * s[0] * scale 
                boxSize.length_y = 2 * s[1] * scale 
                boxSize.length_z = 2 * s[2] * scale 
                rigid_body = sim.RigidBody(boxSize, transform)
                shape_type_str = 'BOX' 
            elif shape_type_i == newton.GeoType.CYLINDER:
                cylinderSize = sim.CylinderSize()
                rigid_body = sim.RigidBody(cylinderSize, transform)
                shape_type_str = 'CYLINDER' 
            else:
                print('unknown geometry type!')
                continue

            rigid_body_attrib = sim.RigidBodyAttrib()
            rigid_body.set_attrib(rigid_body_attrib)

            rigid_body.set_pin(True)
            # rigid_body.set_collision_group(contype[geom_id])
            # rigid_body.set_collision_mask(conaffinity[geom_id])

            rigid_body.attach(self.world)

            self.sim_rigid_bodies.append(rigid_body)
            self.sim_2_body_index.append(mi)
            print(f"add shape {si} {shape_type_str} to simulation , rigid body {len(self.sim_rigid_bodies)-1}, model body {mi}")


    def _quaternion_to_matrix(self,q):
        w, x, y, z = q.w, q.x, q.y, q.z

        # Compute the rotation matrix components
        R = np.array([
            [1 - 2 * (y ** 2 + z ** 2), 2 * (x * y - w * z), 2 * (x * z + w * y)],
            [2 * (x * y + w * z), 1 - 2 * (x ** 2 + z ** 2), 2 * (y * z - w * x)],
            [2 * (x * z - w * y), 2 * (y * z + w * x), 1 - 2 * (x ** 2 + y ** 2)]
        ])

        return R

    def _extract_cloth_mesh(self):

        x = self.model. particle_q.numpy()
        t = self.model. tri_indices.numpy()

        return x, t

    def _update_rigidbody_cloth_collision_force(self ):
        self.collision_force = []
        for sim_ri, model_ri in enumerate(self.sim_2_body_index):
            if model_ri < 0: # save some time
                self.collision_force.append([])
            else:
                rb = self.sim_rigid_bodies[sim_ri]
                f_rb = rb.get_collision_force_piece()
                self.collision_force.append(f_rb)

    def _apply_collision_force_to_rigidbody(self,state_in: State):

        trans_in = state_in.body_q.numpy()
        body_f_np = state_in.body_f.numpy()

        for sim_ri, model_ri in enumerate(self.sim_2_body_index):

            if model_ri < 0:
                continue

            if len(self.collision_force) <= 0:
                continue

            rb_force = self.collision_force[sim_ri]

            for f, bary in zip(*rb_force):  # force and bary

                trans_0 = trans_in[model_ri]
                begin_trans = _to_sim_transform(trans_0, self.scale_model_2_sim)

                orientation = self._quaternion_to_matrix( begin_trans.rotation )

                orientation = orientation.reshape(3, 3)
                r = orientation @ bary
                torque = np.cross(r, f)

                body_f_np[model_ri] += [f[0], f[1], f[2], torque[0], torque[1], torque[2]]

            state_in.body_f.assign(body_f_np)

    def _update_rigidbody_pos_to_simulation(self,state_in: State, state_out: State):
        trans_in = state_in.body_q.numpy()
        trans_out = state_out.body_q.numpy()

        for sim_ri, model_ri in enumerate(self.sim_2_body_index):

            if model_ri < 0:
                continue

            trans_0 = trans_in[model_ri]
            trans_1 = trans_out[model_ri]
            begin_trans = _to_sim_transform(trans_0, self.scale_model_2_sim)
            end_trans = _to_sim_transform(trans_1, self.scale_model_2_sim)
            self.sim_rigid_bodies[sim_ri].move(begin_trans, end_trans)


    def _update_cloth_pos_to_state(self, state_out: State):
        if self.cloth is not None:
            cloth_x = self.cloth.get_positions()
            state_out.particle_q.assign( cloth_x/self.scale_model_2_sim )

    @override
    def step(self, state_in: State, state_out: State, control: Control, contacts: Contacts, dt: float):

        ## apply collision force
        #self._update_rigidbody_cloth_collision_force()
        #self._apply_collision_force_to_rigidbody( state_in)

        self.rigid_solver.step(state_in, state_out, control, contacts, dt)

        #simulation step
        self.world.step_sim()

        #set new rigid body position to simulation
        self.world.fetch_sim(0)

        self._update_cloth_pos_to_state(state_out)

        self._update_rigidbody_pos_to_simulation( state_in, state_out)


    def rebuild_bvh(self, state: State):
        pass