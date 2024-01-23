from typing import Dict, List, Union

import brax
import jax
import jax.numpy as jnp
import pytest
from brax.v1 import jumpy as jp
from brax.v1.physics.base import vec_to_arr
from brax.v1.physics.config_pb2 import Joint

from qdax import environments


def get_joints_ordered_by_dof_pbd(sys: brax.System) -> List[Joint]:
    dict_joints: Dict[int, Dict[str, Union[List[Joint], List[int]]]] = {}

    # Getting joints ordered by dof
    dofs = {len(j.angle_limit) for j in sys.config.joints}
    # check to see if we should make all joints spherical
    sphericalize = len(dofs) > 1  # multiple joint types present
    sphericalize |= 2 in dofs  # ... or 2-dof joints present
    sphericalize &= sys.config.dynamics_mode == "pbd"  # only sphericalize if pbd

    for joint in sys.config.joints:
        dof = len(joint.angle_limit)
        if sys.config.dynamics_mode == "pbd":
            free_dofs = dof
            while sphericalize and dof < 3:
                joint.angle_limit.add()
                dof += 1
            if dof not in dict_joints:
                dict_joints[dof] = {"joint": [], "free_dofs": []}
            dict_joints[dof]["joint"].append(joint)
            dict_joints[dof]["free_dofs"].append(free_dofs)

    sorted_joints_dof = sorted(dict_joints.items(), key=lambda kv: kv[0])
    sorted_joints: List[Joint] = []
    for _, joints in sorted_joints_dof:
        sorted_joints.extend(joints["joint"])

    return sorted_joints


def get_joints_ordered_by_dof_legacy_spring(sys: brax.System) -> List[Joint]:
    dict_joints: Dict[int, List[Joint]] = {}
    for joint in sys.config.joints:
        dof = len(joint.angle_limit)
        springy = joint.stiffness > 0
        if springy:
            if dof not in dict_joints:
                dict_joints[dof] = []
            dict_joints[dof].append(joint)

    # ensure stable order for joint application: dof
    sorted_joints_dof = sorted(dict_joints.items(), key=lambda kv: kv[0])

    sorted_joints = []
    for _, joints in sorted_joints_dof:
        sorted_joints.extend(joints)

    return sorted_joints


def default_angle_sorted_by_dof(sys: brax.System, default_index: int = 0) -> jp.ndarray:
    """Returns the default joint angles for the system."""
    sorted_joints = get_joints_ordered_by_dof_pbd(
        sys
    ) + get_joints_ordered_by_dof_legacy_spring(sys)

    if not sorted_joints:
        return jp.array([])

    dofs = {}
    for j in sorted_joints:
        dofs[j.name] = sum(
            [angle.min != 0 or angle.max != 0 for angle in j.angle_limit]
        )
    angles = {}

    # check overrides in config defaults
    if default_index < len(sys.config.defaults):
        defaults = sys.config.defaults[default_index]
        for ja in defaults.angles:
            angles[ja.name] = vec_to_arr(ja.angle)[: dofs[ja.name]] * jp.pi / 180

    # set remaining joint angles set from angle limits, and add jitter
    for joint in sorted_joints:
        if joint.name not in angles:
            dof = dofs[joint.name]
            angles[joint.name] = jp.array(
                [
                    (angle_lim.min + angle_lim.max) * jp.pi / 360
                    for angle_lim in joint.angle_limit
                ][:dof]
            )

    return jp.concatenate([angles[j.name] for j in sorted_joints])


@pytest.mark.parametrize(
    "env_name",
    ["walker2d_uni", "ant_uni", "hopper_uni", "humanoid_uni", "halfcheetah_uni"],
)
def test_wrapper(env_name: str) -> None:
    """Test the wrapper running."""
    seed = 10

    # Init environment
    env = environments.create(env_name, fixed_init_state=True)
    print("Observation size: ", env.observation_size)
    print("Action size: ", env.action_size)

    random_key = jax.random.PRNGKey(seed)
    init_state = env.reset(random_key)

    joint_angle = jp.concatenate(
        [joint.angle_vel(init_state.qp)[0] for joint in env.sys.joints]
    )
    joint_vel = jp.concatenate(
        [joint.angle_vel(init_state.qp)[1] for joint in env.sys.joints]
    )

    default_angle = default_angle_sorted_by_dof(env.sys)

    # check position and velocity
    pytest.assume(jnp.array_equal(joint_angle, default_angle))
    pytest.assume(jnp.array_equal(joint_vel, jp.zeros((env.sys.num_joint_dof,))))
