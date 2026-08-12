"""Standard LIBERO / robosuite adapter for deterministic physical branches."""

from __future__ import annotations

import copy
import json
import os
import xml.etree.ElementTree as ET

import h5py
import numpy as np

from . import config


POSE_NAMES = ("x", "y", "z", "r1", "r2", "r3")


def _rot6(matrix):
    matrix = np.asarray(matrix, dtype=np.float64).reshape(3, 3)
    return matrix[:, :2].reshape(-1, order="F")


def consequence_feature_names():
    names = []
    for prefix in ("primary", "tcp", "tcp_to_primary", "primary_to_target"):
        names.extend("%s_pos_%s" % (prefix, axis) for axis in "xyz")
        names.extend("%s_rot6_%d" % (prefix, i) for i in range(6))
    names.extend(
        [
            "gripper_width",
            "articulated_qpos_0",
            "articulated_qpos_1",
            "articulated_qpos_2",
            "task_progress",
            "contact_force_gripper_primary",
            "contact_force_primary_target",
            "contact_force_gripper_target",
            "max_penetration",
            "joint_limit_violation",
        ]
    )
    return tuple(names)


FEATURE_NAMES = consequence_feature_names()


def _simple_clone(value):
    if isinstance(value, np.ndarray):
        return value.copy()
    if isinstance(value, (str, int, float, bool, type(None), np.generic)):
        return copy.copy(value)
    if isinstance(value, tuple):
        cloned = [_simple_clone(item) for item in value]
        return tuple(cloned) if all(item is not None for item in cloned) else None
    if isinstance(value, list):
        cloned = [_simple_clone(item) for item in value]
        return cloned if all(item is not None for item in cloned) else None
    if isinstance(value, dict):
        result = {}
        for key, item in value.items():
            cloned = _simple_clone(item)
            if cloned is None and item is not None:
                return None
            result[key] = cloned
        return result
    return None


def _capture_simple_attributes(instance):
    result = {}
    for name, value in instance.__dict__.items():
        cloned = _simple_clone(value)
        if cloned is not None or value is None:
            result[name] = cloned
    return result


def _restore_simple_attributes(instance, values):
    for name, value in values.items():
        if hasattr(instance, name):
            setattr(instance, name, _simple_clone(value))


SIMULATOR_AUXILIARY_ARRAYS = (
    "ctrl",
    "qacc_warmstart",
    "qfrc_applied",
    "xfrc_applied",
    "mocap_pos",
    "mocap_quat",
    "userdata",
)


def _capture_simulator_auxiliary(data):
    result = {}
    for name in SIMULATOR_AUXILIARY_ARRAYS:
        if hasattr(data, name):
            result[name] = np.asarray(getattr(data, name), dtype=np.float64).copy()
    return result


def _restore_simulator_auxiliary(data, values, include_warmstart=True):
    for name, value in values.items():
        if name == "qacc_warmstart" and not include_warmstart:
            continue
        if hasattr(data, name):
            target = getattr(data, name)
            target[...] = value


class LiberoTaskRuntime(object):
    """One simulator instance for one task, reused across deterministic branches."""

    def __init__(self, task, libero_source, dataset_root):
        # Imports are intentionally delayed so pure-analysis processes do not need
        # a working OpenGL backend.
        import robosuite
        from libero.libero.envs.env_wrapper import ControlEnv

        self.task = dict(task)
        self.libero_source = os.path.abspath(libero_source)
        self.package_root = os.path.join(self.libero_source, "libero", "libero")
        self.dataset_root = os.path.abspath(dataset_root)
        self.robosuite_root = os.path.dirname(os.path.abspath(robosuite.__file__))
        self.bddl_path = os.path.join(
            self.package_root,
            "bddl_files",
            config.SUITE,
            self.task["task_name"] + ".bddl",
        )
        self.demo_path = os.path.join(
            self.dataset_root,
            config.SUITE,
            self.task["task_name"] + "_demo.hdf5",
        )
        if not os.path.isfile(self.bddl_path):
            raise FileNotFoundError(self.bddl_path)
        if not os.path.isfile(self.demo_path):
            raise FileNotFoundError(self.demo_path)
        self.env = ControlEnv(
            bddl_file_name=self.bddl_path,
            robots=["Panda"],
            controller=config.CONTROL_MODE,
            use_camera_obs=False,
            has_renderer=False,
            has_offscreen_renderer=False,
            ignore_done=True,
            reward_shaping=True,
            control_freq=config.CONTROL_FREQUENCY_HZ,
            camera_names=[],
            horizon=1000,
        )
        self.env.seed(config.GLOBAL_SEED)
        self.env.reset()
        self.current_episode = None
        self._geom_sets = None

    def close(self):
        if self.env is not None:
            self.env.close()
            self.env = None

    def load_episode(self, episode_id):
        key = "demo_%d" % int(episode_id)
        with h5py.File(self.demo_path, "r") as handle:
            group = handle["data"][key]
            result = {
                "episode_id": int(episode_id),
                "actions": np.asarray(group["actions"], dtype=np.float64),
                "states": np.asarray(group["states"], dtype=np.float64),
                "rewards": np.asarray(group["rewards"], dtype=np.uint8),
                "dones": np.asarray(group["dones"], dtype=np.uint8),
                "model_xml": self.patch_model_xml(group.attrs["model_file"]),
                "num_samples": int(group.attrs.get("num_samples", len(group["actions"]))),
            }
        return result

    def dataset_metadata(self):
        with h5py.File(self.demo_path, "r") as handle:
            data = handle["data"]
            env_args = json.loads(data.attrs["env_args"])
            lengths = [len(data["demo_%d" % i]["actions"]) for i in range(int(data.attrs["num_demos"]))]
            successes = [int(np.max(data["demo_%d" % i]["rewards"])) for i in range(int(data.attrs["num_demos"]))]
            return {
                "demo_path": self.demo_path,
                "num_demos": int(data.attrs["num_demos"]),
                "successful_demos": int(np.sum(np.asarray(successes) > 0)),
                "length_min": int(np.min(lengths)),
                "length_median": float(np.median(lengths)),
                "length_max": int(np.max(lengths)),
                "env_args": env_args,
                "tag": str(data.attrs.get("tag", "")),
                "problem_info": json.loads(data.attrs["problem_info"]),
            }

    def patch_model_xml(self, xml_string):
        if isinstance(xml_string, bytes):
            xml_string = xml_string.decode("utf-8")
        tree = ET.fromstring(xml_string)
        asset = tree.find("asset")
        elements = list(asset.findall("mesh")) + list(asset.findall("texture"))
        for element in elements:
            old = element.get("file")
            if not old:
                continue
            pieces = old.replace("\\", "/").split("/")
            if "robosuite" in pieces:
                index = max(i for i, value in enumerate(pieces) if value == "robosuite")
                element.set("file", os.path.join(self.robosuite_root, *pieces[index + 1 :]))
            elif "chiliocosm" in pieces:
                index = max(i for i, value in enumerate(pieces) if value == "chiliocosm")
                element.set("file", os.path.join(self.package_root, *pieces[index + 1 :]))
        return ET.tostring(tree, encoding="unicode")

    def initialize_episode_model(self, episode):
        self.env.reset_from_xml_string(episode["model_xml"])
        self.env.sim.reset()
        self.env.sim.set_state_from_flattened(episode["states"][0])
        self.env.sim.forward()
        self.env._post_process()
        self.current_episode = int(episode["episode_id"])
        self._geom_sets = self._build_geom_sets()

    def capture_snapshot(self):
        controller = self.env.robots[0].controller
        return {
            "sim_state": self.env.sim.get_state().flatten().copy(),
            # MjSimState omits applied controls, solver warm starts, and mocap
            # state. Those omissions are enough to make the first restored
            # branch differ from later branches even in a deterministic model.
            "sim_auxiliary": _capture_simulator_auxiliary(self.env.sim.data),
            "controller": _capture_simple_attributes(controller),
            "gripper": _capture_simple_attributes(self.env.robots[0].gripper),
            "timestep": int(getattr(self.env.env, "timestep", 0)),
            "cur_time": float(getattr(self.env.env, "cur_time", 0.0)),
            "done": bool(getattr(self.env.env, "done", False)),
        }

    def snapshot_from_recorded_state(self, state, action_history=None):
        self.env.sim.set_state_from_flattened(np.asarray(state, dtype=np.float64))
        self.env.sim.forward()
        self.env._post_process()
        if action_history is not None:
            # Panda's formatter integrates an internal command across MuJoCo
            # substeps. Reconstruct it from the frozen demonstration history;
            # qpos/qvel alone do not contain this action-semantic state.
            gripper = self.env.robots[0].gripper
            gripper.current_action = np.zeros_like(gripper.current_action)
            repeats = int(self.env.env.control_timestep / self.env.env.model_timestep)
            for action in np.asarray(action_history, dtype=np.float64):
                grip = np.asarray([action[config.GRIPPER_ACTION_INDEX]], dtype=np.float64)
                for _ in range(repeats):
                    gripper.format_action(grip)
        controller = self.env.robots[0].controller
        if hasattr(controller, "update"):
            controller.update(force=True)
        if hasattr(controller, "reset_goal"):
            controller.reset_goal()
        return self.capture_snapshot()

    def restore_snapshot(self, snapshot):
        self.env.sim.set_state_from_flattened(snapshot["sim_state"])
        _restore_simulator_auxiliary(
            self.env.sim.data,
            snapshot.get("sim_auxiliary", {}),
            include_warmstart=False,
        )
        self.env.sim.forward()
        # Forward recomputes derived quantities and may overwrite the solver
        # warm start; restore the complete auxiliary record afterwards.
        _restore_simulator_auxiliary(self.env.sim.data, snapshot.get("sim_auxiliary", {}))
        _restore_simple_attributes(self.env.robots[0].controller, snapshot["controller"])
        _restore_simple_attributes(self.env.robots[0].gripper, snapshot.get("gripper", {}))
        if hasattr(self.env.env, "timestep"):
            self.env.env.timestep = int(snapshot["timestep"])
        if hasattr(self.env.env, "cur_time"):
            self.env.env.cur_time = float(snapshot.get("cur_time", 0.0))
        if hasattr(self.env.env, "done"):
            self.env.env.done = bool(snapshot["done"])
        self.env._post_process()

    def _object_rotation(self, name):
        base = self.env.env
        if name in base.object_sites_dict:
            site_id = self.env.sim.model.site_name2id(name)
            return np.asarray(self.env.sim.data.site_xmat[site_id], dtype=np.float64).reshape(3, 3)
        body_id = int(base.obj_body_id[name])
        return np.asarray(self.env.sim.data.body_xmat[body_id], dtype=np.float64).reshape(3, 3)

    def _object_position(self, name):
        base = self.env.env
        if name in base.object_sites_dict:
            site_id = self.env.sim.model.site_name2id(name)
            return np.asarray(self.env.sim.data.site_xpos[site_id], dtype=np.float64).copy()
        body_id = int(base.obj_body_id[name])
        return np.asarray(self.env.sim.data.body_xpos[body_id], dtype=np.float64).copy()

    def _model_contact_geoms(self, name):
        model = self.env.env.get_object(name)
        return set(getattr(model, "contact_geoms", []) or [])

    def _build_geom_sets(self):
        base = self.env.env
        gripper_names = set(base.robots[0].gripper.contact_geoms)
        primary_names = self._model_contact_geoms(self.task["primary_object"])
        target_parent = self.task.get("target_contact_parent")
        target_names = (
            self._model_contact_geoms(target_parent)
            if target_parent and target_parent != self.task["primary_object"]
            else set()
        )

        def ids(names):
            output = set()
            for name in names:
                try:
                    output.add(int(self.env.sim.model.geom_name2id(name)))
                except Exception:
                    pass
            return output

        return {
            "gripper": ids(gripper_names),
            "primary": ids(primary_names),
            "target": ids(target_names),
        }

    def _contact_summary(self):
        import mujoco

        groups = self._geom_sets or self._build_geom_sets()
        pairs = {
            "gripper_primary": (groups["gripper"], groups["primary"]),
            "primary_target": (groups["primary"], groups["target"]),
            "gripper_target": (groups["gripper"], groups["target"]),
        }
        forces = dict((key, 0.0) for key in pairs)
        active = dict((key, False) for key in pairs)
        max_penetration = 0.0
        for index in range(int(self.env.sim.data.ncon)):
            contact = self.env.sim.data.contact[index]
            g1, g2 = int(contact.geom1), int(contact.geom2)
            max_penetration = max(max_penetration, max(-float(contact.dist), 0.0))
            force = np.zeros(6, dtype=np.float64)
            try:
                mujoco.mj_contactForce(self.env.sim.model._model, self.env.sim.data._data, index, force)
                normal = max(float(force[0]), 0.0)
            except Exception:
                normal = 1.0
            for key, (left, right) in pairs.items():
                if not left or not right:
                    continue
                if (g1 in left and g2 in right) or (g2 in left and g1 in right):
                    active[key] = True
                    forces[key] += normal
        relevant = bool(active["gripper_primary"] or active["primary_target"] or active["gripper_target"])
        return {
            "active": active,
            "relevant": relevant,
            "forces": forces,
            "max_penetration": float(max_penetration),
        }

    def _joint_values(self, name):
        model = self.env.env.get_object(name)
        values = []
        for joint in getattr(model, "joints", []) or []:
            address = self.env.sim.model.get_joint_qpos_addr(joint)
            if isinstance(address, tuple):
                start, stop = address
                values.extend(np.asarray(self.env.sim.data.qpos[start:stop], dtype=np.float64).tolist())
            else:
                values.append(float(self.env.sim.data.qpos[address]))
        return values

    def _task_progress(self, primary_position, target_position):
        if self.task["goal_predicate"] == "turnon":
            joints = self._joint_values(self.task["primary_object"])
            progress = np.clip((joints[0] + 0.005) / 0.505, 0.0, 1.0) if joints else 0.0
        else:
            distance = float(np.linalg.norm(primary_position - target_position))
            scale = 0.12
            target = self.task["target_object"]
            if target in self.env.env.object_sites_dict:
                site = self.env.env.object_sites_dict[target]
                size = np.asarray(getattr(site, "size", [0.06, 0.06, 0.06]), dtype=np.float64)
                scale = max(float(np.linalg.norm(size)), 0.05)
            progress = float(np.exp(-distance / scale))
        if bool(self.env.check_success()):
            progress = 1.0
        return float(np.clip(progress, 0.0, 1.0))

    def measure(self):
        base = self.env.env
        primary = self.task["primary_object"]
        target = self.task["target_object"]
        p_pos = self._object_position(primary)
        p_rot = self._object_rotation(primary)
        if self.task["goal_predicate"] == "turnon":
            t_pos, t_rot = p_pos.copy(), p_rot.copy()
            target_pose_valid = False
        else:
            t_pos, t_rot = self._object_position(target), self._object_rotation(target)
            target_pose_valid = True
        eef_site = int(base.robots[0].eef_site_id)
        tcp_pos = np.asarray(self.env.sim.data.site_xpos[eef_site], dtype=np.float64).copy()
        tcp_rot = np.asarray(self.env.sim.data.site_xmat[eef_site], dtype=np.float64).reshape(3, 3).copy()
        tcp_to_primary_rot = tcp_rot.T.dot(p_rot)
        primary_to_target_rot = t_rot.T.dot(p_rot)
        gripper_qpos = []
        for joint in base.robots[0].gripper.joints:
            gripper_qpos.append(float(self.env.sim.data.get_joint_qpos(joint)))
        gripper_width = float(np.sum(np.abs(gripper_qpos)))
        articulated = self._joint_values(primary)
        contacts = self._contact_summary()
        qpos = np.asarray(self.env.sim.data.qpos, dtype=np.float64)
        joint_violation = 0.0
        for joint_id in range(int(self.env.sim.model.njnt)):
            if int(self.env.sim.model.jnt_limited[joint_id]) == 0:
                continue
            address = int(self.env.sim.model.jnt_qposadr[joint_id])
            low, high = np.asarray(self.env.sim.model.jnt_range[joint_id], dtype=np.float64)
            joint_violation += max(low - qpos[address], 0.0) + max(qpos[address] - high, 0.0)

        values = []
        mask = []

        def add_pose(position, rotation, valid=True):
            values.extend(np.asarray(position, dtype=np.float64).tolist())
            values.extend(_rot6(rotation).tolist())
            mask.extend([bool(valid)] * 9)

        add_pose(p_pos, p_rot, True)
        add_pose(tcp_pos, tcp_rot, True)
        add_pose(p_pos - tcp_pos, tcp_to_primary_rot, True)
        add_pose(p_pos - t_pos, primary_to_target_rot, target_pose_valid)
        values.append(gripper_width)
        mask.append(True)
        for index in range(3):
            values.append(float(articulated[index]) if index < len(articulated) else 0.0)
            mask.append(index < len(articulated) and self.task["goal_predicate"] == "turnon")
        values.append(self._task_progress(p_pos, t_pos))
        mask.append(True)
        for key in ("gripper_primary", "primary_target", "gripper_target"):
            values.append(float(np.log1p(contacts["forces"][key])))
            if key == "gripper_primary":
                mask.append(True)
            else:
                mask.append(bool(self.task.get("target_contact_parent")))
        values.extend([contacts["max_penetration"], float(joint_violation)])
        mask.extend([True, True])
        vector = np.asarray(values, dtype=np.float64)
        valid = np.asarray(mask, dtype=bool)
        if len(vector) != len(FEATURE_NAMES):
            raise AssertionError((len(vector), len(FEATURE_NAMES)))
        return {
            "vector": vector,
            "mask": valid,
            "success": bool(self.env.check_success()),
            "progress": float(vector[FEATURE_NAMES.index("task_progress")]),
            "contacts": contacts,
        }

    def execute_chunk(self, snapshot, actions, settle_steps=config.SETTLE_STEPS):
        actions = np.asarray(actions, dtype=np.float64)
        if actions.shape != (config.CHUNK_HORIZON, config.ACTION_DIM):
            raise ValueError("bad chunk shape %r" % (actions.shape,))
        self.restore_snapshot(snapshot)
        initial = self.measure()
        contact_sequence = [bool(initial["contacts"]["relevant"])]
        step_rewards = []
        step_dones = []
        for action in actions:
            _, reward, done, _ = self.env.step(action)
            state = self.measure()
            contact_sequence.append(bool(state["contacts"]["relevant"]))
            step_rewards.append(float(reward))
            step_dones.append(bool(done))
        immediate = self.measure()
        zero = np.zeros(config.ACTION_DIM, dtype=np.float64)
        zero[config.GRIPPER_ACTION_INDEX] = actions[-1, config.GRIPPER_ACTION_INDEX]
        for _ in range(int(settle_steps)):
            self.env.step(zero)
        settled = self.measure()
        before, after = contact_sequence[0], contact_sequence[-1]
        if not before and any(contact_sequence[1:]):
            mode = "onset"
        elif before and after:
            mode = "persistent"
        elif before and not after:
            mode = "release"
        else:
            mode = "no_contact"
        return {
            "initial": initial,
            "immediate": immediate,
            "settled": settled,
            "contact_sequence": contact_sequence,
            "contact_mode": mode,
            "step_rewards": step_rewards,
            "step_dones": step_dones,
            "final_state": self.env.sim.get_state().flatten().copy(),
        }

    @staticmethod
    def continuous_chunk(actions):
        actions = np.asarray(actions, dtype=np.float64)
        return actions[:, config.CONTINUOUS_ACTION_INDICES].reshape(-1).copy()

    @staticmethod
    def replace_continuous_chunk(base_actions, continuous):
        output = np.asarray(base_actions, dtype=np.float64).copy()
        continuous = np.asarray(continuous, dtype=np.float64).reshape(
            config.CHUNK_HORIZON, len(config.CONTINUOUS_ACTION_INDICES)
        )
        output[:, config.CONTINUOUS_ACTION_INDICES] = continuous
        return output

    def scan_episode_states(self, episode):
        self.initialize_episode_model(episode)
        contacts = []
        target_contacts = []
        progress = []
        success = []
        for state in episode["states"]:
            self.env.sim.set_state_from_flattened(state)
            self.env.sim.forward()
            self.env._post_process()
            measured = self.measure()
            contacts.append(bool(measured["contacts"]["relevant"]))
            target_contacts.append(bool(measured["contacts"]["active"]["primary_target"]))
            progress.append(float(measured["progress"]))
            success.append(bool(measured["success"]))
        return {
            "contact": np.asarray(contacts, dtype=bool),
            "target_contact": np.asarray(target_contacts, dtype=bool),
            "progress": np.asarray(progress, dtype=np.float64),
            "success": np.asarray(success, dtype=bool),
        }

    def select_phase_indices(self, episode):
        scan = self.scan_episode_states(episode)
        actions = np.asarray(episode["actions"], dtype=np.float64)
        last = len(actions) - config.CHUNK_HORIZON
        if last < 4:
            raise RuntimeError("episode too short")
        pose = actions[:, config.CONTINUOUS_ACTION_INDICES]
        eligible = np.zeros(len(actions), dtype=bool)
        for index in range(last + 1):
            eligible[index] = bool(np.max(np.abs(pose[index : index + config.CHUNK_HORIZON])) <= 0.895)
        contact = scan["contact"]
        target_contact = scan["target_contact"]
        onsets = np.flatnonzero((~contact[:-1]) & contact[1:]) + 1
        onsets = onsets[onsets <= last]
        first_onset = int(onsets[0]) if len(onsets) else max(4, last // 3)
        target_onsets = np.flatnonzero((~target_contact[:-1]) & target_contact[1:]) + 1
        target_onsets = target_onsets[(target_onsets > first_onset + 2) & (target_onsets <= last)]
        if len(target_onsets):
            post_target = int(target_onsets[0])
        else:
            later = np.flatnonzero(contact & (np.arange(len(contact)) > first_onset + 4) & (np.arange(len(contact)) <= last))
            post_target = int(later[len(later) // 2]) if len(later) else int(np.clip(round(0.75 * last), 0, last))
        desired = {
            "free_space": max(0, first_onset // 3),
            "pre_contact": max(0, first_onset - config.CHUNK_HORIZON),
            "contact_onset": first_onset,
            "post_contact": post_target,
        }

        predicates = {
            "free_space": lambda i: not bool(contact[i]),
            "pre_contact": lambda i: not bool(contact[i]),
            "contact_onset": lambda i: bool(contact[i]) if len(onsets) else True,
            "post_contact": lambda i: True,
        }
        chosen = {}
        used = set()
        notes = []
        for phase in config.PHASES:
            candidates = [
                index
                for index in range(last + 1)
                if eligible[index] and index not in used and predicates[phase](index)
            ]
            fallback = False
            if not candidates:
                candidates = [index for index in range(last + 1) if eligible[index] and index not in used]
                fallback = True
            if not candidates:
                candidates = [index for index in range(last + 1) if index not in used]
                fallback = True
            index = min(candidates, key=lambda item: (abs(item - desired[phase]), item))
            chosen[phase] = int(index)
            used.add(int(index))
            notes.append(
                {
                    "phase": phase,
                    "desired_index": int(desired[phase]),
                    "selected_index": int(index),
                    "fallback": bool(fallback),
                    "contact_at_snapshot": bool(contact[index]),
                    "target_contact_at_snapshot": bool(target_contact[index]),
                    "progress_at_snapshot": float(scan["progress"][index]),
                    "max_abs_pose_action": float(np.max(np.abs(pose[index : index + config.CHUNK_HORIZON]))),
                }
            )
        return chosen, notes, scan


def perturb_continuous_chunk(base, direction, radius, sign):
    base = np.asarray(base, dtype=np.float64).reshape(-1)
    direction = np.asarray(direction, dtype=np.float64).reshape(-1)
    delta = float(sign) * float(radius) * direction
    candidate = base + delta
    if np.max(np.abs(candidate)) > 1.0 + 1e-12:
        raise ValueError("perturbation crosses normalized action bounds")
    return candidate, delta
