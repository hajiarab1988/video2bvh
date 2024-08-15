from . import math3d
from . import bvh_helper

import numpy as np
from pprint import pprint


class CMUSkeleton(object):

    def __init__(self):
        self.root = 'Hips'
        self.keypoint2index = {
            'Hips': 0,
            'RightUpLeg': 1,
            'RightLeg': 2,
            'RightFoot': 3,
            'LeftUpLeg': 4,
            'LeftLeg': 5,
            'LeftFoot': 6,
            'Neck': 7,
            'LeftArm': 8,
            'LeftForeArm': 9,
            'LeftHand': 10,
            'RightArm': 11,
            'RightForeArm': 12,
            'RightHand': 13,
            'RightHipJoint': -1,
            'RightFootEndSite': -1,
            'LeftHipJoint': -1,
            'LeftFootEndSite': -1,
            'LeftShoulder': -1,
            'LeftHandEndSite': -1,
            'RightShoulder': -1,
            'RightHandEndSite': -1,
            'LowerBack': -1,
            'Spine': -1,
            'Spine1': -1,
            'HeadEndSite': -1
        }
        self.index2keypoint = {v: k for k, v in self.keypoint2index.items()}
        self.keypoint_num = len(self.keypoint2index)

        # Define the new hierarchy without Spine, Spine1, and HeadEndSite
        self.children = {
            'Hips': ['LeftHipJoint', 'LowerBack', 'RightHipJoint'],
            'LeftHipJoint': ['LeftUpLeg'],
            'LeftUpLeg': ['LeftLeg'],
            'LeftLeg': ['LeftFoot'],
            'LeftFoot': ['LeftFootEndSite'],
            'LowerBack': ['Neck'],
            'Neck': ['LeftArm', 'RightArm'],
            'LeftArm': ['LeftForeArm'],
            'LeftForeArm': ['LeftHand'],
            'RightArm': ['RightForeArm'],
            'RightForeArm': ['RightHand'],
            'RightHipJoint': ['RightUpLeg'],
            'RightUpLeg': ['RightLeg'],
            'RightLeg': ['RightFoot'],
            'RightFoot': ['RightFootEndSite']
        }
        self.parent = {self.root: None}
        for parent, children in self.children.items():
            for child in children:
                self.parent[child] = parent
                
        self.left_joints = [
            joint for joint in self.keypoint2index
            if 'Left' in joint
        ]
        self.right_joints = [
            joint for joint in self.keypoint2index
            if 'Right' in joint
        ]

        self.initial_directions = {
            'Hips': [0, 0, 0],
            'LeftHipJoint': [1, 0, 0],
            'LeftUpLeg': [1, 0, 0],
            'LeftLeg': [0, 0, -1],
            'LeftFoot': [0, 0, -1],
            'LeftFootEndSite': [0, -1, 0],
            'Neck': [0, 0, 1],
            'LeftArm': [1, 0, 0],
            'LeftForeArm': [1, 0, 0],
            'LeftHand': [1, 0, 0],
            'RightArm': [-1, 0, 0],
            'RightForeArm': [-1, 0, 0],
            'RightHand': [-1, 0, 0],
            'RightHipJoint': [-1, 0, 0],
            'RightUpLeg': [-1, 0, 0],
            'RightLeg': [0, 0, -1],
            'RightFoot': [0, 0, -1],
            'RightFootEndSite': [0, -1, 0]
        }


    def get_initial_offset(self, poses_3d):
        # TODO: RANSAC
        bone_lens = {self.root: [0]}
        stack = [self.root]
        while stack:
            parent = stack.pop()
            p_idx = self.keypoint2index[parent]
            p_name = parent
            while p_idx == -1:
                # find real parent
                p_name = self.parent[p_name]
                p_idx = self.keypoint2index[p_name]
            for child in self.children[parent]:
                stack.append(child)

                if self.keypoint2index[child] == -1:
                    bone_lens[child] = [0.1]
                else:
                    c_idx = self.keypoint2index[child]
                    bone_lens[child] = np.linalg.norm(
                        poses_3d[:, p_idx] - poses_3d[:, c_idx],
                        axis=1
                    )

        bone_len = {}
        for joint in self.keypoint2index:
            if 'Left' in joint or 'Right' in joint:
                base_name = joint.replace('Left', '').replace('Right', '')
                left_len = np.mean(bone_lens['Left' + base_name])
                right_len = np.mean(bone_lens['Right' + base_name])
                bone_len[joint] = (left_len + right_len) / 2
            else:
                bone_len[joint] = np.mean(bone_lens[joint])

        initial_offset = {}
        for joint, direction in self.initial_directions.items():
            direction = np.array(direction) / max(np.linalg.norm(direction), 1e-12)
            initial_offset[joint] = direction * bone_len[joint]

        return initial_offset


    def get_bvh_header(self, poses_3d):
        initial_offset = self.get_initial_offset(poses_3d)

        nodes = {}
        for joint in self.keypoint2index:
            is_root = joint == self.root
            is_end_site = 'EndSite' in joint
            nodes[joint] = bvh_helper.BvhNode(
                name=joint,
                offset=initial_offset[joint],
                rotation_order='zxy' if not is_end_site else '',
                is_root=is_root,
                is_end_site=is_end_site,
            )
        for joint, children in self.children.items():
            nodes[joint].children = [nodes[child] for child in children]
            for child in children:
                nodes[child].parent = nodes[joint]

        header = bvh_helper.BvhHeader(root=nodes[self.root], nodes=nodes)
        return header


    def pose2euler(self, pose, header):
        channel = []
        quats = {}
        eulers = {}
        stack = [header.root]
        while stack:
            node = stack.pop()
            joint = node.name
            joint_idx = self.keypoint2index.get(joint, -1)
            
            if joint_idx == -1:
                continue  # Skip joints not present in the pose data
    
            if node.is_root:
                channel.extend(pose[joint_idx])
    
            order = None
            if joint == 'Hips':
                x_dir = pose[self.keypoint2index['LeftUpLeg']] - pose[self.keypoint2index['RightUpLeg']]
                z_dir = pose[self.keypoint2index['Neck']] - pose[joint_idx]
                order = 'zyx'
            elif joint in ['RightUpLeg', 'RightLeg']:
                child_idx = self.keypoint2index[node.children[0].name]
                x_dir = pose[self.keypoint2index['Hips']] - pose[self.keypoint2index['RightUpLeg']]
                z_dir = pose[joint_idx] - pose[child_idx]
                order = 'zyx'
            elif joint in ['LeftUpLeg', 'LeftLeg']:
                child_idx = self.keypoint2index[node.children[0].name]
                x_dir = pose[self.keypoint2index['LeftUpLeg']] - pose[self.keypoint2index['Hips']]
                z_dir = pose[joint_idx] - pose[child_idx]
                order = 'zyx'
            elif joint == 'Neck':
                x_dir = pose[self.keypoint2index['LeftArm']] - pose[self.keypoint2index['RightArm']]
                z_dir = pose[joint_idx] - pose[self.keypoint2index['Neck']]
                order = 'zyx'
            elif joint == 'LeftArm':
                x_dir = pose[self.keypoint2index['LeftForeArm']] - pose[joint_idx]
                z_dir = pose[joint_idx] - pose[self.keypoint2index['LeftHand']]
                order = 'xzy'
            elif joint == 'LeftForeArm':
                x_dir = pose[self.keypoint2index['LeftHand']] - pose[joint_idx]
                z_dir = pose[joint_idx] - pose[self.keypoint2index['LeftArm']]
                order = 'xzy'
            elif joint == 'RightArm':
                x_dir = pose[joint_idx] - pose[self.keypoint2index['RightForeArm']]
                z_dir = pose[self.keypoint2index['RightForeArm']] - pose[self.keypoint2index['RightHand']]
                order = 'xzy'
            elif joint == 'RightForeArm':
                x_dir = pose[joint_idx] - pose[self.keypoint2index['RightHand']]
                z_dir = pose[joint_idx] - pose[self.keypoint2index['RightArm']]
                order = 'xzy'
            
            if order:
                dcm = math3d.dcm_from_axis(x_dir, None, z_dir, order)
                quats[joint] = math3d.dcm2quat(dcm)
            else:
                quats[joint] = quats.get(self.parent.get(joint, None), None)
    
            local_quat = quats.get(joint, None)
            if local_quat is not None and node.parent:
                local_quat = math3d.quat_divide(
                    q=quats[joint], r=quats[node.parent.name]
                )
    
            euler = math3d.quat2euler(
                q=local_quat, order=node.rotation_order
            )
            euler = np.rad2deg(euler)
            eulers[joint] = euler
            channel.extend(euler)
    
            for child in node.children[::-1]:
                if not child.is_end_site:
                    stack.append(child)
    
        return channel


    def poses2bvh(self, poses_3d, header=None, output_file=None):
        if not header:
            header = self.get_bvh_header(poses_3d)

        channels = []
        for frame, pose in enumerate(poses_3d):
            channels.append(self.pose2euler(pose, header))

        if output_file:
            bvh_helper.write_bvh(output_file, header, channels)
        
        return channels, header
