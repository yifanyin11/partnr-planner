import os
import numpy as np
import shutil
import random
from math import pi, acos
from numpy.linalg import norm

# Constants
ANGLE_THRESHOLD = {"on the left": pi / 4, "on the right": pi / 4, "on the back of self": pi / 4, "on the back of ref": pi / 4}
# ANGLE_THRESHOLD = {"on the left": pi / 6, "on the right": pi / 6, "on the back of self": pi / 3, "on the back of ref": pi / 3}
DISTANCE_THRESHOLD = 5

def calculate_angle(v1, v2):
    """Calculate the angle between two vectors."""
    cos_theta = np.dot(v1, v2) / (norm(v1) * norm(v2))
    cos_theta = np.clip(cos_theta, -1.0, 1.0)  # Avoid numerical errors
    return acos(cos_theta)

def load_world_graph(world_graph_path):
    """Load the world graph."""
    return np.load(world_graph_path, allow_pickle=True).item()  # Load as dictionary

def parse_world_description(file_path):
    """Parses a world description file."""
    with open(file_path, 'r') as file:
        data = file.readlines()

    furniture = {}
    objects = {}

    current_category = None
    for line in data:
        line = line.strip()
        if line.startswith("Furniture:"):
            current_category = "furniture"
        elif line.startswith("Objects:"):
            current_category = "objects"
        elif current_category == "furniture" and ":" in line:
            room, items = line.split(":")
            furniture[room.strip()] = [item.strip() for item in items.split(",")]
        elif current_category == "objects" and ":" in line:
            obj, loc = line.split(":")
            objects[obj.strip()] = loc.strip()
    return furniture, objects

def save_segment(segment_dir, segment_frames, segment_data):
    """Save a segment to the output directory."""
    os.makedirs(segment_dir, exist_ok=True)

    # Create modality folders
    modality_folders = {modality: os.path.join(segment_dir, modality) for modality in [
        "depth", "panoptic", "pose", "rgb", "world_desc", "world_desc_accum", "world_graph", "partial_world_graph"]}
    for folder in modality_folders.values():
        os.makedirs(folder, exist_ok=True)

    for frame, frame_data in zip(segment_frames, segment_data):
        for modality, file_path in frame_data.items():
            modality_type, ext = modality.rsplit("_", 1)
            save_path = os.path.join(modality_folders[modality_type], f"{frame}.{ext}")
            shutil.copy(file_path, save_path)

def save_entity_description(segment_dir, ref_name, target_name, relationship):
    """Save entity description for the segment."""
    desc_file = os.path.join(segment_dir, "entity_desc.txt")
    if relationship == "on the left" or relationship == "on the right":
        description = f"{target_name} is {relationship} of {ref_name}."
    elif relationship == "on the back of self":
        description = f"{target_name} is behind you."
    elif relationship == "on the back of ref":
        description = f"{target_name} is behind {ref_name}."
    elif relationship == "on":
        description = f"Explore around, you will find that {target_name} is on {ref_name}."
    elif relationship == "next to":
        description = f"Explore around, you will find that {target_name} is next to {ref_name}."
    elif relationship == "in":
        description = f"Explore around, you will find that {target_name} is in {ref_name}."
    elif relationship == "inside":
        description = f"Explore around, you will find that {target_name} is inside {ref_name}."
    else:
        raise ValueError(f"Invalid relationship: {relationship}")
    with open(desc_file, "w") as f:
        f.write(description)

def on_the_left(ref_camera_pose, obj, furniture, angle_thred, distance_thred):
    """Check if object is on the left of furniture using the reference camera pose."""
    obj_pos = np.array(obj.properties["translation"])
    furn_pos = np.array(furniture.properties["translation"])
    ref_pos = ref_camera_pose[:3, 3]  # Extract translation from the pose matrix
    ref_dir = -ref_camera_pose[:3, 0]  # Extract the x-axis (left direction) from the pose matrix

    dist = norm(obj_pos - furn_pos)
    angle = calculate_angle(obj_pos - furn_pos, ref_dir)

    return dist <= distance_thred and angle <= angle_thred

def on_the_right(ref_camera_pose, obj, furniture, angle_thred, distance_thred):
    """Check if object is on the right of furniture using the reference camera pose."""
    obj_pos = np.array(obj.properties["translation"])
    furn_pos = np.array(furniture.properties["translation"])
    ref_pos = ref_camera_pose[:3, 3]  # Extract translation from the pose matrix
    ref_dir = ref_camera_pose[:3, 0]  # Negative x-axis (right direction)

    dist = norm(obj_pos - furn_pos)
    angle = calculate_angle(obj_pos - furn_pos, ref_dir)

    return dist <= distance_thred and angle <= angle_thred

def on_the_back_of_self(ref_camera_pose, obj, furniture, angle_thred, distance_thred):
    """Check if object is on the back of the camera frame using the reference camera pose."""
    obj_pos = np.array(obj.properties["translation"])
    ref_pos = ref_camera_pose[:3, 3]  # Extract translation from the pose matrix
    ref_dir = ref_camera_pose[:3, 2]  # Negative z-axis (back direction)

    dist = norm(obj_pos - ref_pos)
    angle = calculate_angle(obj_pos - ref_pos, ref_dir)

    return dist <= distance_thred and angle <= angle_thred

def on_the_back_of_ref(ref_camera_pose, obj, furniture, angle_thred, distance_thred):
    """Check if object is on the back of the furniture using the reference camera pose."""
    obj_pos = np.array(obj.properties["translation"])
    furn_pos = np.array(furniture.properties["translation"])
    ref_pos = ref_camera_pose[:3, 3]  # Extract translation from the pose matrix
    ref_dir = -ref_camera_pose[:3, 2]  # Positive z-axis (front direction)

    dist = norm(obj_pos - furn_pos)
    angle = calculate_angle(obj_pos - furn_pos, ref_dir)

    return dist <= distance_thred and angle <= angle_thred

def next_to(world_graph, obj, furniture, distance_thred=1.5):
    """Check if object is next to furniture in the world graph."""
    """ Using 
    get_closest_object_or_furniture(
        self, obj_node, n: int, dist_threshold: float = 1.5
    )
    """
    obj_name = obj.name  # Access the object's name
    furn_name = furniture.name  # Access the furniture's name

    # Iterate through the keys in world_graph.graph, which are instances
    for graph_obj, graph_furns in world_graph.graph.items():
        if graph_obj.name == obj_name:  # Match object name
            # Get closest object or furniture
            closest_furns = world_graph.get_closest_object_or_furniture(graph_obj, 10, distance_thred)
            # Get names
            closest_furns_names = [furn.name for furn in closest_furns]
            if furn_name in closest_furns_names:
                print(f"Found relationship: next to")
                return True
    return False

def check_graph_relationship(world_graph, obj, furn, target_rel):
    """
    Check if the relationship between object and furniture in the world graph matches the target relationship.
    """
    obj_name = obj.name  # Access the object's name
    furn_name = furn.name  # Access the furniture's name

    # Iterate through the keys in world_graph.graph, which are instances
    for graph_obj, graph_furns in world_graph.graph.items():
        if graph_obj.name == obj_name:  # Match object name
            for graph_furn, rel in graph_furns.items():
                if furn_name in graph_furn.name:  # Match furniture name
                    print(f"Found relationship: {rel}")
                    return rel == target_rel  # Check if the relationship matches
    return False

def generate_segments(data_root, new_root):
    """Generate segments based on the new criteria."""
    # Counter for statistics
    stats = {
        "on": 0,
        "in": 0,
        "inside": 0,
        "next to": 0,
        "on the left": 0,
        "on the right": 0,
        "on the back of self": 0,
        "on the back of ref": 0,
    }
    for episode_id in os.listdir(data_root):
        print(f"Processing episode {episode_id}...")
        episode_path = os.path.join(data_root, episode_id, "main_agent")

        if not os.path.isdir(episode_path):
            continue

        room_names = [name for name in os.listdir(episode_path) if os.path.isdir(os.path.join(episode_path, name))]
        for room_name in room_names:
            print(f"\tProcessing room {room_name}...")
            room_path = os.path.join(episode_path, room_name)
            world_desc_path = os.path.join(room_path, "world_desc")
            pose_path = os.path.join(room_path, "pose")
            world_graph_path = os.path.join(room_path, "world_graph.npy")

            all_objects_files = sorted(os.listdir(os.path.join(room_path, "all_objects")))
            all_furnitures_files = sorted(os.listdir(os.path.join(room_path, "all_furnitures")))

            if not all_objects_files or not all_furnitures_files or not os.path.exists(world_graph_path):
                print(f"\t\tMissing required files for room {room_name}.")
                continue

            all_objects_path = os.path.join(room_path, "all_objects", all_objects_files[0])
            all_furnitures_path = os.path.join(room_path, "all_furnitures", all_furnitures_files[0])

            # Load object and furniture metadata
            all_objects = np.load(all_objects_path, allow_pickle=True)
            all_furnitures = np.load(all_furnitures_path, allow_pickle=True)

            # Load the world graph
            world_graph = load_world_graph(world_graph_path)

            # Track discovery steps for objects
            object_discovery = {}
            object_disappear = {}
            desc_files = sorted(
                [f for f in os.listdir(world_desc_path) if f.endswith(".txt")],
                key=lambda x: int(os.path.splitext(x)[0])  # Sort by numeric index
            )
            for desc_file in desc_files:
                frame_idx = int(os.path.splitext(desc_file)[0])  # Extract numeric index
                current_desc = parse_world_description(os.path.join(world_desc_path, desc_file))
                ##############
                ##  Filter  ##
                ##############
                one_world_graph_path = os.path.join(room_path, "world_graph", f"{frame_idx}.npy")
                one_world_graph = np.load(one_world_graph_path, allow_pickle=True).item()
                # import ipdb; ipdb.set_trace()
                if one_world_graph.get_room_for_entity(one_world_graph.get_spot_robot()).name != room_name:
                    continue
                ##############
                ##  Filter  ##
                ##############
                for obj in current_desc[1].keys():  # Objects in the current view
                    if obj not in object_discovery:
                        object_discovery[obj] = frame_idx
                for obj in object_discovery.keys():
                    if obj not in current_desc[1] and obj not in object_disappear:
                        object_disappear[obj] = frame_idx

            # Define all relationships to process
            relationships = [
                ("on", lambda ref_camera_pose, obj, furn: check_graph_relationship(world_graph, obj, furn, "on")),
                ("in", lambda ref_camera_pose, obj, furn: check_graph_relationship(world_graph, obj, furn, "in")),
                ("inside", lambda ref_camera_pose, obj, furn: check_graph_relationship(world_graph, obj, furn, "inside")),
                ("next to", lambda ref_camera_pose, obj, furn: next_to(world_graph, obj, furn)),
                ("on the left", lambda ref_camera_pose, obj, furn: on_the_left(ref_camera_pose, obj, furn, ANGLE_THRESHOLD["on the left"], DISTANCE_THRESHOLD)),
                ("on the right", lambda ref_camera_pose, obj, furn: on_the_right(ref_camera_pose, obj, furn, ANGLE_THRESHOLD["on the right"], DISTANCE_THRESHOLD)),
                ("on the back of self", lambda ref_camera_pose, obj, furn: on_the_back_of_self(ref_camera_pose, obj, furn, ANGLE_THRESHOLD["on the back of self"], DISTANCE_THRESHOLD)),
                ("on the back of ref", lambda ref_camera_pose, obj, furn: on_the_back_of_ref(ref_camera_pose, obj, furn, ANGLE_THRESHOLD["on the back of ref"], DISTANCE_THRESHOLD)),
            ]

            # Process each discovered object
            for obj_name, discovery_step in object_discovery.items():
                # import ipdb; ipdb.set_trace()
                obj_entity = [obj for obj in world_graph.graph.keys() if obj.name == obj_name][0]
                try:
                    detected_room_name = world_graph.get_room_for_entity(world_graph.find_furniture_for_object(obj_entity)).name
                except:
                    print(f"Object {obj_name} not found in the world graph.")
                    continue
                if detected_room_name != room_name:
                    print(f"Object {obj_name} not found in room {room_name}.")
                    continue
                if obj_name not in object_disappear:
                    continue
                disappear_step = object_disappear[obj_name]
                print(f"discovery_step: {discovery_step}, disappear_step: {disappear_step}")
                processed_furnitures = set()  # Track furniture already considered for this object
                first_frame_flag = False
                for frame_idx in sorted([int(os.path.splitext(f)[0]) for f in desc_files if int(os.path.splitext(f)[0]) < discovery_step]):
                    # record the first frame of the object
                    if first_frame_flag==False:
                        first_idx = frame_idx
                        first_frame_flag = True

                    current_desc = parse_world_description(os.path.join(world_desc_path, f"{frame_idx}.txt"))
                    ref_camera_pose = np.load(os.path.join(pose_path, f"{frame_idx}.npy"))  # Load camera pose
                    for furn_names in current_desc[0].values():  # Furnitures in the current view
                        if not isinstance(furn_names, list):
                            furn_names = [furn_names]  # Ensure we handle non-list values

                        for furn_name in furn_names:
                            if furn_name in processed_furnitures:
                                continue

                            # Find the corresponding furniture and object
                            for furn in all_furnitures:
                                if furn.name == furn_name:
                                    ##############
                                    ##  Filter  ##
                                    ##############
                                    if world_graph.get_room_for_entity(furn).name != room_name:
                                        continue
                                    ##############
                                    ##  Filter  ##
                                    ##############
                                    for obj in all_objects:
                                        if obj.name == obj_name:
                                            for relationship, func in relationships:
                                                if func(ref_camera_pose, obj, furn):
                                                    # Record stats
                                                    stats[relationship] += 1
                                                    # Save the segment
                                                    segment_dir = os.path.join(new_root, episode_id, "main_agent", room_name, f"{furn_name}_{obj_name}_{relationship.replace(' ', '_')}")
                                                    if relationship not in ["on the left", "on the right", "on the back of self", "on the back of ref"]:
                                                        pre_length = random.randint(5, 50)
                                                        start_frame_idx = max(frame_idx-pre_length, first_idx)
                                                    else:
                                                        start_frame_idx = frame_idx
                                                    segment_frames = list(range(start_frame_idx, disappear_step + 1))
                                                    segment_data = []  # Collect the modality files here
                                                    for frame in segment_frames:
                                                        frame_data = {}
                                                        for modality in [
                                                            "depth",
                                                            "panoptic",
                                                            "pose",
                                                            "rgb",
                                                            "world_desc",
                                                            "world_desc_accum",
                                                            "world_graph",
                                                            "partial_world_graph",
                                                        ]:
                                                            modality_dir = os.path.join(room_path, modality)

                                                            if modality in ["depth", "panoptic"]:
                                                                # Save both .npy and .png for depth and panoptic
                                                                frame_data[f"{modality}_npy"] = os.path.join(
                                                                    modality_dir, f"{frame}.npy"
                                                                )
                                                                frame_data[f"{modality}_png"] = os.path.join(
                                                                    modality_dir, f"{frame}.png"
                                                                )
                                                            elif modality == "pose":
                                                                # Save only .npy for pose
                                                                frame_data[f"{modality}_npy"] = os.path.join(
                                                                    modality_dir, f"{frame}.npy"
                                                                )
                                                            elif modality == "rgb":
                                                                # Save both .npy and .jpg for rgb
                                                                frame_data[f"{modality}_npy"] = os.path.join(
                                                                    modality_dir, f"{frame}.npy"
                                                                )
                                                                frame_data[f"{modality}_jpg"] = os.path.join(
                                                                    modality_dir, f"{frame}.jpg"
                                                                )
                                                            elif modality == "world_desc":
                                                                # Save only .txt for world_desc
                                                                frame_data[f"{modality}_txt"] = os.path.join(
                                                                    modality_dir, f"{frame}.txt"
                                                                )
                                                            elif modality == "world_desc_accum":
                                                                # Save only .txt for world_desc
                                                                frame_data[f"{modality}_txt"] = os.path.join(
                                                                    modality_dir, f"{frame}.txt"
                                                                )
                                                            elif modality == "world_graph":
                                                                # Save only .npy for world_graph
                                                                frame_data[f"{modality}_npy"] = os.path.join(
                                                                    modality_dir, f"{frame}.npy"
                                                                )
                                                            elif modality == "partial_world_graph":
                                                                # Save only .npy for world_graph
                                                                frame_data[f"{modality}_npy"] = os.path.join(
                                                                    modality_dir, f"{frame}.npy"
                                                                )

                                                        segment_data.append(frame_data)

                                                    save_segment(segment_dir, segment_frames, segment_data)
                                                    save_entity_description(segment_dir, furn_name, obj_name, relationship)

                                                    # Mark furniture as processed for this object
                                                    processed_furnitures.add(furn_name)
                                                    break
    
    # Print statistics
    print(stats)
    # calculate total number of segments
    total_segments = 0
    for key in stats.keys():
        total_segments += stats[key]
    print(f"Total number of segments: {total_segments}")

# Directories
input_dataset_path = "/media/yyin34/ExtremePro/projects/home_robot/partnr-planner/data/trajectories/v4"
output_dataset_path = "/media/yyin34/ExtremePro/projects/home_robot/partnr-planner/data/trajectories/dataset_v4"

# Generate the segments
generate_segments(input_dataset_path, output_dataset_path)
