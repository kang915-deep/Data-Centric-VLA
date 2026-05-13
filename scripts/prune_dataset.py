import os
import argparse
import h5py
import numpy as np
from tqdm import tqdm
from PIL import Image
import io

from utils.kinematic_filter import KinematicFilter
from utils.semantic_filter import SemanticFilter

def prune_dataset(args):
    print(f"Loading filters... Device: {args.device}")
    k_filter = KinematicFilter(variance_threshold=args.k_var_thresh)
    s_filter = SemanticFilter(device=args.device) if not args.skip_semantic else None

    if not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir)

    # Process all h5 files in input directory
    files = [f for f in os.listdir(args.input_dir) if f.endswith('.h5')]
    
    for filename in tqdm(files, desc="Pruning datasets"):
        input_path = os.path.join(args.input_dir, filename)
        output_path = os.path.join(args.output_dir, filename)
        
        with h5py.File(input_path, 'r') as f_in, h5py.File(output_path, 'w') as f_out:
            # Common structure: 'data' group with 'obs', 'action', etc.
            # This is a simplified version; real datasets vary.
            for demo_id in f_in.keys():
                demo = f_in[demo_id]
                states = demo['obs']['joint_states'][:]
                actions = demo['actions'][:]
                instruction = demo.attrs.get('instruction', 'robot action')
                
                # 1. Kinematic Pruning
                active_mask = np.ones(len(states), dtype=bool)
                # Note: filter_idle_frames returns filtered arrays, but we might want masks
                # for consistency across multiple filters. Let's adapt.
                for i in range(len(states) - args.window_size + 1):
                    if k_filter.is_idle(states[i:i+args.window_size]):
                        active_mask[i] = False
                
                # 2. Semantic Pruning (if enabled)
                if s_filter:
                    # Convert encoded images in h5 to PIL
                    images = []
                    # Assuming images are stored as byte strings or arrays in 'obs/image'
                    raw_images = demo['obs']['image'][:]
                    for img_data in raw_images:
                        images.append(Image.open(io.BytesIO(img_data)))
                    
                    sem_scores = s_filter.get_scores(images, instruction)
                    sem_mask = sem_scores >= args.s_thresh
                    combined_mask = active_mask & sem_mask
                else:
                    combined_mask = active_mask

                # Save pruned data
                if np.any(combined_mask):
                    d_out = f_out.create_group(demo_id)
                    obs_out = d_out.create_group('obs')
                    obs_out.create_dataset('joint_states', data=states[combined_mask])
                    obs_out.create_dataset('image', data=demo['obs']['image'][combined_mask])
                    d_out.create_dataset('actions', data=actions[combined_mask])
                    d_out.attrs['instruction'] = instruction
                    d_out.attrs['pruning_ratio'] = 1.0 - (np.sum(combined_mask) / len(states))

    print("Pruning complete.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_dir", type=str, required=True)
    parser.add_argument("--output_dir", type=str, required=True)
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--k_var_thresh", type=float, default=1e-4)
    parser.add_argument("--s_thresh", type=float, default=0.2)
    parser.add_argument("--window_size", type=int, default=5)
    parser.add_argument("--skip_semantic", action="store_true")
    
    args = parser.parse_args()
    prune_dataset(args)
