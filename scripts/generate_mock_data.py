import h5py
import numpy as np
import os
from PIL import Image
import io
from tqdm import tqdm

def generate_mock_h5(output_path, num_demos=5, seq_len=50):
    """
    Generates a mock HDF5 dataset for testing the VLA pipeline.
    """
    print(f"Generating mock data: {output_path}")
    
    with h5py.File(output_path, 'w') as f:
        for i in range(num_demos):
            demo_grp = f.create_group(f"demo_{i}")
            obs_grp = demo_grp.create_group("obs")
            
            # 1. Joint States (7-DOF)
            # Add some movement and some idle segments
            states = np.zeros((seq_len, 7))
            states[10:40] = np.linspace(0, 1, 30).reshape(-1, 1) * np.random.rand(7)
            obs_grp.create_dataset("joint_states", data=states.astype(np.float32))
            
            # 2. Images (Random noise but encoded as PNG)
            image_data = []
            for _ in range(seq_len):
                img = Image.fromarray(np.uint8(np.random.rand(224, 224, 3) * 255))
                buf = io.BytesIO()
                img.save(buf, format='PNG')
                image_data.append(np.frombuffer(buf.getvalue(), dtype=np.uint8))
            
            # Use variable length byte strings for images
            dt = h5py.vlen_dtype(np.dtype('uint8'))
            ds = obs_grp.create_dataset("image", (seq_len,), dtype=dt)
            for j in range(seq_len):
                ds[j] = image_data[j]
                
            # 3. Actions
            actions = np.random.randn(seq_len, 7).astype(np.float32)
            demo_grp.create_dataset("actions", data=actions)
            
            # 4. Instruction Attribute
            demo_grp.attrs['instruction'] = "pick up the red cube and move it to the tray"

    print(f"Generated {num_demos} demos in {output_path}")

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--output_dir", type=str, default="data/raw")
    parser.add_argument("--num_demos", type=int, default=3)
    args = parser.parse_args()
    
    if not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir)
        
    generate_mock_h5(os.path.join(args.output_dir, "mock_dataset.h5"), num_demos=args.num_demos)
