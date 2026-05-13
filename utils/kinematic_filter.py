import numpy as np
from typing import List, Tuple, Optional

class KinematicFilter:
    """
    Utility to filter robot trajectory data based on kinematic properties.
    """
    
    def __init__(self, variance_threshold: float = 1e-4, velocity_threshold: float = 1e-3):
        """
        Args:
            variance_threshold: Minimum variance in joint states to consider a segment 'moving'.
            velocity_threshold: Minimum average velocity to consider a segment 'active'.
        """
        self.variance_threshold = variance_threshold
        self.velocity_threshold = velocity_threshold

    def is_idle(self, trajectory: np.ndarray) -> bool:
        """
        Checks if a trajectory segment is idle (minimal movement).
        Trajectory shape: (time_steps, state_dim)
        """
        if len(trajectory) < 2:
            return True
            
        # Calculate variance across time for each joint
        variances = np.var(trajectory, axis=0)
        max_variance = np.max(variances)
        
        return max_variance < self.variance_threshold

    def get_velocity(self, trajectory: np.ndarray, dt: float = 1.0) -> np.ndarray:
        """
        Calculate velocities between consecutive frames.
        """
        return np.diff(trajectory, axis=0) / dt

    def filter_idle_frames(self, states: np.ndarray, actions: np.ndarray, window_size: int = 5) -> Tuple[np.ndarray, np.ndarray]:
        """
        Removes chunks of frames where the robot is stationary.
        Uses a sliding window to determine activity.
        """
        n_frames = len(states)
        mask = np.ones(n_frames, dtype=bool)
        
        for i in range(0, n_frames - window_size + 1):
            window = states[i:i+window_size]
            if self.is_idle(window):
                mask[i:i + 1] = False # Mark current frame as idle
        
        return states[mask], actions[mask]

    def detect_anomalies(self, trajectory: np.ndarray, threshold: float = 1.0) -> np.ndarray:
        """
        Detect sudden spikes in movement (potential teleop noise or sensor errors).
        Returns a mask where True means 'normal'.
        """
        velocities = self.get_velocity(trajectory)
        norms = np.linalg.norm(velocities, axis=1)
        
        # Simple thresholding on velocity magnitude
        mask = np.ones(len(trajectory), dtype=bool)
        anomalous_indices = np.where(norms > threshold)[0]
        
        for idx in anomalous_indices:
            mask[idx] = False
            mask[idx + 1] = False # Both ends of the spike frame
            
        return mask

if __name__ == "__main__":
    # Example usage / Simple test
    filter = KinematicFilter(variance_threshold=0.01)
    
    # Mock data: stationary then moving
    mock_states = np.zeros((20, 7))
    mock_states[10:] = np.arange(10).reshape(-1, 1) * 0.1
    mock_actions = np.random.rand(20, 7)
    
    filtered_states, filtered_actions = filter.filter_idle_frames(mock_states, mock_actions, window_size=4)
    print(f"Original length: {len(mock_states)}, Filtered length: {len(filtered_states)}")
