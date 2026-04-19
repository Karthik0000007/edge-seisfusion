"""
Zenodo Hell Bridge Test Arena (HBTA) Dataset Loader

Dataset: "A data set from an extensive experimental benchmark study of the
Hell Bridge Test Arena subject to imposed damage"
Source: Zenodo, DOI: 10.5281/zenodo.14028239 (v3)
Authors: Structural Dynamics Group, NTNU (Norway)

Loads HDF5 accelerometer and strain data from HBTA benchmark.
Contains 10 structural states (0=undamaged, 1-9=progressive damage).
"""

import numpy as np
import h5py
from pathlib import Path
from typing import Dict, Tuple, Optional, List
import json
from datetime import datetime


class ZenodoHBTALoader:
    """Load and preprocess Zenodo HBTA benchmark dataset."""

    # Dataset metadata
    DATASET_NAME = "Hell Bridge Test Arena (HBTA, Zenodo)"
    DOI = "10.5281/zenodo.14028239"
    AUTHORS = "Structural Dynamics Group, NTNU (Norway)"
    ORIGINAL_SAMPLING_RATE = 100  # Hz
    TARGET_SAMPLING_RATE = 1000   # Hz
    NUM_STATES = 10               # 0=undamaged, 1-9=damage
    NUM_TESTS_PER_STATE = 8       # Approximate
    TOTAL_TESTS = NUM_STATES * NUM_TESTS_PER_STATE

    def __init__(self, h5_file_path: Path):
        """
        Initialize loader.

        Args:
            h5_file_path: Path to data_100Hz.h5 file
        """
        self.h5_file = Path(h5_file_path)
        if not self.h5_file.exists():
            raise FileNotFoundError(f"HDF5 file not found: {self.h5_file}")
        self._inspect_hdf5_structure()

    def _inspect_hdf5_structure(self):
        """Inspect and print HDF5 file structure."""
        with h5py.File(self.h5_file, 'r') as f:
            print(f"[HBTA] HDF5 structure of {self.h5_file.name}:")
            self._print_hdf5_structure(f)

    def _print_hdf5_structure(self, group, indent=0):
        """Recursively print HDF5 group/dataset structure."""
        for key in group.keys():
            item = group[key]
            if isinstance(item, h5py.Dataset):
                print(f"  {'  ' * indent}[Dataset] {key}: shape={item.shape}, dtype={item.dtype}")
            elif isinstance(item, h5py.Group):
                print(f"  {'  ' * indent}[Group] {key}/")
                self._print_hdf5_structure(item, indent + 1)

    def get_dataset_info(self) -> Dict:
        """
        Get high-level information about the dataset.

        Returns:
            Dictionary with available groups/datasets and metadata.
        """
        info = {'groups': [], 'datasets': [], 'total_size_bytes': 0}

        with h5py.File(self.h5_file, 'r') as f:
            for key in f.keys():
                item = f[key]
                if isinstance(item, h5py.Dataset):
                    info['datasets'].append({
                        'name': key,
                        'shape': item.shape,
                        'dtype': str(item.dtype),
                        'size_bytes': item.nbytes
                    })
                    info['total_size_bytes'] += item.nbytes
                elif isinstance(item, h5py.Group):
                    info['groups'].append(key)

        return info

    def load_all_states(self, select_channels: Optional[List[str]] = None) -> Dict:
        """
        Load all structural states and tests.

        Args:
            select_channels: List of channel names to extract
                            (e.g., ['accel_z', 'strain_main'])
                            If None, attempt to auto-detect

        Returns:
            Dictionary with:
                - 'data': List of np.array per state
                - 'state_labels': State indices (0-9)
                - 'metadata': Dataset metadata
        """
        print(f"[HBTA] Loading all states from {self.h5_file}...")

        all_states_data = []
        metadata_list = []

        with h5py.File(self.h5_file, 'r') as f:
            # Attempt to detect structure (group per state or flat structure)
            if 'acceleration' in f:
                # Flat structure: single 'acceleration' group with subgroups per state
                print(f"[HBTA] Detected structure: flat with 'acceleration' group")
                accel_group = f['acceleration']
                num_states = len(accel_group)

                for state_idx in range(num_states):
                    state_key = f'state_{state_idx}' if f'state_{state_idx}' in accel_group \
                               else str(state_idx)

                    if state_key in accel_group:
                        state_data = accel_group[state_key][:]
                        all_states_data.append(state_data)
                        metadata_list.append({'state': state_idx, 'shape': state_data.shape})
                        print(f"  Loaded state {state_idx}: {state_data.shape}")
            else:
                print(f"[HBTA] Could not auto-detect structure. Groups available: {list(f.keys())}")

        return {
            'states_data': all_states_data,
            'state_labels': list(range(len(all_states_data))),
            'metadata': {
                'dataset': self.DATASET_NAME,
                'doi': self.DOI,
                'num_states': len(all_states_data),
                'original_sampling_rate': self.ORIGINAL_SAMPLING_RATE,
                'metadata_list': metadata_list
            }
        }

    def load_state(self, state_index: int) -> Dict:
        """
        Load all tests for a specific structural state.

        Args:
            state_index: State (0=undamaged, 1-9=damage)

        Returns:
            Dictionary with:
                - 'data': np.array([N_tests, N_samples, N_channels])
                - 'state_label': state_index
                - 'metadata': dict
        """
        if not (0 <= state_index < self.NUM_STATES):
            raise ValueError(f"state_index must be 0-{self.NUM_STATES - 1}")

        print(f"[HBTA] Loading state {state_index}...")

        with h5py.File(self.h5_file, 'r') as f:
            # Attempt to locate state data
            if 'acceleration' in f:
                accel_group = f['acceleration']
                state_key = f'state_{state_index}' if f'state_{state_index}' in accel_group \
                           else str(state_index)

                if state_key in accel_group:
                    data = accel_group[state_key][:]
                    return {
                        'data': data.astype(np.float32),
                        'state_label': state_index,
                        'num_tests': len(data) if len(data.shape) >= 2 else 1,
                        'shape': data.shape,
                        'metadata': {
                            'state_index': state_index,
                            'state_description': ['undamaged', 'damage_1', 'damage_2',
                                                   'damage_3', 'damage_4', 'damage_5',
                                                   'damage_6', 'damage_7', 'damage_8', 'damage_9']
                            [state_index],
                            'original_sampling_rate': self.ORIGINAL_SAMPLING_RATE
                        }
                    }

            raise KeyError(f"Could not find state {state_index} in HDF5 file")

    def extract_canonical_channels(self, data: np.ndarray) -> np.ndarray:
        """
        Extract/map channels to canonical [accel_x, accel_y, accel_z, strain, acoustic].

        HBTA provides ~75 channels (40 vertical + 18 triaxial + strain).
        Extract most informative ones:
        - Channel 0: Primary vertical acceleration (Z)
        - Channel 1: Second vertical or transverse
        - Channel 2: Transverse or third axis
        - Channel 3-4: Strain gauge (if available)

        Args:
            data: Raw data with multiple channels

        Returns:
            Canonical 5-channel np.array([N_samples, 5])
        """
        if len(data.shape) == 1:
            # 1D data, expand to 5 channels
            data = data.reshape(-1, 1)

        num_samples, num_channels = data.shape

        if num_channels >= 5:
            # Select first 5 channels (primary acceleration + strain)
            canonical = data[:, :5].copy()
        else:
            # Pad with zeros
            canonical = np.zeros((num_samples, 5), dtype=np.float32)
            canonical[:, :num_channels] = data[:, :num_channels]

        return canonical.astype(np.float32)

    def resample_to_1khz(self, data: np.ndarray) -> np.ndarray:
        """
        Upsample from 100 Hz to 1 kHz (10× interpolation).

        Args:
            data: np.array([N_samples, N_channels])

        Returns:
            np.array([N_samples * 10, N_channels])
        """
        from scipy import signal

        num_samples, num_channels = data.shape
        target_num_samples = num_samples * 10

        resampled = np.zeros((target_num_samples, num_channels), dtype=np.float32)

        for ch in range(num_channels):
            resampled[:, ch] = signal.resample(data[:, ch], target_num_samples)

        return resampled

    def z_score_normalize(self, data: np.ndarray) -> np.ndarray:
        """
        Z-score normalization per channel (zero-mean, unit-variance).

        Args:
            data: np.array([N_samples, N_channels])

        Returns:
            Normalized np.array([N_samples, N_channels])
        """
        normalized = np.zeros_like(data)

        for ch in range(data.shape[1]):
            mean = np.mean(data[:, ch])
            std = np.std(data[:, ch])
            if std > 1e-6:
                normalized[:, ch] = (data[:, ch] - mean) / std
            else:
                normalized[:, ch] = data[:, ch]

        return normalized

    def create_windows(self, data: np.ndarray, window_size: int = 1024,
                      overlap: float = 0.5) -> Tuple[np.ndarray, np.ndarray]:
        """
        Create sliding windows from time-series data.

        Args:
            data: np.array([N_samples, N_channels])
            window_size: Samples per window (1024 = 1 sec @ 1 kHz)
            overlap: Overlap fraction (0.5 = 50%)

        Returns:
            (windows, window_indices)
        """
        stride = int(window_size * (1 - overlap))
        windows = []
        indices = []

        for start_idx in range(0, len(data) - window_size, stride):
            window = data[start_idx:start_idx + window_size]
            windows.append(window)
            indices.append(start_idx)

        return np.array(windows, dtype=np.float32), np.array(indices)

    def process_all_states(self, output_dir: Optional[Path] = None) -> Dict:
        """
        Full processing pipeline for all states: load → extract → resample → normalize → window.

        Args:
            output_dir: Optional output directory to save processed data

        Returns:
            Dictionary with processed windows and metadata
        """
        print(f"[HBTA] Starting full pipeline...")

        all_states_windows = {}
        all_states_metadata = {}

        for state_idx in range(self.NUM_STATES):
            print(f"\n[HBTA] Processing state {state_idx}/{self.NUM_STATES - 1}...")

            try:
                state_dict = self.load_state(state_idx)
            except KeyError:
                print(f"  Warning: State {state_idx} not found, skipping")
                continue

            raw_data = state_dict['data']
            print(f"  Loaded: {raw_data.shape} @ {self.ORIGINAL_SAMPLING_RATE} Hz")

            print(f"  Extracting canonical channels...")
            canonical = self.extract_canonical_channels(raw_data)

            print(f"  Resampling 100 Hz → 1000 Hz...")
            resampled = self.resample_to_1khz(canonical)

            print(f"  Z-score normalization...")
            normalized = self.z_score_normalize(resampled)

            print(f"  Creating 1-second windows...")
            windows, window_indices = self.create_windows(normalized, window_size=1024)
            print(f"    Created {len(windows)} windows")

            all_states_windows[state_idx] = windows
            all_states_metadata[state_idx] = {
                'state_label': state_idx,
                'raw_shape': raw_data.shape,
                'num_windows': len(windows),
                'window_indices': window_indices.tolist()
            }

        result = {
            'all_states_windows': all_states_windows,
            'all_states_metadata': all_states_metadata,
            'metadata': {
                'dataset': self.DATASET_NAME,
                'doi': self.DOI,
                'num_states': len(all_states_windows),
                'original_sampling_rate': self.ORIGINAL_SAMPLING_RATE,
                'target_sampling_rate': self.TARGET_SAMPLING_RATE,
                'channels': ['accel_x', 'accel_y', 'accel_z', 'strain', 'acoustic'],
                'missing_channels': ['acoustic'],
                'total_windows': sum(len(w) for w in all_states_windows.values()),
                'processed_timestamp': datetime.utcnow().isoformat()
            }
        }

        if output_dir:
            self._save_processed_data(result, output_dir)

        return result

    def _save_processed_data(self, result: Dict, output_dir: Path):
        """Save processed data to HDF5 and metadata to JSON."""
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        import h5py

        h5_file = output_dir / "hbta_processed.h5"
        print(f"\n[HBTA] Saving to {h5_file}...")

        with h5py.File(h5_file, 'w') as f:
            for state_idx, windows in result['all_states_windows'].items():
                f.create_dataset(f'state_{state_idx}', data=windows, compression='gzip')

        metadata_file = output_dir / "hbta_metadata.json"
        with open(metadata_file, 'w') as f:
            json.dump(result['metadata'], f, indent=2)

        print(f"  Saved processed states: {h5_file}")
        print(f"  Saved metadata: {metadata_file}")


# Example usage
if __name__ == '__main__':
    from pathlib import Path

    # Example (adjust path to actual data location)
    hbta_h5 = Path("./datasets/raw/hbta/data_100Hz.h5")
    loader = ZenodoHBTALoader(hbta_h5)

    result = loader.process_all_states(output_dir=Path("./datasets/normalized"))
    print(f"\n✓ Processed {result['metadata']['total_windows']} total windows across all states")
