"""
Mendeley Bridge Vibration Dataset Loader

Dataset: "Bridge vibration monitoring dataset" (Version 1)
Source: Mendeley Data, DOI: 10.17632/d3by55pjh7.1
Authors: Premjeet Singh & Ayan Sadhu (Western University)

Loads plain-text accelerometer data from 8 test files (test1.txt - test8.txt)
and normalizes to canonical 1 kHz schema.
"""

import numpy as np
import pandas as pd
from pathlib import Path
from typing import Dict, Tuple, Optional
import json
from datetime import datetime


class MendeleyBridgeLoader:
    """Load and preprocess Mendeley bridge vibration dataset."""

    # Dataset metadata
    DATASET_NAME = "Bridge Vibration Monitoring (Mendeley)"
    DOI = "10.17632/d3by55pjh7.1"
    AUTHORS = "Premjeet Singh & Ayan Sadhu (Western University)"
    ORIGINAL_SAMPLING_RATE = 200  # Hz
    TARGET_SAMPLING_RATE = 1000   # Hz
    NUM_CHANNELS = 5              # Uniaxial accelerometers
    NUM_TESTS = 8

    def __init__(self, data_dir: Path):
        """
        Initialize loader.

        Args:
            data_dir: Directory containing test1.txt - test8.txt files
        """
        self.data_dir = Path(data_dir)
        self._validate_data_directory()

    def _validate_data_directory(self):
        """Check that all required test files exist."""
        for i in range(1, self.NUM_TESTS + 1):
            test_file = self.data_dir / f"test{i}.txt"
            if not test_file.exists():
                raise FileNotFoundError(f"Missing {test_file}")

    def load_single_test(self, test_index: int) -> Dict:
        """
        Load a single test file (test1.txt - test8.txt).

        Args:
            test_index: Test number (1-8)

        Returns:
            Dictionary with:
                - 'data': np.array([N_samples, 5_channels])
                - 'sampling_rate': 200 (original)
                - 'test_number': test_index
                - 'metadata': dict with provenance
        """
        if not (1 <= test_index <= self.NUM_TESTS):
            raise ValueError(f"test_index must be 1-{self.NUM_TESTS}")

        test_file = self.data_dir / f"test{test_index}.txt"

        # Load as CSV (space or comma separated)
        try:
            df = pd.read_csv(test_file, sep=None, engine='python', header=None)
        except Exception as e:
            raise ValueError(f"Failed to parse {test_file}: {e}")

        # Extract acceleration data (5 columns)
        if df.shape[1] < 5:
            raise ValueError(f"Expected >= 5 columns, got {df.shape[1]}")

        accel_data = df.iloc[:, :5].values.astype(np.float32)

        return {
            'data': accel_data,
            'sampling_rate': self.ORIGINAL_SAMPLING_RATE,
            'test_number': test_index,
            'num_samples': len(accel_data),
            'metadata': {
                'source': 'mendeley',
                'doi': self.DOI,
                'authors': self.AUTHORS,
                'test_index': test_index,
                'original_sampling_rate': self.ORIGINAL_SAMPLING_RATE,
                'channels': ['accel_x', 'accel_y', 'accel_z', 'accel_vert', 'accel_vert2'],
                'missing_channels': ['strain', 'acoustic'],
                'structure': 'real steel truss bridge (in-service)',
                'condition': 'traffic loading (undamaged baseline)'
            }
        }

    def load_all_tests(self) -> Tuple[np.ndarray, Dict]:
        """
        Load all 8 test files and concatenate.

        Returns:
            (concatenated_data, metadata_list)
            - concatenated_data: np.array([total_samples, 5])
            - metadata_list: List of metadata dicts for each test
        """
        all_data = []
        metadata_list = []

        for test_idx in range(1, self.NUM_TESTS + 1):
            test_dict = self.load_single_test(test_idx)
            all_data.append(test_dict['data'])
            metadata_list.append(test_dict['metadata'])

        concatenated = np.vstack(all_data)

        return concatenated, metadata_list

    def resample_to_1khz(self, data: np.ndarray) -> np.ndarray:
        """
        Upsample from 200 Hz to 1 kHz (5× interpolation).

        Args:
            data: np.array([N_samples, 5_channels])

        Returns:
            np.array([N_samples * 5, 5_channels])
        """
        from scipy import signal

        # Resample each channel independently
        num_samples, num_channels = data.shape
        target_num_samples = num_samples * 5

        resampled = np.zeros((target_num_samples, num_channels), dtype=np.float32)

        for ch in range(num_channels):
            resampled[:, ch] = signal.resample(data[:, ch], target_num_samples)

        return resampled

    def normalize_to_canonical(self, data: np.ndarray) -> np.ndarray:
        """
        Map 5 Mendeley channels to canonical [X, Y, Z, Strain, Acoustic].

        Mendeley provides: [accel_1, accel_2, accel_3, accel_4, accel_5]
        Canonical schema: [accel_x, accel_y, accel_z, strain, acoustic]

        Action: First 3 → X/Y/Z, channels 4-5 → padding
                Strain/Acoustic filled with zeros (+ flag)

        Args:
            data: np.array([N_samples, 5])

        Returns:
            Canonical np.array([N_samples, 5])
        """
        # Mendeley already has 5 channels, map directly
        # (In real case, might reorder or select subsets)
        canonical = np.zeros_like(data)
        canonical[:, 0:3] = data[:, 0:3]  # Accel X/Y/Z
        canonical[:, 3] = 0.0              # Strain (not available)
        canonical[:, 4] = 0.0              # Acoustic (not available)

        return canonical

    def z_score_normalize(self, data: np.ndarray) -> np.ndarray:
        """
        Z-score normalization per channel (zero-mean, unit-variance).

        Args:
            data: np.array([N_samples, 5])

        Returns:
            Normalized np.array([N_samples, 5])
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
            data: np.array([N_samples, 5])
            window_size: Samples per window (1024 = 1 sec @ 1 kHz)
            overlap: Overlap fraction (0.5 = 50%)

        Returns:
            (windows, window_indices)
            - windows: np.array([N_windows, window_size, 5])
            - window_indices: Starting sample index for each window
        """
        stride = int(window_size * (1 - overlap))
        windows = []
        indices = []

        for start_idx in range(0, len(data) - window_size, stride):
            window = data[start_idx:start_idx + window_size]
            windows.append(window)
            indices.append(start_idx)

        return np.array(windows, dtype=np.float32), np.array(indices)

    def process_full_pipeline(self, output_dir: Optional[Path] = None) -> Dict:
        """
        Full processing pipeline: load → resample → normalize → window.

        Args:
            output_dir: Optional output directory to save processed data

        Returns:
            Dictionary with:
                - 'windows': np.array([N_windows, 1024, 5])
                - 'metadata': Processing metadata
        """
        print(f"[Mendeley] Loading all {self.NUM_TESTS} test files...")
        all_data, test_metadata = self.load_all_tests()
        print(f"  Loaded: {all_data.shape} @ {self.ORIGINAL_SAMPLING_RATE} Hz")

        print(f"[Mendeley] Resampling 200 Hz → 1000 Hz...")
        resampled = self.resample_to_1khz(all_data)
        print(f"  Resampled: {resampled.shape}")

        print(f"[Mendeley] Mapping to canonical schema...")
        canonical = self.normalize_to_canonical(resampled)

        print(f"[Mendeley] Z-score normalization...")
        normalized = self.z_score_normalize(canonical)

        print(f"[Mendeley] Creating 1-second windows...")
        windows, window_indices = self.create_windows(normalized, window_size=1024)
        print(f"  Created {len(windows)} windows")

        result = {
            'windows': windows,
            'window_indices': window_indices,
            'original_data': all_data,
            'resampled_data': resampled,
            'normalized_data': normalized,
            'metadata': {
                'dataset': self.DATASET_NAME,
                'doi': self.DOI,
                'num_tests': self.NUM_TESTS,
                'original_sampling_rate': self.ORIGINAL_SAMPLING_RATE,
                'target_sampling_rate': self.TARGET_SAMPLING_RATE,
                'total_samples_original': len(all_data),
                'total_samples_resampled': len(resampled),
                'num_windows': len(windows),
                'window_size': 1024,
                'window_duration_sec': 1.0,
                'channels': ['accel_x', 'accel_y', 'accel_z', 'strain', 'acoustic'],
                'missing_channels': ['strain', 'acoustic'],
                'processed_timestamp': datetime.utcnow().isoformat(),
                'test_metadata': test_metadata
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

        h5_file = output_dir / "mendeley_processed.h5"
        print(f"[Mendeley] Saving to {h5_file}...")

        with h5py.File(h5_file, 'w') as f:
            f.create_dataset('windows', data=result['windows'], compression='gzip')
            f.create_dataset('original_data', data=result['original_data'], compression='gzip')
            f.create_dataset('window_indices', data=result['window_indices'])

        metadata_file = output_dir / "mendeley_metadata.json"
        with open(metadata_file, 'w') as f:
            json.dump(result['metadata'], f, indent=2)

        print(f"  Saved windows: {h5_file}")
        print(f"  Saved metadata: {metadata_file}")


# Example usage
if __name__ == '__main__':
    # Example (adjust path to actual data location)
    from pathlib import Path

    mendeley_dir = Path("./datasets/raw/mendeley")
    loader = MendeleyBridgeLoader(mendeley_dir)

    result = loader.process_full_pipeline(output_dir=Path("./datasets/normalized"))
    print(f"\n✓ Processed {result['metadata']['num_windows']} windows")
