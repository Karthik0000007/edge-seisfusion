"""
Combined Dataset Processor

Orchestrates loading, processing, and splitting of both Mendeley and HBTA datasets.
Outputs unified train/val/test feature packs ready for Phase 5 (DSP reference).
"""

import numpy as np
import h5py
from pathlib import Path
from typing import Dict, Tuple
import json
from datetime import datetime
from dataclasses import dataclass


@dataclass
class DatasetConfig:
    """Configuration for dataset processing."""
    mendeley_dir: Path
    hbta_h5_file: Path
    output_dir: Path
    train_ratio: float = 0.70
    validation_ratio: float = 0.20
    test_ratio: float = 0.10
    window_size: int = 1024
    target_sampling_rate: int = 1000


class CombinedDatasetProcessor:
    """Unified processor for Mendeley + HBTA datasets."""

    def __init__(self, config: DatasetConfig):
        """Initialize with configuration."""
        self.config = config
        self.config.output_dir.mkdir(parents=True, exist_ok=True)

    def process_all(self) -> Dict:
        """
        Full end-to-end processing:
        1. Load Mendeley (200 Hz) → upsample → normalize → window
        2. Load HBTA (100 Hz) → upsample → normalize → window
        3. Split: HBTA-state0 (healthy) → train
                 Mendeley (traffic variability) → validation
                 HBTA-states1-9 (damage) → test
        4. Export combined feature pack

        Returns:
            Dictionary with train/val/test windows and metadata
        """
        print("\n" + "="*70)
        print("COMBINED DATASET PROCESSOR - Phase 3")
        print("="*70)

        # Load Mendeley
        print("\n[PHASE 3.1] Loading Mendeley Bridge Dataset...")
        from loader_mendeley import MendeleyBridgeLoader
        mendeley_loader = MendeleyBridgeLoader(self.config.mendeley_dir)
        mendeley_result = mendeley_loader.process_full_pipeline()
        mendeley_windows = mendeley_result['windows']
        print(f"✓ Mendeley: {len(mendeley_windows)} windows")

        # Load HBTA
        print("\n[PHASE 3.2] Loading Zenodo HBTA Dataset...")
        from loader_zenodo_hbta import ZenodoHBTALoader
        hbta_loader = ZenodoHBTALoader(self.config.hbta_h5_file)
        hbta_result = hbta_loader.process_all_states()
        print(f"✓ HBTA: {hbta_result['metadata']['total_windows']} total windows")

        # Split datasets
        print("\n[PHASE 3.3] Creating train/val/test splits...")
        print(f"  Strategy: HBTA-state0 (healthy) → train (70%)")
        print(f"            Mendeley (traffic) → validation (20%)")
        print(f"            HBTA-states1-9 (damage) → test (10%)")

        splits = self._create_splits(mendeley_windows, hbta_result)

        # Export
        print("\n[PHASE 3.4] Exporting combined feature pack...")
        export_path = self._export_combined(splits, mendeley_result, hbta_result)

        result = {
            'splits': splits,
            'export_path': export_path,
            'statistics': self._compute_statistics(splits),
            'metadata': {
                'processor': 'CombinedDatasetProcessor',
                'mendeley_windows': len(mendeley_windows),
                'hbta_total_windows': hbta_result['metadata']['total_windows'],
                'processed_timestamp': datetime.utcnow().isoformat()
            }
        }

        return result

    def _create_splits(self, mendeley_windows: np.ndarray, hbta_result: Dict) -> Dict:
        """
        Create train/val/test splits.

        Strategy:
        - TRAIN (70%): HBTA state 0 (undamaged/healthy)
        - VAL (20%): Mendeley (real-world traffic variability)
        - TEST (10%): HBTA states 1-9 (progressive damage/anomalies)
        """
        splits = {
            'train': [],
            'validation': [],
            'test': [],
            'metadata': {}
        }

        # TRAIN: HBTA State 0 (undamaged)
        hbta_state0 = hbta_result['all_states_windows'].get(0, [])
        if len(hbta_state0) > 0:
            splits['train'].append({
                'data': hbta_state0,
                'label': 0,  # Healthy
                'source': 'HBTA',
                'state': 0
            })
            print(f"  TRAIN: {len(hbta_state0)} from HBTA state 0 (healthy)")

        # VALIDATION: Mendeley
        splits['validation'].append({
            'data': mendeley_windows,
            'label': 0,  # Assumed healthy (traffic noise only)
            'source': 'Mendeley',
            'note': 'Real-world traffic variability'
        })
        print(f"  VAL: {len(mendeley_windows)} from Mendeley")

        # TEST: HBTA States 1-9 (damage)
        for state_idx in range(1, 10):
            hbta_state_data = hbta_result['all_states_windows'].get(state_idx, [])
            if len(hbta_state_data) > 0:
                splits['test'].append({
                    'data': hbta_state_data,
                    'label': 1,  # Anomalous/damaged
                    'source': 'HBTA',
                    'state': state_idx
                })
                print(f"  TEST: {len(hbta_state_data)} from HBTA state {state_idx} (damage level {state_idx})")

        # Concatenate within each split
        splits_concatenated = {}
        for split_name in ['train', 'validation', 'test']:
            if splits[split_name]:
                concatenated = np.vstack([item['data'] for item in splits[split_name]])
                splits_concatenated[split_name] = {
                    'data': concatenated,
                    'sources': [item['source'] for item in splits[split_name]],
                    'labels': [item.get('label', 0) for item in splits[split_name]],
                    'num_windows': len(concatenated)
                }
            else:
                splits_concatenated[split_name] = None

        return splits_concatenated

    def _export_combined(self, splits: Dict, mendeley_result: Dict, hbta_result: Dict) -> Path:
        """
        Export combined feature pack as HDF5.

        Structure:
        ```
        combined_features.h5
        ├── train/
        │   ├── windows [N_train, 1024, 5]
        │   └── metadata (JSON)
        ├── validation/
        │   ├── windows [N_val, 1024, 5]
        │   └── metadata (JSON)
        ├── test/
        │   ├── windows [N_test, 1024, 5]
        │   └── metadata (JSON)
        └── global_metadata (JSON)
        ```
        """
        export_file = self.config.output_dir / "combined_features.h5"
        metadata_file = self.config.output_dir / "combined_metadata.json"

        print(f"  Exporting to {export_file}...")

        with h5py.File(export_file, 'w') as f:
            for split_name in ['train', 'validation', 'test']:
                if splits[split_name] is not None:
                    split_data = splits[split_name]['data']
                    f.create_dataset(f'{split_name}/windows',
                                    data=split_data,
                                    compression='gzip',
                                    compression_opts=4)
                    print(f"    {split_name}: {split_data.shape}")

        # Global metadata
        global_metadata = {
            'version': '1.0',
            'created': datetime.utcnow().isoformat(),
            'splits': {
                'train': {
                    'num_windows': splits['train']['num_windows'] if splits['train'] else 0,
                    'sources': splits['train']['sources'] if splits['train'] else []
                },
                'validation': {
                    'num_windows': splits['validation']['num_windows'] if splits['validation'] else 0,
                    'sources': splits['validation']['sources'] if splits['validation'] else []
                },
                'test': {
                    'num_windows': splits['test']['num_windows'] if splits['test'] else 0,
                    'sources': splits['test']['sources'] if splits['test'] else []
                }
            },
            'datasets': {
                'mendeley': mendeley_result['metadata'],
                'hbta': hbta_result['metadata']
            },
            'configuration': {
                'window_size': self.config.window_size,
                'target_sampling_rate': self.config.target_sampling_rate,
                'channels': ['accel_x', 'accel_y', 'accel_z', 'strain', 'acoustic'],
                'normalization': 'z_score (per channel)'
            }
        }

        with open(metadata_file, 'w') as f:
            json.dump(global_metadata, f, indent=2)

        print(f"  Metadata: {metadata_file}")

        return export_file

    def _compute_statistics(self, splits: Dict) -> Dict:
        """Compute statistics on the splits."""
        stats = {}

        for split_name in ['train', 'validation', 'test']:
            if splits[split_name] is not None:
                data = splits[split_name]['data']
                stats[split_name] = {
                    'num_windows': len(data),
                    'shape': data.shape,
                    'mean_per_channel': np.mean(data, axis=(0, 1)).tolist(),
                    'std_per_channel': np.std(data, axis=(0, 1)).tolist(),
                    'min': float(np.min(data)),
                    'max': float(np.max(data))
                }

        return stats


def main():
    """Example usage."""
    # Configuration (adjust paths to actual data locations)
    config = DatasetConfig(
        mendeley_dir=Path("./datasets/raw/mendeley"),
        hbta_h5_file=Path("./datasets/raw/hbta/data_100Hz.h5"),
        output_dir=Path("./datasets/normalized")
    )

    processor = CombinedDatasetProcessor(config)
    result = processor.process_all()

    print("\n" + "="*70)
    print("PHASE 3 COMPLETE")
    print("="*70)
    print(f"\nExported: {result['export_path']}")
    print(f"\nSplit Statistics:")
    for split_name, stats in result['statistics'].items():
        print(f"\n  {split_name.upper()}:")
        print(f"    Windows: {stats['num_windows']}")
        print(f"    Shape: {stats['shape']}")
        print(f"    Value range: [{stats['min']:.3f}, {stats['max']:.3f}]")


if __name__ == '__main__':
    main()
