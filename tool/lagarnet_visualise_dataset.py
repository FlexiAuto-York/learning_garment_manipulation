"""
Actoris-Harena DataLoader Augmentation Visualizer (Multi-Batch 5x5 Master Grid).

This script fetches multiple training batches, applies augmentations, and outputs a 
single master grid. It processes 5 batches of size 5, arranging 25 samples into 5 rows. 
Each sample is a 2x2 block showing:
  [ Orig RGB ] [ Orig Goal ]
  [ Aug RGB  ] [ Aug Goal  ]
"""

import os
import argparse
import copy
import yaml
import numpy as np
import torch
import cv2
from dotmap import DotMap

# Import your dataset and augmentor
from actoris_harena.utilities.trajectory_dataset import TrajectoryDataset
from data_augmentation.pixel_based_multi_primitive_data_augmenter_for_diffusion import PixelBasedMultiPrimitiveDataAugmenterForDiffusion

# Import your existing drawing utilities
from tool.magpie.visualise_data import (
    format_image, draw_keypoints, decode_action_vector, 
    draw_action, draw_text_with_bg, TEXT_Y_STEP
)

# ==========================================
# Hardcoded Dataset Configuration
# ==========================================
DATASET_CONFIG = {
    "data_path": "multi_longsleeve_multi_primitive_alignment_human_demo",
    "data_dir": "./data/datasets",
    "split_ratios": [0.0, 0.05, 0.95],
    "seq_length": 1,
    "io_mode": "r", 
    "cache_in_memory": True,
    "obs_config": {
        "mask": {"shape": [128, 128, 1], "output_key": "mask"},
        "depth": {"shape": [128, 128, 1], "output_key": "depth"},
        "rgb": {"shape": [128, 128, 3], "output_key": "rgb"},
        "semkey_norm_pixel": {"shape": [15, 2], "output_key": "semkey_norm_pixel"},
        "goal_rgb": {"shape": [128, 128, 3], "output_key": "goal_rgb"},
        "goal_depth": {"shape": [128, 128, 1], "output_key": "goal_depth"},
        "goal_mask": {"shape": [128, 128, 1], "output_key": "goal_mask"},
        "flattened_semkey_norm_pixel": {"shape": [15, 2], "output_key": "flattened_goal_semkey_norm_pixel"}
    },
    "act_config": {
        "default": {"shape": [9], "output_key": "default"}
    }
}

def unformat_tensor_image(img_tensor: torch.Tensor, is_normalized: bool = False) -> np.ndarray:
    """
    Converts a PyTorch image tensor (C, H, W) back to a numpy array (H, W, C).
    """
    img_np = img_tensor.detach().cpu().numpy()
    if img_np.shape[0] == 3:  
        img_np = np.transpose(img_np, (1, 2, 0))
        
    if is_normalized or img_np.max() <= 1.0:
        img_np = (img_np * 255.0)
        
    return img_np.clip(0, 255).astype(np.uint8)

def main(args):
    # 1. Load Augmentor Configuration via DotMap
    print(f"Loading Augmentor Config from: {args.aug_config_path}")
    with open(args.aug_config_path, 'r') as f:
        raw_aug_config = yaml.safe_load(f)
    aug_config = DotMap(raw_aug_config)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # 2. Initialize Dataset and DataLoader
    print("Initializing TrajectoryDataset...")
    dataset = TrajectoryDataset(**DATASET_CONFIG, sample_mode='train')
    
    dataloader = torch.utils.data.DataLoader(
        dataset, 
        batch_size=args.batch_size, 
        shuffle=True, 
        num_workers=0 
    )

    print("Initializing Augmentor...")
    augmentor = PixelBasedMultiPrimitiveDataAugmenterForDiffusion(config=aug_config)

    # 3. Process Multiple Batches
    img_size = 512
    os.makedirs(os.path.dirname(args.output_path), exist_ok=True)
    
    all_comp_blocks = []
    dataloader_iter = iter(dataloader)
    total_samples = 0

    for batch_idx in range(args.num_batches):
        print(f"\n--- Fetching Batch {batch_idx + 1}/{args.num_batches} ---")
        try:
            raw_batch = next(dataloader_iter)
        except StopIteration:
            print("DataLoader ran out of data!")
            break
        
        # 4. Replicate MagpieTrainer Batch Restructuring
        obs = raw_batch['observation']
        action = raw_batch['action']['default']
        
        formatted_batch = {k: v for k, v in obs.items()}
        formatted_batch['action'] = action.reshape(*action.shape[:2], -1)

        orig_batch = copy.deepcopy(formatted_batch)

        # 5. Apply Augmentations
        print("Applying Augmentations...")
        aug_batch = augmentor(formatted_batch, train=True, device=device)

        # 6. Visualization Generation Loop for Current Batch
        B = orig_batch['rgb'].shape[0]
        
        for b in range(B):
            total_samples += 1
            print(f"Processing Sample {total_samples} (Batch {batch_idx + 1}, Item {b+1}/{B})...")
            
            t = 0 # Since seq_length=1, there is only 1 action transition per sample
            
            # --- Extract Original ---
            orig_rgb_raw = unformat_tensor_image(orig_batch['rgb'][b, t], is_normalized=False)
            orig_goal_raw = unformat_tensor_image(orig_batch['goal_rgb'][b, t], is_normalized=False)
            
            img_cur_orig = format_image(orig_rgb_raw, img_size)
            img_goal_orig = format_image(orig_goal_raw, img_size)
            
            orig_semkey = orig_batch['semkey_norm_pixel'][b, t].numpy()
            goal_key = 'flattened_goal_semkey_norm_pixel' if 'flattened_goal_semkey_norm_pixel' in orig_batch else 'flattened_semkey_norm_pixel'
            orig_goal_semkey = orig_batch[goal_key][b, t].numpy()
            orig_act = orig_batch['action'][b, t].numpy()

            draw_keypoints(img_cur_orig, orig_semkey)
            draw_keypoints(img_goal_orig, orig_goal_semkey)
            prim_name_orig, params_orig = decode_action_vector(orig_act)
            draw_action(img_cur_orig, prim_name_orig, params_orig)
            
            draw_text_with_bg(img_cur_orig, "Orig RGB", (10, TEXT_Y_STEP), (255, 255, 255))
            draw_text_with_bg(img_goal_orig, "Orig Goal", (10, TEXT_Y_STEP), (200, 200, 200))

            # --- Extract Augmented ---
            aug_rgb_raw = unformat_tensor_image(aug_batch['rgb'][b, t], is_normalized=True)
            aug_goal_raw = unformat_tensor_image(aug_batch['goal_rgb'][b, t], is_normalized=True)
            
            img_cur_aug = format_image(aug_rgb_raw, img_size)
            img_goal_aug = format_image(aug_goal_raw, img_size)
            
            aug_semkey = aug_batch['semkey_norm_pixel'][b, t].detach().cpu().numpy()
            aug_goal_semkey = aug_batch[goal_key][b, t].detach().cpu().numpy()
            aug_act = aug_batch['action'][b, t].detach().cpu().numpy()

            draw_keypoints(img_cur_aug, aug_semkey)
            draw_keypoints(img_goal_aug, aug_goal_semkey)
            prim_name_aug, params_aug = decode_action_vector(aug_act)
            draw_action(img_cur_aug, prim_name_aug, params_aug)
            
            draw_text_with_bg(img_cur_aug, "Aug RGB", (10, TEXT_Y_STEP), (50, 255, 50))
            draw_text_with_bg(img_goal_aug, "Aug Goal", (10, TEXT_Y_STEP), (50, 255, 50))

            # --- Assemble 2x2 Block ---
            top_row = cv2.hconcat([img_cur_orig, img_goal_orig])
            bottom_row = cv2.hconcat([img_cur_aug, img_goal_aug])
            comp_block = cv2.vconcat([top_row, bottom_row])
            
            # Add Sample Banner
            banner_height = 50
            banner = np.zeros((banner_height, comp_block.shape[1], 3), dtype=np.uint8)
            cv2.putText(
                banner, 
                f"Sample {total_samples}", 
                (20, 35), 
                cv2.FONT_HERSHEY_SIMPLEX, 
                1.2, 
                (255, 255, 0), 
                3, 
                cv2.LINE_AA
            )
            sample_with_banner = cv2.vconcat([banner, comp_block])
            
            # Add a visual border to separate samples clearly
            border_size = 10
            sample_bordered = cv2.copyMakeBorder(
                sample_with_banner, border_size, border_size, border_size, border_size, 
                cv2.BORDER_CONSTANT, value=[100, 100, 100]
            )
            
            all_comp_blocks.append(sample_bordered)

    # 7. Final Master Concatenation (Stitching the accumulated batches)
    if all_comp_blocks:
        print("\nStitching final multi-batch master grid...")
        cols_per_row = 5 # Force a 5-column layout
        master_rows = []
        
        for i in range(0, len(all_comp_blocks), cols_per_row):
            row_blocks = all_comp_blocks[i : i + cols_per_row]
            
            # Pad with blank images if the final row has fewer samples
            while len(row_blocks) < cols_per_row:
                blank = np.zeros_like(all_comp_blocks[0])
                row_blocks.append(blank)
                
            row_img = cv2.hconcat(row_blocks)
            master_rows.append(row_img)
            
        master_grid = cv2.vconcat(master_rows)
        
        output_file = f"{os.path.splitext(args.output_path)[0]}_master_multi_batch.png"
        cv2.imwrite(output_file, master_grid)
        print(f"-> Successfully saved Master Comparison Grid to: {output_file}")
    else:
        print("-> No valid samples to visualize.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Visualize DataLoader Output and Augmentations across multiple batches.")
    parser.add_argument('--aug_config_path', type=str, required=True, help="Path to the augmentor YAML config file")
    parser.add_argument('--batch_size', type=int, default=5, help="Batch size to request from DataLoader")
    parser.add_argument('--num_batches', type=int, default=5, help="Number of batches to sample and stack")
    parser.add_argument('--output_path', type=str, default='./tmp/dataloader_augment.png')
    args = parser.parse_args()
    
    main(args)