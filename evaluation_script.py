import os
import time
import argparse
import glob
import numpy as np
import cv2
import torch
from models.restoration_model import SemiconRestorationNet

def load_image_file(file_path):
    """Loads image from .npy array or standard image format (PNG, JPG, TIFF)."""
    ext = os.path.splitext(file_path)[1].lower()
    if ext == '.npy':
        img = np.load(file_path).astype(np.float32)
    else:
        # Load grayscale image and normalize to [0, 1]
        img = cv2.imread(file_path, cv2.IMREAD_GRAYSCALE)
        if img is None:
            raise ValueError(f"Could not read image file: {file_path}")
        img = img.astype(np.float32) / 255.0
    return img

def run_evaluation(input_path, output_dir, weights_path='weights/best_model.pth', device=None):
    """
    Standalone Evaluation Script for KLA Benchmarking Team.
    Accepts:
      - input_path: Path to directory containing test images OR path to a single image file (.npy, .png, .jpg)
      - output_dir: Path to directory where restored 256x256 .npy / image files will be saved
    """
    if device is None:
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    print(f"============================================================")
    print(f" KLA Semiconductor Image Restoration Evaluation")
    print(f" Input Path      : {input_path}")
    print(f" Output Directory: {output_dir}")
    print(f" Device          : {device}")
    print(f"============================================================")

    os.makedirs(output_dir, exist_ok=True)

    # Load Model Architecture & Weights
    model = SemiconRestorationNet().to(device)
    if os.path.exists(weights_path):
        model.load_state_dict(torch.load(weights_path, map_location=device))
        print(f"Loaded trained weights from: {weights_path}")
    else:
        fallback = 'weights/semicon_restoration_model.pth'
        if os.path.exists(fallback):
            model.load_state_dict(torch.load(fallback, map_location=device))
            print(f"Loaded trained weights from: {fallback}")
        else:
            raise FileNotFoundError(f"Weights file not found at {weights_path} or {fallback}")

    model.eval()

    # Handle Single File vs Directory
    if os.path.isfile(input_path):
        test_files = [input_path]
    elif os.path.isdir(input_path):
        test_files = sorted(glob.glob(os.path.join(input_path, '*.npy')))
        if len(test_files) == 0:
            # Check for PNG/JPG if no .npy found
            test_files = sorted(glob.glob(os.path.join(input_path, '*.png')) + glob.glob(os.path.join(input_path, '*.jpg')))
    else:
        raise FileNotFoundError(f"Input path does not exist: {input_path}")

    if len(test_files) == 0:
        print(f"Warning: No valid .npy or image files found in {input_path}")
        return

    print(f"Processing {len(test_files)} file(s)...")
    start_time = time.time()

    with torch.no_grad():
        for file_path in test_files:
            # Load degraded image (.npy or PNG/JPG, float32)
            img = load_image_file(file_path)
            
            # Prepare tensor (1, 1, H, W)
            input_tensor = torch.from_numpy(img).unsqueeze(0).unsqueeze(0).to(device)

            # Model Forward Pass -> Restored 256x256 tensor
            output_tensor = model(input_tensor)

            # Convert back to numpy array (256, 256) bounded [0.0, 1.0]
            restored_img = output_tensor.cpu().numpy().squeeze(0).squeeze(0)

            # Save restored .npy file to output directory
            filename = os.path.basename(file_path)
            base_name, ext = os.path.splitext(filename)

            # Always save .npy file for benchmarking
            out_npy_path = os.path.join(output_dir, f"{base_name}.npy")
            np.save(out_npy_path, restored_img)

            # Also save PNG visual if single file or if PNG input
            if len(test_files) == 1 or ext in ['.png', '.jpg']:
                out_png_path = os.path.join(output_dir, f"{base_name}_restored.png")
                img_uint8 = (np.clip(restored_img, 0.0, 1.0) * 255.0).astype(np.uint8)
                cv2.imwrite(out_png_path, img_uint8)

    total_time = time.time() - start_time
    avg_fps = len(test_files) / total_time
    avg_latency = (total_time / len(test_files)) * 1000

    print(f"============================================================")
    print(f" Restored {len(test_files)} image(s) in {total_time:.3f} seconds.")
    print(f" Average Speed: {avg_fps:.1f} FPS ({avg_latency:.2f} ms per image)")
    print(f" Outputs saved to: {output_dir}")
    print(f"============================================================")

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Semiconductor Image Restoration Evaluation Script')
    parser.add_argument('--input_dir', type=str, help='Path to input test directory OR single image file (.npy, .png)')
    parser.add_argument('--input_file', type=str, help='Path to a single input image file (.npy, .png, .jpg)')
    parser.add_argument('--output_dir', type=str, default='restored_test_outputs', help='Path to output directory')
    parser.add_argument('--weights', type=str, default='weights/best_model.pth', help='Path to trained PyTorch weights file')

    args = parser.parse_args()
    
    # Priority: --input_file if provided, else --input_dir, default to 'dataset/NoisyLR'
    target_input = args.input_file or args.input_dir or 'dataset/NoisyLR'
    run_evaluation(target_input, args.output_dir, args.weights)
