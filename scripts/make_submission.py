"""Script to generate test predictions for Kaggle submission.

This loads a trained model, runs inference on `test.csv`, and
generates the `submission_pen.csv` file.
"""
import argparse
import sys
from pathlib import Path
import torch
import pandas as pd
from tqdm import tqdm
from torch.utils.data import DataLoader

# Add project root to sys.path to allow absolute imports from 'src'
root_dir = Path(__file__).resolve().parent.parent
if str(root_dir) not in sys.path:
    sys.path.append(str(root_dir))

from src.data.dataset import CircleDataset
from src.data.transforms import get_val_transform
from src.models.factory import create_model


def parse_args():
    parser = argparse.ArgumentParser(description="Generate Kaggle Submission")
    parser.add_argument("--test_csv", type=str, default="icdar-2026-circleid-pen-classification/test.csv")
    parser.add_argument("--image_dir", type=str, default="icdar-2026-circleid-pen-classification")
    parser.add_argument("--weights", type=str, default="weights/convnext_tiny_fold0.pth")
    parser.add_argument("--model_name", type=str, default="convnext_tiny")
    parser.add_argument("--output", type=str, default="submissions/submission_pen.csv")
    parser.add_argument("--batch_size", type=int, default=64)
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    # 1. Device selection
    if torch.backends.mps.is_available():
        device = torch.device("mps")
    elif torch.cuda.is_available():
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")
    print(f"Using device: {device}")

    # 2. Data Loading
    print(f"Loading test data from {args.test_csv}...")
    df = pd.read_csv(args.test_csv)
    
    # We pass dummy values for writer_id and pen_id to satisfy the CircleDataset requirements
    annotations = []
    for _, row in df.iterrows():
        annotations.append({
            "image_id": str(row["image_id"]),
            "image_path": str(Path(args.image_dir) / row["image_path"]),
            "writer_id": "dummy_writer",
            "pen_id": -1  # Dummy label
        })

    test_ds = CircleDataset(
        annotations=annotations,
        transform=get_val_transform(image_size=224),
        image_size=224,
    )
    
    test_dl = DataLoader(
        test_ds, 
        batch_size=args.batch_size, 
        shuffle=False, 
        num_workers=4,
        pin_memory=False # Avoid warnings on MPS
    )

    # 3. Model Setup
    print(f"Loading model '{args.model_name}' from {args.weights}...")
    model = create_model(model_name=args.model_name, pretrained=False)
    model.load_state_dict(torch.load(args.weights, map_location="cpu", weights_only=True))
    model.to(device)
    model.eval()

    # 4. Inference
    predictions = []
    image_ids = []

    print("Running inference...")
    # Inference handles AMP similarly to evaluate
    with torch.no_grad():
        for batch in tqdm(test_dl, desc="Predicting"):
            images = batch["image"].to(device)
            ids = batch["image_id"]
            
            if device.type in ('cuda', 'mps'):
                dtype = torch.bfloat16 if device.type == 'mps' else torch.float16
                with torch.autocast(device_type=device.type, dtype=dtype):
                    logits = model(images)
            else:
                logits = model(images)
                
            # Get Top-1 class
            preds = torch.argmax(logits, dim=1).cpu().numpy()
            
            predictions.extend(preds)
            image_ids.extend(ids)

    # 5. Save Submission
    submission_df = pd.DataFrame({
        "image_id": image_ids,
        "pen_id": predictions
    })

    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    submission_df.to_csv(out_path, index=False)
    
    print(f"\nSubmission saved to {out_path}")
    print(submission_df.head())


if __name__ == "__main__":
    main()
