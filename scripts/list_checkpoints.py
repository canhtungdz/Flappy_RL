import os
import sys
from datetime import datetime

ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT_DIR not in sys.path:
    sys.path.append(ROOT_DIR)


def list_checkpoints(checkpoint_dir="checkpoints"):
    """Liệt kê tất cả checkpoints có sẵn."""
    if not os.path.exists(checkpoint_dir):
        print(f"Checkpoint directory not found: {checkpoint_dir}")
        return
    
    checkpoints = []
    for file in os.listdir(checkpoint_dir):
        if file.endswith('.pth') and not file.endswith('_stats.pth'):
            path = os.path.join(checkpoint_dir, file)
            size = os.path.getsize(path)
            mtime = os.path.getmtime(path)
            
            # Check if stats file exists
            stats_path = path.replace('.pth', '_stats.pth')
            has_stats = os.path.exists(stats_path)
            
            checkpoints.append({
                'name': file,
                'path': path,
                'size': size,
                'modified': datetime.fromtimestamp(mtime),
                'has_stats': has_stats
            })
    
    if not checkpoints:
        print("No checkpoints found.")
        return
    
    # Sort by modified time
    checkpoints.sort(key=lambda x: x['modified'], reverse=True)
    
    print("=" * 90)
    print("Available Checkpoints")
    print("=" * 90)
    print(f"{'Filename':<30} {'Size (KB)':<12} {'Stats':<8} {'Modified':<20}")
    print("-" * 90)
    
    for cp in checkpoints:
        size_kb = cp['size'] / 1024
        modified_str = cp['modified'].strftime('%Y-%m-%d %H:%M:%S')
        stats_str = "✓" if cp['has_stats'] else "✗"
        print(f"{cp['name']:<30} {size_kb:<12.1f} {stats_str:<8} {modified_str:<20}")
    
    print("=" * 90)
    print(f"\nTotal: {len(checkpoints)} checkpoints")
    print("\n📝 To resume training from a checkpoint:")
    print("   python3 scripts/train_dqn.py --resume checkpoints/dqn_episode_5000.pth --episodes 10000")
    print("\n🎮 To test a checkpoint:")
    print("   python3 scripts/evaluate_dqn.py --checkpoint checkpoints/best_model.pth --episodes 10")


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="List available checkpoints")
    parser.add_argument("--checkpoint-dir", type=str, default="checkpoints",
                       help="Checkpoint directory")
    args = parser.parse_args()
    
    list_checkpoints(args.checkpoint_dir)
