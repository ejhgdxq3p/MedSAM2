import os
import sys
import hydra
from omegaconf import DictConfig
from training.trainer import Trainer
from training.utils.checkpoint_utils import load_checkpoint_and_apply_kernels
import torch

def train_single_class(cfg, class_id, checkpoint_path=None):
    """训练单个类别"""
    print(f"开始训练类别 {class_id}")
    
    # 设置当前类别ID
    cfg.trainer.data.train.datasets[0].dataset.datasets[0].video_dataset.current_class_id = class_id
    
    # 创建训练器
    trainer = Trainer(cfg.trainer)
    
    # 如果有checkpoint，加载它
    if checkpoint_path and os.path.exists(checkpoint_path):
        print(f"加载checkpoint: {checkpoint_path}")
        checkpoint = torch.load(checkpoint_path, map_location='cpu')
        trainer.model.load_state_dict(checkpoint['model'])
    
    # 开始训练
    trainer.train()
    
    # 保存checkpoint
    save_path = f"checkpoints/class_{class_id}_checkpoint.pt"
    trainer.save_checkpoint(save_path)
    
    return save_path

@hydra.main(version_base=None, config_path="../sam2/configs", config_name="sam2.1_hiera_tiny_finetune512")
def main(cfg: DictConfig):
    """顺序训练所有类别"""
    total_classes = 46
    checkpoint_path = None
    
    for class_id in range(1, total_classes + 1):
        print(f"\n{'='*50}")
        print(f"训练类别 {class_id}/{total_classes}")
        print(f"{'='*50}")
        
        try:
            checkpoint_path = train_single_class(cfg, class_id, checkpoint_path)
            print(f"类别 {class_id} 训练完成，checkpoint保存到: {checkpoint_path}")
        except Exception as e:
            print(f"训练类别 {class_id} 时出错: {e}")
            continue
    
    print("\n所有类别训练完成！")

if __name__ == "__main__":
    main() 