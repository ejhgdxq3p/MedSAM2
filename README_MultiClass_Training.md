# MedSAM2 多类别训练策略

## 概述

本项目实现了MedSAM2的多类别训练策略，支持从nifti格式的医学图像数据中训练模型，能够区分46个不同的医学图像类别。每次训练时随机选择一个类别作为前景，其他类别作为背景，从而提高模型的泛化能力。

## 训练策略

### 核心思想
- **多类别模式**：每次训练时将一个类别作为前景，其他类别作为背景
- **随机类别选择**：每次加载数据时随机选择一个类别作为目标
- **动态类别轮换**：通过随机选择确保所有类别都能被充分训练
- **Mask-only Prompt模式**：只使用mask作为prompt，不使用box和click输入

### 优势
1. **提高泛化能力**：模型学会区分不同类别
2. **避免过拟合**：防止模型只学会识别单一类别
3. **更接近实际应用**：模拟真实场景中的多类别分割需求
4. **简化训练流程**：只使用mask prompt，减少训练复杂度
5. **提高训练效率**：避免point/box采样开销，专注于mask学习

## 文件结构

```
MedSAM2/
├── sam2/configs/
│   └── sam2.1_hiera_tiny_finetune512.yaml    # 主配置文件
├── training/dataset/
│   ├── vos_raw_dataset.py                     # 数据集加载器
│   └── vos_segment_loader.py                  # 分割标签加载器
└── README_MultiClass_Training.md              # 本文档
```

## 关键文件详解

### 1. 配置文件: `sam2/configs/sam2.1_hiera_tiny_finetune512.yaml`

#### 关键配置项
```yaml
# 数据路径配置
dataset:
  folder: /root/autodl-fs/medsam2/meddata    # 数据根目录

# 训练器配置
trainer:
  _target_: training.trainer.Trainer
  mode: train_only
  max_epochs: ${times:${scratch.num_epochs},${scratch.phases_per_epoch}}
  accelerator: cuda
  seed_value: 123

# 数据集配置
data:
  train:
    datasets:
      - video_dataset:
          _target_: training.dataset.vos_raw_dataset.NiftiRawDataset
          img_folder: /root/autodl-fs/medsam2/meddata/ImagesTr  # 图像文件夹
          gt_folder: /root/autodl-fs/medsam2/meddata/LabelsTr   # 标签文件夹
          normalize: true
          lower_bound: -1000    # CT HU值下限
          upper_bound: 1000     # CT HU值上限
          multi_class_mode: true  # 启用多类别模式
          target_class_id: null   # 动态设置目标类别

# Prompt模式配置
model:
  # mask-only input (no box/point prompts)
  prob_to_use_pt_input_for_train: 0.0  # 禁用训练时的point输入
  prob_to_use_pt_input_for_eval: 0.0   # 禁用评估时的point输入
  prob_to_use_box_input_for_train: 0.0 # 禁用训练时的box输入
  prob_to_use_box_input_for_eval: 0.0  # 禁用评估时的box输入
  num_init_cond_frames_for_train: 1    # 只使用第一帧作为初始条件帧
  rand_init_cond_frames_for_train: False  # 总是使用第一帧

# 数据集配置
data:
  train:
    datasets:
      - video_dataset:
          include_class_id: true  # 启用ID-aware模式，将ID和mask一起作为prompt
```

### 2. 数据集加载器: `training/dataset/vos_raw_dataset.py`

#### NiftiRawDataset 类
```python
class NiftiRawDataset(VOSRawDataset):
    def __init__(
        self,
        img_folder,  # ImagesTr folder
        gt_folder,   # labelsTr folder
        file_list_txt=None,
        excluded_videos_list_txt=None,
        sample_rate=1,
        truncate_video=-1,
        normalize=True,
        lower_bound=None,
        upper_bound=None,
        multi_class_mode=True,
        target_class_id=None,
    ):
        # 初始化参数
        self.img_folder = img_folder
        self.gt_folder = gt_folder
        self.multi_class_mode = multi_class_mode
        self.target_class_id = target_class_id
        
        # 处理文件名（移除_0000后缀）
        for file in os.listdir(self.img_folder):
            if file.endswith('.nii.gz'):
                base_name = os.path.splitext(os.path.splitext(file)[0])[0]
                if base_name.endswith('_0000'):
                    base_name = base_name[:-5]
                subset.append(base_name)
```

#### 关键方法: get_video()
```python
def get_video(self, idx):
    video_name = self.video_names[idx]
    
    # 加载图像和标签文件
    img_path = os.path.join(self.img_folder, f"{video_name}_0000.nii.gz")
    gt_path = os.path.join(self.gt_folder, f"{video_name}.nii.gz")
    
    # 使用SimpleITK加载nifti文件
    import SimpleITK as sitk
    nii_image = sitk.ReadImage(img_path)
    nii_image_data = sitk.GetArrayFromImage(nii_image)
    nii_gt = sitk.ReadImage(gt_path)
    nii_gt_data = sitk.GetArrayFromImage(nii_gt)
    
    # 图像预处理和归一化
    if self.normalize:
        if self.lower_bound is not None and self.upper_bound is not None:
            nii_image_data = np.clip(nii_image_data, self.lower_bound, self.upper_bound)
            nii_image_data = (nii_image_data - np.min(nii_image_data)) / (np.max(nii_image_data) - np.min(nii_image_data)) * 255.0
        nii_image_data = np.uint8(nii_image_data)
    
    # 转换为RGB格式
    frames = np.repeat(nii_image_data[:, np.newaxis, :, :], 3, axis=1)
    
    # 创建VOSVideo对象
    video = VOSVideo(video_name, idx, vos_frames)
    
    # 创建NiftiSegmentLoader
    segment_loader = NiftiSegmentLoader(
        nii_gt_data[::self.sample_rate], 
        target_class_id=self.target_class_id,
        multi_class_mode=self.multi_class_mode
    )
    
    return video, segment_loader
```

### 3. 分割标签加载器: `training/dataset/vos_segment_loader.py`

#### NiftiSegmentLoader 类
```python
class NiftiSegmentLoader:
    def __init__(self, masks, target_class_id=None, multi_class_mode=True):
        """
        初始化NiftiSegmentLoader
        
        Args:
            masks (numpy.ndarray): 形状为(D, H, W)的掩码数组
            target_class_id (int, optional): 训练时关注的目标类别ID
            multi_class_mode (bool): 是否启用多类别模式
        """
        self.masks = masks
        self.target_class_id = target_class_id
        self.multi_class_mode = multi_class_mode

    def load(self, frame_idx):
        """
        加载指定帧的掩码并转换为二进制分割
        
        Args:
            frame_idx (int): 帧索引
            
        Returns:
            dict: 键为对象ID，值为二进制掩码的字典
        """
        mask = self.masks[frame_idx]
        
        # 找到所有非零的对象ID
        object_ids = np.unique(mask)
        object_ids = object_ids[object_ids != 0]
        
        if len(object_ids) == 0:
            return {}
        
        if self.multi_class_mode:
            # 多类别模式：将一个类别作为前景，其他作为背景
            if self.target_class_id is not None:
                target_id = self.target_class_id
            else:
                # 随机选择一个类别作为目标
                target_id = np.random.choice(object_ids)
            
            # 创建二进制掩码：目标类别=1，其他=0
            binary_mask = (mask == target_id)
            return {1: torch.from_numpy(binary_mask).bool()}
        else:
            # 单类别模式：每个类别都作为独立对象
            binary_segments = {}
            for obj_id in object_ids:
                binary_mask = (mask == obj_id)
                binary_segments[int(obj_id)] = torch.from_numpy(binary_mask).bool()
            return binary_segments

    def set_target_class(self, target_class_id):
        """设置目标类别ID"""
        self.target_class_id = target_class_id
```

## 数据格式要求

### 文件命名规则
- **图像文件**: `case_001_0000.nii.gz` (在ImagesTr文件夹中)
- **标签文件**: `case_001.nii.gz` (在labelsTr文件夹中)
- **对应关系**: 图像和标签文件的基础名称相同，图像文件多一个`_0000`后缀

### 文件夹结构
```
/root/autodl-fs/medsam2/meddata/
├── ImagesTr/
│   ├── case_001_0000.nii.gz
│   ├── case_002_0000.nii.gz
│   └── ...
└── labelsTr/
    ├── case_001.nii.gz
    ├── case_002.nii.gz
    └── ...
```

### 数据格式
- **图像数据**: 3D nifti格式，支持CT、MR、PET等医学图像
- **标签数据**: 3D nifti格式，每个体素值为对应的类别ID (1-46)
- **背景**: 标签值为0表示背景

## 使用方法

### 1. 环境准备
```bash
# 安装依赖
pip install SimpleITK>=2.4.0
pip install torch>=2.5.1
pip install hydra-core>=1.3.2
```

### 2. 数据准备
```bash
# 将数据放置在指定目录
mkdir -p /root/autodl-fs/medsam2/meddata/ImagesTr
mkdir -p /root/autodl-fs/medsam2/meddata/labelsTr

# 复制数据文件
cp your_images/*.nii.gz /root/autodl-fs/medsam2/meddata/ImagesTr/
cp your_labels/*.nii.gz /root/autodl-fs/medsam2/meddata/labelsTr/
```

### 3. 开始训练
```bash
cd /root/autodl-fs/medsam2

# 使用脚本训练
./single_node_train.sh

# 或直接运行
python training/train.py \
    -c sam2/configs/sam2.1_hiera_tiny_finetune512.yaml \
    --output-path ./exp_log/MedSAM2_TF \
    --use-cluster 0 \
    --num-gpus 1 \
    --num-nodes 1
```

## 训练参数说明

### 关键参数
- `multi_class_mode: true` - 启用多类别训练模式
- `target_class_id: null` - 动态随机选择目标类别
- `lower_bound: -1000` - CT图像HU值下限（可根据数据类型调整）
- `upper_bound: 1000` - CT图像HU值上限（可根据数据类型调整）
- `normalize: true` - 启用图像归一化

### Prompt模式参数
- `prob_to_use_pt_input_for_train: 0.0` - 禁用训练时的point输入
- `prob_to_use_pt_input_for_eval: 0.0` - 禁用评估时的point输入
- `prob_to_use_box_input_for_train: 0.0` - 禁用训练时的box输入
- `prob_to_use_box_input_for_eval: 0.0` - 禁用评估时的box输入
- `num_init_cond_frames_for_train: 1` - 只使用第一帧作为初始条件帧
- `rand_init_cond_frames_for_train: False` - 总是使用第一帧

### ID-aware参数
- `include_class_id: true` - 启用ID-aware模式，将class ID和mask一起作为prompt
- `multi_class_mode: true` - 启用多类别模式
- `target_class_id: null` - 动态随机选择目标类别

### 数据增强参数
```yaml
vos:
  train_transforms:
    - RandomHorizontalFlip
    - RandomAffine
    - RandomResizeAPI
    - ColorJitter
    - RandomGrayscale
    - ToTensorAPI
    - NormalizeAPI
```

## 训练策略详解

### 1. 多类别模式 (multi_class_mode=True)
- **前景类别**: 随机选择一个类别作为前景
- **背景类别**: 其他所有类别合并为背景
- **输出**: 二进制掩码，前景=1，背景=0

### 2. 单类别模式 (multi_class_mode=False)
- **独立对象**: 每个类别都作为独立的对象
- **输出**: 多个二进制掩码，每个类别一个

### 3. 随机类别选择
```python
# 在NiftiSegmentLoader.load()方法中
if self.target_class_id is not None:
    target_id = self.target_class_id
else:
    # 随机选择一个类别作为目标
    target_id = np.random.choice(object_ids)
```

### 4. Mask-only Prompt模式
- **输入方式**: 只使用mask作为prompt，不使用box和click
- **训练流程**: 
  - 第一帧的GT mask作为初始prompt
  - 模型基于mask prompt进行分割预测
  - 不使用交互式point/box采样
- **优势**: 
  - 简化训练流程，减少计算开销
  - 专注于mask-to-mask的学习
  - 更适合医学图像分割任务

### 5. ID-aware Prompt模式
- **输入方式**: 同时使用mask和class ID作为prompt
- **训练流程**: 
  - 第一帧的GT mask + 对应的class ID作为prompt
  - 模型学会将ID和mask形状关联起来
  - 提高多类别分割的准确性
- **优势**: 
  - 更接近实际推理场景
  - 避免ID混淆问题
  - 增强模型对多类别的理解能力

## 性能优化

### 1. 内存优化
- 使用 `sample_rate` 控制帧采样率
- 使用 `truncate_video` 限制视频长度
- 启用 `pin_memory` 和 `num_workers` 优化数据加载

### 2. 训练优化
- 使用混合精度训练 (`amp_dtype: bfloat16`)
- 梯度裁剪 (`max_norm: 0.1`)
- 学习率调度 (`CosineParamScheduler`)

## 故障排除

### 常见问题
1. **文件路径错误**: 确保ImagesTr和labelsTr文件夹路径正确
2. **文件命名错误**: 确保图像文件有`_0000`后缀，标签文件没有
3. **类别ID不匹配**: 确保标签文件中的类别ID在1-46范围内
4. **内存不足**: 减少batch_size或num_frames参数

### 调试技巧
```python
# 检查数据集长度
print(f"Dataset length: {len(dataset)}")

# 检查可用类别
print(f"Available classes: {dataset.get_available_classes()}")

# 检查单个样本
video, segment_loader = dataset.get_video(0)
print(f"Video frames: {len(video)}")
print(f"Segment loader: {type(segment_loader)}")
```

## 扩展功能

### 1. 自定义类别轮换策略
可以修改 `NiftiSegmentLoader` 类来实现更复杂的类别选择策略：
```python
def custom_class_selection(self, object_ids):
    # 实现自定义的类别选择逻辑
    return selected_class_id
```

### 2. 类别权重训练
可以为不同类别设置不同的训练权重：
```python
def get_class_weights(self):
    # 返回类别权重字典
    return {class_id: weight for class_id, weight in zip(self.available_classes, weights)}
```

### 3. 多模态支持
可以扩展支持多种医学图像模态：
```python
def load_multimodal_data(self, img_paths):
    # 加载多种模态的数据
    return multimodal_data
```

## 总结

这个多类别训练策略为MedSAM2提供了强大的多类别分割能力，通过随机类别选择和动态轮换，确保模型能够学会区分所有46个医学图像类别。该实现具有良好的扩展性和可配置性，可以根据具体需求进行调整和优化。

## 配置修改说明

### 本次修改内容
1. **禁用Point/Box输入**: 将所有与point和box相关的概率参数设置为0
2. **启用Mask-only模式**: 只使用mask作为prompt输入
3. **简化训练流程**: 减少交互式采样，专注于mask学习

### 修改的文件
- `sam2/configs/sam2.1_hiera_tiny_finetune512.yaml` - 主配置文件
- `README_MultiClass_Training.md` - 本文档

### 关键配置变更
```yaml
# 修改前
prob_to_use_pt_input_for_train: 0.5
prob_to_use_box_input_for_train: 1.0
num_init_cond_frames_for_train: 2
rand_init_cond_frames_for_train: True

# 修改后
prob_to_use_pt_input_for_train: 0.0  # 禁用point输入
prob_to_use_box_input_for_train: 0.0 # 禁用box输入
num_init_cond_frames_for_train: 1    # 只使用第一帧
rand_init_cond_frames_for_train: False # 总是使用第一帧
```

### 预期效果
- 训练速度更快（减少point/box采样开销）
- 内存使用更少（不需要存储point/box信息）
- 训练更稳定（专注于mask学习）
- 更适合医学图像分割任务

## ID-aware功能详解

### 实现原理
ID-aware功能通过在训练时同时传递mask和对应的class ID，让模型学会将ID和mask形状关联起来。这样在推理时，模型能够更准确地根据给定的ID进行分割。

### 数据结构变化
```python
# 修改前：只返回mask
return {1: torch.from_numpy(binary_mask).bool()}

# 修改后：返回mask和ID的组合
return {
    1: {
        'mask': torch.from_numpy(binary_mask).bool(),
        'class_id': torch.tensor(target_id, dtype=torch.long)
    }
}
```

### 向后兼容性
为了保持向后兼容性，代码会自动检测数据格式：
```python
# 在vos_dataset.py中
if isinstance(segments[obj_id], dict):
    # 新格式：包含mask和ID
    segment = segments[obj_id]['mask'].to(torch.uint8)
else:
    # 旧格式：直接是tensor
    segment = segments[obj_id].to(torch.uint8)
```

### 训练优势
1. **语义关联**: 模型学会将ID和mask形状建立强关联
2. **避免混淆**: 当不同ID的mask形状相似时，模型仍能正确区分
3. **推理一致性**: 训练和推理场景更加一致
4. **多类别理解**: 增强模型对多类别分割的理解能力

### 使用场景
- **医学图像分割**: 不同器官/病变的ID对应不同的mask形状
- **多类别检测**: 需要同时处理多个类别的分割任务
- **ID-guided分割**: 根据指定的ID进行精确分割

### 配置说明
```yaml
# 在配置文件中启用ID-aware模式
video_dataset:
  include_class_id: true  # 启用ID-aware功能
  multi_class_mode: true  # 多类别模式
  target_class_id: null   # 动态选择目标类别
```

## 故障排除

### 常见错误及解决方案

#### 1. `'dict' object has no attribute 'sum'` 错误
**原因**: 代码期望tensor对象，但收到了字典
**解决方案**: 确保已更新`vos_dataset.py`和`vos_sampler.py`文件以支持新的数据格式

#### 2. 数据加载失败
**原因**: ID-aware模式与现有代码不兼容
**解决方案**: 
- 检查`include_class_id`参数设置
- 确保所有相关文件都已更新
- 运行测试脚本验证功能

#### 3. 训练时ID信息丢失
**原因**: SAM2模型层未处理ID信息
**解决方案**: 需要进一步修改SAM2模型以支持ID-aware训练（下一步工作）

### 调试技巧
```python
# 检查segment loader的输出格式
segment_loader = NiftiSegmentLoader(masks, include_class_id=True)
result = segment_loader.load(0)
print(f"输出类型: {type(result)}")
for obj_id, data in result.items():
    print(f"对象 {obj_id}: {type(data)}")
    if isinstance(data, dict):
        print(f"  mask形状: {data['mask'].shape}")
        print(f"  class_id: {data['class_id'].item()}")
``` 