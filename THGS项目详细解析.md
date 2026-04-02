# THGS项目详细解析

## 项目概述

**THGS (Training-Free Hierarchical Scene Understanding for Gaussian Splatting with Superpoint Graphs)** 是一个基于3D高斯散射(Gaussian Splatting)的无需训练的分层场景理解系统。该项目实现了开放词汇的3D场景分割和交互功能。

### 核心特性
- **无需训练**: 不需要额外的训练过程，直接在预训练的2DGS场景上工作
- **分层理解**: 构建多层次的超点图结构，支持不同粒度的场景分割
- **开放词汇**: 支持任意文本查询的3D场景分割
- **实时交互**: 提供GUI界面进行可视化和交互

## 项目架构

### 目录结构
```
THGS/
├── configs/                    # 配置文件
│   └── lerf.yml               # LERF数据集配置
├── ext/spt/                   # SPT (Superpoint Transformer) 库
├── gaussian_renderer/         # 高斯渲染器
├── scene/                     # 场景相关模块
├── utils/                     # 工具函数
├── scripts/                   # 脚本文件
├── gui/                       # 图形用户界面
├── submodules/                # 子模块
├── sp_partition.py            # 超点分割主程序
├── graph_weight.py            # 图权重计算
├── merge_proj.py              # 超点合并和投影
├── test_lerf.py              # LERF数据集测试
└── nag_data.py               # NAG数据结构
```

## 核心算法流程

### 1. 预处理阶段 (Preprocessing)

#### 场景重建
- 使用2DGS (2D Gaussian Splatting) 进行场景重建
- 提取2D语义图和特征
- 生成高斯点云表示

#### 语义特征编码
```python
# scripts/image_encoding.py
python scripts/image_encoding.py --source_path <scene_path>
```
- 为每个场景生成语言特征
- 兼容LangSplat的特征格式

### 2. 对比高斯分割 (Contrastive Gaussian Partitioning)

#### 步骤1: 高斯邻接图构建 (`sp_partition.py`)

**核心功能**:
- 从PLY文件加载3D高斯点
- 构建高斯中心点的邻接图
- 使用图切割算法将邻接图分割为超点

**关键代码解析**:
```python
def load_ply(path, pos_center=torch.tensor([0, 0, 0]), semantic=None, use_normal=False):
    """加载PLY文件并构建数据结构"""
    # 1. 读取高斯点的位置、颜色和球谐系数
    xyz = np.stack((np.asarray(plydata.elements[0]["x"]),
                    np.asarray(plydata.elements[0]["y"]),
                    np.asarray(plydata.elements[0]["z"])), axis=1)
    
    # 2. 提取球谐系数特征
    features_dc = np.zeros((xyz.shape[0], 3, 1))
    features_dc[:, 0, 0] = np.asarray(plydata.elements[0]["f_dc_0"])
    # ... 更多球谐系数
    
    # 3. 计算RGB颜色
    sh2rgb = eval_sh(3, shs_view, dir_pp_normalized)
    colors_precomp = torch.clamp(sh2rgb + 0.5, 0.0, 1.0)
    
    # 4. 构建数据对象
    data = Data()
    data.pos = xyz
    data.rgb = colors_precomp
    return data
```

**分割过程**:
```python
def partition(data, filepath, transforms_dict, graph_cut=None, **kwargs):
    """执行图分割"""
    if graph_cut is not None:
        # 加载预计算的邻居信息
        neibor = torch.load(graph_cut)
        data.neighbor_index = neibor['neighbors'].cuda()
        data.neighbor_distance = neibor['distances'].cuda()
    
    # 应用变换进行分割
    nag = transforms_dict['cut_transform'](data)
    return nag
```

#### 步骤2: SAM引导的图边权重调整 (`graph_weight.py`)

**核心思想**:
- 使用SAM (Segment Anything Model) 提供的分割线索
- 调整高斯邻接图的边权重
- 增强分割边界的准确性

**算法流程**:
```python
def extract_gaussian_features(model_path, views, gaussians, pipeline, background, feature_level, scale_factor=1):
    """提取高斯特征并调整边权重"""
    for i, view in enumerate(views):
        # 1. 获取分割图和前景掩码
        seg_map = view.semantic["seg_map"][feature_level]
        img_mask = view.semantic['fg_mask'][feature_level]
        
        # 2. 生成随机编码
        enc = torch.normal(mean=0, std=1, size=(seg_num, dim_latent), device="cuda")
        enc = torch.nn.functional.normalize(enc, p=2, dim=-1)
        
        # 3. 光线追踪获取高斯语义
        render_pkg = trace(view, gaussians, feature_map, None, pipeline, background)
        gau_sem = render_pkg["gaussian_semantics"]
        
        # 4. 计算相似度并过滤
        sim = torch.matmul(gau_sem, enc.T)
        sim_filter = sim.max(dim=-1)[0] > tau
        
        # 5. 更新边权重
        neg_dist[valid_gau] += ((knn_label != seen_gau_label) & (knn_label < seg_num-1)).float() * depth_weight
        pos_dist[valid_gau] += ((knn_label == seen_gau_label) & (knn_label < seg_num-1)).float() * depth_weight
```

**权重更新公式**:
```python
# 最终距离计算
new_distance = ori_distance + neg_dist * neg_w - pos_dist * pos_w
new_distance = torch.clamp(new_distance, min=0)
```

### 3. 分层语义表示 (Hierarchical Semantic Representation)

#### 超点合并和语义特征重投影 (`merge_proj.py`)

**核心类 SAI3D**:
```python
class SAI3D:
    def __init__(self, args):
        self.max_neighbor_distance = args.max_neighbor_distance
        self.view_freq = args.view_freq
        self.dis_decay = args.dis_decay
```

**主要功能**:

1. **数据初始化**:
```python
def init_data(self, args):
    """初始化场景数据"""
    # 1. 加载高斯模型和场景
    self.gaussians = GaussianModel(3, 0)
    self.scene = Scene(args, self.gaussians, load_iteration=30000)
    
    # 2. 加载超点分割结果
    self.seg_ids = torch.load(join(args.model_path, 'nag-l1.pt'))
    
    # 3. 构建KD树用于邻居查找
    points_kdtree = scipy.spatial.KDTree(self.points)
    points_neighbors = points_kdtree.query(self.points, 8, workers=n_workers)[1]
    self.points_neighbors = torch.tensor(points_neighbors, dtype=torch.long).cuda()
```

2. **语义特征投影**:
```python
def proj_gaussian_features(self, gaussians, views, feature_level, gau2sp):
    """将语义特征投影到超点"""
    sp_feature = torch.zeros((sp_uni.shape[0], 512), device="cuda")
    
    for _, view in enumerate(views):
        # 1. 渲染获取权重
        render_pkg = render_point(view, gaussians, pipeline, background)
        weight = render_pkg["weight"]
        
        # 2. 获取语义特征
        gt_feature = view.semantic["sem"].cuda()
        
        # 3. 计算超点特征
        seen_gau_sem = significance.unsqueeze(-1) * gt_batch_mask.unsqueeze(-1) * gt_feature[gt_batch_seg]
        
        # 4. 聚合到超点级别
        sp_feature[sp_uni == sp] += F.normalize(seen_gau_sem[sp_mask].sum(dim=0), p=2, dim=-1) * portion
    
    return sp_feature
```

3. **区域生长算法**:
```python
def assign_seg_label_torch(self, adj, thres_connect, max_neighbor_distance):
    """基于邻接矩阵的区域生长"""
    assign_id = 1
    seg_labels = torch.zeros(self.seg_num, dtype=torch.int, device=adj.device)
    
    for i in range(self.seg_num):
        if seg_labels[i] == 0:
            queue = deque([i])
            seg_labels[i] = assign_id
            
            while queue:
                v = queue.popleft()
                js = self.seg_direct_neighbors[v].nonzero(as_tuple=True)[0]
                js = js[seg_labels[js] == 0]  # 只选择未标记的邻居
                
                for j in js:
                    connect = self.judge_connect_torch_opt(adj, v, j, thres_connect, ...)
                    if connect:
                        seg_labels[j] = assign_id
                        queue.append(j)
            assign_id += 1
    
    return seg_labels
```

4. **连接性判断**:
```python
def judge_connect_torch_opt(self, adj, p1_id, p2_id, thres_connect, ...):
    """判断两个超点是否应该连接"""
    # 使用距离衰减权重
    weight = decay ** torch.arange(max_neighbor_distance, device=adj.device)
    
    # 计算加权相似度分数
    adj_sum = (weight[:, None] * adj[p2_id, neighbor_ids] * self.seg_member_count[neighbor_ids]).sum()
    weight_sum = (weight[:, None] * self.seg_member_count[neighbor_ids]).sum()
    
    score = adj_sum / weight_sum if weight_sum != 0 else 0
    return score >= thres_connect
```

### 4. 查询和分解 (Query and Decomposition)

#### NAG数据结构 (`nag_data.py`)

**SemanticNAG类**:
```python
class SemanticNAG():
    def __init__(self, labels, feat):
        """构建语义NAG结构"""
        self.labels = labels  # 多层级标签 [level0, level1, level2, ...]
        self.nag = self.build_nag_from_multilevel_labels(labels)
        self.feat = feat      # 语义特征
        self.gaussian_num = labels[0].shape[0]
```

**关键方法**:

1. **相关高斯点检索**:
```python
def get_related_gaussian(self, sim: List[torch.Tensor], topk: int = 1, level: int = -1):
    """根据相似度检索相关的高斯点"""
    # 1. 在指定层级找到topk相似的超点
    related_sp_lvl = []
    for i in lvls:
        sim_array = sim[i]
        sim_val, indices = torch.topk(sim_array, topk)
        related_sp_lvl.extend(list(zip([i+1] * topk, sim_val, indices)))
    
    # 2. 按相似度排序
    related_sp_lvl.sort(key=lambda x: x[1], reverse=True)
    related_sp_lvl = related_sp_lvl[:topk]
    
    # 3. 映射到底层高斯点
    rel_gaussians = torch.zeros(self.gaussian_num, 1, dtype=torch.float32)
    for level, _, index in related_sp_lvl:
        lowest_idx = torch.where(self.labels[level] == index)[0]
        rel_gaussians[lowest_idx, 0] = 1
    
    return rel_gaussians
```

2. **多层级NAG构建**:
```python
@staticmethod
def build_nag_from_multilevel_labels(labels: List[torch.Tensor]) -> NAG:
    """从多层级标签构建NAG结构"""
    data_list = [Data(num_nodes=N, super_index=labels[0])]
    prev_sub = None
    
    for i in range(num_levels):
        if i == 0:
            # 构建第一层聚类
            sorted_upper_labels, perm = torch.sort(upper_labels)
            cluster_sizes = torch.bincount(sorted_upper_labels, minlength=num_clusters)
            pointers = torch.cumsum(cluster_sizes, dim=0)
            prev_sub = Cluster(pointers=pointers, points=sorted_lower_labels, dense=False)
        else:
            # 构建后续层级
            unique_lower_labels, inv = torch.unique(lower_labels, sorted=True, return_inverse=True)
            data.super_index = unique_upper_labels
            # ... 构建聚类结构
    
    return NAG(data_list)
```

### 5. 开放词汇分割测试 (`test_lerf.py`)

**测试流程**:
```python
def training(dataset, pipe):
    """执行开放词汇分割测试"""
    # 1. 加载模型和场景
    gaussians = GaussianModel(dataset.sh_degree, 20)
    scene = Scene(dataset, gaussians, 30000, load_sem=False)
    
    # 2. 加载NAG结构
    nag = torch.load(os.path.join(dataset.model_path, f"sai_nag.pt"))
    snag = SemanticNAG(nag['nag'], nag['nag_feat'])
    
    # 3. 初始化CLIP模型
    vlm = ClipSimMeasure()
    vlm.load_model()
    
    # 4. 处理每个查询
    for prompt in prompt_list:
        # 编码文本查询
        vlm.encode_text(prompt)
        
        # 检索相关高斯点
        point_valid = snag.get_related_gaussian(
            [vlm.compute_similarity(f) for f in snag.feat], 
            topk=3, level=[2,3]
        )
        
        # 渲染分割掩码
        gaussians._semantics = point_valid.expand(-1, 20).cuda()
        embd_sim = render(cam, gaussians, pipe, background)["semantics"]
        binary_mask = embd_sim.reshape(20, -1)[0] > 0.5
```

## 配置系统

### 主配置文件 (`configs/lerf.yml`)

```yaml
dataset: 
  name: LERF-OVS
  data_path: data/lerf_ovs
  save_folder: lerf_ovs
  scenes: ['figurines', 'ramen', 'teatime', 'waldo_kitchen']

graph_weight:
  tau: 0.85              # 相似度阈值
  neg_w: 0.1             # 负权重
  pos_w: 0.02            # 正权重
  neg_b: 25              # 负权重上界
  pos_b: 25              # 正权重上界
  zero_scale: 0.2        # 零缩放因子
  level: 1               # 特征层级

merge_proj:
  thres_connect: 0.9,0.7,0.7  # 连接阈值（多层级）
  thres_merge: 20             # 合并阈值
  feat_assign: 2              # 特征分配方法

spt:
  pcp_regularization: 0.1     # PCP正则化
  pcp_spatial_weight: 1e-1    # 空间权重
  aligned_normal: True        # 法向量对齐
```

## 运行流程

### 完整流水线执行

```bash
# 运行完整流水线
bash scripts/run.sh configs/lerf.yml [scene_names]
```

**流水线步骤** (`scripts/run.sh`):
```bash
# 1. 构建邻接图
python scripts/launcher.py -f sp_partition.py -cf $config_file

# 2. 调整图权重
python scripts/launcher.py -f graph_weight.py -cf $config_file

# 3. 执行图分割
python scripts/launcher.py -f sp_partition.py -cf $config_file -k

# 4. 合并和投影
python scripts/launcher.py -f merge_proj.py -cf $config_file
```

### 评估流程

```bash
# 1. 对每个场景进行测试
for sc in figurines ramen teatime waldo_kitchen; do
    python test_lerf.py -s data/lerf/$sc -m output/lerf/$sc --path_pred output/render/lerf
done

# 2. 评估分割结果
python scripts/eval_seg.py \
    --dataset lerf \
    --scene_list figurines ramen teatime waldo_kitchen \
    --path_pred output/render/lerf \
    --path_gt data/lerf/label
```

## 技术创新点

### 1. 无训练分层分割
- 直接在预训练的2DGS场景上工作
- 不需要额外的神经网络训练
- 利用SAM提供的分割先验

### 2. 多层级超点图
- 构建分层的超点表示
- 支持不同粒度的查询和分割
- 高效的层级间映射

### 3. 对比学习增强
- 使用对比线索调整图边权重
- 提高分割边界的准确性
- 结合深度信息进行权重调整

### 4. 开放词汇支持
- 基于CLIP的文本-视觉匹配
- 支持任意自然语言查询
- 实时的语义检索和分割

## 依赖关系

### 核心依赖
- **PyTorch 2.2.0**: 深度学习框架
- **CUDA 11.8**: GPU计算支持
- **Open3D**: 3D数据处理
- **OpenCV**: 图像处理
- **Scipy**: 科学计算
- **PyTorch Geometric**: 图神经网络

### 专用模块
- **SPT库**: 超点变换器
- **FRNN**: 快速最近邻搜索
- **2DGS**: 2D高斯散射渲染
- **SAM**: 分割一切模型

## 性能特点

### 内存优化
- 分块处理大规模点云
- 稀疏矩阵表示邻接关系
- GPU内存管理和缓存清理

### 计算效率
- 并行化的邻居搜索
- 优化的区域生长算法
- 高效的特征投影和聚合

### 可扩展性
- 模块化的设计架构
- 可配置的参数系统
- 支持不同数据集和场景类型

## 应用场景

1. **3D场景理解**: 自动分析和理解复杂3D场景
2. **机器人导航**: 为机器人提供语义地图
3. **AR/VR应用**: 实时的场景分割和交互
4. **内容创作**: 3D场景的自动标注和编辑
5. **工业检测**: 基于语义的质量控制和检测

## 总结

THGS项目实现了一个完整的无训练3D场景理解系统，通过巧妙地结合高斯散射、超点图和对比学习技术，实现了高质量的开放词汇3D分割。其分层的架构设计和高效的算法实现使得系统具有良好的性能和可扩展性，为3D场景理解领域提供了新的技术路径。