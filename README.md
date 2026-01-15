# PDF 工具与 Mask2Former 示例

## 概览
- 提供 PDF 图片提取脚本、Mask2Former/Detectron2 推理示例，以及将流程图图片转换为 Graph IR 的 Dify 工作流。
- 命令从仓库根目录执行，路径均相对 `./`。

## 目录说明
- `extract_pdf_images.py` — 从 PDF 抽取图片，跳过过小资源、去重 xref，并可添加白色边框的命令行工具。
- `maskformer/config.py` — 在 Detectron2 配置中增加 Mask2Former/Swin 相关设置。
- `maskformer/demo_single_image_once.py` — 单张图片推理，支持按顺序标注框（字母或数字）并可显示置信度。
- `maskformer/demo_batch_images.py` — 批量遍历文件夹做推理（可递归），输出可视化与对应的 bbox 版本。
- `maskformer/flow_swin_tiny_single.yaml` — 基于上游 Mask2Former coco 模板扩展的 FlowCoco Swin-Tiny 训练配置。
- `Graph IR—单个.yml` — Dify 工作流（使用 `langgenius/ollama` 插件），将上传的单张流程图转为 JSON Graph IR。

## 环境准备（推荐）
1) 创建虚拟环境（可选）
2) PDF 提取依赖：`pip install pymupdf pillow`
3) Mask2Former 推理：安装 PyTorch（按需 CUDA 版），再运行 `pip install opencv-python fvcore iopath pycocotools` 以及 `pip install 'git+https://github.com/facebookresearch/detectron2.git'`
4) Dify 工作流：在 Dify 控制台导入 YAML，并按 `langgenius/ollama` 配置 Ollama 端点/插件。

## PDF 图片提取
示例：
```
python extract_pdf_images.py input.pdf out_dir \
  --border 20 --min-width 200 --min-height 200 --min-area 40000
```
- 输出文件名包含页码/序号，边框宽度可控（`--border > 0` 时添加白边）。
- 低于尺寸或面积阈值的图片会被跳过，跨页的重复 xref 会被过滤。

## Mask2Former 推理
- 单张图片：
```
python maskformer/demo_single_image_once.py \
  --config /path/to/flow_swin_tiny_single.yaml \
  --weights /path/to/model_final.pth \
  --input /path/to/image.png \
  --output output/vis/single_vis.png \
  --confidence 0.5 --label-style alpha --show-score
```
- 批量文件夹：
```
python maskformer/demo_batch_images.py \
  --config /path/to/flow_swin_tiny_single.yaml \
  --weights /path/to/model_final.pth \
  --input-dir /path/to/images \
  --output-dir output/vis/batch \
  --confidence 0.5 --recursive
```
- 当元数据不在配置的 TEST 列表中时使用 `--test-dataset`；仅 bbox 的可视化会使用 `_bbox` 后缀或写到 `--bbox-output-dir`。

## 训练配置说明
- `maskformer/flow_swin_tiny_single.yaml` 默认放在上游 Mask2Former 仓库结构下（基础配置位于 `configs/coco/...`）。放好后用 Detectron2 的 `train_net.py` 运行，例如：`python train_net.py --config-file configs/flow/flow_swin_tiny_single.yaml --num-gpus 1`。

## Graph IR 工作流
- 在 Dify Workflow 应用中导入 `Graph IR—单个.yml`。起始节点接收一张图片（变量 `pic`），后续 LLM/代码迭代生成符合提示要求的 JSON Graph IR。
