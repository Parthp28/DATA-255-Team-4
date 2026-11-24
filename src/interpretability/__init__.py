"""Interpretability (Execution Plan 12, 15.1, 10.6): Grad-CAM (ResNet-18; owner Kaushik), attention rollout (DeiT-S;
owner Khushi), region masking and the attribution-to-flip link (owner Farzaan). All of it runs on seed-0
checkpoints of frozen configurations over the D15 sample, and every tool is tested on CPU.

Every method uses one pixel partition of each image, computed from the renderer's own draw list
(`regions.region_masks`), so heatmaps and masks agree exactly on what is a candle, a grid line or a margin.
"""
