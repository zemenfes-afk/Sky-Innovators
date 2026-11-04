# Sky Innovators Model

## Project Purpose
The Sky Innovators Model is developed to provide a highly resilient, AI-driven spatial analysis platform for forestry management and environmental monitoring. The primary goal is to process aerial drone imagery and massive terrain maps (orthomosaics) to accurately identify and segment 8 distinct land cover classes, with a specialized focus on tracking forest health, standing dead trees, and urban infrastructure.

## Key Advantages
- Computational Efficiency: Designed to support sliding-window inference on stitched terrain maps, eliminating the GPU compute waste and redundancy associated with processing overlapping video frames.
- High Algorithmic Resilience: Utilizes a progressive unfreezing training schedule and a custom hybrid loss function (Cross-Entropy + Dice) to strictly enforce spatial boundary integrity and protect against dataset imbalances.
- Unified Domain Adaptation: Seamlessly amalgamates diverse public datasets, capturing both nadir and oblique drone angles, into a single cohesive training pipeline to ensure the model generalizes effectively to real-world environments.

## Project Summary
This repository provides the complete PyTorch training pipeline for the v10.0 baseline model. It features a memory-efficient SegFormer-B4 architecture optimized with Automatic Mixed Precision (AMP) and gradient accumulation to run effectively within strict hardware limits. By leveraging advanced data augmentation, balanced class weighting, and a robust dataset caching system, the pipeline successfully transforms raw drone telemetry into highly accurate, production-ready geospatial intelligence.

## Requirements, Tools, and Libraries Used
- Python 3.x
- PyTorch (with CUDA and Automatic Mixed Precision support)
- Hugging Face Transformers (SegformerForSemanticSegmentation)
- Albumentations (for spatial and photometric data augmentations)
- OpenCV (cv2)
- Pillow (PIL)
- NumPy
- tqdm

## Datasets Used
- Semantic Drone Dataset (Graz): Used for high-resolution nadir and oblique urban and rural scenes.
- UAVid Dataset: Used for oblique urban infrastructure and geometric edge capture.
- USA Standing Dead Tree Dataset: Used for high-altitude aerial RGB captures targeting defoliated timber.
- Our own collected dataset

## Final Output of the Model
- Checkpoint File: sky_innovators_v10_landscape.pth
- Network Backbone: SegFormer-B4 (nvidia/segformer-b4-finetuned-ade-512-512)
- Predicted Classes: 8-Class Semantic Segmentation (Background, HealthyTree, DeadTree, LowVegetation, BareSoil, Water, Road, Building)
- Performance Metric: Achieves a peak Validation Mean Intersection-over-Union (mIoU) of approximately 0.77 across the combined multi-domain validation split after a 40-epoch progressive unfreezing training cycle.
