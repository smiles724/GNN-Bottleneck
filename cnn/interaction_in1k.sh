inter_type="pixel";

# ===== basic settings =====
gpu_id=0; dirname="result"; pretrained=True; ckpt_DIR="timm_hub/";
dataset="imagenet"; class_number=1000; image_size=224; grid=14; seed=1;  # ImageNet: 224x224, grid 14x14


# ===== model settings =====
arch="resnet";
ckpt="resnet50_a1_0-14fe96d1.pth"

# arch="mlpmixer";
# ckpt="jx_mixer_b16_224-76587d61.pth"

# arch="swin";
# ckpt="swin_tiny_patch4_window7_224.pth"

# arch="convmixer";
# ckpt="convmixer_768_32_ks7_p7_relu.pth"

# arch="convnext";
# ckpt="convnext_tiny_1k_224_ema.pth"


# ===== evaluation of interactions =====
python sampler.py --gpu_id=$gpu_id --arch=$arch --dataset=$dataset --output_dirname=$dirname \
    --grid_size=$grid --inter_type=$inter_type --seed=$seed
# python gen_pairs_pixel.py --gpu_id=$gpu_id --arch=$arch --dataset=$dataset --output_dirname=$dirname \
#     --grid_size=$grid --inter_type=$inter_type --seed=$seed
# python m_order_interaction_logit_pixel.py --gpu_id=$gpu_id --arch=$arch \
#     --checkpoint_path=$ckpt_DIR --checkpoint_name=$ckpt \
#     --dataset=$dataset --class_number=$class_number --image_size=$image_size --output_dirname=$dirname --grid_size=$grid --inter_type=$inter_type --seed=$seed
# python compute_interactions.py --gpu_id=$gpu_id --arch=$arch --pretrained=$pretrained \
#     --dataset=$dataset --class_number=$class_number --output_dirname=$dirname  --grid_size=$grid --inter_type=$inter_type --seed=$seed
# python draw_figures.py --arch=$arch --pretrained=$pretrained --dataset=$dataset --class_number=$class_number --output_dirname=$dirname \
#     --grid_size=$grid --inter_type=$inter_type --seed=$seed --save_name=$ckpt
