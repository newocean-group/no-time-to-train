from src.libs import *
from src.helper_functions import *


class ModelManager:
    def __init__(self, device="cpu"):
        self.device = device
        self.dinov2_model = None
        self.sam2_model = None
        self.sam2_predictor = None
        self.sam2_mask_generator = None

    def load_dino_model(
        self,
        dino_model_name="dinov3_vits16",
        weights="checkpoints\dinov3_vits16_pretrain.pth",
    ):
        """
        dino_model_name:  dinov2_vits14, dinov2_vitb14, dinov2_vitl14, dinov2_vitg14,
                        dinov3_vits16, dinov3_vits16plus, dinov3_vitb16, dinov3_vitl16, dinov3_vith16plus, dinov3_vit7b16
        """
        if "dinov3" in dino_model_name:
            """with model size: (592, 592) -> feature map: (37, 37)"""
            model_size = (592, 592)  # divisible by 16
            self.dino_model = torch.hub.load(
                repo_or_dir="dinov3",
                model=dino_model_name,
                source="local",
                weights=weights,
            ).to(torch.bfloat16)
        else:
            """with model size: (518, 518) -> feature map: (37, 37)"""
            model_size = (518, 518)  # divisible by 14
            self.dino_model = torch.hub.load(
                repo_or_dir="facebookresearch/dinov2",
                model=dino_model_name,
            ).to(torch.bfloat16)

        self.dino_model.eval()
        self.dino_model.to(self.device)

        self.dino_transform = transforms.Compose(
            [
                transforms.Resize(
                    size=model_size,
                    interpolation=transforms.InterpolationMode.BICUBIC,
                    antialias=True,
                ),
                transforms.ToTensor(),
                transforms.Normalize(
                    mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)
                ),
            ]
        )

        return self.dino_model, self.dino_transform

    def load_sam2_model(
        self,
        sam2_model_type="tiny",
        model_cfg=None,
        sam2_model_ckpt_path=None,
        points_per_side=32,
    ):
        if self.sam2_model is None:
            if sam2_model_ckpt_path is None:
                model_cfg, ckpt_path = get_sam2_model_cfg_and_ckpt_path(
                    model_type=sam2_model_type
                )
            else:
                assert model_cfg is None, "Please provide model_cfg"
                model_cfg = model_cfg
                ckpt_path = sam2_model_ckpt_path

            self.sam2_model = build_sam2(
                model_cfg, ckpt_path, device=self.device, apply_postprocessing=False
            ).to(self.device)
            self.sam2_predictor = SAM2ImagePredictor(sam_model=self.sam2_model)
            self.sam2_mask_generator = SAM2AutomaticMaskGenerator(
                model=self.sam2_model,
                points_per_side=points_per_side,
                points_per_batch=128,
                pred_iou_thresh=0.7,
                stability_score_thresh=0.92,
                stability_score_offset=0.7,
                crop_n_layers=1,
                box_nms_thresh=0.7,
                crop_n_points_downscale_factor=2,
                min_mask_region_area=0.0,
                use_m2m=True,
            )
        return self.sam2_model, self.sam2_predictor, self.sam2_mask_generator
