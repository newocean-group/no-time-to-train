from src.libs import *


def encode_image(filepath):
    with open(filepath, "rb") as f:
        image_bytes = f.read()
    encoded = str(base64.b64encode(image_bytes), "utf-8")
    return "data:image/jpg;base64," + encoded


def get_sam2_model_cfg_and_ckpt_path(model_type):
    if model_type == "tiny":
        model_cfg = "configs/sam2.1/sam2.1_hiera_t.yaml"
        ckpt_path = "checkpoints\sam2.1_hiera_tiny.pt"
    elif model_type == "small":
        model_cfg = "configs/sam2.1/sam2.1_hiera_s.yaml"
        ckpt_path = "checkpoints\sam2.1_hiera_small.pt"
    elif model_type == "base_plus":
        model_cfg = "configs/sam2.1/sam2.1_hiera_b+.yaml"
        ckpt_path = "checkpoints\sam2.1_hiera_base_plus.pt"
    else:
        model_cfg = "configs/sam2.1/sam2.1_hiera_l.yaml"
        ckpt_path = "checkpoints\sam2.1_hiera_large.pt"
    return model_cfg, ckpt_path


def show_anns(anns, borders=True):
    if len(anns) == 0:
        return
    sorted_anns = sorted(anns, key=(lambda x: x["area"]), reverse=True)
    ax = plt.gca()
    ax.set_autoscale_on(False)

    img = np.ones(
        (
            sorted_anns[0]["segmentation"].shape[0],
            sorted_anns[0]["segmentation"].shape[1],
            4,
        )
    )
    img[:, :, 3] = 0
    for ann in sorted_anns:
        m = ann["segmentation"]
        color_mask = np.concatenate([np.random.random(3), [0.5]])
        img[m] = color_mask
        if borders:
            import cv2

            contours, _ = cv2.findContours(
                m.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE
            )
            # Try to smooth contours
            contours = [
                cv2.approxPolyDP(contour, epsilon=0.01, closed=True)
                for contour in contours
            ]
            cv2.drawContours(img, contours, -1, (0, 0, 1, 0.4), thickness=1)

    ax.imshow(img)


def mask_to_bbox_xyxy_cv2(mask_np, min_area=None, max_area=None):
    """
    Converts a binary mask to its tightest bounding box in [x_min, y_min, x_max, y_max] format using OpenCV.
    Returns None if the mask is empty.
    """
    if mask_np.sum() == 0:
        return None

    mask_uint8 = (mask_np * 255).astype(np.uint8)

    contours, _ = cv2.findContours(
        mask_uint8, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
    )

    if not contours:
        return None  # No contours found (e.g., very small or scattered pixels)

    largest_contour = max(contours, key=cv2.contourArea)

    x, y, w, h = cv2.boundingRect(largest_contour)

    area = w * h

    if min_area is not None:
        if area < min_area:
            return None

    if max_area is not None:
        if area > max_area:
            return None

    x_min, y_min = x, y
    x_max, y_max = x + w, y + h

    return [x_min, y_min, x_max, y_max]


def visualize_results(
    image_path,
    instances,
    ax=None,
    title="Reference-Based Instance Segmentation Results with Bounding Boxes",
):
    image_np = cv2.imread(image_path)
    image_np = cv2.cvtColor(image_np, cv2.COLOR_BGR2RGB)

    if ax is None:
        fig, ax = plt.subplots(figsize=(12, 12))

    ax.imshow(image_np)
    ax.set_autoscale_on(False)

    colors = plt.cm.get_cmap("tab20", len(instances))

    for i, (mask, score, class_name, bbox_xyxy) in enumerate(instances):
        if mask.sum() == 0 or bbox_xyxy is None:
            continue
        color = colors(i % 20)

        # Create a colored mask
        colored_mask = np.zeros((*mask.shape, 4), dtype=np.uint8)
        colored_mask[mask] = np.array(
            [color[0] * 255, color[1] * 255, color[2] * 255, 150]
        )
        ax.imshow(colored_mask)

        # Draw bounding box
        x_min, y_min, x_max, y_max = bbox_xyxy
        rect = patches.Rectangle(
            (x_min, y_min),
            x_max - x_min,
            y_max - y_min,
            linewidth=2,
            edgecolor=color[:3],
            facecolor="none",
            linestyle="--",
        )
        ax.add_patch(rect)

        # Add text label
        ax.text(
            x_min,
            y_min - 5,
            f"{class_name} ({score:.2f})",
            bbox=dict(
                facecolor=color[:3],
                alpha=0.7,
                edgecolor="none",
                boxstyle="round,pad=0.2",
            ),
            color="white",
            fontsize=8,
        )

    ax.axis("off")
    ax.set_title(title, fontsize=14, fontweight="bold")

    return ax


def preprocess_image_for_dino(img_path, dino_transform, device="cpu"):
    img_pil = Image.open(img_path).convert("RGB")
    return dino_transform(img_pil)[None].to(device)


def get_dino_features(dino_model, image_tensor, model_name, device="cpu"):
    h, w = image_tensor.shape[2:]
    scale_factor = 14 if "dinov2" in model_name else 16
    with torch.inference_mode(), torch.autocast(device, dtype=torch.bfloat16):
        features = dino_model.get_intermediate_layers(image_tensor.to(torch.bfloat16))[
            0
        ]
        features_map = (
            features.reshape(
                features.shape[0],
                h // scale_factor,
                w // scale_factor,
                features.shape[2],
            )
            .permute(0, 3, 1, 2)
            .contiguous()
        )
        return features_map


def get_embedding_visualization(tokens, grid_size, resized_mask=None):
    pca = PCA(n_components=3)
    if resized_mask is not None:
        tokens = tokens[resized_mask]
    reduced_tokens = pca.fit_transform(tokens.astype(np.float32))
    if resized_mask is not None:
        tmp_tokens = np.zeros((*resized_mask.shape, 3), dtype=reduced_tokens.dtype)
        tmp_tokens[resized_mask] = reduced_tokens
        reduced_tokens = tmp_tokens
    reduced_tokens = reduced_tokens.reshape((*grid_size, -1))
    normalized_tokens = (reduced_tokens - np.min(reduced_tokens)) / (
        np.max(reduced_tokens) - np.min(reduced_tokens)
    )
    return normalized_tokens


def visualize_dino_embedding(
    img_path, dino_model, dino_transform, model_name, device="cpu"
):
    img_pil = Image.open(img_path).convert("RGB")
    ori_w, ori_h = img_pil.size
    img_tensor = preprocess_image_for_dino(img_path, dino_transform, device=device)
    features_map = get_dino_features(dino_model, img_tensor, model_name, device)
    _, c, h, w = features_map.shape
    tokens = features_map.flatten(-2).permute(0, 2, 1).contiguous().reshape(-1, c)
    tokens = (
        tokens.float().cpu().numpy() if device == "cuda" else tokens.float().numpy()
    )
    grid_size = (h, w)
    vis_image = get_embedding_visualization(tokens, grid_size, resized_mask=None)

    vis_image = cv2.resize(vis_image, dsize=(ori_w, ori_h))
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 20))
    ax1.imshow(img_pil)
    ax2.imshow(vis_image)
    fig.tight_layout()


def resize_mask_to_features(mask_np, feature_map_shape):
    H_feat, W_feat = feature_map_shape[1], feature_map_shape[2]
    resized_mask = cv2.resize(mask_np.astype(np.float32), dsize=(W_feat, H_feat))
    return (resized_mask > 0.5).astype(np.float32)


def build_memory_bank(
    reference_data, dino_model, dino_transform, model_name, device="cpu"
):
    """
    reference_data: {
        'cls_nm' : [
            {
                "img_path": img_path ,
                "masks": masks with shape [n_objects, h, w]
            },
            {
                ....
            }
        ]
    }
    """
    memory_bank = {}  # Stores {class_name: class_prototype_feature_vector}

    print("Building Memory Bank...")
    for class_name, ref_images_masks in tqdm(reference_data.items()):
        all_instance_features_for_class = []
        for data in ref_images_masks:
            img_path = data["img_path"]
            instance_masks = data["masks"]
            img_tensor_dino = preprocess_image_for_dino(
                img_path, dino_transform, device=device
            )
            full_img_features = get_dino_features(
                dino_model, img_tensor_dino, model_name, device=device
            )  # (1, C, H_feat, W_feat)

            for mask_np in instance_masks:
                # Resize mask to feature map dimensions
                resized_mask = resize_mask_to_features(
                    mask_np, full_img_features.shape[1:]
                )  # (H_feat, W_feat)
                resized_mask_tensor = (
                    torch.from_numpy(resized_mask).unsqueeze(0).unsqueeze(0).to(device)
                )  # (1, 1, H_feat, W_feat)

                # Mask the features and take the average (instance-wise prototype)
                masked_features = full_img_features * resized_mask_tensor
                # Only average over non-zero elements
                num_pixels = resized_mask_tensor.sum()
                if num_pixels > 0:
                    instance_prototype = masked_features.sum(dim=[2, 3]) / num_pixels
                    all_instance_features_for_class.append(instance_prototype)
                else:
                    print(
                        f"Warning: Empty mask for instance in class {class_name} from {img_path}. Skipping."
                    )

        if all_instance_features_for_class:
            # Average all instance prototypes to get the class-wise prototype
            class_prototype = torch.cat(all_instance_features_for_class, dim=0).mean(
                dim=0
            )
            memory_bank[class_name] = (
                class_prototype.squeeze().cpu().numpy()
            )  # Store as numpy array
        else:
            print(
                f"Warning: No valid instances found for class {class_name}. Skipping class."
            )

    print("Memory Bank Built.")
    return memory_bank


def preprocess_image_for_sam2(img_path):
    img_pil = Image.open(img_path).convert("RGB")
    return img_pil


def get_candidate_masks(sam2_mask_generator, img_pil, device="cpu"):
    with torch.inference_mode(), torch.autocast(device, dtype=torch.bfloat16):
        masks = sam2_mask_generator.generate(np.array(img_pil))
    candidate_masks = [m["segmentation"] for m in masks]
    return masks, candidate_masks


def extract_mask_features_dino(
    img_pil,
    candidate_masks_with_boxes,
    dino_model,
    dino_transform,
    model_name,
    device="cpu",
):
    img_tensor_dino = dino_transform(img_pil)[None].to(device)
    full_img_features = get_dino_features(
        dino_model, img_tensor_dino, model_name, device
    )  # (1, C, H_feat, W_feat)

    mask_features = []
    for mask_np, _ in candidate_masks_with_boxes:
        resized_mask = resize_mask_to_features(mask_np, full_img_features.shape[1:])
        resized_mask_tensor = (
            torch.from_numpy(resized_mask).unsqueeze(0).unsqueeze(0).to(device)
        )

        masked_features = full_img_features * resized_mask_tensor
        num_pixels = resized_mask_tensor.sum()
        if num_pixels > 0:
            # Average pool and L2-normalize
            feature_vector = masked_features.sum(dim=[2, 3]) / num_pixels
            feature_vector = torch.nn.functional.normalize(feature_vector, p=2, dim=-1)
            mask_features.append(feature_vector.squeeze(0))  # Squeeze batch dim
        else:
            mask_features.append(
                torch.zeros(full_img_features.shape[1]).to(device)
            )  # Empty mask feature

    return mask_features  # List of (C,) tensors


def calculate_similarity(mask_features, memory_bank, device="cpu"):
    class_names = list(memory_bank.keys())
    prototypes = torch.tensor(np.array(list(memory_bank.values()))).to(device)
    prototypes = torch.nn.functional.normalize(prototypes, p=2, dim=-1)

    scores = torch.matmul(torch.stack(mask_features, dim=0), prototypes.T).squeeze(0)

    best_scores, best_idxs = torch.max(scores, dim=-1)

    similarities = [
        (class_names[best_idx.item()], best_score.item())
        for best_idx, best_score in zip(best_idxs, best_scores)
    ]
    return similarities


def predict_single_image(
    img_path,
    memory_bank,
    sam2_mask_generator,
    dino_model,
    dino_transform,
    model_name,
    min_aera=None,
    max_area=None,
    iou_threshold=0.25,
    conf_threshold=0.5,
    device="cpu",
):

    img_pil = preprocess_image_for_sam2(img_path)

    masks, candidate_masks = get_candidate_masks(
        sam2_mask_generator, img_pil, device=device
    )

    # Get bounding boxes for each candidate mask
    candidate_masks_with_boxes = []
    for mask in candidate_masks:
        bbox = mask_to_bbox_xyxy_cv2(mask, min_aera, max_area)
        if bbox is not None:
            candidate_masks_with_boxes.append((mask, bbox))

    mask_features = extract_mask_features_dino(
        img_pil,
        candidate_masks_with_boxes,
        dino_model,
        dino_transform,
        model_name,
        device=device,
    )

    mask_classifications = calculate_similarity(
        mask_features, memory_bank, device=device
    )

    # Filter out masks with no clear classification or very low score
    predicted_masks_info = []  # (mask_np, score, class_name, bbox_xyxy)

    for i, (class_name, score) in enumerate(mask_classifications):
        if class_name is not None and score >= conf_threshold:
            mask_np, bbox_xyxy = candidate_masks_with_boxes[i]
            mask_feature = mask_features[i]
            predicted_masks_info.append(
                (mask_np, mask_feature, score, class_name, bbox_xyxy)
            )

    masks_by_class = {}
    for i, (mask_np, mask_feature, score, class_name, bbox_xyxy) in enumerate(
        predicted_masks_info
    ):
        if class_name not in masks_by_class:
            masks_by_class[class_name] = []
        masks_by_class[class_name].append((mask_np, mask_feature, score, bbox_xyxy))

    final_segmented_instances = []

    for cls_name, values in masks_by_class.items():
        current_masks = [torch.tensor(v[0]) for v in values]
        current_mask_features = [v[1] for v in values]
        current_scores = [torch.tensor(v[2]) for v in values]
        current_boxes = [torch.tensor(v[3]) for v in values]

        current_masks = torch.stack(current_masks, dim=0)
        current_mask_features = torch.stack(current_mask_features, dim=0)
        current_scores = torch.stack(current_scores, dim=0)
        current_boxes = torch.stack(current_boxes, dim=0)

        keep_indices = torchvision.ops.nms(
            current_boxes.float(), current_scores, iou_threshold=iou_threshold
        )
        final_masks, final_scores, final_boxes = (
            current_masks[keep_indices],
            current_scores[keep_indices],
            current_boxes[keep_indices],
        )

        final_segmented_instances += [
            [mask, score, cls_name, box]
            for mask, score, box in zip(final_masks, final_scores, final_boxes)
        ]

    return img_pil, masks, final_segmented_instances
