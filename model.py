"""Segmentation model = segmentation_models_pytorch decoder + a configurable encoder.
The point of this experiment: drop Lunit's pathology-pretrained ResNet50 weights
straight into the encoder (native PyTorch — no conversion needed, unlike the TF path)."""
import torch
import segmentation_models_pytorch as smp

_DECODERS = {
    "unet": smp.Unet,
    "unetpp": smp.UnetPlusPlus,
    "deeplabv3plus": smp.DeepLabV3Plus,
}


def build_model(encoder="resnet50", decoder="deeplabv3plus", encoder_weights="imagenet", classes=1):
    ew = None if str(encoder_weights).lower() in ("none", "") else encoder_weights
    return _DECODERS[decoder](
        encoder_name=encoder, encoder_weights=ew, in_channels=3, classes=classes
    )


def load_lunit_weights(model, ckpt_path):
    """Load a Lunit (or any torchvision-style ResNet) SSL checkpoint into model.encoder.
    Mirrors the defensive unwrapping we needed for the TF conversion, minus the pain."""
    sd = torch.load(ckpt_path, map_location="cpu")
    for k in ("state_dict", "model", "teacher", "student", "network"):
        if isinstance(sd, dict) and k in sd and isinstance(sd[k], dict):
            sd = sd[k]
            break
    clean = {}
    for k, v in sd.items():
        for p in ("module.", "backbone.", "encoder.", "resnet."):
            if k.startswith(p):
                k = k[len(p):]
        clean[k] = v
    missing, unexpected = model.encoder.load_state_dict(clean, strict=False)
    loaded = len(clean) - len(unexpected)
    print(f"Lunit weights: ~{loaded} tensors loaded | {len(missing)} missing | "
          f"{len(unexpected)} unexpected")
    # money check: if key names didn't line up, almost nothing loads — fail loud.
    assert loaded > 50, "almost nothing loaded — encoder/checkpoint key mismatch; inspect it"
    return model
