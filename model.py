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
    enc_keys = set(model.encoder.state_dict().keys())
    matched = sum(1 for k in clean if k in enc_keys)
    # NB: smp encoders override load_state_dict and may return None -> don't unpack it.
    model.encoder.load_state_dict(clean, strict=False)
    print(f"Lunit weights: {matched}/{len(clean)} tensors matched the encoder "
          f"(encoder has {len(enc_keys)})")
    # money check: if key names didn't line up, almost nothing matches — fail loud.
    assert matched > 50, "almost nothing matched — encoder/checkpoint key mismatch; inspect it"
    return model
