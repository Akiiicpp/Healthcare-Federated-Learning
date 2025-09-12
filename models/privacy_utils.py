from typing import Optional, Tuple

try:
    from opacus import PrivacyEngine  # type: ignore
except Exception:  # pragma: no cover
    PrivacyEngine = None  # fallback if opacus is not installed yet


def attach_privacy_engine(optimizer, model, data_loader, noise_multiplier: float = 0.8, max_grad_norm: float = 1.0):
    """
    Attach Opacus PrivacyEngine to optimizer and model if available.
    Returns (model, optimizer, privacy_engine) so caller can track epsilon.
    """
    if PrivacyEngine is None:
        return model, optimizer, None

    privacy_engine = PrivacyEngine()
    model, optimizer, data_loader = privacy_engine.make_private(
        module=model,
        optimizer=optimizer,
        data_loader=data_loader,
        noise_multiplier=noise_multiplier,
        max_grad_norm=max_grad_norm,
    )
    return model, optimizer, privacy_engine
