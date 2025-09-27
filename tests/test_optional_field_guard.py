from rcwa_app.orchestration.session import default_config


def test_optional_field_guard_pattern() -> None:
    # Use the surface from a fully-typed config (no guessing about ctor defaults)
    cfg = default_config()
    surf = cfg.geometry.surface
    duty = getattr(surf, "duty", None)
    if duty is not None:
        assert 0.0 <= float(duty) <= 1.0
