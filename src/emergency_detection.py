"""Emergency-priority helper.

The bundled YOLO model is a general COCO model and cannot reliably identify
ambulances or fire engines. Therefore this module does not classify trucks as
emergency vehicles. A production implementation should use a custom trained
emergency-vehicle detector.
"""


def emergency_priority(enabled: bool = False) -> dict:
    """Return the signal action for the manually controlled emergency demo."""
    return {
        "emergency": bool(enabled),
        "signal": "GREEN" if enabled else "NORMAL",
        "message": "EMERGENCY PRIORITY - GREEN" if enabled else "NORMAL TRAFFIC CONTROL",
    }
