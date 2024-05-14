__all__ = []

try:
    from medkit.audio.segmentation.pa_speaker_detector import PASpeakerDetector

    __all__ += ["PASpeakerDetector"]
except ModuleNotFoundError:
    pass

try:
    from medkit.audio.segmentation.webrtc_voice_detector import WebRTCVoiceDetector

    __all__ += ["WebRTCVoiceDetector"]
except ModuleNotFoundError:
    pass
