from .generalized_vl_rcnn_new import GeneralizedVLRCNN_New
from groundingdino_new.models import build_model

_DETECTION_META_ARCHITECTURES = {
                                 "GeneralizedVLRCNN_New": GeneralizedVLRCNN_New,
                                 }


def build_detection_model(cfg, **kwargs):
    if cfg.GROUNDINGDINO.enabled:
        return build_model(cfg.GROUNDINGDINO, cfg)
    else:
        meta_arch = _DETECTION_META_ARCHITECTURES[cfg.MODEL.META_ARCHITECTURE]
        return meta_arch(cfg, **kwargs)
