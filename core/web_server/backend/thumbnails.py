import io
import flask
from flask import Blueprint, abort

from context import local_context as context
from stacks.composite_stack import Stack
from targets.types import Target
from mosaics.types import Mosaic
from products.registry import get_producer, MediaType
from previews.types import QualityLevel, Implementation
from stacks.run import Run
from stacks.composite_stack import CalibrationStack

thumbnails_bp = Blueprint("thumbnails", __name__)


_ALL_LEVELS = [
    QualityLevel.POST_PROCESSED,
    QualityLevel.MASTER,
    QualityLevel.DOWNSCALED_MASTER,
    QualityLevel.SUBFRAME,
]


def _thumbnail_response(fn, level, *args):
    """Try to render a thumbnail from a specific producer."""
    if fn is None or not hasattr(fn, "thumbnail_fn"):
        return None
    thumbnail = fn.thumbnail_fn(*args)
    if not thumbnail:
        return None
    resp = flask.Response(thumbnail, mimetype="image/png", direct_passthrough=True)
    resp.headers["X-Quality-Level"] = level.value
    return resp


def get_thumbnail(object_type: type, *args):
    """Return the best-available thumbnail (used by convenience routes)."""
    implementation = context.settings.implementation
    for level in _ALL_LEVELS:
        fn = get_producer(MediaType.IMAGE, object_type, level, implementation)
        resp = _thumbnail_response(fn, level, *args)
        if resp:
            return resp
    return abort(404)


def get_specific_thumbnail(object_type: type, implementation: Implementation,
                           level: QualityLevel, *args):
    """Return the thumbnail for an exact (implementation, level)."""
    fn = get_producer(MediaType.IMAGE, object_type, level, implementation)
    resp = _thumbnail_response(fn, level, *args)
    if resp:
        return resp

    # Fallback: for mosaics, serve the in-progress merge thumbnail so the
    # dashboard has something to show before the final store runs. args is
    # (mosaic, filter); merge_images keys its thumbnail by (mode, mosaic, filter).
    if object_type is Mosaic:
        from mosaics.merge_images import MergedImage
        merge_fn = get_producer(MediaType.IMAGE, MergedImage, level, implementation)
        if merge_fn is not None and hasattr(merge_fn, "thumbnail_fn"):
            resp = _thumbnail_response(merge_fn, level, level, *args)
            if resp:
                return resp

    return abort(404)


@thumbnails_bp.route("/thumbnail/stack/<stack_id>")
def stack_thumbnail(stack_id: str):
    return get_thumbnail(Stack, context.stacks[stack_id].get())


@thumbnails_bp.route("/thumbnail/target/<target_id>")
def target_thumbnail(target_id: str):
    return get_thumbnail(Target, context.targets[target_id].get())


@thumbnails_bp.route("/thumbnail/mosaic/<mosaic_id>/<filter>")
def mosaic_thumbnail(mosaic_id: str, filter: str):
    return get_thumbnail(Mosaic, context.mosaics[mosaic_id].get(), filter)


@thumbnails_bp.route("/thumbnail/mosaic/<mosaic_id>/<implementation>/<quality>/<filter>")
def mosaic_thumbnail_specific(mosaic_id: str, implementation: str, quality: str, filter: str):
    """Thumbnail for an exact (mosaic, implementation, quality, filter)."""
    return get_specific_thumbnail(
        Mosaic, Implementation(implementation), QualityLevel(quality),
        context.mosaics[mosaic_id].get(), filter,
    )


# TODO: Use better stack types.
@thumbnails_bp.route("/thumbnail/calibration_stack/<stack_id>")
def calibration_stack_thumbnail(stack_id: str):
    return get_thumbnail(CalibrationStack, context.calibration_stacks[stack_id])


@thumbnails_bp.route("/thumbnail/run/<stack_id>/<substack_id>/<flat_stack_id>")
def run_thumbnail(stack_id: str, substack_id: str, flat_stack_id: str):
    stacks = context.stacks
    stack = stacks[stack_id].get()
    substack = stack.contents[substack_id]
    flat_stack = context.calibration_stacks[flat_stack_id]
    return get_thumbnail(Run, stack, substack, flat_stack)


def get_video(object_type: type, *args):
    implementation = context.settings.implementation
    for level in [
        QualityLevel.POST_PROCESSED,
        QualityLevel.MASTER,
        QualityLevel.DOWNSCALED_MASTER,
        QualityLevel.SUBFRAME,
    ]:
        if fn := get_producer(MediaType.VIDEO, object_type, level, implementation):
            if hasattr(fn, "video_fn"):
                if video := fn.video_fn(*args):
                    buffer = io.BytesIO()
                    buffer.write(video)
                    buffer.seek(0)
                    return flask.send_file(buffer, mimetype="video/webm")

    return abort(404)


    # Video endpoints moved to videos.py
