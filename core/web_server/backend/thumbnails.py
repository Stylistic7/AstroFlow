import io
import flask
from flask import Blueprint, abort

from context import context
from stacks.composite_stack import Stack
from targets.types import Target
from mosaics.types import Mosaic
from products.registry import get_producer, MediaType
from previews.types import QualityLevel
from stacks.run import Run
from stacks.composite_stack import CalibrationStack

thumbnails_bp = Blueprint("thumbnails", __name__)


def get_thumbnail(object_type: type, *args):
    for level in [
        QualityLevel.POST_PROCESSED,
        QualityLevel.MASTER,
        QualityLevel.SUBFRAME,
    ]:
        if fn := get_producer(MediaType.IMAGE, object_type, level):
            if hasattr(fn, "thumbnail_fn"):
                if thumbnail := fn.thumbnail_fn(*args):
                    resp = flask.Response(
                        thumbnail, mimetype="image/png", direct_passthrough=True
                    )
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
    for level in [
        QualityLevel.POST_PROCESSED,
        QualityLevel.MASTER,
        QualityLevel.SUBFRAME,
    ]:
        if fn := get_producer(MediaType.VIDEO, object_type, level):
            if hasattr(fn, "video_fn"):
                if video := fn.video_fn(*args):
                    buffer = io.BytesIO()
                    buffer.write(video)
                    buffer.seek(0)
                    return flask.send_file(buffer, mimetype="video/webm")

    return abort(404)


@thumbnails_bp.route("/video/stack/<stack_id>")
def stack_video(stack_id: str):
    return get_video(Stack, context.stacks[stack_id].get())
