import os
import io
import flask
from flask import Blueprint, abort

from context import context
from stacks.simple_stack import CalibrationStack
from stacks.composite_stack import CompositeStack
from products.registry import get_producer, MediaType
from previews.types import QualityLevel

videos_bp = Blueprint("videos", __name__)


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


@videos_bp.route("/video/stack/<stack_id>")
def stack_video(stack_id: str):
    return get_video(CompositeStack, context.stacks[stack_id].get())


@videos_bp.route("/video/calibration_stack/<stack_id>")
def calibration_stack_video(stack_id: str):
    return get_video(CalibrationStack, context.calibration_stacks[stack_id])
