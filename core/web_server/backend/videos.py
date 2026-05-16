import os
import io
import flask
from flask import Blueprint, abort

from context import local_context as context
from stacks.composite_stack import CalibrationStack
from stacks.composite_stack import Stack
from products.registry import get_producer, MediaType
from products.identity import StackId, CalibrationStackId
from previews.types import QUALITY_PRIORITY

videos_bp = Blueprint("videos", __name__)


def get_video(identity: type, *args):
    for level in QUALITY_PRIORITY:
        if fn := get_producer(MediaType.VIDEO, identity, level):
            if hasattr(fn, "video_fn"):
                if video := fn.video_fn(*args):
                    buffer = io.BytesIO()
                    buffer.write(video)
                    buffer.seek(0)
                    return flask.send_file(buffer, mimetype="video/webm")

    return abort(404)


@videos_bp.route("/video/stack/<stack_id>")
def stack_video(stack_id: str):
    return get_video(StackId, context.stacks[stack_id].get())


@videos_bp.route("/video/calibration_stack/<stack_id>")
def calibration_stack_video(stack_id: str):
    return get_video(CalibrationStackId, context.calibration_stacks[stack_id])
