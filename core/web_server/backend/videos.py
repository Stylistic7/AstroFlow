import io
import flask
from flask import Blueprint, abort

from context import local_context as context
from products.registry import producers_for

videos_bp = Blueprint("videos", __name__)


def get_video(entity, *extra_args):
    """First registered video producer for `type(entity)` that yields a
    stream. Each producer attaches a `video_fn` callable that, given
    the entity args, returns bytes or None."""
    args = (entity,) + extra_args
    for producer in producers_for(type(entity)):
        video_fn = getattr(producer, "video_fn", None)
        if video_fn is None:
            continue
        video = video_fn(*args)
        if video is None:
            continue
        buffer = io.BytesIO()
        buffer.write(video)
        buffer.seek(0)
        return flask.send_file(buffer, mimetype="video/webm")
    return abort(404)


@videos_bp.route("/video/stack/<stack_id>")
def stack_video(stack_id: str):
    return get_video(context.stacks[stack_id].get())


@videos_bp.route("/video/calibration_stack/<stack_id>")
def calibration_stack_video(stack_id: str):
    return get_video(context.calibration_stacks[stack_id])
