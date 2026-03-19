from flask import request, jsonify, Blueprint, Response

from context import context
from customization.settings import Settings, StackSettings
from state.actor import state

# I need to figure out two things:
#
# 1. gunicorn is starting 20+ workers, but each one has its own _action_queue.
# I need to find a way for each worker to send actions to the single worker.
#
# Maybe the best way to do this is just to use the diskcache with an atomic
# transaction.
#
# But I can't serialize Graphs, which might be a significant problem.
#
# On the hand, creating Graphs inside the web server is a poor separation of
# concerns. The UI should just pass along actions to the AstroFlow business
# logic, which can then create and run Graphs.
#
# So I should have a global Ray singleton actor that responds to actions. What
# are some examples of actions? Basically the same stuff that it's in the CLI!
#
# The Ray actor can certainly accept actions via function calls. Maybe it
# should only do that.
#
# The web server already makes heavy use of the main AstroFlow diskcache --
# it serves content out of it -- so it might as well use it for action dispatch.
# The workers can just drop actions into the queue.
#
# Who calls main_event_loop() to service that queue? main.py calls it, so it
# runs in that main process. ALL of the gunicorn workers, even the router
# instance, run in separate processes.
#
# Another problem is that the SSE announcer is duplicated on every gunicorn
# worker. I can make the SSE endpoint a @stateful_route, which will at least
# ensure that it only runs on a deterministic port with a single worker.
#
# Then how does that worker get announcements? Through a different queue,
# of course!

settings_bp = Blueprint("settings", __name__)


@settings_bp.route("/update_settings/stack", methods=["POST"])
def update_stack_settings():
    port = request.environ.get("SERVER_PORT")
    print(f"### update_stack_settings SERVER_PORT {port}")

    data = request.get_json()
    stack_id = data["stack_id"]
    run_id = data["run_id"]
    dependencies = data["dependencies"]
    print(f"### update_stack_settings {stack_id=} {dependencies=}")

    # TODO: Watch out for race conditions here.
    settings = context.settings
    stack_settings = settings.stack_settings.get(run_id, StackSettings())
    stack_settings.dependencies |= dependencies
    settings.stack_settings[run_id] = stack_settings
    context.settings = settings
    print(f"### context.settings is now {context.settings}")

    # TODO: Maybe don't pass the entire settings struct?
    state.enqueue(
        dict(type="update_settings", stack_id=stack_id, settings=settings)
    )

    return jsonify({"status": "ok"})


@settings_bp.route("/quarantine_stack/<run_id>")
def quarantine_stack(run_id):
    context.quarantined_stacks = tuple(
        set(context.quarantined_stacks or tuple()) | {run_id}
    )

    return Response(status=200)
