import tempfile
import os
import subprocess
import time
from contextlib import nullcontext
from typing import ContextManager, cast
import os
import sys
import uuid
import re
from contextlib import contextmanager
import psutil

# TODO: Maybe inject this somewhere so pix can stay independent.
from context import get_cache
from pix.classes import ShellScriptTask

# Put these in the UI somewhere.
_PIXINSIGHT_EXECUTABLE = (
    "/Applications/PixInsight/PixInsight.app/Contents/MacOS/PixInsight"
)
_DEFAULT_SLOTS: list[int] = [1]


def _find_running_pixinsight_instances() -> dict[int, int]:
    try:
        command = [_PIXINSIGHT_EXECUTABLE, "-e"]
        result = subprocess.run(command, text=True, capture_output=True, check=True)
        output = result.stdout

        slot_pid_map = {}
        lines = output.splitlines()

        # Look for lines with the slot and pid information
        for line in lines:
            match = re.match(r"^\s*(\d+)\s+(\d+)$", line)
            if match:
                slot = int(match.group(1))
                pid = int(match.group(2))
                slot_pid_map[slot] = pid

        return slot_pid_map
    except:
        return []


def _start_pixinsight_instance(slot: int) -> int:
    # This should be start and detach. If it's already started, this
    # attempt should just exit immediately with no other effect.
    subprocess.Popen(
        [
            _PIXINSIGHT_EXECUTABLE,
            f"-n={slot}",
            "--no-startup-check-updates",
            "--no-startup-gui-messages",
            "--no-splash",
            "--without-printer",
        ],
        start_new_session=True,
    )

    print(f"Started new PixInsight instance at slot {slot}")

    while slot not in _find_running_pixinsight_instances():
        time.sleep(5)

    return slot


def _kill_pixinsight_instance(slot: int):
    pid_map = _find_running_pixinsight_instances()
    pid = pid_map.get(slot)
    if pid:
        try:
            print(f"Killing PixInsight slot {slot} (PID {pid}) due to timeout...")
            os.kill(pid, 9)
            psutil.Process(pid).wait()
            print(f"PID {pid} has exited.")
        except psutil.NoSuchProcess:
            print(f"PID {pid} already exited.")
        except OSError as e:
            print(f"Failed to kill PID {pid}: {e}")


def _save_javascript_file(tmp_dir: str, task: ShellScriptTask) -> ShellScriptTask:
    if task.javascript and not task.javascript_filename:
        script_filename = os.path.join(tmp_dir, f"{uuid.uuid4()}.js")
        with open(script_filename, "w") as f:
            print(f"### Writing JavaScript to file {script_filename}")
            print(task.javascript)
            f.write(task.javascript)

        task.javascript_filename = script_filename

    return task


def _create_run_command(task: ShellScriptTask) -> str:
    arguments = " ".join(
        f'-a="{name}={value}"' for name, value in task.arguments.items()
    )
    parameters = " ".join(
        f'-p="{name},{value}"' for name, value in task.parameters.items()
    )
    return f'run -x {arguments} {parameters} "{task.javascript_filename}"'


def prepare_cache(prefix="pixinsight-slot-"):
    print(f"### Preparing pix cache...")
    cache = get_cache()
    for key in cache.iterkeys():
        if isinstance(key, str) and key.startswith(prefix):
            del cache[key]


@contextmanager
def _acquire_slot(slots=None):
    cache = get_cache()

    if slots is None:
        slots = _DEFAULT_SLOTS

    pid = os.getpid()

    while True:
        for slot in slots:
            key = f"pixinsight-slot-{slot}"

            acquired = False
            with cache.transact():
                if key not in cache:
                    cache[key] = pid
                    acquired = True

            if acquired:
                try:
                    yield slot
                finally:
                    del cache[key]
                return

        # All slots were taken — back off briefly
        time.sleep(1)


# TODO: Maybe allow multiple PixInsight instances.
# TODO: Add a start sentinel that gets written to disk when the script starts.
# If, after some timeout, the script hasn't started, kill it and either retry
# or more on to the next one. Obey Ray's retry mechanism.
def run_script(
    tasks: list[ShellScriptTask],
    slots: list[int] | None = None,
    tmp_dir_in: str | None = None,
    close_after: bool = True,
    sleep_after: bool = False,
    timeout: float | None = None,
    force_to_foreground: bool = False,
) -> None:
    assert tasks

    with _acquire_slot(slots) as slot:
        assert slot is not None
        _start_pixinsight_instance(slot)

        with (
            tempfile.TemporaryDirectory()
            if tmp_dir_in is None
            else cast(ContextManager, nullcontext(tmp_dir_in))
        ) as tmp_dir:
            start_sentinel_filename = os.path.join(tmp_dir, "start_sentinel.txt")
            end_sentinel_filename = os.path.join(tmp_dir, "end_sentinel.txt")
            script_filename = os.path.join(tmp_dir, "script.scp")

            tasks = [_save_javascript_file(tmp_dir, t) for t in tasks]
            script = f'![echo "start" > {start_sentinel_filename}\n'
            script += "\n\n".join(_create_run_command(t) for t in tasks) + "\n\n"
            if close_after:
                script += f"close --force *\n"
            if sleep_after:
                script += f"!sleep 30\n"
            script += f'![echo "end" > {end_sentinel_filename}\n'

            with open(script_filename, "w") as f:
                print("### toplevel script is", script)
                f.write(script)

            print(f"Running PixInsight shell script...")

            start_time = time.monotonic()
            ret = subprocess.run(
                [_PIXINSIGHT_EXECUTABLE, f"--execute={slot}:{script_filename}"]
            ).returncode
            print(f"--execute returned")

            while not os.path.exists(start_sentinel_filename):
                if timeout and time.monotonic() - start_time > timeout:
                    _kill_pixinsight_instance(slot)
                    _start_pixinsight_instance(slot)

                # TODO: Bring the PixInsight window to the front on other platforms, too.
                # TODO: Does this work with any / all slots?
                if force_to_foreground:
                    print(f"Forcing PixInsight to the foreground...")
                    if sys.platform == "darwin":
                        subprocess.run(
                            """osascript -e 'tell application "PixInsight" to activate'""",
                            shell=True,
                        )

                time.sleep(5)

            while not os.path.exists(end_sentinel_filename):
                if timeout and time.monotonic() - start_time > timeout:
                    _kill_pixinsight_instance(slot)
                    raise Exception(
                        f"### PixInsight script did not finish within {timeout} seconds"
                    )
                time.sleep(1)
