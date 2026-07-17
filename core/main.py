import atexit
import subprocess

from base_path import get_bin_dir
from state.state_machine import Action, enqueue_action, main_event_loop
from dependency_graph.serialization import enable_function_fqname_pickling


def _force_cleanup():
    ray_path = get_bin_dir() / "ray"
    subprocess.run(["pkill", "-f", "gunicorn"], capture_output=True)
    # Kill any PixInsight too. The pix runner keeps a persistent out-of-process
    # PI instance; on a CRASH (Ray/raylet dies, main.py exits with an error) it's
    # orphaned — its in-flight WBPP output is never cached (the dispatching Ray
    # task is gone), it can sit on tens of GB of RAM, and a survivor collides
    # with the next launch's PI ("yielded execution to running instance" →
    # SIGSEGV). This atexit fires on clean exit AND on error exit, so a failed
    # app cleans up its own PI. (SIGKILL of main.py itself can still skip atexit;
    # `make stop` remains the belt-and-suspenders.)
    subprocess.run(["pkill", "-f", "PixInsight"], capture_output=True)
    subprocess.run([ray_path, "stop"], capture_output=True)


def main():
    atexit.register(_force_cleanup)
    enable_function_fqname_pickling()
    enqueue_action(Action(type="init"))
    main_event_loop()


if __name__ == "__main__":
    main()
