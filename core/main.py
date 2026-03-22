import atexit
import subprocess

from base_path import get_bin_dir
from state.state_machine import Action, enqueue_action, main_event_loop
from dependency_graph.serialization import enable_function_fqname_pickling


def _force_cleanup():
    ray_path = get_bin_dir() / "ray"
    subprocess.run(["pkill", "-f", "gunicorn"], capture_output=True)
    subprocess.run([ray_path, "stop"], capture_output=True)


def main():
    atexit.register(_force_cleanup)
    enable_function_fqname_pickling()
    enqueue_action(Action(type="init"))
    main_event_loop()


if __name__ == "__main__":
    main()
