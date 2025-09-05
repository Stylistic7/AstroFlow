from state.state_machine import Action, enqueue_action, main_event_loop
from dependency_graph.serialization import enable_function_fqname_pickling


def main():
  enable_function_fqname_pickling()
  enqueue_action(Action(type='init'))
  main_event_loop()


if __name__ == '__main__':
  main()
