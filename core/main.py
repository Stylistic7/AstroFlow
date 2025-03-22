from state import Action, enqueue_action, main_event_loop


def main():
  enqueue_action(Action(type='init'))
  main_event_loop()


if __name__ == '__main__':
  main()
