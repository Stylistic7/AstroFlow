import curses
import time
import random
import sys
from datetime import timedelta
import queue


# Placeholder function to simulate node status updates
def get_node_status():
  """Simulate getting node status"""
  node_types = ['stack_files', 'merge_images', 'process_data']
  running_nodes = {node_type: random.randint(0, 10) for node_type in node_types}
  total_nodes = {
      node_type: 10 for node_type in node_types
  }  # Example: each task has 10 nodes in total
  exceptions = random.sample(node_types, random.randint(0, len(node_types)))
  eta = timedelta(minutes=random.randint(30, 90))
  return running_nodes, total_nodes, exceptions, eta


def spinner(queue):
  """Spinner function running in the background"""
  symbols = '|/-\\'
  while True:
    for symbol in symbols:
      queue.put(symbol)
      time.sleep(0.2)


def parse_command(command):
  """Parse command like `astroflow merge mosaic abc123`"""
  parts = command.strip().split()
  if len(parts) < 2:
    raise ValueError("Command must include a verb and noun.")

  verb = parts[0]
  noun = parts[1]
  arguments = parts[2:] if len(parts) > 2 else []

  return verb, noun, arguments


def run_cli(stdscr, spinner_queue):
  """Main CLI loop"""
  curses.curs_set(0)  # Hide cursor
  stdscr.nodelay(1)  # Non-blocking input
  stdscr.timeout(1000)  # Update screen every second

  # Window setup
  height, width = stdscr.getmaxyx()
  start_x = 2
  start_y = 2

  # Parse the command from the arguments
  command = ' '.join(sys.argv[1:])  # Command is taken from the arguments
  try:
    verb, noun, arguments = parse_command(command)
    stdscr.addstr(0, start_x, f"Executing: {verb} {noun} {' '.join(arguments)}",
                  curses.A_BOLD)
  except ValueError as e:
    stdscr.addstr(0, start_x, f"Error: {str(e)}", curses.A_BOLD)
    stdscr.refresh()
    time.sleep(2)
    return

  while True:
    # Get the status of the nodes
    running_nodes, total_nodes, exceptions, eta = get_node_status()

    stdscr.clear()  # Clear the screen

    # Title of the screen
    stdscr.addstr(0, start_x, "Compute Graph Progress", curses.A_BOLD)

    # Display running nodes with spinner on the left and "2/10"
    row = start_y
    for node_type, count in running_nodes.items():
      try:
        # Get the spinner symbol from the queue
        spinner_symbol = spinner_queue.get(block=False)
      except queue.Empty:
        spinner_symbol = ' '  # If the queue is empty, just leave the spinner blank
      total_count = total_nodes.get(node_type, 0)
      stdscr.addstr(
          row, start_x,
          f"[{spinner_symbol}] Running {node_type} tasks: {count}/{total_count}"
      )
      row += 1

    # Display exceptions
    if exceptions:
      stdscr.addstr(row, start_x, "Exceptions:")
      row += 1
      for exception in exceptions:
        stdscr.addstr(row, start_x, f"- {exception}")
        row += 1

    # Display ETA
    stdscr.addstr(row, start_x, f"Estimated Time Remaining: {eta}")
    row += 1

    # Refresh the window
    stdscr.refresh()

    # Handle keypresses (esc to quit)
    key = stdscr.getch()
    if key == 27:  # ESC key
      break


def main():
  """
    Main function to run the `astroflow` CLI.
    It parses the command and then invokes the curses-based CLI.
    """
  if len(sys.argv) < 3:
    print("Usage: astroflow <verb> <noun> [arguments...]")
    sys.exit(1)

  try:
    # The first two arguments are the verb and noun
    command = ' '.join(sys.argv[1:])  # Command passed from the executable
    print(f"Executing command: {command}")

    # Initialize the queue for the spinner symbols
    spinner_queue = queue.Queue()

    # Start the spinner in a separate thread
    import threading
    threading.Thread(target=spinner, args=(spinner_queue,), daemon=True).start()

    # Run the curses-based interface
    curses.wrapper(run_cli, spinner_queue)  # Start the curses-based CLI

  except ValueError as e:
    print(f"Error: {str(e)}")
    sys.exit(1)


if __name__ == "__main__":
  main()
