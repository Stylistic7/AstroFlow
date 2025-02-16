import threading

from web_server.backend.app import app

stop_event = threading.Event()


def flask_thread_body():
  app.run(host='0.0.0.0', port=80)


def start_flask():
  flask_thread = threading.Thread(target=flask_thread_body)
  flask_thread.start()
