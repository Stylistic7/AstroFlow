from state.gunicorn_control import GunicornConfig


def guincorn_configs() -> list[GunicornConfig]:
  return [
      GunicornConfig(name='static_server',
                     config_file=None,
                     module="web_server.backend.app:app",
                     worker_type='sync',
                     num_servers=1,
                     start_port=8000,
                     num_workers=None,
                     timeout=300),
      GunicornConfig(name='stream_server',
                     config_file="web_server/backend/gunicorn_config.py",
                     module="web_server.backend.stream:app",
                     worker_type='gevent',
                     num_servers=1,
                     start_port=8001,
                     num_workers=1,
                     timeout=0),
      GunicornConfig(name='tile_server',
                     config_file=None,
                     module="web_server.backend.app:app",
                     worker_type='sync',
                     num_servers=4,
                     start_port=8002,
                     num_workers=1,
                     timeout=300)
  ]
