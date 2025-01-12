import subprocess
import os
import tempfile
import shutil


def start_test_ipfs_node(api_port: int):
  """Initialize and configure a new IPFS node for test purposes."""
  # Create a temporary directory for the node's data.
  temp_dir = tempfile.mkdtemp()
  os.environ['IPFS_PATH'] = temp_dir

  # Initialize the repo.
  subprocess.run(['ipfs', 'init'], check=True)

  # Disable the web UI and gateway.
  subprocess.run(['ipfs', 'config', 'Addresses.Gateway', ''], check=True)

  # Prevent the IPFS node from downloading stuff off the internet.
  subprocess.run(['ipfs', 'config', 'Repo.AutoFetch', 'false'], check=True)
  subprocess.run(['ipfs', 'config', 'Repo.Provider', 'false'], check=True)

  # Start the IPFS daemon.
  ipfs_daemon = subprocess.Popen(
      ['ipfs', 'daemon', '--api', f'/ip4/127.0.0.1/tcp/{api_port}'])

  return ipfs_daemon, temp_dir


def stop_test_ipfs_node(ipfs_daemon, temp_dir):
  ipfs_daemon.terminate()
  ipfs_daemon.wait()
  shutil.rmtree(temp_dir)
