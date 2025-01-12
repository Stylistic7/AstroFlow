import unittest
import subprocess
import time
import requests
import hashlib
import json
import os
import tempfile
import shutil
import uuid
from pathlib import Path


class TestIPFSDiscoveryAndDownload(unittest.TestCase):

  # Helper function to start an IPFS node subprocess
  def start_ipfs_node(node_id: str, ipfs_path: str, api_port: int):
    """Initialize and configure a new IPFS node."""
    # Set the IPFS_PATH environment variable for each node
    os.environ['IPFS_PATH'] = ipfs_path

    # Initialize the repo
    subprocess.run(['ipfs', 'init'], check=True)

    # Disable the web UI and gateway
    subprocess.run(['ipfs', 'config', 'Addresses.Gateway', ''], check=True)

    # Prevent the IPFS node from downloading stuff off the internet.
    subprocess.run(['ipfs', 'config', 'Repo.AutoFetch', 'false'], check=True)
    subprocess.run(['ipfs', 'config', 'Repo.Provider', 'false'], check=True)

    # Start the IPFS daemon
    ipfs_daemon = subprocess.Popen(
        ['ipfs', 'daemon', '--api', f'/ip4/127.0.0.1/tcp/{api_port}'])

    return ipfs_daemon

  def get_node_id(self, node_api_url):
    response = requests.post(f"{node_api_url}/api/v0/id")
    response.raise_for_status()
    return response.json()['ID']

  # Helper function to add a file to IPFS and return its CID
  def add_file(self, node_api_url, file_path):
    files = {'file': open(file_path, 'rb')}
    response = requests.post(f"{node_api_url}/api/v0/add", files=files)
    files['file'].close()
    response.raise_for_status()
    return response.json()['Hash']

  # Helper function to provide a file on IPFS
  def provide_file(self, node_api_url, cid):
    response = requests.post(f"{node_api_url}/api/v0/routing/provide?arg={cid}")
    response.raise_for_status()

  def cat_file(self, node_api_url, cid):
    response = requests.post(f'{node_api_url}/api/v0/cat?arg={cid}')
    response.raise_for_status()
    return response.content

  # Helper function to publish metadata to IPNS (using default IPNS name)
  def publish_name(self, node_api_url, cid):
    response = requests.post(f"{node_api_url}/api/v0/name/publish?arg={cid}")
    response.raise_for_status()
    return response.json()

  # Helper function to get metadata from IPNS (using default IPNS name)
  def resolve_name(self, node_api_url, ipns_name):
    response = requests.post(
        f"{node_api_url}/api/v0/name/resolve?arg={ipns_name}")
    response.raise_for_status()
    return response.json()['Path']

  # Helper function to find providers for a CID.
  def find_providers(self, node_api_url, cid):
    response = requests.post(
        f"{node_api_url}/api/v0/routing/findprovs?arg={cid}")
    response.raise_for_status()

    # Read and process the NDJSON response line by line
    providers = []
    for line in response.iter_lines():
      if line:
        data = json.loads(line)
        # See discussion: https://github.com/ipfs/js-ipfs/issues/2868
        if data.get("Type") == 4:
          for response in data.get("Responses"):
            providers.append(response.get("ID"))

    return providers

  def get_file(self, node_api_url, cid):
    response = requests.post(f"{node_api_url}/api/v0/get?arg={cid}")
    response.raise_for_status()
    return response

  # Helper function to get the status of a downloaded file using /dag/stat
  def check_download_status(self, node_api_url, cid):
    response = requests.post(
        f"{node_api_url}/api/v0/dag/stat?arg={cid}&progress=false")
    response.raise_for_status()
    return response.json()

  # Setup and teardown for each test case
  def setUp(self):
    # Create temporary directories for two independent IPFS nodes
    self.temp_dir_A = tempfile.mkdtemp()
    self.temp_dir_B = tempfile.mkdtemp()

    # Start two IPFS nodes (Node A and Node B) on fixed ports 5501 and 5502
    self.node_A_process = self.start_ipfs_node(self.temp_dir_A, 5501)
    self.node_B_process = self.start_ipfs_node(self.temp_dir_B, 5502)

    # Give time for the nodes to start up
    time.sleep(10)

  def tearDown(self):
    # Clean up temporary files and processes
    shutil.rmtree(self.temp_dir_A)
    shutil.rmtree(self.temp_dir_B)
    self.node_A_process.terminate()
    self.node_B_process.terminate()
    self.node_A_process.wait()
    self.node_B_process.wait()

  def test_ipfs_nodes_interaction(self):
    api_A = "http://127.0.0.1:5501"
    api_B = "http://127.0.0.1:5502"

    discovery_id = f"AstroFlow v1 {uuid.uuid4()}"

    print('### Getting node IDs...')
    node_id_A = self.get_node_id(api_A)
    node_id_B = self.get_node_id(api_B)
    print(f'### node_id_A {node_id_A} node_id_B {node_id_B}')

    # Node A adds and provides a small file containing "AstroFlow v1"
    print("### Adding discovery file to IPFS on Nodes A")
    print(f'### discovery_id {discovery_id}')
    discovery_file = tempfile.NamedTemporaryFile(delete=False)
    discovery_file.write(discovery_id.encode('utf-8'))
    discovery_file.close()
    discovery_cid = self.add_file(api_A, discovery_file.name)
    print(f'### discovery_cid {discovery_cid}')
    self.provide_file(api_A, discovery_cid)

    # Node A adds and provides a file from the testdata directory
    print('### Adding image to IPFS on Node A')
    image_file_A = "testdata/input/Light_M42_180.0s_Bin1_2600MM_20231130-233344_0021.fit"  # Replace with your path
    with open(image_file_A, 'rb') as f:
      image_data_original = f.read()
    checksum_original = hashlib.md5(image_data_original).hexdigest()
    print(f'### len(image_data_original) {len(image_data_original)} '
          f'checksum_original {checksum_original}')
    cid_image_A = self.add_file(api_A, image_file_A)
    self.provide_file(api_A, cid_image_A)

    # Node A publishes metadata with the image file CID
    print('### Publishing metadata to default IPNS name on Node A')
    metadata_A = tempfile.NamedTemporaryFile(delete=False)
    metadata = {"image_file": cid_image_A}
    with open(metadata_A.name, 'w') as f:
      json.dump(metadata, f)
    cid_metadata_A = self.add_file(api_A, metadata_A.name)
    print(f'### cid_metadata_A {cid_metadata_A}')
    self.publish_name(api_A, cid_metadata_A)

    # Node B uses findprovs to find nodes providing "AstroFlow v1"
    print('### Finding providers of the discovery file on Node B')
    other_nodes = self.find_providers(api_B, discovery_cid)
    print(f'### other_nodes {other_nodes}')
    self.assertEqual(len(other_nodes), 1)
    self.assertEqual(other_nodes[0], node_id_A)

    # Node B retrieves the metadata from Node A's IPNS entry
    print("### Getting file for Node A's default IPNS name on Node B")
    cid_metadata_B = self.resolve_name(api_B, f'/ipns/{node_id_A}')
    print(f'### cid_metadata_B {cid_metadata_B}')
    metadata_B = json.loads(self.cat_file(api_B, cid_metadata_B))
    print(f'### metadata_B {metadata_B}')
    image_cid = metadata_B['image_file']

    # Node B downloads the image file
    print('### Downloading image file on Node B')
    self.get_file(api_B, image_cid)

    # Poll the /dag/stat API until the download is complete
    print('### Polling until download is done...')
    while True:
      stats = self.check_download_status(api_B, image_cid)
      print(f'### stats {stats}')
      if stats['Ratio'] == 1:
        break
      time.sleep(1)

    image_data_downloaded = self.cat_file(api_B, image_cid)
    print(f'### type(image_data_downloaded) {type(image_data_downloaded)}')
    checksum_downloaded = hashlib.md5(image_data_downloaded).hexdigest()
    print(f'### len(image_data_downloaded) {len(image_data_downloaded)} '
          f'checksum_downloaded {checksum_downloaded}')

    # Compare MD5 checksums of the original file and the downloaded file
    self.assertEqual(checksum_original, checksum_downloaded,
                     "Checksum mismatch!")


if __name__ == "__main__":
  unittest.main()
