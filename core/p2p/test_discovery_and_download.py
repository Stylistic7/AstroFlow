import unittest
import time
import hashlib
import json
import tempfile
import uuid

from p2p.test_ipfs_node import start_test_ipfs_node, stop_test_ipfs_node
from p2p.ipfs_api import get_node_id, add_file, provide_file, cat_file, publish_name, resolve_name, find_providers, get_file, check_download_status


class TestIPFSDiscoveryAndDownload(unittest.TestCase):

  def setUp(self):
    self.node_A, self.temp_dir_A = start_test_ipfs_node(5501)
    self.node_B, self.temp_dir_B = start_test_ipfs_node(5502)

    # Give time for the nodes to start up
    time.sleep(10)

  def tearDown(self):
    stop_test_ipfs_node(self.node_A, self.temp_dir_A)
    stop_test_ipfs_node(self.node_B, self.temp_dir_B)

  def test_ipfs_nodes_interaction(self):
    api_A = "http://127.0.0.1:5501"
    api_B = "http://127.0.0.1:5502"

    discovery_id = f"AstroFlow v1 {uuid.uuid4()}"

    print('### Getting node IDs...')
    node_id_A = get_node_id(api_A)
    node_id_B = get_node_id(api_B)
    print(f'### node_id_A {node_id_A} node_id_B {node_id_B}')

    # Node A adds and provides a small file containing "AstroFlow v1"
    print("### Adding discovery file to IPFS on Nodes A")
    print(f'### discovery_id {discovery_id}')
    discovery_file = tempfile.NamedTemporaryFile(delete=False)
    discovery_file.write(discovery_id.encode('utf-8'))
    discovery_file.close()
    discovery_cid = add_file(api_A, discovery_file.name)
    print(f'### discovery_cid {discovery_cid}')
    provide_file(api_A, discovery_cid)

    # Node A adds and provides a file from the testdata directory
    print('### Adding image to IPFS on Node A')
    image_file_A = "testdata/input/Light_M42_180.0s_Bin1_2600MM_20231130-233344_0021.fit"  # Replace with your path
    with open(image_file_A, 'rb') as f:
      image_data_original = f.read()
    checksum_original = hashlib.md5(image_data_original).hexdigest()
    print(f'### len(image_data_original) {len(image_data_original)} '
          f'checksum_original {checksum_original}')
    cid_image_A = add_file(api_A, image_file_A)
    provide_file(api_A, cid_image_A)

    # Node A publishes metadata with the image file CID
    print('### Publishing metadata to default IPNS name on Node A')
    metadata_A = tempfile.NamedTemporaryFile(delete=False)
    metadata = {"image_file": cid_image_A}
    with open(metadata_A.name, 'w') as f:
      json.dump(metadata, f)
    cid_metadata_A = add_file(api_A, metadata_A.name)
    print(f'### cid_metadata_A {cid_metadata_A}')
    publish_name(api_A, cid_metadata_A)

    # Node B uses findprovs to find nodes providing "AstroFlow v1"
    print('### Finding providers of the discovery file on Node B')
    other_nodes = find_providers(api_B, discovery_cid)
    print(f'### other_nodes {other_nodes}')
    self.assertEqual(len(other_nodes), 1)
    self.assertEqual(other_nodes[0], node_id_A)

    # Node B retrieves the metadata from Node A's IPNS entry
    print("### Getting file for Node A's default IPNS name on Node B")
    cid_metadata_B = resolve_name(api_B, f'/ipns/{node_id_A}')
    print(f'### cid_metadata_B {cid_metadata_B}')
    metadata_B = json.loads(cat_file(api_B, cid_metadata_B))
    print(f'### metadata_B {metadata_B}')
    image_cid = metadata_B['image_file']

    # Node B downloads the image file
    print('### Downloading image file on Node B')
    get_file(api_B, image_cid)

    # Poll the /dag/stat API until the download is complete
    print('### Polling until download is done...')
    while True:
      if check_download_status(api_B, image_cid) == 1:
        break
      time.sleep(1)

    image_data_downloaded = cat_file(api_B, image_cid)
    print(f'### type(image_data_downloaded) {type(image_data_downloaded)}')
    checksum_downloaded = hashlib.md5(image_data_downloaded).hexdigest()
    print(f'### len(image_data_downloaded) {len(image_data_downloaded)} '
          f'checksum_downloaded {checksum_downloaded}')

    # Compare MD5 checksums of the original file and the downloaded file
    self.assertEqual(checksum_original, checksum_downloaded,
                     "Checksum mismatch!")


if __name__ == "__main__":
  unittest.main()
