import requests
import json


def get_node_id(node_api_url):
  response = requests.post(f"{node_api_url}/api/v0/id")
  response.raise_for_status()
  return response.json()['ID']


# Helper function to add a file to IPFS and return its CID
def add_file(node_api_url, file_path):
  files = {'file': open(file_path, 'rb')}
  response = requests.post(f"{node_api_url}/api/v0/add", files=files)
  files['file'].close()
  response.raise_for_status()
  return response.json()['Hash']


# Helper function to provide a file on IPFS
def provide_file(node_api_url, cid):
  response = requests.post(f"{node_api_url}/api/v0/routing/provide?arg={cid}")
  response.raise_for_status()


def cat_file(node_api_url, cid):
  response = requests.post(f'{node_api_url}/api/v0/cat?arg={cid}')
  response.raise_for_status()
  return response.content


# Helper function to publish metadata to IPNS (using default IPNS name)
def publish_name(node_api_url, cid):
  response = requests.post(f"{node_api_url}/api/v0/name/publish?arg={cid}")
  response.raise_for_status()
  return response.json()


# Helper function to get metadata from IPNS (using default IPNS name)
def resolve_name(node_api_url, ipns_name):
  response = requests.post(
      f"{node_api_url}/api/v0/name/resolve?arg={ipns_name}")
  response.raise_for_status()
  return response.json()['Path']


# Helper function to find providers for a CID.
def find_providers(node_api_url, cid):
  response = requests.post(f"{node_api_url}/api/v0/routing/findprovs?arg={cid}")
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


def get_file(node_api_url, cid):
  response = requests.post(f"{node_api_url}/api/v0/get?arg={cid}")
  response.raise_for_status()
  return response


# Helper function to get the status of a downloaded file using /dag/stat
def check_download_status(node_api_url, cid):
  response = requests.post(
      f"{node_api_url}/api/v0/dag/stat?arg={cid}&progress=false")
  response.raise_for_status()
  return response.json()['Ratio']
