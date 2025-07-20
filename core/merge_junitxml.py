import sys
import xml.etree.ElementTree as ET
from pathlib import Path


def merge_junitxml(paths: list[str]) -> ET.Element:
  testsuite = ET.Element("testsuite")
  total = {"tests": 0, "failures": 0, "errors": 0, "skipped": 0}

  for path in paths:
    root = ET.parse(path).getroot()
    suite = root.find(
        "testsuite"
    ) or root  # handle both <testsuites><testsuite> and bare <testsuite>
    for case in suite.findall("testcase"):
      testsuite.append(case)
    for k in total:
      total[k] += int(suite.attrib.get(k, 0))

  for k, v in total.items():
    testsuite.set(k, str(v))
  return testsuite


if __name__ == "__main__":
  if len(sys.argv) < 2:
    print(
        "Usage: python merge_junitxml.py file1.xml file2.xml ... > merged.xml",
        file=sys.stderr)
    sys.exit(1)

  merged = merge_junitxml(sys.argv[1:])
  tree = ET.ElementTree(merged)
  ET.indent(tree, space="  ")  # Python 3.9+
  tree.write(sys.stdout, encoding="unicode", xml_declaration=True)
