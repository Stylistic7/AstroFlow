import subprocess


def print_tree(path):
    result = subprocess.run(
        ["tree"],
        cwd=path,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        check=True,
    )
    print(f"Contents of {path} are {result.stdout}")
