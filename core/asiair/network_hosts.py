import netifaces as net
import ipaddress as ip
import traceback
from concurrent.futures import ThreadPoolExecutor
import socket
from contextlib import closing
import itertools


def check_socket(host, port):
    with closing(socket.socket(socket.AF_INET, socket.SOCK_STREAM)) as sock:
        sock.settimeout(0.5)
        return sock.connect_ex((host, port)) == 0


def find_hosts_with_open_port(port):
    hosts = []
    for iface in net.interfaces():
        print(f"looking at interface {iface}")
        try:
            networks = net.ifaddresses(iface)
            print(f"got networks", networks)
            if net.AF_INET not in networks:
                continue
            networks = networks[net.AF_INET]
            print(f"interface {iface} has networks iwth AF_INET {networks}")
            for network in networks:
                print(f"looking at network {network}")
                if "broadcast" not in network:
                    continue
                if "netmask" not in network:
                    continue
                net_string = f"{network['broadcast']}/{network['netmask']}"
                print(f"net_string {net_string}")
                network = ip.ip_network(net_string, strict=False)
                print(f"network {network}")
                hosts.extend(list(network.hosts()))
        except:
            pass

    hosts = [str(h) for h in hosts]
    hosts = list(set(hosts))
    print(f"### checking {len(hosts)} hosts")

    active_hosts = []
    with ThreadPoolExecutor(max_workers=32) as executor:
        ports_open = executor.map(check_socket, hosts, itertools.repeat(port))

        # TODO: Look for the ASIAIR identification, instead of just the port being open.
        for host, port_open in zip(hosts, ports_open):
            if port_open:
                print(f"host {host} has port open")
                active_hosts.append(host)

    return active_hosts


if __name__ == "__main__":
    hosts = find_hosts_with_open_port(4700)
    print("Found hosts", hosts)
