import miniupnpc
import time
import traceback

def map_local(port):
    while True:
        try:
            upnp = miniupnpc.UPnP()

            upnp.discoverdelay = 10
            upnp.discover()
            upnp.selectigd()

            # addportmapping(external-port, protocol, internal-host, internal-port, description, remote-host)
            upnp.addportmapping(port, 'TCP', upnp.lanaddr, port, 'StableDiffusion', '')
            print(f"added upnp port mapping, forwarding {upnp.externalipaddress()}:{port}")
        except Exception:
            print("failed to map upnp")
            traceback.print_exc()
        time.sleep(600)
