# pose_server_discovery.py
import socket
import threading

DISCOVERY_PORT = 8091
DISCOVERY_REQUEST = "WHO_IS_POSE_SERVER"
DISCOVERY_RESPONSE_PREFIX = "POSE_SERVER"

def get_local_ip():
    s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    try:
        s.connect(("8.8.8.8", 80))
        ip = s.getsockname()[0]
    except Exception:
        ip = "127.0.0.1"
    finally:
        s.close()
    return ip

def discovery_server():
    sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    sock.bind(("", DISCOVERY_PORT))

    print(f"[DISCOVERY] Escuchando UDP en puerto {DISCOVERY_PORT}")

    while True:
        try:
            data, addr = sock.recvfrom(1024)
            message = data.decode("utf-8", errors="ignore").strip()

            if message == DISCOVERY_REQUEST:
                local_ip = get_local_ip()
                response = f"{DISCOVERY_RESPONSE_PREFIX}:{local_ip}:8090"
                sock.sendto(response.encode("utf-8"), addr)
                print(f"[DISCOVERY] Respondido a {addr} con {response}")
        except Exception as e:
            print(f"[DISCOVERY] Error: {e}")

def start_discovery_server():
    thread = threading.Thread(target=discovery_server, daemon=True)
    thread.start()
