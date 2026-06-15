import socket
import struct
import pickle
import cv2
import numpy as np

from landmarks_unified import LandmarksEngine


HOST = "127.0.0.1"
PORT = 5055


def recv_all(sock, size):
    data = b""
    while len(data) < size:
        packet = sock.recv(size - len(data))
        if not packet:
            return None
        data += packet
    return data


def recv_message(sock):
    header = recv_all(sock, 4)
    if header is None:
        return None
    msg_len = struct.unpack("!I", header)[0]
    payload = recv_all(sock, msg_len)
    return payload


def send_message(sock, payload_bytes):
    header = struct.pack("!I", len(payload_bytes))
    sock.sendall(header + payload_bytes)


def encode_image_to_jpg_bytes(img_bgr, quality=90):
    encode_params = [int(cv2.IMWRITE_JPEG_QUALITY), quality]
    ok, enc = cv2.imencode(".jpg", img_bgr, encode_params)
    if not ok:
        return None
    return enc.tobytes()


def decode_jpg_bytes_to_bgr(jpg_bytes):
    arr = np.frombuffer(jpg_bytes, np.uint8)
    return cv2.imdecode(arr, cv2.IMREAD_COLOR)


def serialize_result(result):
    out = {}

    out["success"] = result["success"]

    if result["image"] is not None:
        out["image_jpg"] = encode_image_to_jpg_bytes(result["image"])
    else:
        out["image_jpg"] = None

    if result["landmarks"] is not None:
        lm = {}
        for k, v in result["landmarks"].items():
            if v is None:
                lm[k] = None
            else:
                lm[k] = [float(v[0]), float(v[1])]
        out["landmarks"] = lm
    else:
        out["landmarks"] = None

    if result["keypoints"] is not None:
        out["keypoints"] = result["keypoints"].tolist()
    else:
        out["keypoints"] = None

    if result["scores"] is not None:
        out["scores"] = result["scores"].tolist()
    else:
        out["scores"] = None

    return pickle.dumps(out)


def deserialize_request(payload):
    return pickle.loads(payload)


def main():
    engine = LandmarksEngine(device="cuda:0", thr=0.3)

    server = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    server.bind((HOST, PORT))
    server.listen(1)

    print(f"Servidor escuchando en {HOST}:{PORT}")

    while True:
        conn, addr = server.accept()
        print(f"Cliente conectado: {addr}")

        try:
            while True:
                payload = recv_message(conn)
                if payload is None:
                    break

                req = deserialize_request(payload)
                jpg_bytes = req["image_jpg"]

                frame = decode_jpg_bytes_to_bgr(jpg_bytes)
                if frame is None:
                    response = pickle.dumps({"success": False, "error": "No se pudo decodificar la imagen"})
                    send_message(conn, response)
                    continue

                result = engine.process_frame(frame)
                response = serialize_result(result)
                send_message(conn, response)

        except Exception as e:
            print(f"Error con cliente {addr}: {e}")

        finally:
            conn.close()
            print(f"Cliente desconectado: {addr}")


if __name__ == "__main__":
    main()
