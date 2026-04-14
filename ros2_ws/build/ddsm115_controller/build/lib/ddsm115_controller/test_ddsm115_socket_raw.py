import serial
import struct
import time
import crcmod.predefined


def crc_attach(data_bytes: bytes) -> bytes:
    crc8 = crcmod.predefined.mkPredefinedCrcFun('crc-8-maxim')
    crc_int = crc8(data_bytes)
    out = bytearray(data_bytes)
    out.append(crc_int)
    return bytes(out)


def main():
    device = "socket://192.168.1.201:4196"
    baudrate = 115200

    ser = serial.serial_for_url(device, baudrate=baudrate, timeout=1.0)

    print(f"Opened {device} at {baudrate} baud")

    for motor_id in range(1, 11):
        try:
            ser.reset_input_buffer()
        except Exception:
            pass

        cmd = struct.pack(
            ">BBBBBBBBB",
            motor_id, 0x74, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00
        )
        cmd = crc_attach(cmd)

        print(f"\nID {motor_id}")
        print(f"TX: {cmd.hex()}")

        ser.write(cmd)
        ser.flush()

        time.sleep(1.0)

        data = ser.read(100)

        if data:
            print(f"RX ({len(data)} bytes): {data.hex()}")
        else:
            print("RX: no se recibió ningún byte")

    ser.close()


if __name__ == "__main__":
    main()
