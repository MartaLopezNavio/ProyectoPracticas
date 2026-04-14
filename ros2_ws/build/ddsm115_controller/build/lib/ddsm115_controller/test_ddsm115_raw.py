import serial
import serial.rs485
import struct
import time
import crcmod.predefined


def crc_attach(data_bytes: bytes) -> bytes:
    crc8 = crcmod.predefined.mkPredefinedCrcFun('crc-8-maxim')
    crc_int = crc8(data_bytes)
    data_bytesarray = bytearray(data_bytes)
    data_bytesarray.append(crc_int)
    return bytes(data_bytesarray)


def main():
    device = "/dev/ttyUSB0"
    baudrate = 115200

    ser = serial.rs485.RS485(device, baudrate, timeout=0.2)
    ser.rs485_mode = serial.rs485.RS485Settings()

    print(f"Opened {device} at {baudrate} baud")

    for motor_id in range(1, 11):
        ser.reset_input_buffer()
        ser.reset_output_buffer()

        cmd = struct.pack(">BBBBBBBBB", motor_id, 0x74, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00)
        cmd = crc_attach(cmd)

        print(f"\nID {motor_id}")
        print(f"TX: {cmd.hex()}")

        ser.write(cmd)
        ser.flush()
        time.sleep(0.2)

        data = ser.read(100)

        if data:
            print(f"RX ({len(data)} bytes): {data.hex()}")
        else:
            print("RX: no se recibió ningún byte")

    ser.close()


if __name__ == "__main__":
    main()

