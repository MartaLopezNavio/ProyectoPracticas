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

def int16_to_bytes(value: int):
    value &= 0xFFFF
    return [(value >> 8) & 0xFF, value & 0xFF]

device = "socket://192.168.1.201:4196"
motor_id = 1
rpm = 100

ser = serial.serial_for_url(device, baudrate=115200, timeout=1.0)

rpm_bytes = int16_to_bytes(rpm)
cmd = struct.pack(
    ">BBBBBBBBB",
    motor_id, 0x64, rpm_bytes[0], rpm_bytes[1],
    0x00, 0x00, 0x00, 0x00, 0x00
)
cmd = crc_attach(cmd)

print("TX:", cmd.hex())
ser.write(cmd)
ser.flush()

time.sleep(1.0)

# parar
rpm = 0
rpm_bytes = int16_to_bytes(rpm)
cmd = struct.pack(
    ">BBBBBBBBB",
    motor_id, 0x64, rpm_bytes[0], rpm_bytes[1],
    0x00, 0x00, 0x00, 0x00, 0x00
)
cmd = crc_attach(cmd)
print("TX stop:", cmd.hex())
ser.write(cmd)
ser.flush()

ser.close()
