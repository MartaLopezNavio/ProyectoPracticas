'''
    This code is RS485 driver interface to control DDSM115 motor.
    Developed by Rasheed Kittinanthapanya

    https://github.com/rasheeddo/
    https://www.youtube.com/@stepbystep-robotics
'''

import serial
import serial.rs485
import struct
import crcmod.predefined
import numpy as np
import time


class MotorControl:

    def __init__(self, device="/dev/ttyUSB0"):
        self.device = device

        # CAMBIO: timeout mayor para conexiones por red
        self.ser = serial.serial_for_url(device, baudrate=115200, timeout=0.5)

        # CAMBIO: rs485_mode solo si es puerto local
        if not str(device).startswith("socket://"):
            try:
                self.ser.rs485_mode = serial.rs485.RS485Settings()
            except Exception:
                pass

        self.crc8 = crcmod.predefined.mkPredefinedCrcFun('crc-8-maxim')
        self.str_10bytes = ">BBBBBBBBBB"
        self.str_9bytes = ">BBBBBBBBB"

        self.prev_fb_rpm = [0, 0, 0, 0]
        self.prev_fb_cur = [0, 0, 0, 0]

    def close(self):
        self.ser.close()

    ######################
    ### Math Functions ###
    ######################
    def Int16ToBytesArray(self, data: int):
        byte1 = (data & 0xFF00) >> 8
        byte2 = (data & 0x00FF)
        return [byte1, byte2]

    def TwoBytesTo16Int(self, high_byte: int, lo_byte: int):
        int16 = ((high_byte & 0xFF) << 8) | (lo_byte & 0xFF)
        return np.int16(int16).item()

    def map(self, val, in_min, in_max, out_min, out_max):
        return (val - in_min) * (out_max - out_min) / (in_max - in_min) + out_min

    def crc_attach(self, data_bytes: bytes):
        crc_int = self.crc8(data_bytes)
        data_bytesarray = bytearray(data_bytes)
        data_bytesarray.append(crc_int)
        return bytes(data_bytesarray)

    def currentRawToCurrentAmp(self, cur_raw: int):
        return self.map(cur_raw, -32767, 32767, -8.0, 8.0)

    ######################
    ### Read/Write cmd ###
    ######################
    def set_id(self, _id: int):
        set_id_cmd = struct.pack(
            self.str_10bytes,
            0xAA, 0x55, 0x53, _id, 0x00, 0x00, 0x00, 0x00, 0x00, 0xDE
        )

        for _ in range(5):
            self.ser.write(set_id_cmd)
            try:
                self.ser.flush()
            except Exception:
                pass
            time.sleep(0.01)

    def send_rpm(self, _id: int, rpm):
        rpm = int(rpm)
        rpm_ints = self.Int16ToBytesArray(rpm)
        cmd_bytes = struct.pack(
            self.str_9bytes,
            _id, 0x64, rpm_ints[0], rpm_ints[1], 0x00, 0x00, 0x00, 0x00, 0x00
        )
        cmd_bytes = self.crc_attach(cmd_bytes)

        while not self.ser.writable():
            pass

        self.ser.write(cmd_bytes)
        try:
            self.ser.flush()
        except Exception:
            pass

        _ = self.read_reply(_id)

    def send_degree(self, _id: int, deg):
        raw = int(self.map(deg, 0, 360, 0, 32767))
        deg_ints = self.Int16ToBytesArray(raw)

        cmd_bytes = struct.pack(
            self.str_9bytes,
            _id, 0x64, deg_ints[0], deg_ints[1], 0x00, 0x00, 0x00, 0x00, 0x00
        )
        cmd_bytes = self.crc_attach(cmd_bytes)

        self.ser.write(cmd_bytes)
        try:
            self.ser.flush()
        except Exception:
            pass

        _ = self.read_reply(_id)

    def set_brake(self, _id: int):
        cmd_bytes = struct.pack(
            self.str_9bytes,
            _id, 0x64, 0x00, 0x00, 0x00, 0x00, 0x00, 0xFF, 0x00
        )
        cmd_bytes = self.crc_attach(cmd_bytes)

        self.ser.write(cmd_bytes)
        try:
            self.ser.flush()
        except Exception:
            pass

        _ = self.read_reply(_id)

    def set_drive_mode(self, _id: int, _mode: int):
        if _mode == 1:
            mode_text = f"Set {_id} as current (torque) mode"
        elif _mode == 2:
            mode_text = f"Set {_id} as velocity mode"
        elif _mode == 3:
            mode_text = f"Set {_id} as position mode"
        else:
            mode_text = f"Error {_mode} is unknown"

        cmd_bytes = struct.pack(
            self.str_10bytes,
            _id, 0xA0, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, _mode
        )

        self.ser.write(cmd_bytes)
        try:
            self.ser.flush()
        except Exception:
            pass

        return mode_text

    def get_motor_feedback(self, _id: int):
        fb_req_cmd = struct.pack(
            self.str_9bytes,
            _id, 0x74, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00
        )
        fb_req_cmd = self.crc_attach(fb_req_cmd)

        while not self.ser.writable():
            pass

        self.ser.write(fb_req_cmd)
        try:
            self.ser.flush()
        except Exception:
            pass

        # CAMBIO: timeout mayor para que la respuesta tenga margen por TCP
        data_fb = self.read_reply(_id, timeout=1.0)
        return data_fb

    def read_reply(self, _id, timeout=1.0):
        got_reply = False
        ring_buffer = bytearray()
        start_time = time.time()

        data_fb = {
            'id': None,
            'mode': 0,
            'fb_cur': 0.0,
            'fb_rpm': 0,
            'wind_temp': 0,
            'pos': 0,
            'error': 0
        }

        while not got_reply:
            try:
                res = self.ser.read(1)
            except serial.serialutil.SerialException:
                break

            if len(res) != 0:
                if (len(ring_buffer) == 0) and (res == _id.to_bytes(1, 'big')):
                    ring_buffer.append(int.from_bytes(res, 'big'))

                elif (len(ring_buffer) == 1) and (res in [b'\x01', b'\x02', b'\x03']):
                    ring_buffer.append(int.from_bytes(res, 'big'))

                elif (len(ring_buffer) >= 2) and (len(ring_buffer) < 10):
                    ring_buffer.append(int.from_bytes(res, 'big'))

                    if len(ring_buffer) == 10:
                        crc_value = ring_buffer[-1]
                        raw_non_crc_bytes = bytes(ring_buffer[:-1])
                        crc_check = self.crc8(raw_non_crc_bytes)

                        if crc_value == crc_check:
                            data_fb['id'] = ring_buffer[0]
                            data_fb['mode'] = ring_buffer[1]

                            cur_hi = ring_buffer[2]
                            cur_lo = ring_buffer[3]
                            rpm_hi = ring_buffer[4]
                            rpm_lo = ring_buffer[5]

                            data_fb['wind_temp'] = ring_buffer[6]
                            data_fb['pos'] = ring_buffer[7]
                            data_fb['error'] = ring_buffer[8]
                            data_fb['fb_cur'] = self.currentRawToCurrentAmp(
                                self.TwoBytesTo16Int(cur_hi, cur_lo)
                            )
                            data_fb['fb_rpm'] = self.TwoBytesTo16Int(rpm_hi, rpm_lo)
                            got_reply = True
                        else:
                            ring_buffer = bytearray()
                else:
                    ring_buffer = bytearray()

            period = time.time() - start_time
            if period > timeout:
                got_reply = True
                break

        return data_fb


if __name__ == "__main__":
    a = MotorControl()
