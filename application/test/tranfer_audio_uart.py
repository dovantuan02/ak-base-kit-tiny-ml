import sounddevice as sd
import serial

ser = serial.Serial('/dev/ttyUSB0', 115200)

def callback(indata, frames, time, status):
    # print(indata.shape, len(indata.tobytes()))
    ser.write(indata.tobytes()) # send 320 x 2 = 640 bytes

with sd.InputStream(samplerate=16000, channels=1, dtype='int16', blocksize=320, callback=callback):
    input("Recording...\n")