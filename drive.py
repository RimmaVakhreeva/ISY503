import base64
from io import BytesIO

import eventlet
import eventlet.wsgi
import numpy as np
import socketio
from PIL import Image
from flask import Flask

MODEL_PATH = 'model.h5'


class TelemetryServer:
    def __init__(self):
        self.sio = socketio.Server()
        self.app = socketio.Middleware(self.sio, Flask(__name__))
        self.sio.on('connect', handler=self.connect)
        self.sio.on('telemetry', handler=self.telemetry)

        eventlet.wsgi.server(eventlet.listen(('', 4567)), self.app)

    def connect(self, sid, environ):
        print("connect ", sid)
        self._send_control(0, 1)

    def telemetry(self, sid, data):
        if data:
            steering_angle = data["steering_angle"]
            throttle = data["throttle"]
            speed = data["speed"]
            imgString = data["image"]

            image = Image.open(BytesIO(base64.b64decode(imgString)))
            image_array = np.asarray(image)
            # steering_angle = float(model.predict(image_array[None, :, :, :], batch_size=1))
            steering_angle = 0
            throttle = 0

            # print(steering_angle, throttle)
            self._send_control(steering_angle, throttle)

            # save frame

            # timestamp = datetime.utcnow().strftime('%Y_%m_%d_%H_%M_%S_%f')[:-3]
            # image_filename = os.path.join(r"E:\PythonScripts\Ass\photo", timestamp)
            # image.save('{}.jpg'.format(image_filename))
        else:
            self.sio.emit('manual', data={}, skip_sid=True)

    def _send_control(self, steering_angle, throttle):
        print(f"{steering_angle=} | {throttle=}")
        self.sio.emit(
            "steer",
            data={
                'steering_angle': steering_angle.__str__(),
                'throttle'      : throttle.__str__()
            },
            skip_sid=True)


class Driver:
    def __init__(self, max_speed: int = 25, min_speed: int = 10):
        self.max_speed = max_speed
        self.min_speed = min_speed

    def get_throttle(self, speed):
        speed_limit = self.min_speed if speed > self.max_speed else self.max_speed
        return 1.0 - (speed / speed_limit) ** 2


if __name__ == '__main__':
    app = TelemetryServer()
