import base64
from io import BytesIO

import cv2
import eventlet
import eventlet.wsgi
import numpy as np
import socketio
from PIL import Image
from flask import Flask
from keras.models import load_model

from utils import preprocess

MODEL_PATH = 'model.h5'


class TelemetryServer:
    def __init__(self, model):
        self.sio = socketio.Server()
        self.app = socketio.Middleware(self.sio, Flask(__name__))
        self.sio.on('connect', handler=self.connect)
        self.sio.on('telemetry', handler=self.telemetry)
        self.driver = Driver(model)

        eventlet.wsgi.server(eventlet.listen(('', 4567)), self.app)

    def connect(self, sid, environ):
        print("connect ", sid)
        self._send_control(0, 1)

    def telemetry(self, sid, data):
        if data:
            self.driver.get_properties_from_data(data)
            self._send_control(self.driver.steering_angle, self.driver.throttle)
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
    def __init__(self, model, max_speed: int = 25, min_speed: int = 10):
        self.imgString = None
        self.speed = None
        self.max_speed = max_speed
        self.min_speed = min_speed

        self.model = model

    def get_properties_from_data(self, data):
        self.speed = float(data["speed"])
        self.imgString = data["image"]

    @property
    def throttle(self):
        return 1.2 - (self.speed / self.max_speed)


    @property
    def steering_angle(self):
        if self.model:
            image = Image.open(BytesIO(base64.b64decode(self.imgString)))
            image_array = np.asarray(image)
            return float(self.model.predict(preprocess(image_array)[None, :, :, :], batch_size=1))
        else:
            return 0.0


if __name__ == '__main__':
    app = TelemetryServer(load_model("model-010.h5"))

