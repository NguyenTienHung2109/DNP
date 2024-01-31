import cv2
import numpy as np
from kafka import KafkaConsumer
from io import BytesIO

consumer = KafkaConsumer('detection',
                        group_id='detection',
                        bootstrap_servers=['localhost:9092'])

for message in consumer:
    try:
        image_cv2 = cv2.imdecode(np.frombuffer(message.value, dtype=np.uint8), cv2.IMREAD_UNCHANGED)
        if image_cv2 is not None:
            print("Image shape:", image_cv2.shape)
        else:
            print("Failed to decode image")
    except Exception as e:
        print("Error:", e)