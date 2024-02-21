from io import BytesIO
from confluent_kafka import Consumer, KafkaError, Producer
from ultralytics import YOLO
from ultralytics.utils.plotting import Annotator
import numpy as np
import cv2
import json

class KafkaHumanDetection:
    def __init__(self, bootstrap_servers='localhost:9092', detection_topic='detection', result_topic='bbox', group_id='detection'):
        self.bootstrap_servers = bootstrap_servers
        self.detection_topic = detection_topic
        self.result_topic = result_topic
        self.frame_count = 0
        self.group_id = group_id
        self.received_frames = []  # List to store received frames
        self.model = YOLO('yolov8n.pt')
        #CONSUMER
        self.consumer_config = {
            'bootstrap.servers': self.bootstrap_servers,
            'group.id': self.group_id,
            'auto.offset.reset': 'earliest'
        }
        self.consumer = Consumer(self.consumer_config)
        self.consumer.subscribe([self.detection_topic])

        #PRODUCER
        self.producer_config = {
            'bootstrap.servers': self.bootstrap_servers
        }
        self.producer = Producer(self.producer_config)
    def delivery_report(self, err, msg):
        if err is not None:
            print('Message delivery failed: {}'.format(err))
        else:
            print("Result delivered to topic: {}, partition: {}, offset: {}".format(msg.topic(), msg.partition(), msg.offset()))

    def process_frame(self, frame_data, offset):

        results = self.model(frame_data, classes=0)
        # Prepare the data to send
        detection_data = {
            'offset': offset,
            'detection_results': []
        }

        for r in results:
            boxes = r.boxes
            for box in boxes:
                b = box.xyxy[0]  # get box coordinates in (left, top, right, bottom) format
                c = box.cls
                detection_data['detection_results'].append({
                    'box': b,
                    'class': int(c)
                })

        # Send the detection results along with the offset to another Kafka topic
        self.producer.produce(self.result_topic, value=json.dumps(detection_data, default=lambda x: x.tolist()).encode('utf-8'), callback=self.delivery_report)
        self.producer.poll(0)

    def receive_and_process_frames(self):
        while True:  # Keep polling for messages
            message = self.consumer.poll(timeout=1000)  # Adjust the timeout as needed
            if message is None:
                continue
            if message.error():
                if message.error().code() == KafkaError._PARTITION_EOF:
                    continue
                else:
                    print(message.error())
                    break

            stream = BytesIO(message.value())
            frame_data = cv2.imdecode(np.frombuffer(message.value(), 'u1'), cv2.IMREAD_UNCHANGED)

            offset = message.offset()  # Get the offset of the received frame
            
            self.frame_count += 1
            if self.frame_count % 5 == 0:  # Process every 5th frame
                self.process_frame(frame_data, offset)
            stream.close()

if __name__ == "__main__":
    kafka_detection = KafkaHumanDetection()
    kafka_detection.receive_and_process_frames()
