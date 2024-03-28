from io import BytesIO
from confluent_kafka import Consumer, KafkaError, Producer, TopicPartition, OFFSET_END, OFFSET_BEGINNING
from ultralytics import YOLO
from ultralytics.utils.plotting import Annotator
import time
import numpy as np
import torch
from pathlib import Path
import cv2
import json
from boxmot import OCSORT

class KafkaHumanDetection:
    def __init__(self, bootstrap_servers='localhost:9092', detection_topic='detection', result_topic='bbox', group_id='detection'):
        self.bootstrap_servers = bootstrap_servers
        self.detection_topic = detection_topic
        self.result_topic = result_topic
        self.frame_count = 0
        self.group_id = group_id
        self.received_frames = []  # List to store received frames
        self.model = torch.hub.load('ultralytics/yolov5', 'yolov5s', pretrained=True)

        self.tracker = OCSORT(
            # model_weights=Path('osnet_x0_25_msmt17.pt'),
            # device='cuda:0' if torch.cuda.is_available() else 'cpu',
            # fp16=False
        )

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

        # results = self.model(frame_data, classes=0)
        # # Prepare the data to send
        detection_data = {
            'offset': offset,
            'detection_results': []
        }

        # for r in results:
        #     boxes = r.boxes
        #     for box in boxes:
        #         b = box.xyxy[0]  # get box coordinates in (left, top, right, bottom) format
        #         c = box.cls
        #         res = self.tracker.update(np.array(b), np.array(frame_data))
        #         print(res)
        #         detection_data['detection_results'].append({
        #             'box': b,
        #             'class': int(c)
        #         })

        bounding_box = np.array(self.model(frame_data).xyxy[0].cpu())
        bounding_box = bounding_box[bounding_box[:, 5] == 0]
        results = self.tracker.update(bounding_box, np.array(frame_data))
        for tracked_box in results:
            # Append each tracked box to the 'detection_results' list in the detection_data dictionary
            detection_data['detection_results'].append({
                'box': tracked_box.tolist(),  # Convert to list for JSON serialization
                'class': 0  # Assuming class 0 for all detected objects
            })

        # Send the detection results along with the offset to another Kafka topic
        self.producer.produce(self.result_topic, value=json.dumps(detection_data, default=lambda x: x.tolist()).encode('utf-8'), callback=self.delivery_report)
        self.producer.poll(0)


    def receive_and_process_frames(self):
        while True:  # Keep polling for messages
            message = self.consumer.poll(0)  # Adjust the timeout as needed
            topic_with_latest_offset = TopicPartition("detection", partition = 0, offset = OFFSET_END)
            self.consumer.assign([topic_with_latest_offset])
            time.sleep(0.2)

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
            print(f"Processing message with offset: {offset}")
            
            self.process_frame(frame_data, offset)

            stream.close()

if __name__ == "__main__":
    kafka_detection = KafkaHumanDetection()
    kafka_detection.receive_and_process_frames()
