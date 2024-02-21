from confluent_kafka import Producer, KafkaError
import cv2
from datetime import datetime
import json

class KafkaFrameProducer:
    def __init__(self, bootstrap_servers='localhost:9092', topic='detection'):
        self.bootstrap_servers = bootstrap_servers
        self.topic = topic
        self.producer_config = {
            'bootstrap.servers': self.bootstrap_servers
        }
        self.producer = Producer(self.producer_config)

    def delivery_report(self, err, msg):
        if err is not None:
            print('Message delivery failed: {}'.format(err))
        else:
            print("frame: {}, topic name: {}, partition: {}, offset: {}".format(
                self.count, msg.topic(), msg.partition(), msg.offset()))

    def send_frame(self, image):
        ret, buffer = cv2.imencode('.jpeg', image)
    

        self.producer.produce(self.topic, value=buffer.tobytes(), callback=self.delivery_report)
        self.producer.poll(0)  # Trigger delivery report callbacks

    def send_video(self, video_path):
        vidcap = cv2.VideoCapture(video_path)
        self.count = 0

        while True:
            success, image = vidcap.read()
            if not success:
                break

            self.count += 1
            self.send_frame(image)

        vidcap.release()
        self.producer.flush()

if __name__ == "__main__":
    kafka_producer = KafkaFrameProducer()
    kafka_producer.send_video('siu.mp4')
