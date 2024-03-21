from confluent_kafka import Producer, KafkaError, Consumer
import cv2
from datetime import datetime
from io import BytesIO
import numpy as np
from datetime import datetime

class VideoSaver:
    def __init__(self, topic = 'savevid', bootstrap_servers='localhost:9092', group_id = 'save_info'):
        now = datetime.now()
        date = now.strftime("%d_%m_%Y_%H_%M_%S")
        filename = "./output/video_" + date + ".avi"
        fourcc = cv2.VideoWriter_fourcc(*'MPEG')
        self.video = cv2.VideoWriter(filename, fourcc, 20.0, (720,  1280))

        self.topic = topic
        self.group_id = group_id
        self.bootstrap_servers = bootstrap_servers
   
        self.consumer_config = {
            'bootstrap.servers': self.bootstrap_servers,
            'group.id': self.group_id,
            'auto.offset.reset': 'earliest'
        }
        self.consumer = Consumer(self.consumer_config)
        self.consumer.subscribe([self.topic])

    def receive_and_process_frames(self):
        while True:  # Keep polling for messages
            message = self.consumer.poll(0)  # Adjust the timeout as needed
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

            print(frame_data.shape)
            offset = message.offset()  # Get the offset of the received frame
            # print(f"Processing message with offset: {offset}")
            
            self.video.write(frame_data)

            stream.close()
    
if __name__ == "__main__":
    video_saver = VideoSaver()
    video_saver.receive_and_process_frames()