from confluent_kafka import Consumer, KafkaError
from io import BytesIO
from pose_estimation import PoseEstimation
import json
import cv2
import numpy as np

class KafkaPoseEstimation:
    def __init__(self, bootstrap_servers='localhost:9092', detection_topic='detection', bbox_topic='bbox', group_id='pose_estimation'):
        self.bootstrap_servers = bootstrap_servers
        self.detection_topic = detection_topic
        self.bbox_topic = bbox_topic
        self.group_id = group_id
        self.detection_data_list = []
        self.pose_estimation = PoseEstimation(model_type='rtmpose | body')

        # Consumer setup
        self.consumer_config = {
            'bootstrap.servers': self.bootstrap_servers,
            'group.id': self.group_id,
            'auto.offset.reset': 'earliest'
        }
        self.consumer = Consumer(self.consumer_config)
        self.consumer.subscribe([self.detection_topic, self.bbox_topic])

        # ... (rest of the setup)

    def process_frames(self, detection_data, bbox_data):
        detection_offset = detection_data['offset']
        bbox_offset = bbox_data['offset']

        # Check if offsets match
        if detection_offset == bbox_offset:
            # Perform pose estimation with the input frame and bbox information
            detection_frame = np.array(detection_data['image'], dtype=np.uint8)
            bbox_info_list = bbox_data['detection_results']

            # Your pose estimation logic here using detection_frame and bbox_info
            for bbox_info in bbox_info_list:
                x, y, w, h = bbox_info['box']
                x, y, w, h = int(x), int(y), int(w), int(h)
                cv2.rectangle(detection_frame, (x, y), (w, h), (0, 255, 0), 2)
                cropped_image = detection_frame[y:h, x:w]

                # Perform pose estimation on the cropped image
                pose_result = self.pose_estimation.predict(cropped_image)

                # Apply the pose estimation result to the original image
                detection_frame[y:h, x:w] = pose_result

            # Display the frame with bounding boxes
            cv2.imshow('Pose Estimation', detection_frame)
            cv2.waitKey(0)
            cv2.destroyAllWindows()

            # Print some information for demonstration purposes
            print(f"Pose estimation for frame at offset {detection_offset}")

    def receive_and_process_frames(self):
        detection_data = None
        bbox_data = None

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
            offset = message.offset()

            # Check the topic and process accordingly
            if message.topic() == self.detection_topic:
                frame_data = cv2.imdecode(np.frombuffer(message.value(), 'u1'), cv2.IMREAD_UNCHANGED)
                detection_data = {'offset': offset, 'image': frame_data}
                self.detection_data_list.append(detection_data)

            elif message.topic() == self.bbox_topic:
                decoded_data = json.loads(message.value().decode('utf-8'))
                bbox_data = {'offset': decoded_data['offset'], 'detection_results': decoded_data['detection_results']}
                for detection_data in self.detection_data_list:
                    if detection_data['offset'] == bbox_data['offset']:
                        self.process_frames(detection_data, bbox_data)

                # Remove processed detection data
                self.detection_data_list = [d for d in self.detection_data_list if d['offset'] != bbox_data['offset']]

            # Process frames if both detection and bbox data are available
            stream.close()

if __name__ == "__main__":
    kafka_pose_estimation = KafkaPoseEstimation()
    kafka_pose_estimation.receive_and_process_frames()
