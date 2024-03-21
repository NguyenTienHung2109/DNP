from confluent_kafka import Consumer, KafkaError, Producer
from io import BytesIO
from pose_estimation import PoseEstimation
import json
import cv2
import numpy as np

class KafkaPoseEstimation:
    def __init__(self, bootstrap_servers='localhost:9092', detection_topic='detection', bbox_topic='bbox', save_video_topic='savevid', save_json_topic='savejson', group_id='pose_estimation'):
        self.bootstrap_servers = bootstrap_servers
        self.save_json_topic = save_json_topic
        self.save_video_topic= save_video_topic
        self.detection_topic = detection_topic
        self.bbox_topic = bbox_topic
        self.group_id = group_id
        self.detection_data_list = []
        self.bbox_data_list = []
        
        self.pose_estimation = PoseEstimation()

        # Consumer setup
        self.consumer_config = {
            'bootstrap.servers': self.bootstrap_servers,
            'group.id': self.group_id,
            'auto.offset.reset': 'earliest'
        }
        self.consumer = Consumer(self.consumer_config)
        self.consumer.subscribe([self.detection_topic, self.bbox_topic])

        self.producer_config = {
            'bootstrap.servers': self.bootstrap_servers
        }
        self.producer = Producer(self.producer_config)

    def delivery_report(self, err, msg):
        if err is not None:
            print('Message delivery failed: {}'.format(err))
        else:
            print("Result delivered to topic: {}, partition: {}, offset: {}".format(msg.topic(), msg.partition(), msg.offset()))

    def process_frames(self, detection_data, bbox_data):
        detection_offset = detection_data['offset']
        bbox_offset = bbox_data['offset']

        # Check if offsets match
        if detection_offset == bbox_offset:
            # Perform pose estimation with the input frame and bbox information
            detection_frame = np.array(detection_data['image'], dtype=np.uint8)
            bbox_info_list = bbox_data['detection_results']

            # Your pose estimation logic here using detection_frame and bbox_info
            keypoints_list = []
            for bbox_info in bbox_info_list:
                x, y, w, h, id, confidence, cls, idx = bbox_info['box']
                x, y, w, h = int(x), int(y), int(w), int(h)
                cv2.rectangle(detection_frame, (x, y), (w, h), (0, 255, 0), 2)
                text = f"id: {id:.2f}"
                cv2.putText(detection_frame, text, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
                cropped_image = detection_frame[y:h, x:w]

            #     # Perform pose estimation on the cropped image
                pose_result, keypoints = self.pose_estimation.predict(cropped_image)
                keypoints_list.append(keypoints)
                # Apply the pose estimation result to the original image
                detection_frame[y:h, x:w] = pose_result

            self.send_frame(detection_frame)
            self.send_json(bbox_info_list, keypoints_list, bbox_offset)

    def send_frame(self, image):
        ret, buffer = cv2.imencode('.jpeg', image)
    
        self.producer.produce(self.save_video_topic, value=buffer.tobytes(), callback=self.delivery_report)
        self.producer.poll(0)  # Trigger delivery report callbacks

    def send_json(self, bbox_info_list, keypoints_list, offset):

        bbox_info_list = list(bbox_info_list)

        # Convert keypoints_list to a list of lists
        keypoints_list_converted = []
        for keypoints in keypoints_list:
            keypoints_as_list = [keypoint.tolist() for keypoint in keypoints]
            keypoints_list_converted.append(keypoints_as_list)

        data = {'bbox_info_list': bbox_info_list, 'keypoints_list': keypoints_list_converted, 'offset': offset}
        json_data = json.dumps(data)
        
        
        self.producer.produce(self.save_json_topic, value=json_data.encode('utf-8'), callback=self.delivery_report)
        self.producer.poll(0)  # Trigger delivery report callbacks
        

    def receive_and_process_frames(self):
        detection_data = None
        bbox_data = None

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
            offset = message.offset()

            # Check the topic and process accordingly
            if message.topic() == self.detection_topic:
                frame_data = cv2.imdecode(np.frombuffer(message.value(), 'u1'), cv2.IMREAD_UNCHANGED)
                detection_data = {'offset': offset, 'image': frame_data}
                self.detection_data_list.append(detection_data)

            elif message.topic() == self.bbox_topic:
                decoded_data = json.loads(message.value().decode('utf-8'))
                bbox_data = {'offset': decoded_data['offset'], 'detection_results': decoded_data['detection_results']}
                self.bbox_data_list.append(bbox_data)
            
            for bbox_data in self.bbox_data_list:
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
