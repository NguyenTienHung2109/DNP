from confluent_kafka import Consumer, KafkaError
import json
from datetime import datetime
import numpy as np

class JSONSaver:
    def __init__(self, topic='savejson', bootstrap_servers='localhost:9092', group_id='save_json_group'):
        self.topic = topic
        self.bootstrap_servers = bootstrap_servers
        self.group_id = group_id
        now = datetime.now()
        date = now.strftime("%d_%m_%Y_%H_%M_%S")
        self.filename = "./output/data_" + date + ".json"
        
        self.consumer_config = {
            'bootstrap.servers': self.bootstrap_servers,
            'group.id': self.group_id,
            'auto.offset.reset': 'earliest'
        }
        self.consumer = Consumer(self.consumer_config)
        self.consumer.subscribe([self.topic])

    def receive_and_process_json(self):
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
            
            json_data = json.loads(message.value().decode('utf-8'))

            # Process the JSON data as needed
            bbox_info_list = json_data['bbox_info_list']
            keypoints_list = json_data['keypoints_list']
            offset = json_data['offset']
            self.write_data(bbox_info_list, keypoints_list, offset)

    def write_data(self, bbox_info_list, keypoints_list, offset):
        # Convert keypoints_list from NumPy arrays to lists
    
        data = {
            "offset": offset,
            "data": []
        }

        for bbox_info, keypoints in zip(bbox_info_list, keypoints_list):
            box = bbox_info['box']
            box_dict = {
                "xmin": box[0],
                "ymin": box[1],
                "xmax": box[2],
                "ymax": box[3]
            }
            id = box[4]
            bounding_box_data = {
                "bounding_box": box_dict,
                "id": id,
                "keypoints": [{"x": kp[0], "y": kp[1]} for kp in keypoints]
            }
            data["data"].append(bounding_box_data)

        with open(self.filename, 'a') as f:
            json.dump(data, f)

if __name__ == "__main__":
    json_saver = JSONSaver()
    json_saver.receive_and_process_json()
