from confluent_kafka import Producer, KafkaError
import cv2
from datetime import datetime

# create a Kafka producer
bootstrap_servers = 'localhost:9092'
producer_config = {
    'bootstrap.servers': bootstrap_servers
}
producer = Producer(producer_config)

vidcap = cv2.VideoCapture('siu.mp4')
success,image = vidcap.read()
count = 0

# start_date = datetime.now()
    
while success:
    success, image = vidcap.read()
    if success == False:
        break
    count += 1
    ret, buffer = cv2.imencode('.jpeg', image)
    def delivery_report(err, msg):
        if err is not None:
            print('Message delivery failed: {}'.format(err))
        else:
            print("frame: {}, topic name: {}, partition: {}, offset: {}".format(count, msg.topic(), msg.partition(), msg.offset()))
    # Produce the message using confluent_kafka's Producer
    producer.produce('detection', value=buffer.tobytes(), callback=delivery_report)
    producer.poll(0)  # Trigger delivery report callbacks

producer.flush()
