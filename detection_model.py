from io import BytesIO
from confluent_kafka import Producer, Consumer, KafkaError
import numpy as np
import cv2
import torch
bootstrap_servers = 'localhost:9092'
producer_config = {
    'bootstrap.servers': bootstrap_servers
}
producer = Producer(producer_config)

topic = 'detection'
group_id = 'detection'
consumer_config = {
    'bootstrap.servers': bootstrap_servers,
    'group.id': group_id,
    'auto.offset.reset': 'earliest'
}
consumer = Consumer(consumer_config)
consumer.subscribe([topic])

model = torch.hub.load('ultralytics/yolov5', 'yolov5s', pretrained=True)

while True:  # Keep polling for messages
    message = consumer.poll(timeout=1000)  # Adjust the timeout as needed

    if message is None:
        continue
    if message.error():
        if message.error().code() == KafkaError._PARTITION_EOF:
            continue
        else:
            print(message.error())
            break

    stream = BytesIO(message.value())
    image_cv2 = cv2.imdecode(np.frombuffer(message.value(), 'u1'), cv2.IMREAD_UNCHANGED)
    stream.close()

    # Your processing logic here
    results = model(image_cv2)
    result_string = str(results.pandas())
    print(result_string)
    output_image = results.render()[0]
    
    ret, buffer = cv2.imencode('.jpeg', output_image)
    decoded_image = cv2.imdecode(buffer, cv2.IMREAD_COLOR)

    # Draw the image
    cv2.imshow('Image with Bounding Boxes', decoded_image)
    cv2.waitKey(0)  # Wait for a key press to close the window
    #future = producer.produce('bbox', bytes(result_string, "utf-8"))


cv2.destroyAllWindows()
