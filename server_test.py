import logging
import time

import ntcore

if __name__ == "__main__":
    logging.basicConfig(level=logging.DEBUG)

    # initialize networktables server (on a robot this is already done)
    inst = ntcore.NetworkTableInstance.getDefault()
    inst.startServer()

    # Initialize two subscriptions
    table = inst.getTable("vision")

    # only keep the latest value for this topic
    sub1 = table.getFloatTopic("distance").subscribe(0.0)

    # keep the last 10 values for this topic
    sub2 = table.getFloatTopic("pixels").subscribe(0.0)

    # Periodically read from them
    # - note sub1 only has 1 value, but sub2 sometimes has more than 1
    while True:
        print("---", ntcore._now())
        print("/data/distance:", sub1.get())
        print("/data/pixels:", sub2.get())