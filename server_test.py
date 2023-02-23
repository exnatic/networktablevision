import logging
import ntcore
import time

if __name__ == "__main__":
    logging.basicConfig(level=logging.DEBUG)

    # initialize networktables server (on a robot this is already done)
    inst = ntcore.NetworkTableInstance.getDefault()
    inst.startServer()

    table = inst.getTable("vision")

    sub1 = table.getFloatTopic("distance").subscribe(0.0)

    sub2 = table.getFloatTopic("pixels").subscribe(0.0)

    sub3 = table.getFloatTopic("angle").subscribe(0.0)

    while True:
        time.sleep(0.25)
        print("---", ntcore._now())
        print("/vision/distance:", sub1.get())
        print("/vision/pixels:", sub2.get())
        print("/vision/angle:", sub3.get())