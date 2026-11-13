#!/usr/bin/env python3
"""Enable torque on OpenMANIPULATOR-X gripper motor (dxl5 / ID 15)."""

import sys
import rclpy
from rclpy.node import Node
from dynamixel_interfaces.srv import SetDataToDxl


def main():
    rclpy.init()
    node = Node("enable_gripper_torque")
    client = node.create_client(SetDataToDxl, "/dynamixel_hardware_interface/set_dxl_data")

    if not client.wait_for_service(timeout_sec=3.0):
        print("[ERROR] Service /dynamixel_hardware_interface/set_dxl_data not available.")
        node.destroy_node()
        rclpy.shutdown()
        sys.exit(1)

    req = SetDataToDxl.Request()
    req.id = 15
    req.item_name = "Torque Enable"
    req.item_data = 1

    future = client.call_async(req)
    rclpy.spin_until_future_complete(node, future, timeout_sec=5.0)

    if future.done() and future.result() is not None and future.result().result:
        print("[SUCCESS] Torque ENABLED on gripper (Dynamixel ID 15).")
        exit_code = 0
    else:
        print("[ERROR] Failed to enable torque on Dynamixel ID 15.")
        exit_code = 1

    node.destroy_node()
    rclpy.shutdown()
    sys.exit(exit_code)


if __name__ == "__main__":
    main()

