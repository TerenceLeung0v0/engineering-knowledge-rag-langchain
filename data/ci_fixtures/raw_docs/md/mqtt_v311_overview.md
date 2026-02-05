# MQTT v3.1.1 Overview

MQTT is a lightweight publish/subscribe messaging protocol designed for unreliable networks.

Key concepts:
- Broker: server that routes messages
- Client: connects to broker and publishes/subscribes
- Topic: UTF-8 string used to route messages
- QoS levels:
  - QoS 0: at most once
  - QoS 1: at least once
  - QoS 2: exactly once
- Retained message: broker stores last retained message per topic
- Keep Alive: client sends periodic control packets to keep connection alive

Typical pattern:
devices publish telemetry to topics and subscribe to command topics.
