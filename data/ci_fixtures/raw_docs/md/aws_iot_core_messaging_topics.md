# AWS IoT Core Messaging & Topics

AWS IoT Core commonly uses MQTT topics to route device-to-cloud and cloud-to-device messages.

Topic design:
- Use hierarchical names like: devices/{thingName}/telemetry
- Avoid overly broad wildcards for security and cost reasons
- Use separate topics for telemetry vs commands

MQTT wildcards:
- '+' matches exactly one topic level
- '#' matches multiple levels (must be last)

Rules of thumb:
- Keep topic structure consistent across devices
- Use per-device topic prefixes to simplify authorization policies
