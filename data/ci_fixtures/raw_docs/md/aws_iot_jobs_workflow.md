# AWS IoT Jobs Overview
AWS IoT Jobs helps you perform remote operations across one or many devices.

A Job typically includes:
- job document: the task description a device can execute (e.g., firmware update)
- targets: the devices or thing groups
- job executions: per-device progress

Common steps:
1) Create a job with a document and targets
2) Devices receive job notifications
3) Devices fetch the job document, execute the task, and report status
