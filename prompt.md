# Lemtik Security AI Platform — Addendum 01

# Multi-Camera Multiple Object Tracking (MCMOT), Person Re-Identification (Re-ID) & Intelligent CCTV Extension

> **IMPORTANT**
>
> This document is an **extension** to the existing **Lemtik Security AI Platform – Qwen Intelligence & Human-Governed Autonomous Infrastructure** implementation specification.
>
> The original specification remains the primary implementation guide.
>
> This document only introduces additional capabilities related to intelligent CCTV perception, cross-camera tracking, visual intelligence, and enhanced AI reasoning.
>
> **Do NOT redesign the platform.**
>
> **Do NOT rebuild existing modules.**
>
> **Do NOT change the existing architecture.**
>
> Only extend existing functionality.

---

# Purpose

Enhance the existing CCTV Integration module with AI-powered perception capable of:

* Multi-Camera Multiple Object Tracking (MCMOT)
* Person Re-Identification (Re-ID)
* Cross-camera target continuity
* Blind spot prediction
* Visual anomaly verification
* Intelligent telemetry generation
* Qwen Vision analysis
* Integration with the existing AI Orchestrator
* Integration with the existing Master AI
* Integration with the existing Autonomous Controller

The purpose is to allow Lemtik Security to maintain continuous situational awareness across multiple cameras without changing the existing command-and-control workflow.

---

# Existing Modules To Extend

Extend the existing modules only.

Existing CCTV Integration

Add:

* AI video processing
* Object tracking
* Re-ID
* Cross-camera matching

Existing AI Orchestrator

Add:

* Vision requests
* Telemetry ingestion
* Re-ID event reasoning

Existing Incident Management

Add:

* Visual telemetry
* Target timeline
* Cross-camera history

Existing Dashboard

Add:

* Live tracked targets
* Camera transitions
* Confidence indicators

Existing Command Center

Add:

* Active target visualization
* Cross-camera movement history

No other modules should be recreated.

---

# New Component

## CCTV Perception Service

Introduce a dedicated GPU-enabled perception service.

This service exists only to enhance the existing CCTV Integration.

It is **NOT** responsible for reasoning.

It is **NOT** responsible for automation.

It is **NOT** responsible for dispatching.

Its only responsibility is perception.

Responsibilities

* Receive RTSP streams
* Decode video
* Detect people
* Track objects
* Maintain target identities
* Extract visual embeddings
* Predict movement
* Publish telemetry
* Trigger Qwen Vision when necessary

---

# Technology Stack

Recommended technologies

Python

OpenCV

PyTorch

Ultralytics YOLOv8

ByteTrack (preferred) or DeepSORT

FastReID or OSNet

ONVIF

FFmpeg

CUDA

TensorRT (optional)

Qwen-VL

FastAPI

Redis (optional)

The technology stack may evolve without affecting the existing platform.

---

# Multi-Camera Multiple Object Tracking (MCMOT)

The CCTV service must maintain persistent identities across multiple cameras.

Example

Camera 1

↓

Target enters blind spot

↓

Predicted movement

↓

Camera 4

↓

Re-ID confirms same individual

↓

Tracking continues

↓

Existing incident updated

The target must retain the same internal tracking identifier throughout the event.

Example

REID-000381

instead of

Camera1-Target5

Camera4-Target2

---

# Person Re-Identification (Re-ID)

Every tracked individual should receive a visual embedding.

The embedding should represent:

* Clothing appearance
* Body shape
* Color distribution
* Accessories
* Motion characteristics

Embeddings are used only for matching within the operational environment.

The system should avoid making identity claims about unknown individuals.

The Re-ID service should produce similarity scores and confidence values rather than absolute assertions.

---

# Camera Topology Awareness

Extend the existing CCTV configuration by introducing camera topology metadata.

Each camera may define:

* Neighboring cameras
* Estimated travel times
* Transition probabilities
* Blind spot relationships

This topology should be used to reduce unnecessary Re-ID searches and improve cross-camera tracking performance.

---

# Blind Spot Prediction

When a target disappears from one camera, the perception service should estimate likely reappearance locations using the configured camera topology.

This prediction should be attached to telemetry for downstream reasoning.

The Master AI may use this information when recommending patrol movements or infrastructure actions.

---

# Qwen Vision Verification

The perception service should **not** invoke Qwen-VL continuously.

Instead, invoke Qwen-VL only when predefined conditions are met.

Examples include:

* Unauthorized access
* Tailgating
* Loitering in restricted zones
* Camera obstruction
* Camera freeze
* Forced entry
* Object abandonment
* Suspicious behavior
* Manual operator verification

Qwen-VL should receive:

* Snapshot(s)
* Relevant event metadata
* Existing incident context

Expected outputs include:

* Threat summary
* Confidence score
* Visual explanation
* Recommended follow-up actions

The visual analysis augments existing detections and should not replace deterministic detection logic.

---

# Telemetry Extension

Extend the existing telemetry pipeline.

Each tracked target should publish updates including:

* Tracking identifier
* Camera identifier
* Timestamp
* Zone
* Bounding box
* Movement vector
* Re-ID confidence
* Event type
* Snapshot reference
* Predicted destination
* Qwen verification status (if applicable)

Telemetry should integrate with the existing AI Orchestrator and Incident Management modules.

---

# Existing AI Orchestrator Integration

Add support for:

* Vision verification requests
* Re-ID telemetry ingestion
* Cross-camera event correlation

Suggested additions:

POST /ai/vision/verify

POST /ai/reid/analyze

POST /ai/reid/correlate

POST /ai/reid/history

These endpoints extend the orchestrator and should follow the same authentication, logging, and response conventions already defined.

---

# Existing Master AI Integration

The Master AI should consume Re-ID telemetry as additional context.

Examples:

* Determine likely destination of a tracked individual.
* Correlate movement with previous incidents.
* Recommend patrol positioning based on predicted routes.
* Include visual telemetry in containment reasoning.

The Master AI remains responsible for recommendations only.

---

# Existing Autonomous Controller Integration

The Autonomous Controller should remain unchanged.

It should simply receive richer recommendations generated from enhanced situational awareness.

For example:

Instead of:

"Unauthorized entry detected."

It may receive:

"Target REID-000381 was observed moving from Gate 3 toward Elevator Lobby B, verified across three cameras. Confidence: 0.93. Recommended actions: hold Elevator B, lock Gate C, dispatch nearest patrol."

Execution still follows existing approval and policy workflows.

---

# Dashboard Enhancements

Extend the existing dashboard with optional widgets for:

* Live tracked targets
* Cross-camera movement paths
* Camera transition history
* Active Re-ID confidence
* Visual verification status
* Predicted movement path

These additions should complement existing dashboards without replacing them.

---

# Incident Timeline Enhancement

Each incident may include:

* Camera appearances
* Camera transitions
* Snapshot references
* Qwen visual analyses
* Predicted destinations
* Movement history

This creates a continuous visual timeline for operators.

---

# Safety & Privacy

The Re-ID system is intended for operational continuity within monitored environments.

It should not infer personal identities without authorized identity sources.

Confidence values should accompany all visual matches.

Operators should be able to manually confirm or reject suggested matches.

Visual evidence should be retained and audited according to existing organizational policies.

---

# Logging & Audit Extensions

In addition to existing AI logs, record:

* Camera transitions
* Re-ID similarity scores
* Vision verification requests
* Vision model outputs
* Telemetry publication events
* Operator confirmations
* Manual target corrections

These logs should integrate with the platform's existing audit trail.

---

# Performance Guidelines

The perception service should be optimized for near real-time operation.

Recommended practices include:

* GPU acceleration where available.
* Batched inference when appropriate.
* Asynchronous processing.
* Efficient frame sampling for Qwen-VL.
* Caching of recent embeddings.
* Topology-based candidate filtering.
* Graceful degradation under heavy load.

---

# Success Criteria

After implementing this extension, the existing Lemtik Security platform should additionally be capable of:

* Tracking individuals across multiple cameras.
* Maintaining persistent target identities.
* Predicting movement through blind spots.
* Using Qwen-VL for selective visual verification.
* Providing richer telemetry to the existing AI Orchestrator.
* Improving Master AI reasoning with cross-camera intelligence.
* Enhancing operator situational awareness without altering existing command workflows.
* Preserving human oversight and existing automation policies while extending visual intelligence capabilities.
