# Anvaya: Hackathon Changes & Feature List

Based on the gap analysis in `hackathon_status.md`, the following goals and specific features must be implemented to ensure a flawless and impactful hackathon demonstration.

## Goal 1: Create a Realistic, High-Stakes Demonstration Environment
The dashboard must visually demonstrate the system's ability to differentiate between low, medium, and high-risk customers, and effectively highlight the SHAP drivers.
*   **Feature 1.1: Mock Data Generator Script**: Develop a Python script to synthesize and seed 100-200 customer records into the Supabase database.
*   **Feature 1.2: Engineered Risk Profiles**: Ensure the data generator intentionally creates specific cohorts:
    *   **Red Path (>35% PD)**: E.g., multiple auto-debit failures, high peer stress, dropping savings.
    *   **Yellow Path (15-35% PD)**: Emerging risks.
    *   **Green Path (<15% PD)**: Stable customers.

## Goal 2: Demonstrate Empathetic Intervention (Closing the Loop)
The project is an "Intervention Engine," not just a risk dashboard. We must show the actual action taken by the Relationship Manager (RM).
*   **Feature 2.1: GenAI "Action Plan" Integration**: Add an endpoint in the FastAPI backend that takes a customer's SHAP drivers (e.g., "F5 Auto-Debit Fails") and calls an LLM (OpenAI/Gemini) to generate an empathetic outreach script or restructuring offer.
*   **Feature 2.2: Frontend Action Modal**: Build a button in the UI (`Generate Action Plan`) that triggers the GenAI endpoint and displays the suggested script to the RM.

## Goal 3: Prove Real-Time Alerting Capabilities
The PRD emphasizes a 20ms SLA for real-time Kafka event ingestion. During a 3-minute pitch, we need to visually prove the system reacts instantly to live events.
*   **Feature 3.1: Live Event Injector Script**: Create a lightweight CLI tool to fire a simulated "auto-debit failure" or "salary drop" event.
*   **Feature 3.2: SSE Alerting Pipeline**: Ensure the frontend's SSE connection immediately catches the injected event and pops up a live RED alert on the dashboard without requiring a page refresh.

## Goal 4: Highlight Ethical Constraints & Vulnerable Customer Protection
Judges look for ethical AI implementations. We must prove the system knows when *not* to use standard collections strategies.
*   **Feature 4.1: Vulnerability Detection Rules**: Configure the backend to flag specific conditions (e.g., `F10_peer_stress` indicating bereavement or extreme hardship).
*   **Feature 4.2: UI Vulnerability Badge**: Add a prominent visual badge (e.g., Purple "VULNERABLE - HUMAN ROUTING ONLY") in the UI when these conditions are met, overriding the standard risk scores.
