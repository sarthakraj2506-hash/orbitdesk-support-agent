# Video walkthrough outline (4–7 minutes)

Use this script while screen-recording the project.

1. **Graph overview (45–60s)**  
   Open `docs/graph.png` and explain: `triage → retrieve → generate → verify → (revise once | finalize)`.

2. **Model load (45–60s)**  
   Run:
   ```powershell
   .\.venv\Scripts\Activate.ps1
   python -m orbitdesk_agent.cli --preload
   ```
   Call out model names, revisions, and `cpu`/`cuda` device.

3. **Live run A — answerable (60–90s)**  
   ```powershell
   python -m orbitdesk_agent.cli "I am a read-only Viewer. Can I create an API credential for a reporting script?"
   ```
   Show node trace, sources (`KB-002` / `KB-005`), and structured JSON.

4. **Live run B — clarification or out-of-scope (60–90s)**  
   ```powershell
   python -m orbitdesk_agent.cli "Our data sync is not working. Can you tell me how to fix it?"
   ```
   or the refund question. Show route difference in logs.

5. **Verification / revision path (60–90s)**  
   ```powershell
   python -m orbitdesk_agent.cli --force-verify-fail "Our daily dashboard exports stopped after a timezone change. What should we check?"
   ```
   Point to `verify → revise → generate → verify → finalize` in the trace.

6. **Trade-off / limitation / next step (45–60s)**  
   Mention deterministic triage for stable routing, small-model quality limits, and a future cross-encoder reranker.
