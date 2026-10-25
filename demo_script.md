# VoiceOps 2026: ZentroMart AI Customer Support Agent
## 2-Minute Video Demonstration Script & Shot List

**Target Duration**: Exactly 2:00 (120 seconds)  
**Presenter**: Senior Voice-AI & RAG Engineer  
**Engine**: ZentroMart Unified Engine (Gemini 3.5 Flash Lite + Gnani Prisma v2.5 STT & Timbre v2.5 TTS)  
**Reference Date**: Fixed at `2026-10-20`

---

### Shot 1: Architecture & Unified Omnichannel Engine (0:00 – 0:25)
- **Visual**: Screen capture of Gradio Web Application Header (`http://localhost:7860`) showing the dark modern SaaS UI, active models (`gemini-3.5-flash-lite`, `gnani-prisma-v2.5`, `timbre-v2.5`), 154 indexed chunks, and fixed reference date `2026-10-20`.
- **Spoken Audio**:
  > *"Welcome to VoiceOps 2026. This is the ZentroMart Voice Support Agent, built for Consulting & Analytics Club IIT Guwahati and Gnani.ai. Customer support at ZentroMart handles everything from return windows and delayed refunds to damaged electronics. Rather than building fragmented bots, one shared conversational engine powers our web chat, voice assistant, and batch CLI. It enforces strict grounding in company policies and caller identity verification, with speech powered entirely by Gnani.ai."*

---

### Shot 2: Grounded RAG & Policy Precedence Defense (0:25 – 0:50)
- **Visual**: Click "Customer Support (Text Chat)" tab. Click Scenario 1: *"How fast are refunds to the ZentroMart Wallet?"*. Show the agent reply: *"Refunds to your ZentroMart Wallet are instant, within 2 hours of initiation."* Expand the Turn Inspector showing retrieved chunks: `REF-2` (Policy, precedence score 5) and `FAQ-1` (FAQ, precedence score 2).
- **Spoken Audio**:
  > *"Let's test policy precedence. FAQ documents commonly state refunds take 5 to 7 days, but authoritative policy `REF-2` mandates that wallet refunds are instant within 2 hours. Our hybrid retriever uses Reciprocal Rank Fusion, combining 3072-dimensional Gemini embeddings with BM25 keyword matching and strict precedence boosting. The agent correctly quotes the 2-hour window and rejects the FAQ trap, citing policy `REF-2` directly in the turn audit."*

---

### Shot 3: Multi-Turn Context & Business Calendar Math (0:50 – 1:15)
- **Visual**: Ask a multi-turn question: *"What about a credit card refund initiated on 17 October 2026?"*. The agent calculates: *"Between 24 October and 27 October 2026"*.
- **Spoken Audio**:
  > *"Next: multi-turn context and business day calculations. The customer asks when their credit card refund will arrive if started on 17 October 2026. Credit card refunds take 5 to 7 business days. Our Python business calendar skips Sundays and warehouse holidays like 19 October, correctly computing the credit window between 24 and 27 October 2026 relative to our fixed reference date."*

---

### Shot 4: AGT-IDV Identity Verification & Escalation Policy (1:15 – 1:40)
- **Visual**: Switch to Tab 4 (Order Verification). Enter order `89782837`. Click "Initiate Lookup". The UI challenges the caller for the last 4 digits of phone or pincode. Enter `4533`. Click "Verify Identity". The dossier unlocks: ₹34,999 4K Smart TV. In Chat, ask: *"The TV arrived damaged. Approve my replacement right now."* The agent refuses unauthorized compensation, triggers `create_ticket(intent='escalation')`, and routes to Priority Resolution.
- **Spoken Audio**:
  > *"Security is non-negotiable. Before disclosing order facts, our AGT-IDV state machine challenges the caller for proof—either the last 4 digits of their phone or 6-digit delivery pincode, with a strict one-retry limit. Once verified, the order dossier unlocks. When the customer demands immediate replacement for a ₹34,999 damaged TV, our deterministic policy engine recognizes damage exceeding ₹10,000, refuses instant compensation, creates audit ticket `TKT-1002`, and escalates to our Priority Resolution team."*

---

### Shot 5: Native Voice Assistant (Gnani STT & TTS) & Batch CLI (1:40 – 2:00)
- **Visual**: Switch to Tab 2 (Voice Assistant). Play an uploaded customer voice clip. Show real-time Gnani STT transcript, turn latency breakdown (STT 280ms, Agent 1200ms, TTS 450ms), and play the synthesized reply from Gnani Timbre v2.5 (`Kaveri`). Show terminal executing `python run.py --input-dir ... --output-dir ...`.
- **Spoken Audio**:
  > *"Finally, voice interaction. Gnani Prisma v2.5 transcribes spoken input with full Indian English and Hindi-English code-switching support. The reply is normalized for speech—converting acronyms and formatting currency—then synthesized via Gnani Timbre v2.5 using the Kaveri voice. The exact same engine runs our batch CLI, processing entire audio directories into structured JSON and reply WAVs. Thank you."*
