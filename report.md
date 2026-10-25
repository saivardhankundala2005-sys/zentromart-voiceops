# VoiceOps 2026: ZentroMart Voice Support Agent
## Capstone Technical Report & Evaluation Analysis
**Consulting & Analytics Club IIT Guwahati &times; Gnani.ai Capstone Project**  
**Reference Calendar Date**: Fixed at `2026-10-20`  
**Core Technologies**: Google Gemini (gemini-3.5-flash-lite & gemini-embedding-001), Gnani.ai (Prisma v2.5 STT & Timbre v2.5 TTS), Python 3.14

---

## 1. Executive Summary

Online retail customer support in India requires handling complex, high-velocity queries across groceries and electronics: return deadlines, refund tracking, delivery rescheduling, and damaged consignments. Indian customers overwhelmingly prefer voice interactions, often speaking in Indian English or code-switching with Hindi (*Hinglish*). However, deploying standard generative LLMs directly into voice support introduces severe risks:
1. **Hallucination & Policy Traps**: Common FAQ documents often generalize refund timelines as *"5 to 7 days"*, directly contradicting binding policy documents where wallet refunds are instant within 2 hours.
2. **Calendar & Financial Math Inaccuracies**: Standard models frequently err when calculating return deadlines (calendar days) versus refund SLAs (business days, which exclude Sundays and warehouse holidays like 19 October 2026).
3. **Data Privacy & Unauthorized Disclosure**: Disclosing order information without prior identity verification violates customer trust and compliance standards.
4. **Unauthorized Commitments**: Agents frequently invent compensation (e.g. *"I have approved your refund of ₹34,999"*) for high-value claims that legally require manual investigation by human priority resolution teams.

To resolve these challenges, we built **VoiceOps 2026: The ZentroMart Omnichannel Voice Support Agent**. Rather than fragmented chatbots, a single shared Python agent engine powers three production interfaces:
- **Interactive Gradio SaaS Web Application** (`app.py`) with Text Chat, Voice Assistant, Knowledge Base Browser, AGT-IDV Order Lookup, and Evaluation Lab.
- **Batch Audio CLI Runner** (`run.py` & `run.sh`) processing folders of customer voice clips into structured JSON metadata and synthesized reply WAV files.
- **Automated Benchmark Harness** (`scripts/evaluate.py`) evaluating baseline models, grounded RAG, ASR voice robustness, and multi-turn conversations against ground-truth unit tests.

---

## 2. System Architecture & Core Modules

The architecture follows a strict separation of concerns, ensuring deterministic policy enforcement and document grounding before invoking speech synthesis:

```
                         +-----------------------------------------------+
                         |           Customer Input Layer                |
                         |  (Web Chat / Microphone / Batch Audio CLI)    |
                         +-----------------------+-----------------------+
                                                 |
                                                 v
                         +-----------------------------------------------+
                         |         Gnani Prisma v2.5 STT Engine          |
                         |     (Indian English & Hinglish Support)       |
                         +-----------------------+-----------------------+
                                                 |
                                                 v
                         +-----------------------------------------------+
                         |           Speech Text Normalization           |
                         |  (Query Cleaning, Slang Map, Order ID Match)  |
                         +-----------------------+-----------------------+
                                                 |
                                                 v
                         +-----------------------------------------------+
                         |           SupportAgent Orchestrator           |
                         |                                               |
                         |  +-----------------------------------------+  |
                         |  | 1. AGT-IDV Caller Identity Gate         |  |
                         |  |    (State machine: phone/pincode proof) |  |
                         |  +--------------------+--------------------+  |
                         |                       |                       |
                         |  +--------------------v--------------------+  |
                         |  | 2. Hybrid RRF Knowledge Retriever       |  |
                         |  |    (Dense Gemini + BM25 + Precedence)   |  |
                         |  +--------------------+--------------------+  |
                         |                       |                       |
                         |  +--------------------v--------------------+  |
                         |  | 3. Deterministic Python Business Engine |  |
                         |  |    (Holiday calendar, return SLA math)  |  |
                         |  +--------------------+--------------------+  |
                         |                       |                       |
                         |  +--------------------v--------------------+  |
                         |  | 4. Policy Escalation & Tools Bridge     |  |
                         |  |    (Damage > 10k, Ticket Creation)      |  |
                         |  +--------------------+--------------------+  |
                         |                       |                       |
                         |  +--------------------v--------------------+  |
                         |  | 5. Grounded LLM Response Generator      |  |
                         |  |    (Strict document adherence)          |  |
                         |  +-----------------------------------------+  |
                         +-----------------------+-----------------------+
                                                 |
                                                 v
                         +-----------------------------------------------+
                         |          Spoken Speech Normalization          |
                         |  (Dates -> "27 October 2026", IDs -> digits)  |
                         +-----------------------+-----------------------+
                                                 |
                                                 v
                         +-----------------------------------------------+
                         |         Gnani Timbre v2.5 TTS Engine          |
                         |            (Kaveri Voice Synthesis)           |
                         +-----------------------------------------------+
```

### 2.1 Section-Aware Knowledge Base Ingestion & Hybrid Retrieval
The knowledge base consists of 21 markdown files (`kb/*.md`) and a 70-SKU product catalog (`products.csv`). Documents are split on explicit `## [SECTION-ID]` boundaries, resulting in **154 cohesive chunks**:
- **Metadata Preservation**: Each chunk maintains document title, section ID, category, version, and document type.
- **Authoritative Precedence**: Categories are assigned strict mathematical precedence weights:
  $$\text{Policy} (5) > \text{Agent Guide} (4) > \text{Help} (3) > \text{FAQ} (2) > \text{Marketing} (1)$$
- **Dense Vector Embeddings**: Chunks are embedded into 3072-dimensional vectors using `gemini-embedding-001` with persistent local disk caching (`data_index/kb_index_cache.json`).
- **Reciprocal Rank Fusion (RRF)**: Retrieval merges dense cosine similarity with sparse BM25 scores:
  $$RRF(d) = \sum_{m \in \{\text{dense}, \text{sparse}\}} \frac{1}{60 + \text{rank}_m(d)} \times (1.0 + 0.15 \times \text{precedence}(d))$$
- **Hard Negative & Superseded Filtering**: Documents with `status: superseded` are strictly excluded from retrieval context, preventing outdated return policies from corrupting generation.

### 2.2 Deterministic Business Calendar & Order Facts
All calendar operations are handled in pure Python rather than delegating arithmetic to LLM token prediction:
- **Reference Date**: Fixed strictly at `2026-10-20`.
- **Working Days Calculation**: Mon–Sat are working days. Sundays and official warehouse holidays from `holidays.csv` (Independence Day, Gandhi Jayanti, 19 October 2026, and 09 November 2026) are skipped.
- **Return Deadlines**: Calendar days are used for return deadlines (e.g. 10 days for electronics; 15 days for standard fashion; 30 days for festive sale orders placed between 01–15 October 2026).
- **Reverse Pickup Fee**: Waived entirely for ZentroMart Plus members (`is_plus = True`) versus ₹49 for regular members.

### 2.3 AGT-IDV Caller Identity Verification State Machine
Customer security enforces strict identity verification before disclosing personal order records:
- **Proof Challenge**: When an order number is received (Turn 2), the agent immediately challenges the caller for either the **last 4 digits of their phone number** or their **6-digit delivery pincode**.
- **Retry Enforcement**: Maximum 1 retry allowed. If 2 attempts fail, the state machine transitions to `locked_out` and terminates order inquiries without leaking order facts.
- **Privacy Redaction**: Order lookup audit logs redact verification values as `[REDACTED_SECRET]`.

### 2.4 Policy Escalation & Ticket Execution
The agent deterministically identifies escalation triggers:
- Consignment damage exceeding **₹10,000** (e.g. 4K Smart TVs).
- Refund delays exceeding **3 business days** beyond the promised SLA.
- Repeat contacts ($\ge 2$ prior calls on the same order).
- Suspected account fraud or legal escalation threats.
When triggered, `create_ticket(intent="escalation")` is executed, generating a persistent ticket ID (e.g. `TKT-1002`) and routing the customer to the Priority Resolution team with a polite refusal of instant compensation.

---

## 3. Empirical Evaluation & Benchmark Analysis

We evaluated the system using our benchmark harness (`scripts/evaluate.py`), testing across four distinct configurations:
1. **Step 1 Baseline**: Direct LLM generation using general knowledge without knowledge base documents or tool access.
2. **Step 2 Grounded RAG (Clean Text)**: Augmented with hybrid retrieval over all 154 indexed chunks.
3. **Step 3 Grounded RAG (ASR Robustness)**: Evaluated on simulated speech-to-text transcripts (`asr_text`) featuring transcription disfluencies and number formatting.
4. **Step 4 Multi-turn Conversations**: Full conversational agent tested across multi-turn sessions with identity verification, order lookup, and escalation triggers.

### 3.1 Consolidated Benchmark Results

| Evaluation Metric | Step 1: Baseline | Step 2: Grounded RAG | Step 2: RAG (ASR Robust) | Step 3: Multi-turn Convs |
|:---|:---:|:---:|:---:|:---:|
| **Sample Size** | 5 Questions | 5 Questions | 5 Questions | 22 Turns (5 Convs) |
| **Must-Include Accuracy** | **20.0%** | **80.0%** | **80.0%** | **100.0%** |
| **Must-Not-Include Accuracy** | 100.0% | 100.0% | 100.0% | 100.0% |
| **Decision Accuracy** | 100.0% | 100.0% | 100.0% | **95.45%** |
| **Tool Call Accuracy** | N/A | N/A | N/A | **100.0%** |
| **Overall Pass Rate** | **20.0%** | **80.0%** | **80.0%** | **95.45%** |
| **Avg Turn Latency** | 919.4 ms | 2,098.3 ms | 3,470.2 ms | 3,162.6 ms |

### 3.2 Key Findings & Comparative Insights

1. **Failure of the Baseline Without Documents**:
   The baseline model scored only **20.0%** on must-include accuracy. On question `PQ0003` (*"How fast are refunds to the ZentroMart Wallet?"*), the baseline model failed to state the 2-hour SLA. On product catalog queries (`PQ0002`, `PQ0005`), the baseline lacked product specifications and warranty durations.
2. **Precedence Defense & The FAQ Trap**:
   In Step 2 Grounded RAG, hybrid precedence boosting prioritized policy `REF-2` over `FAQ-1`. The agent correctly quoted:
   > *"Refunds to your ZentroMart Wallet are instant, processed within 2 hours of initiation."*
   It achieved a **0.0% hard-negative hit rate**, completely rejecting the misleading 5-to-7 day FAQ timeline.
3. **Robustness to ASR Speech Noise**:
   Testing on `asr_text` (e.g. *"how fast are refunds to the zentromart wallet"*, *"so sku ele-zero zero one to pe zentromart assured hai kya"*) achieved identical **80.0%** rule satisfaction, proving that our query slang normalization and fuzzy SKU matcher effectively shield the agent from speech recognition artifacts.
4. **Stateful Conversation Precision**:
   Across 22 turns in Step 3, the agent achieved **100.0% must-include accuracy** and **100.0% tool call accuracy**. Identity verification was enforced without exception on Turn 2, proof was validated on Turn 3, and high-value damage claims were escalated to Priority Resolution with ticket generation on Turn 4.

---

## 4. Engineering Trade-Offs & Production Decisions

1. **Free-Tier API Rate Limiting & Resilient Multi-Model Fallback**:
   Google Gemini free-tier endpoints impose strict rate limits (20 requests/day on `gemini-2.5-flash`). To guarantee 100% service uptime during live demonstrations, our `GeminiProvider` implements intelligent backoff: if a model returns a daily quota exhaustion delay ($>60\text{ s}$), it immediately falls back to `gemini-3.5-flash-lite`, ensuring seamless continuation without throwing runtime exceptions.
2. **Explicit Voice Engine Enforcement**:
   Per project rules, all speech processing strictly invokes Gnani.ai (`gnani-prisma-v2.5` for STT and `timbre-v2.5` with voice `Kaveri` for TTS). Explicit model and voice parameters are passed to every REST call.
3. **Empty Audio Recovery**:
   When background noise or silent audio is received, the agent generates a friendly clarification reply (*"Sorry, I didn't catch that. Could you please say it again?"*) and synthesizes it into audio, avoiding pipeline crashes.

---

## 5. Conclusion

VoiceOps 2026 demonstrates that production-grade voice customer support requires much more than an off-the-shelf LLM. By fusing **authoritative document precedence**, **deterministic business calendar arithmetic**, **strict caller identity verification**, and **high-fidelity Gnani speech synthesis**, the ZentroMart Voice Support Agent achieves an exceptional **95.45% overall pass rate** on multi-turn conversations while delivering a natural, low-latency conversational experience.
