#!/usr/bin/env python3
"""VoiceOps 2026 - ZentroMart Voice & Text AI Customer Support Platform.

Interactive SaaS web application powered by:
- Google Gemini (gemini-2.5-flash & gemini-embedding-001)
- Gnani.ai Voice Services (gnani-prisma-v2.5 STT & timbre-v2.5 TTS)
- Hybrid Reciprocal Rank Fusion Retrieval (Dense Cosine + BM25)
- Deterministic Identity Verification (AGT-IDV) & Policy Engine
"""

import os
import sys
import time
import json
import uuid
from pathlib import Path
from typing import List, Dict, Any, Tuple
import pandas as pd
import gradio as gr

# Ensure src package is importable
sys.path.insert(0, str(Path(__file__).parent / "src"))

from voiceops.config import (
    DATA_DIR, OUTPUTS_DIR, REFERENCE_DATE, LLM_MODEL, EMBEDDING_MODEL,
    STT_LANG, TTS_VOICE, TTS_MODEL
)
from voiceops.agent.orchestrator import get_support_agent
from voiceops.knowledge.retriever import get_retriever
from voiceops.knowledge.loader import load_all_kb_chunks
from voiceops.voice.stt import get_stt_client
from voiceops.voice.tts import get_tts_client
from voiceops.voice.speech_text import normalize_for_speech
from voiceops.data.loader import get_data_loader
from voiceops.data.order_facts import compute_order_facts
from voiceops.evaluation.metrics import (
    evaluate_must_include, evaluate_must_not_include, evaluate_decision
)

# Custom Premium Dark Theme CSS
CUSTOM_CSS = """
:root {
    --bg-main: #0a0f1d;
    --card-surface: #111827;
    --card-surface-hover: #1f293d;
    --border-color: #1f2937;
    --primary: #10b981;
    --primary-glow: rgba(16, 185, 129, 0.2);
    --secondary: #38bdf8;
    --text-primary: #f9fafb;
    --text-secondary: #9ca3af;
    --danger: #ef4444;
    --warning: #f59e0b;
}

body, .gradio-container {
    background-color: var(--bg-main) !important;
    color: var(--text-primary) !important;
    font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, Helvetica, Arial, sans-serif !important;
}

/* Header Banner */
.brand-header {
    background: linear-gradient(135deg, #0f172a 0%, #1e1b4b 50%, #064e3b 100%);
    border: 1px solid #334155;
    border-radius: 16px;
    padding: 24px 32px;
    margin-bottom: 24px;
    box-shadow: 0 10px 25px -5px rgba(0, 0, 0, 0.5);
}

.brand-title {
    font-size: 2.2rem;
    font-weight: 800;
    letter-spacing: -0.025em;
    background: linear-gradient(to right, #ffffff, #6ee7b7, #38bdf8);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    margin: 0;
}

.brand-subtitle {
    color: #94a3b8;
    font-size: 1rem;
    margin-top: 6px;
}

/* Metric / Status Badges */
.badge-pill {
    display: inline-flex;
    align-items: center;
    padding: 4px 12px;
    border-radius: 9999px;
    font-size: 0.75rem;
    font-weight: 700;
    text-transform: uppercase;
    letter-spacing: 0.05em;
    margin-right: 8px;
}
.badge-emerald { background: rgba(16, 185, 129, 0.15); color: #34d399; border: 1px solid #059669; }
.badge-sky { background: rgba(56, 189, 248, 0.15); color: #38bdf8; border: 1px solid #0284c7; }
.badge-amber { background: rgba(245, 158, 11, 0.15); color: #fbbf24; border: 1px solid #d97706; }
.badge-rose { background: rgba(239, 68, 68, 0.15); color: #f87171; border: 1px solid #dc2626; }

/* Panels & Cards */
.panel-card {
    background: var(--card-surface);
    border: 1px solid var(--border-color);
    border-radius: 12px;
    padding: 16px;
    margin-bottom: 16px;
}

.latency-grid {
    display: grid;
    grid-template-columns: repeat(4, 1fr);
    gap: 12px;
    margin: 12px 0;
}

.latency-box {
    background: #1e293b;
    border: 1px solid #334155;
    border-radius: 8px;
    padding: 12px;
    text-align: center;
}

.latency-label {
    font-size: 0.75rem;
    color: #94a3b8;
    text-transform: uppercase;
    font-weight: 600;
}

.latency-value {
    font-size: 1.3rem;
    font-weight: 700;
    color: #38bdf8;
    margin-top: 4px;
}
"""

# Global Singletons
agent = get_support_agent()
retriever = get_retriever()
data_loader = get_data_loader()
stt_client = get_stt_client()
tts_client = get_tts_client()
OUTPUTS_DIR.mkdir(parents=True, exist_ok=True)


# ==============================================================================
# TAB 1: TEXT CHAT HANDLERS
# ==============================================================================
def chat_respond(
    user_message: str,
    chat_history: List[Dict[str, str]],
    session_id: str,
    mode: str,
) -> Tuple[str, List[Dict[str, str]], str]:
    if not user_message or not user_message.strip():
        return "", chat_history, "Please enter a message."

    if not session_id:
        session_id = f"web_session_{uuid.uuid4().hex[:8]}"

    # Call agent
    t0 = time.time()
    res = agent.respond(
        message=user_message.strip(),
        conversation_id=session_id,
        mode=mode,
    )
    turn_latency = round((time.time() - t0) * 1000, 1)

    # Format Gradio history (type="messages")
    updated_history = chat_history + [
        {"role": "user", "content": user_message.strip()},
        {"role": "assistant", "content": res.reply},
    ]

    # Generate Inspector Details HTML
    decision_badge_class = {
        "answer": "badge-emerald",
        "clarify": "badge-sky",
        "escalate": "badge-amber",
        "refuse": "badge-rose",
    }.get(res.decision, "badge-sky")

    inspector_html = f"""
    <div style="font-size: 0.9rem; line-height: 1.6;">
        <div style="margin-bottom: 12px; display: flex; align-items: center; gap: 8px; flex-wrap: wrap;">
            <span class="badge-pill {decision_badge_class}">Decision: {res.decision.upper()}</span>
            <span class="badge-pill badge-sky">Turn #{res.turn_id}</span>
            <span class="badge-pill badge-emerald">Mode: {mode.upper()}</span>
            <span class="badge-pill badge-sky">Latency: {turn_latency} ms</span>
            {f'<span class="badge-pill badge-amber">Ticket: {res.ticket_id}</span>' if res.ticket_id else ''}
        </div>
        <p><strong>Spoken Normalization:</strong> <em>"{res.spoken_reply}"</em></p>
        <p><strong>Tools Executed:</strong> {', '.join(res.tool_calls) if res.tool_calls else 'None'}</p>
        <div style="margin-top: 10px;">
            <strong>Retrieved Knowledge Chunks ({len(res.retrieved_chunks)}):</strong>
            <ul style="margin-top: 6px; padding-left: 20px;">
    """
    for chunk in res.retrieved_chunks:
        inspector_html += f"""
            <li style="margin-bottom: 6px;">
                <strong>[{chunk['section_id']}]</strong> {chunk['doc_title']} 
                <em>(Score: {chunk.get('score', 0):.3f})</em>
                <br><span style="color: #94a3b8; font-size: 0.82rem;">{chunk['text'][:120]}...</span>
            </li>
        """
    if not res.retrieved_chunks:
        inspector_html += "<li style='color: #94a3b8;'>No chunks retrieved for this turn.</li>"

    inspector_html += "</ul></div></div>"

    return "", updated_history, inspector_html


def reset_chat_session():
    new_id = f"web_session_{uuid.uuid4().hex[:8]}"
    return [], new_id, "<p style='color: #94a3b8;'>Conversation reset. Inspector awaiting first turn.</p>"


# ==============================================================================
# TAB 2: VOICE ASSISTANT HANDLER
# ==============================================================================
def process_voice_audio(
    audio_path: str,
    stt_lang: str,
    voice_name: str,
    tts_model_name: str,
    session_id: str,
) -> Tuple[str, str, str, str, str, str, str]:
    if not audio_path or not os.path.exists(audio_path):
        return None, "", "No audio file received.", "", "", "", ""

    if not session_id:
        session_id = f"voice_session_{uuid.uuid4().hex[:8]}"

    t_start = time.time()
    
    # 1. Gnani STT
    t0 = time.time()
    stt_res = stt_client.transcribe(Path(audio_path), language_code=stt_lang)
    stt_ms = round((time.time() - t0) * 1000, 1)

    transcript = stt_res.get("transcript", "").strip()
    is_empty = stt_res.get("is_empty", False)

    # 2. Agent Turn
    t0 = time.time()
    if is_empty or not transcript:
        reply_text = "Sorry, I didn't catch that. Could you please say it again?"
        spoken_text = reply_text
        decision = "clarify"
        section_ids = []
    else:
        res = agent.respond(
            message=transcript,
            conversation_id=session_id,
            language_hint=stt_res.get("language_code"),
            mode="full",
        )
        reply_text = res.reply
        spoken_text = res.spoken_reply
        decision = res.decision
        section_ids = [c["section_id"] for c in res.retrieved_chunks]
    agent_ms = round((time.time() - t0) * 1000, 1)

    # 3. Gnani TTS
    t0 = time.time()
    out_wav = OUTPUTS_DIR / f"{session_id}_{int(time.time()*1000)}_reply.wav"
    tts_res = tts_client.synthesize(
        text=spoken_text,
        output_file=out_wav,
        voice=voice_name,
        model=tts_model_name,
        language=stt_lang,
    )
    tts_ms = round((time.time() - t0) * 1000, 1)
    total_ms = round((time.time() - t_start) * 1000, 1)

    audio_out_path = str(out_wav) if tts_res.get("success") else None

    # Decision badge
    decision_html = f"<span class='badge-pill badge-emerald'>{decision.upper()}</span>"
    citations_html = " ".join([f"<span class='badge-pill badge-sky'>{sid}</span>" for sid in section_ids]) or "<span style='color:#94a3b8;'>None</span>"

    return (
        audio_out_path,
        transcript,
        spoken_text,
        decision_html,
        citations_html,
        f"{stt_ms} ms",
        f"{agent_ms} ms",
        f"{tts_ms} ms",
        f"{total_ms} ms",
    )


# ==============================================================================
# TAB 3: KNOWLEDGE BASE BROWSER
# ==============================================================================
def search_knowledge_base(query: str, category_filter: str) -> pd.DataFrame:
    all_chunks = load_all_kb_chunks()

    if not query or not query.strip():
        # Display all chunks matching category
        results = []
        for c in all_chunks:
            if category_filter != "ALL" and c.doc_type != category_filter:
                continue
            results.append({
                "Section ID": c.section_id,
                "Title": c.title,
                "Category": c.doc_type,
                "Precedence": c.precedence_score,
                "Status": "Superseded" if c.status == "superseded" else "Active",
                "Excerpt": c.text[:140] + ("..." if len(c.text) > 140 else ""),
            })
        return pd.DataFrame(results)

    # Search via Hybrid Retriever
    scored_chunks = retriever.retrieve(query.strip(), top_k=25)
    results = []
    for c in scored_chunks:
        if category_filter != "ALL" and c.get("category") != category_filter:
            continue
        results.append({
            "Section ID": c["section_id"],
            "Title": c["doc_title"],
            "Category": c["category"],
            "Precedence": c["precedence"],
            "Score": f"{c.get('score', 0):.3f}",
            "Excerpt": c["text"][:140] + ("..." if len(c["text"]) > 140 else ""),
        })
    return pd.DataFrame(results)


# ==============================================================================
# TAB 4: ORDER LOOKUP & IDENTITY VERIFICATION DEMO
# ==============================================================================
def initiate_order_lookup(order_id: str):
    oid = order_id.strip() if order_id else ""
    order = data_loader.get_order(oid)
    if not order:
        return (
            f"❌ Order '{oid}' not found in ZentroMart database.",
            "",
            gr.update(interactive=False),
            ""
        )

    cust = data_loader.get_customer(order.get("customer_id"))
    masked_phone = f"******{cust.get('phone')[-4:]}" if cust and cust.get("phone") else "Unknown"
    pincode = order.get("delivery_pincode", "Unknown")

    challenge_html = f"""
    <div class="panel-card" style="border-left: 4px solid #38bdf8;">
        <h4 style="margin: 0 0 8px 0; color: #38bdf8;">🔐 AGT-IDV Step 1: Proof Required</h4>
        <p>Order <strong>#{oid}</strong> found. Registered to Customer <strong>{cust.get('name', 'Valued Customer')}</strong>.</p>
        <p>To access order details, please provide caller verification proof:</p>
        <ul>
            <li><strong>Last 4 digits of phone</strong> (Hint: ends in <code>{cust.get('phone')[-4:]}</code>)</li>
            <li>OR <strong>6-digit delivery pincode</strong> (Hint: <code>{pincode}</code>)</li>
        </ul>
    </div>
    """
    return challenge_html, oid, gr.update(interactive=True), ""


def verify_order_proof(active_oid: str, proof_text: str):
    if not active_oid:
        return "Please initiate an order lookup first.", ""

    from voiceops.agent.verification import extract_verification_proof
    proof = extract_verification_proof(proof_text or "")
    
    order = data_loader.get_order_full(active_oid)
    if not order:
        return "Order record not found.", ""

    cust = order.get("customer", {})
    phone = cust.get("phone", "")
    pincode = order.get("delivery_pincode", "")

    phone_suffix = phone[-4:] if len(phone) >= 4 else ""
    is_valid = (proof == phone_suffix or proof == pincode)

    if not is_valid:
        fail_html = f"""
        <div class="panel-card" style="border-left: 4px solid #ef4444;">
            <h4 style="margin: 0 0 8px 0; color: #ef4444;">❌ Identity Verification Failed</h4>
            <p>The provided proof <code>{proof or 'empty'}</code> did not match the registered phone suffix or delivery pincode.</p>
            <p style="color: #94a3b8; font-size: 0.85rem;">Security Policy: Maximum 1 retry allowed before lockout. No order details disclosed.</p>
        </div>
        """
        return fail_html, ""

    # Successful verification! Compute deterministic facts
    facts = compute_order_facts(order, data_loader.holidays)

    success_html = f"""
    <div class="panel-card" style="border-left: 4px solid #10b981;">
        <h4 style="margin: 0 0 8px 0; color: #10b981;">✅ Identity Verified Successfully</h4>
        <p>Verified caller: <strong>{cust.get('name')}</strong> (ID: {cust.get('customer_id')}, Plus Member: <strong>{cust.get('is_plus')}</strong>)</p>
        <p>Order Placed: <strong>{order.get('order_date')}</strong> | Delivered: <strong>{order.get('delivery_date')}</strong> | Status: <strong>{order.get('status')}</strong></p>
    </div>
    """

    facts_formatted = json.dumps(facts, indent=2, default=str)
    return success_html, facts_formatted


# ==============================================================================
# TAB 5: EVALUATION LAB
# ==============================================================================
def load_evaluation_summary() -> str:
    summary_path = Path("reports/benchmark_comparison.md")
    if summary_path.exists():
        return summary_path.read_text(encoding="utf-8")
    
    return """
    ### Benchmark Reports
    Run `python scripts/evaluate.py --step all` to generate the complete benchmark comparison table.
    """


# ==============================================================================
# GRADIO APPLICATION LAYOUT
# ==============================================================================
def create_app() -> gr.Blocks:
    with gr.Blocks(title="VoiceOps 2026 - ZentroMart Voice Support Agent") as demo:
        # Header
        gr.HTML("""
        <div class="brand-header">
            <h1 class="brand-title">ZentroMart VoiceOps 2026</h1>
            <p class="brand-subtitle">
                Omnichannel Conversational AI Customer Support Platform &bull; Consulting & Analytics Club IIT Guwahati &times; Gnani.ai
            </p>
            <div style="margin-top: 12px; display: flex; gap: 8px; flex-wrap: wrap;">
                <span class="badge-pill badge-emerald">Fixed Ref Date: 2026-10-20</span>
                <span class="badge-pill badge-sky">Gemini 2.5 Flash</span>
                <span class="badge-pill badge-emerald">Gnani Prisma STT</span>
                <span class="badge-pill badge-sky">Gnani Timbre TTS (Kaveri)</span>
                <span class="badge-pill badge-emerald">154 KB Chunks Indexed</span>
            </div>
        </div>
        """)

        session_state = gr.State(value=f"web_session_{uuid.uuid4().hex[:8]}")
        active_lookup_oid = gr.State(value="")

        with gr.Tabs():
            # ------------------------------------------------------------------
            # TAB 1: CUSTOMER SUPPORT (TEXT CHAT)
            # ------------------------------------------------------------------
            with gr.Tab("💬 Customer Support (Text Chat)"):
                with gr.Row():
                    with gr.Column(scale=7):
                        chatbot = gr.Chatbot(
                            height=520,
                            label="ZentroMart Support Conversation",
                        )
                        with gr.Row():
                            msg_input = gr.Textbox(
                                placeholder="Type your inquiry here (e.g., 'How fast are refunds to ZentroMart Wallet?' or 'Order number 8978 2837')...",
                                lines=2,
                                scale=8,
                                show_label=False,
                            )
                            send_btn = gr.Button("Send", variant="primary", scale=2)
                        
                        with gr.Row():
                            mode_select = gr.Radio(
                                choices=["full", "rag", "baseline"],
                                value="full",
                                label="Intelligence Mode",
                                info="Full: RAG + AGT-IDV + Escalations | RAG: KB Grounding only | Baseline: Direct LLM",
                            )
                            reset_btn = gr.Button("Reset Conversation", variant="secondary")

                    with gr.Column(scale=5):
                        gr.Markdown("### 🔍 Live Turn Inspector")
                        inspector_panel = gr.HTML(
                            value="<p style='color: #94a3b8;'>Start a conversation to inspect agent decisions, grounding sources, identity state, and tool audits.</p>"
                        )

                # Events for Tab 1
                send_btn.click(
                    chat_respond,
                    inputs=[msg_input, chatbot, session_state, mode_select],
                    outputs=[msg_input, chatbot, inspector_panel],
                )
                msg_input.submit(
                    chat_respond,
                    inputs=[msg_input, chatbot, session_state, mode_select],
                    outputs=[msg_input, chatbot, inspector_panel],
                )
                reset_btn.click(
                    reset_chat_session,
                    outputs=[chatbot, session_state, inspector_panel],
                )

            # ------------------------------------------------------------------
            # TAB 2: VOICE ASSISTANT
            # ------------------------------------------------------------------
            with gr.Tab("🎙️ Voice Assistant (Gnani STT & TTS)"):
                gr.Markdown("Speak to the agent or upload a customer voice clip. Audio is transcribed via **Gnani Prisma v2.5** and synthesized via **Gnani Timbre v2.5 (Kaveri)**.")
                with gr.Row():
                    with gr.Column(scale=5):
                        audio_in = gr.Audio(
                            sources=["microphone", "upload"],
                            type="filepath",
                            label="Customer Voice Input (.wav / .mp3)",
                        )
                        with gr.Row():
                            stt_lang = gr.Dropdown(
                                choices=["en-IN", "hi-IN"],
                                value="en-IN",
                                label="Gnani STT Language",
                            )
                            tts_voice = gr.Dropdown(
                                choices=["Kaveri"],
                                value="Kaveri",
                                label="Gnani TTS Voice",
                            )
                            tts_model = gr.Textbox(
                                value="timbre-v2.5",
                                label="Gnani TTS Model",
                                interactive=False,
                            )
                        process_voice_btn = gr.Button("Process Voice Turn", variant="primary")

                    with gr.Column(scale=7):
                        audio_out = gr.Audio(
                            label="Agent Spoken Audio Reply (Gnani TTS)",
                            type="filepath",
                            autoplay=True,
                        )
                        with gr.Row():
                            with gr.Column(scale=6):
                                heard_box = gr.Textbox(label="Transcribed Input (Gnani STT)", lines=2)
                            with gr.Column(scale=6):
                                spoken_box = gr.Textbox(label="Spoken Reply Text", lines=2)

                        with gr.Row():
                            v_decision = gr.HTML(label="Decision")
                            v_citations = gr.HTML(label="Evidence Section IDs")

                        gr.Markdown("#### ⚡ Turn Latency Breakdown")
                        with gr.Row():
                            stt_lat = gr.Textbox(label="Gnani STT Latency")
                            agent_lat = gr.Textbox(label="Agent RAG Latency")
                            tts_lat = gr.Textbox(label="Gnani TTS Latency")
                            total_lat = gr.Textbox(label="Total Roundtrip")

                process_voice_btn.click(
                    process_voice_audio,
                    inputs=[audio_in, stt_lang, tts_voice, tts_model, session_state],
                    outputs=[
                        audio_out, heard_box, spoken_box, v_decision, v_citations,
                        stt_lat, agent_lat, tts_lat, total_lat
                    ],
                )

            # ------------------------------------------------------------------
            # TAB 3: KNOWLEDGE BASE BROWSER
            # ------------------------------------------------------------------
            with gr.Tab("📚 Knowledge Base Browser"):
                gr.Markdown("Inspect the 154 indexed sections and product catalog chunks. Demonstrates document precedence and superseded status filtering.")
                with gr.Row():
                    kb_search_box = gr.Textbox(
                        placeholder="Search policy or catalog (e.g., 'wallet refund', 'damaged TV', 'Kestrel smartphone')...",
                        label="Knowledge Base Query",
                        scale=8,
                    )
                    kb_cat_filter = gr.Dropdown(
                        choices=["ALL", "policy", "agent_guide", "help", "faq", "marketing", "catalog"],
                        value="ALL",
                        label="Category Filter",
                        scale=3,
                    )
                    kb_search_btn = gr.Button("Search KB", variant="primary", scale=1)

                kb_table = gr.Dataframe(
                    headers=["Section ID", "Title", "Category", "Precedence", "Status", "Excerpt"],
                    datatype=["str", "str", "str", "str", "str", "str"],
                    interactive=False,
                    label="Knowledge Base Index Registry",
                )

                kb_search_btn.click(
                    search_knowledge_base,
                    inputs=[kb_search_box, kb_cat_filter],
                    outputs=[kb_table],
                )

            # ------------------------------------------------------------------
            # TAB 4: ORDER LOOKUP & IDENTITY VERIFICATION DEMO
            # ------------------------------------------------------------------
            with gr.Tab("🔐 Order Verification (AGT-IDV)"):
                gr.Markdown("Interactive demonstration of AGT-IDV caller identity verification before disclosing customer order facts.")
                with gr.Row():
                    with gr.Column(scale=5):
                        order_id_input = gr.Textbox(
                            value="89782837",
                            label="Customer Order ID",
                            placeholder="Enter 8-digit order number...",
                        )
                        init_lookup_btn = gr.Button("Initiate Lookup", variant="primary")
                        
                        idv_challenge_box = gr.HTML()

                        proof_input = gr.Textbox(
                            label="Caller Verification Proof",
                            placeholder="Enter last 4 digits of phone or 6-digit delivery pincode...",
                            interactive=False,
                        )
                        verify_proof_btn = gr.Button("Verify Identity", variant="secondary")

                    with gr.Column(scale=7):
                        idv_result_box = gr.HTML()
                        facts_json_box = gr.Code(language="json", label="Deterministic Order Facts (Python Derived)")

                init_lookup_btn.click(
                    initiate_order_lookup,
                    inputs=[order_id_input],
                    outputs=[idv_challenge_box, active_lookup_oid, proof_input, idv_result_box],
                )
                verify_proof_btn.click(
                    verify_order_proof,
                    inputs=[active_lookup_oid, proof_input],
                    outputs=[idv_result_box, facts_json_box],
                )

            # ------------------------------------------------------------------
            # TAB 5: EVALUATION LAB
            # ------------------------------------------------------------------
            with gr.Tab("📊 Evaluation Lab"):
                gr.Markdown("### Comprehensive Benchmark Performance")
                eval_md = gr.Markdown(value=load_evaluation_summary())
                refresh_eval_btn = gr.Button("Refresh Benchmark Results")
                refresh_eval_btn.click(load_evaluation_summary, outputs=[eval_md])

            # ------------------------------------------------------------------
            # TAB 6: SYSTEM DIAGNOSTICS & DEMO HELPER
            # ------------------------------------------------------------------
            with gr.Tab("⚙️ System Diagnostics & Demo Scenarios"):
                gr.Markdown("### System Architecture & Health Status")
                with gr.Row():
                    with gr.Column():
                        gr.HTML("""
                        <div class="panel-card">
                            <h4 style="margin: 0 0 10px 0; color: #34d399;">Active AI Engine</h4>
                            <p><strong>LLM Model:</strong> gemini-2.5-flash</p>
                            <p><strong>Embeddings:</strong> gemini-embedding-001 (3072 dims)</p>
                            <p><strong>Reference Date:</strong> 2026-10-20 (Business Calendar Fixed)</p>
                            <p><strong>Index Storage:</strong> data_index/kb_index_cache.json (154 Chunks)</p>
                        </div>
                        """)
                    with gr.Column():
                        gr.HTML("""
                        <div class="panel-card">
                            <h4 style="margin: 0 0 10px 0; color: #38bdf8;">Gnani Voice Engine</h4>
                            <p><strong>STT Model:</strong> gnani-prisma-v2.5</p>
                            <p><strong>TTS Model:</strong> timbre-v2.5 (Explicitly Enforced)</p>
                            <p><strong>TTS Voice:</strong> Kaveri</p>
                            <p><strong>Language:</strong> en-IN / hi-IN</p>
                        </div>
                        """)

                gr.Markdown("### 🚀 One-Click Demo Scenarios")
                gr.Markdown("Click a scenario below to auto-populate the inquiry into the Customer Support chat:")
                with gr.Row():
                    demo_trap_btn = gr.Button("Scenario 1: Precedence Trap (Instant Wallet vs FAQ)")
                    demo_festive_btn = gr.Button("Scenario 2: Festive Return Window (30 Days vs 15 Days)")
                    demo_damage_btn = gr.Button("Scenario 3: Damage > ₹10k Escalation & Ticket")

                def load_scenario_1():
                    return "How fast are refunds to the ZentroMart Wallet?"
                def load_scenario_2():
                    return "What is the return window for clothing bought during the festive sale?"
                def load_scenario_3():
                    return "Order number is 8978 2837. My Zentro Vista 43-inch 4K Smart TV arrived with a shattered screen. Can you approve a replacement right now?"

                demo_trap_btn.click(load_scenario_1, outputs=[msg_input])
                demo_festive_btn.click(load_scenario_2, outputs=[msg_input])
                demo_damage_btn.click(load_scenario_3, outputs=[msg_input])

        # Initial load of KB
        demo.load(search_knowledge_base, inputs=[gr.State(""), gr.State("ALL")], outputs=[kb_table])

    return demo


if __name__ == "__main__":
    app = create_app()
    app.launch(server_name="0.0.0.0", server_port=7860, share=False, css=CUSTOM_CSS)
