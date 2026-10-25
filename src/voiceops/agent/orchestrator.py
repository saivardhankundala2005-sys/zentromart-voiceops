"""Shared conversational SupportAgent orchestrator (Text, Voice, and Batch CLI)."""
import json
import re
from typing import Dict, Any, List, Optional
from voiceops.config import REFERENCE_DATE
from voiceops.llm.provider import get_llm_provider
from voiceops.knowledge.retriever import get_retriever
from voiceops.tools.bridge import get_tools_bridge
from voiceops.data.loader import get_data_loader
from voiceops.data.order_facts import compute_order_facts
from voiceops.agent.state import get_conversation_manager, ConversationState
from voiceops.agent.prompts import (
    BASE_SYSTEM_PROMPT, GROUNDED_RAG_USER_PROMPT, BASELINE_USER_PROMPT
)
from voiceops.agent.policy import check_policy_triggers
from voiceops.voice.speech_text import normalize_for_speech

class AgentResponse:
    """Structured response object returned by SupportAgent.respond()."""
    def __init__(
        self,
        reply: str,
        spoken_reply: str,
        decision: str,
        conversation_id: str,
        turn_id: int,
        retrieved_chunks: List[Dict[str, Any]],
        tool_calls: List[str],
        ticket_id: Optional[str] = None,
        language: str = "en",
        order_id: Optional[str] = None,
        verified: bool = False,
    ):
        self.reply = reply
        self.spoken_reply = spoken_reply
        self.decision = decision
        self.conversation_id = conversation_id
        self.turn_id = turn_id
        self.retrieved_chunks = retrieved_chunks
        self.tool_calls = tool_calls
        self.ticket_id = ticket_id
        self.language = language
        self.order_id = order_id
        self.verified = verified

    def to_dict(self) -> Dict[str, Any]:
        return {
            "reply": self.reply,
            "spoken_reply": self.spoken_reply,
            "decision": self.decision,
            "conversation_id": self.conversation_id,
            "turn_id": self.turn_id,
            "retrieved_chunks": self.retrieved_chunks,
            "tool_calls": self.tool_calls,
            "ticket_id": self.ticket_id,
            "language": self.language,
            "order_id": self.order_id,
            "verified": self.verified,
        }

def detect_language(text: str) -> str:
    """Detect whether query is English, Hindi (Devanagari), or Hinglish."""
    # Check for Devanagari Unicode \u0900-\u097F
    if re.search(r"[\u0900-\u097F]", text):
        return "hi"
    # Check for common Hinglish words
    hinglish_markers = [
        r"\bhai\b", r"\bko\b", r"\bmein\b", r"\bse\b", r"\bkaise\b", r"\bkya\b",
        r"\bkar\b", r"\bdo\b", r"\bhain\b", r"\bkitne\b", r"\bdin\b", r"\bshuru\b",
        r"\bchahiye\b", r"\bhoga\b", r"\bkab\b", r"\btak\b", r"\baayega\b", r"\bpe\b"
    ]
    if any(re.search(pat, text.lower()) for pat in hinglish_markers):
        return "hi-en"
    return "en"

def extract_order_id(text: str) -> Optional[str]:
    """Extract an 8-digit order ID, including spaced or separated formats."""
    # Direct 8-digit match
    m = re.findall(r"\b(\d{8})\b", text)
    if m:
        return m[0]
    # Spaced pairs e.g. 8978 2837
    m2 = re.findall(r"\b(\d{4})\s+(\d{4})\b", text)
    if m2:
        return f"{m2[0][0]}{m2[0][1]}"
    return None

def parse_llm_json(raw_text: str) -> Dict[str, Any]:
    """Robustly parse JSON output from LLM, stripping markdown code fences."""
    text = raw_text.strip()
    if text.startswith("```"):
        text = re.sub(r"^```(?:json)?\s*", "", text, flags=re.MULTILINE)
        text = re.sub(r"```\s*$", "", text, flags=re.MULTILINE).strip()
    try:
        return json.loads(text)
    except Exception:
        m = re.search(r"(\{.*\})", text, re.DOTALL)
        if m:
            try:
                return json.loads(m.group(1))
            except Exception:
                pass
    return {"reply": raw_text, "decision": "answer"}

class SupportAgent:
    """Shared conversational agent for ZentroMart Customer Support."""

    def __init__(self):
        self.llm = get_llm_provider()
        self.retriever = get_retriever()
        self.tools = get_tools_bridge()
        self.data_loader = get_data_loader()
        self.conv_manager = get_conversation_manager()

    def respond(
        self,
        message: str,
        conversation_id: str = "default",
        language_hint: Optional[str] = None,
        mode: str = "full", # "baseline", "rag", or "full"
    ) -> AgentResponse:
        """Process one conversational turn and return structured AgentResponse."""
        conv = self.conv_manager.get_or_create(conversation_id)
        current_turn = conv.turn_count + 1
        lang = language_hint or detect_language(message)
        conv.language = lang

        executed_tool_calls: List[str] = []
        retrieved_chunks: List[Dict[str, Any]] = []
        order_facts_text = "No verified order details."
        ticket_id = None
        decision = "answer"

        # ----------------------------------------------------------------------
        # MODE 1: BASELINE (Prompting only, no documents, no tools)
        # ----------------------------------------------------------------------
        if mode == "baseline":
            prompt = BASELINE_USER_PROMPT.format(
                history_text=conv.get_history_summary(),
                user_message=message
            )
            raw_json = self.llm.generate(
                prompt=prompt,
                system_instruction=BASE_SYSTEM_PROMPT,
                json_mode=True
            )
            parsed = parse_llm_json(raw_json)
            reply = parsed.get("reply", "Hello, I am ZentroMart's AI support assistant.")
            decision = parsed.get("decision", "answer")

            spoken = normalize_for_speech(reply)
            conv.add_turn(message, reply, decision, [], [], None)
            return AgentResponse(
                reply=reply,
                spoken_reply=spoken,
                decision=decision,
                conversation_id=conversation_id,
                turn_id=current_turn,
                retrieved_chunks=[],
                tool_calls=[],
                language=lang,
            )

        # ----------------------------------------------------------------------
        # MODE 2 & 3: RAG & FULL CONVERSATIONAL AGENT
        # ----------------------------------------------------------------------
        extracted_oid = extract_order_id(message)
        active_oid = extracted_oid or conv.order_id

        # 1. Run Verification State Machine if order is involved (in Full mode)
        verification_prompt_needed = False
        verification_failed = False
        is_verified = False

        if mode == "full" and (active_oid or conv.verification_sm.state != "idle"):
            v_res = conv.verification_sm.process_turn(message, extracted_order_id=extracted_oid)
            if v_res.get("tool_call"):
                executed_tool_calls.append(v_res["tool_call"])
                
            conv.order_id = v_res.get("order_id") or conv.order_id
            is_verified = v_res.get("verified", False)
            
            # If verification prompt needed (Turn 2 after order ID)
            if v_res.get("prompt_for_proof"):
                verification_prompt_needed = True
            elif v_res.get("verification_failed"):
                verification_failed = True

            # If order is verified, compute derived facts in Python
            if is_verified:
                full_order = self.data_loader.get_order_full(conv.order_id)
                if full_order:
                    conv.verified_order = full_order
                    facts = compute_order_facts(full_order, self.data_loader.holidays)
                    order_facts_text = json.dumps(facts, indent=2, default=str)

        # 2. Check Policy Escalation & Refusal Triggers
        order_facts_obj = None
        if conv.verified_order:
            order_facts_obj = compute_order_facts(conv.verified_order, self.data_loader.holidays)
        policy_triggers = check_policy_triggers(message, order_facts_obj)

        # 3. Formulate Search Query & Hybrid Retrieval
        search_query = message
        # If follow-up in context of an order, augment search query
        if conv.order_id and len(conv.history) > 0:
            last_turn = conv.history[-1]["user"]
            search_query = f"{last_turn} {message}"

        # Extract SKU hint if any item in verified order or message
        sku_hint = None
        if conv.verified_order and conv.verified_order.get("items"):
            sku_hint = conv.verified_order["items"][0].get("sku")
        
        retrieved_chunks = self.retriever.retrieve(
            query=search_query,
            top_k=6,
            sku_hint=sku_hint,
        )
        context_text = "\n\n---\n\n".join(
            f"[{c['section_id']}] {c['title']} > {c['section_heading']}:\n{c['text']}"
            for c in retrieved_chunks
        )

        # 4. Handle Identity Verification Direct Flow (Standard Conversation Pattern)
        # Turn 1: If user just asks for help with order without giving ID -> clarify, introduce as AI, ask for order ID!
        if current_turn == 1 and not active_oid and ("order" in message.lower() or "help" in message.lower()):
            if lang == "hi":
                reply = "नमस्ते! मैं ज़ेंट्रोमार्ट का एआई सहायता सहायक (AI support assistant) हूँ। मैं आपकी सहायता कर सकता हूँ, कृपया अपना 8-अंकों का ऑर्डर आईडी बताएं।"
            elif lang == "hi-en":
                reply = "Hello! Main ZentroMart ka AI support assistant hoon. Main aapki zaroor help karunga, kripya apna 8-digit order ID share karein."
            else:
                reply = "Hello! I am ZentroMart's AI support assistant. I would be happy to help you with your order. Could you please share your 8-digit order ID?"
            decision = "clarify"
            spoken = normalize_for_speech(reply)
            conv.add_turn(message, reply, decision, [c["section_id"] for c in retrieved_chunks], executed_tool_calls, None)
            return AgentResponse(
                reply=reply,
                spoken_reply=spoken,
                decision=decision,
                conversation_id=conversation_id,
                turn_id=current_turn,
                retrieved_chunks=retrieved_chunks,
                tool_calls=executed_tool_calls,
                language=lang,
                order_id=conv.order_id,
                verified=False,
            )

        # Turn 2: Order ID just provided -> ask for phone last 4 or delivery pincode!
        if verification_prompt_needed and not is_verified:
            if lang == "hi":
                reply = f"ऑर्डर आईडी {conv.order_id} मिल गया है। सुरक्षा जांच के लिए, कृपया अपने पंजीकृत फोन नंबर के आख़िरी 4 अंक (last 4 digits) या डिलीवरी पिनकोड (delivery pincode) बताएं।"
            elif lang == "hi-en":
                reply = f"Order ID {conv.order_id} mil gaya hai. Verification ke liye, please registered phone number ke last 4 digits ya delivery pincode share karein."
            else:
                reply = f"Thank you for sharing order ID {conv.order_id}. For verification, please provide the last 4 digits of your registered phone number or your delivery pincode."
            decision = "clarify"
            spoken = normalize_for_speech(reply)
            conv.add_turn(message, reply, decision, [c["section_id"] for c in retrieved_chunks], executed_tool_calls, None)
            return AgentResponse(
                reply=reply,
                spoken_reply=spoken,
                decision=decision,
                conversation_id=conversation_id,
                turn_id=current_turn,
                retrieved_chunks=retrieved_chunks,
                tool_calls=executed_tool_calls,
                language=lang,
                order_id=conv.order_id,
                verified=False,
            )

        # 5. LLM Prompt Construction & Generation
        prompt = GROUNDED_RAG_USER_PROMPT.format(
            context_text=context_text,
            order_facts_text=order_facts_text,
            history_text=conv.get_history_summary(),
            user_message=message
        )

        system_prompt = BASE_SYSTEM_PROMPT
        if current_turn == 1:
            system_prompt += "\nIMPORTANT: This is Turn 1. You MUST introduce yourself as ZentroMart's AI support assistant."

        raw_json = self.llm.generate(
            prompt=prompt,
            system_instruction=system_prompt,
            json_mode=True
        )

        parsed = parse_llm_json(raw_json)
        reply = parsed.get("reply", "")
        decision = parsed.get("decision", "answer")
        needs_ticket = parsed.get("needs_ticket", False)
        ticket_summary = parsed.get("ticket_summary", "")

        # 6. Apply Policy Escalation & Ticket Execution (in Full mode)
        if mode == "full":
            if policy_triggers["should_escalate"] or needs_ticket:
                decision = "escalate" if not policy_triggers["is_compensation_demand"] else "refuse"
                # Execute create_ticket
                summary = ticket_summary or (
                    f"Customer issue: {message}. Escalation reasons: {', '.join(policy_triggers['escalation_reasons'])}. "
                    f"Order: {conv.order_id or 'None'}. Handed over for Priority Resolution."
                )
                tkt_res = self.tools.create_ticket(
                    summary=summary,
                    order_id=conv.order_id,
                    reason=policy_triggers["escalation_reasons"][0] if policy_triggers["escalation_reasons"] else "Customer Escalation",
                    intent="escalation"
                )
                if tkt_res.get("created"):
                    ticket_id = tkt_res.get("ticket_id")
                    executed_tool_calls.append("create_ticket()")
                    # Ensure ticket ID and handover mention is present in reply if not already there
                    if "ticket" not in reply.lower() and "colleague" not in reply.lower() and "team" not in reply.lower():
                        if lang == "hi":
                            reply += f" मैंने आपकी शिकायत के लिए टिकट (Ticket ID: {ticket_id}) बना दिया है। हमारी टीम जल्द आपसे संपर्क करेगी।"
                        elif lang == "hi-en":
                            reply += f" Maine aapke issue ke liye ticket (ID: {ticket_id}) create kar diya hai. Hamari team aapse jald hi contact karegi."
                        else:
                            reply += f" I have created a priority support ticket ({ticket_id}) for your request. Our specialized team will reach out to you shortly."

        # Ensure first turn introduces as AI assistant
        if current_turn == 1 and not any(w in reply.lower() for w in ("ai", "artificial intelligence", "एआई", "कृत्रिम बुद्धिमत्ता")):
            if lang == "hi":
                reply = "नमस्ते! मैं ज़ेंट्रोमार्ट का एआई सहायता सहायक (AI support assistant) हूँ। " + reply
            else:
                reply = "Hello! I am ZentroMart's AI customer support assistant. " + reply

        spoken = normalize_for_speech(reply)
        conv.add_turn(
            user_message=message,
            assistant_reply=reply,
            decision=decision,
            retrieved_section_ids=[c["section_id"] for c in retrieved_chunks],
            tool_calls=executed_tool_calls,
            ticket_id=ticket_id,
        )

        return AgentResponse(
            reply=reply,
            spoken_reply=spoken,
            decision=decision,
            conversation_id=conversation_id,
            turn_id=current_turn,
            retrieved_chunks=retrieved_chunks,
            tool_calls=executed_tool_calls,
            ticket_id=ticket_id,
            language=lang,
            order_id=conv.order_id,
            verified=is_verified,
        )

# Global singleton
_SUPPORT_AGENT_INSTANCE: Optional[SupportAgent] = None

def get_support_agent() -> SupportAgent:
    global _SUPPORT_AGENT_INSTANCE
    if _SUPPORT_AGENT_INSTANCE is None:
        _SUPPORT_AGENT_INSTANCE = SupportAgent()
    return _SUPPORT_AGENT_INSTANCE
