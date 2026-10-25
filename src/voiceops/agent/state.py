"""Session and conversation state management."""
from typing import Dict, Any, List, Optional
from voiceops.agent.verification import VerificationStateMachine

class ConversationState:
    """Encapsulates context, history, and verification state for one conversation."""

    def __init__(self, conversation_id: str):
        self.conversation_id = conversation_id
        self.turn_count = 0
        self.order_id: Optional[str] = None
        self.verification_sm = VerificationStateMachine()
        self.history: List[Dict[str, Any]] = []
        self.ticket_id: Optional[str] = None
        self.language: str = "en"
        self.verified_order: Optional[Dict[str, Any]] = None
        self.created_tickets: List[str] = []

    def add_turn(
        self,
        user_message: str,
        assistant_reply: str,
        decision: str,
        retrieved_section_ids: List[str],
        tool_calls: List[str],
        ticket_id: Optional[str] = None,
    ):
        self.turn_count += 1
        turn_record = {
            "turn": self.turn_count,
            "user": user_message,
            "assistant": assistant_reply,
            "decision": decision,
            "retrieved_section_ids": retrieved_section_ids,
            "tool_calls": tool_calls,
            "ticket_id": ticket_id,
        }
        self.history.append(turn_record)
        if ticket_id:
            self.ticket_id = ticket_id
            self.created_tickets.append(ticket_id)

    def get_history_summary(self, max_turns: int = 5) -> str:
        """Format recent conversation history for prompt context."""
        if not self.history:
            return "No previous turns."
        recent = self.history[-max_turns:]
        lines = []
        for t in recent:
            lines.append(f"Turn {t['turn']} - Customer: {t['user']}")
            lines.append(f"Turn {t['turn']} - Assistant: {t['assistant']}")
        return "\n".join(lines)

class ConversationManager:
    """Thread-safe multi-session manager preventing cross-conversation leakage."""

    def __init__(self):
        self._conversations: Dict[str, ConversationState] = {}

    def get_or_create(self, conversation_id: str) -> ConversationState:
        cid = str(conversation_id).strip() or "default"
        if cid not in self._conversations:
            self._conversations[cid] = ConversationState(cid)
        return self._conversations[cid]

    def reset_conversation(self, conversation_id: str):
        cid = str(conversation_id).strip()
        if cid in self._conversations:
            del self._conversations[cid]

# Global singleton
_CONVERSATION_MANAGER_INSTANCE: Optional[ConversationManager] = None

def get_conversation_manager() -> ConversationManager:
    global _CONVERSATION_MANAGER_INSTANCE
    if _CONVERSATION_MANAGER_INSTANCE is None:
        _CONVERSATION_MANAGER_INSTANCE = ConversationManager()
    return _CONVERSATION_MANAGER_INSTANCE
