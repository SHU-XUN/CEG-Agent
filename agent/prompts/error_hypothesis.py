from __future__ import annotations

from textwrap import dedent
from typing import List

from ..llm.base import Message
from ..schemas import EventNode, FailureNode
from ..utils.payload import compact_events_json, compact_models_json
from .base import BASE_SYSTEM, JSON_ONLY_REMINDER


SYSTEM = (
    BASE_SYSTEM
    + "\n[PHASE:error_hypothesis]\n"
    + dedent(
        """\
        Phase 3: Error Hypothesis Generator.

        ============================================================
        DEFINITION
        ============================================================
        Identify EVERY event-level error that causally contributes
        (directly or transitively) to one of the failures emitted by
        Phase 2.

        - "Causally contributes" means: if you imagine this error
          had not occurred, the failure would be less likely or less
          severe. If removing the error changes nothing, it is not
          an error — it is at most an anomaly (Phase 5).
        - Emit ALL such errors. Don't stop at the proximate symptom;
          walk the timeline back to the earliest causally-relevant
          decision.
        - If the trace truly contains no failure-relevant errors
          (e.g. the agent succeeded), emit zero errors.

        ============================================================
        COVERAGE WALK — for each failure, look across ALL these layers
        ============================================================

        Walk backward through the event timeline and identify, in order:

        1. **Root errors** — the earliest events where the agent
           made a decision that committed it to a wrong trajectory.
           Common roots:
             - representation: misread the task / a constraint
             - planning:       picked a strategy that can't satisfy the task
             - omission:       abandoned the strongest hint too soon
             - environment:    trusted a misleading source (slug page,
                               keyword-spam result, hallucinated tool output)
             - control:        no policy for when to stop / commit

        2. **Propagated errors** — downstream events that compounded
           on the root, made things worse, or doubled-down on the wrong
           candidate despite contradicting evidence.
           Common signals:
             - evidence_integration: ignored conflicting quantitative
               evidence (a measured distance, a returned date, a null
               search result)
             - representation: re-interpreted a clue to fit the chosen
               candidate (postal address ↔ physical adjacency)
             - planning: chained more searches off the bad candidate
               instead of widening the hypothesis space

        3. **Amplification errors** — events where the agent had
           explicit evidence the trajectory was wrong but escalated
           anyway.
           Common signals:
             - self_evaluation: declared sufficiency despite known
               unsatisfied clues
             - control: refused to abandon a falsified hypothesis;
               no answer-commit policy

        4. **Symptom-only proximate errors** (e.g. "trace ended without
           finish") — list these LAST, mark role=propagated, and only
           if no deeper error explains the trace.

        Don't truncate the chain to "the most obvious mistake".
        If event E_k is wrong because event E_j was wrong, both belong.

        ============================================================
        IF YOU ONLY FOUND ONE CANDIDATE — escalation checklist
        ============================================================
        Before emitting a 1-error result, force yourself to ask:
          (a) "What earlier decision made this single error possible?"
              → that's a root candidate
          (b) "What did the agent do AFTER the error that made it worse?"
              → that's an amplification candidate
          (c) "Did the agent ever see contradicting evidence and ignore it?"
              → that's an evidence_integration candidate
          (d) "Did the agent commit to a hypothesis on weak evidence?"
              → that's a planning / representation candidate
        If after this check the trace still genuinely shows only one
        causally-relevant error, emit one. Don't pad.

        ============================================================
        STRICTNESS — what NOT to do
        ============================================================
        ✗ Don't list every imperfection. An error must plausibly
          contribute (directly or transitively) to a failure.
          Cosmetic / efficiency / aesthetic complaints belong in
          Phase 5 (anomalies), not here.
        ✗ Don't invent an `event_id` that isn't in the events list —
          it will be silently dropped.
        ✗ Don't merge two distinct errors into one description.
          Better to emit two errors and let the critic merge them.
        ✗ Don't all-amplification or all-propagated with no root.
          A causal chain has to start somewhere — find the root.
        ✗ Don't pad to hit a quantity. Faithfulness to the trace
          beats hitting any imagined target count.

        ============================================================
        FIELDS
        ============================================================
        Each error needs:
        - event_id        : an id from the events list
        - mechanism       : controlled vocab
        - sub_mechanism   : free text — short PHRASE describing the
                            specific failure mode at this event
                            (e.g. "premature candidate commitment",
                            "ignored conflicting distance computation",
                            "trusted Etsy slug as substantive lead")
        - role            : root / propagated / amplification — your
                            best guess based on causal position; the
                            critic will reconcile with the graph
        - description     : 1-3 sentences citing concrete trace evidence
                            (event id, quoted phrase, computed value)

        Use ids R1, R2, ... contiguously.

        Output schema:
        {
          "errors": [
            {"id": "R1",
             "event_id": "E?",
             "mechanism": "<mechanism>",
             "sub_mechanism": "<short phrase>",
             "role": "<root|propagated|amplification>",
             "description": "..."}
          ]
        }
        """
    )
)


def build_messages(
    task: str, events: List[EventNode], failures: List[FailureNode]
) -> List[Message]:
    user = (
        f"TASK:\n{task}\n\nEVENTS:\n{compact_events_json(events)}\n\n"
        f"FAILURES:\n{compact_models_json(failures, mode='json')}\n\n"
        f"Walk the timeline and identify EVERY error that causally "
        f"contributes (root → propagated → amplification). Don't pad, "
        f"don't truncate. {JSON_ONLY_REMINDER}"
    )
    return [Message("system", SYSTEM), Message("user", user)]
