from __future__ import annotations

from textwrap import dedent
from typing import List

from ..llm.base import Message
from ..schemas import Edge, EdgeType, ErrorNode, FailureNode
from ..utils.payload import compact_models_json
from .base import BASE_SYSTEM, JSON_ONLY_REMINDER


SYSTEM = (
    BASE_SYSTEM
    + "\n[PHASE:repair_estimator]\n"
    + dedent(
        """\
        Phase 7 (optional): Repair Value Estimator.

        For EACH error r in the input, estimate counterfactually:
            "If only this single error r were fixed, would the failure
             plausibly be averted or significantly mitigated?"

        Bucket the answer into {high, medium, low}.

        ============================================================
        INPUT YOU WILL RECEIVE
        ============================================================
        - errors[]   : every error with id, event_id, mechanism, role,
                       sub_mechanism, description.
        - failures[] : every failure with id, type, description.
        - edges[]    : the SUBSET of edges relevant to your scoring,
                       restricted to types
                       {attached_to, causes, amplifies, contributes_to}.
                       Use this to compute reach(r), severance, and the
                       in/out degrees referenced below.

        Compute graph quantities directly from `edges[]`. Do NOT
        fabricate them from descriptions.

        ============================================================
        DECISION CHECKLIST (apply IN ORDER; later steps refine the score)
        ============================================================

        STEP 1. Connectivity
          Is r on a causal chain that reaches some failure f, via
          {causes, amplifies, contributes_to}?
            - NO  → score is at most LOW. r cannot affect any failure.
                    Stop here; emit "low".
            - YES → continue.

        STEP 2. Counterfactual severance for the primary failure f*
          Imagine deleting r and all its outgoing edges. Does any OTHER
          error still reach f*?
            - NO  → severance = 1.0  (r is the unique gateway; fixing r
                                       removes the only path to failure)
            - YES → severance = 0.0  (failure persists via other errors)

        STEP 3. Role + structural prior
          role_prior(r):
            root           → 1.0
            amplification  → 0.5
            propagated     → 0.3

          upstream_hazard(r) = 1 if BOTH:
            - r.role == "propagated", AND
            - r has at least one incoming `causes` edge in the graph
          else 0.
          (A propagated symptom with intact upstream gets re-introduced
          even after r is fixed → cap its score.)

          damage_amp(r) = 1 - exp(-k), where k = number of `amplifies`
          edges OUTGOING from r. Saturates: 0→0, 1→~0.63, 3→~0.95.

        STEP 4. Description hedging  D(r) ∈ [-0.2, +0.2]
          Read r.description. Adjust:
            +0.20  if it explicitly says r BLOCKED / PREVENTED / KILLED
                   the task (e.g. "no answer ever emitted")
            +0.10  for strong negative wording ("never", "failed to",
                   "refused", "exhausted budget")
             0.00  neutral
            -0.10  hedged ("could have", "might have", "suboptimal")
            -0.20  explicitly minor ("only marginally", "barely affects")

        ============================================================
        SCORING FORMULA
        ============================================================
            severance = severance(r, f*)            ∈ {0, 1}
            connect   = 1 if reach(r) ≠ ∅ else 0    ∈ {0, 1}
            prior     = role_prior(r)               ∈ {0.3, 0.5, 1.0}
            amp       = damage_amp(r)               ∈ [0, 1)
            hedge     = clip01(0.5 + D(r))          ∈ [0.3, 0.7]
            hazard    = upstream_hazard(r)          ∈ {0, 1}

            score(r) = clip01(
                  0.45 * connect * severance
                + 0.25 * prior
                + 0.15 * amp
                + 0.10 * hedge
                - 0.30 * hazard
            )

        ============================================================
        BUCKETING
        ============================================================
            score ≥ 0.66           → "high"
            0.33 ≤ score < 0.66    → "medium"
            score < 0.33           → "low"

        Tie-breakers when right at a boundary:
        - prefer the LOWER bucket if upstream_hazard(r) = 1
        - prefer the HIGHER bucket if r is the unique root in reach⁻¹(f*)

        ============================================================
        OUTPUT
        ============================================================
        {
          "repair_values": {"R1": "high", "R2": "low"}
        }

        Optional but encouraged: also include a short rationale per
        error so reviewers can audit the score:

        {
          "repair_values": {"R1": "high"},
          "rationale": {
            "R1": "connect=1, severance=1, prior=1.0, amp=0, hedge=0.6, hazard=0 → 0.76 → high"
          }
        }

        Hard rules:
        - Never assign "high" to a propagated error whose upstream is
          unfixed in this counterfactual (single-fix scope).
        - Never assign above "low" to an error with reach(r) = ∅.
        - Use the controlled vocabulary {high, medium, low} only.
        """
    )
)


_RELEVANT_EDGE_TYPES = {
    EdgeType.ATTACHED_TO,
    EdgeType.CAUSES,
    EdgeType.AMPLIFIES,
    EdgeType.CONTRIBUTES_TO,
}


def build_messages(
    errors: List[ErrorNode],
    failures: List[FailureNode],
    edges: List[Edge],
) -> List[Message]:
    relevant = [e for e in edges if e.type in _RELEVANT_EDGE_TYPES]
    user = (
        f"ERRORS:\n{compact_models_json(errors, mode='json')}\n\n"
        f"FAILURES:\n{compact_models_json(failures, mode='json')}\n\n"
        f"EDGES (only types relevant to the rubric):\n"
        f"{compact_models_json(relevant, mode='json')}\n\n"
        f"Apply the 4-step checklist + scoring formula in the system prompt "
        f"to every error. Compute graph quantities from EDGES — do not "
        f"fabricate them. {JSON_ONLY_REMINDER}"
    )
    return [Message("system", SYSTEM), Message("user", user)]
