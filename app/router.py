"""Adaptive router. v1 = cheap, transparent heuristic over query features.
The whole point of the project: match query difficulty to pipeline cost.

TODO v2: replace/augment with an LLM-judge or a small classifier trained on
your labeled benchmark queries -- then benchmark v1 vs v2 in the eval harness.
"""
import re
from app.schemas import RouteDecision, TierName

COMPARISON = re.compile(
    r"\b(compare|versus|vs\.?|relate|relationship|connect|cause|caused|impact|because|difference between)\b",
    re.I,
)
MULTI = re.compile(r"\b(and|then|both|across|combined|as well as|subsequent)\b", re.I)
ENTITY_HINT = re.compile(r"\b[A-Z]{2,}\b")  # crude ticker/acronym proxy


class AdaptiveRouter:
    def route(self, question: str) -> RouteDecision:
        q = question.strip()
        n = len(q.split())
        comparison = bool(COMPARISON.search(q))
        multi = bool(MULTI.search(q))
        entities = len(set(ENTITY_HINT.findall(q)))
        hops = 1 + int(comparison) + int(multi) + max(0, entities - 1)

        features = {
            "length": n, "comparison": comparison,
            "multi_clause": multi, "entities": entities,
        }

        if n <= 8 and not comparison and entities <= 1:
            tier, reason = TierName.long_context, "Short, single-entity, no relational language: skip retrieval."
        elif hops <= 1:
            tier, reason = TierName.hybrid, "Single-hop factual lookup: hybrid retrieval + rerank suffices."
        elif comparison and entities >= 2:
            tier, reason = TierName.graph_rag, "Relational across multiple entities: needs graph traversal."
        else:
            tier, reason = TierName.agentic, "Open-ended multi-step reasoning: agentic loop."

        return RouteDecision(tier=tier, estimated_hops=hops, reasoning=reason, features=features)
