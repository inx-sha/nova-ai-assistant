from __future__ import annotations
from dataclasses import dataclass


@dataclass
class PersonaTemplate:
    id: str
    name: str
    description: str
    system_prompt: str
    requires_subject: bool = False
    subject_prompt: str = ""
    keywords: list[str] = None
    anchor_phrase: str = ""
    subject_clause_format: str = ", specializing in {subject}"

    def to_dict(self) -> dict:
        return {
            "id": self.id,
            "name": self.name,
            "description": self.description,
            "requires_subject": self.requires_subject,
            "subject_prompt": self.subject_prompt,
        }


PERSONA_TEMPLATES: dict[str, PersonaTemplate] = {
    "lawyer": PersonaTemplate(
        id="lawyer",
        name="Lawyer",
        description="Structured legal analysis, contractual risk assessment, and statutory principles.",
        requires_subject=False,
        system_prompt=(
            "You are acting as an experienced legal advisor. Analyze questions with precision, "
            "identifying relevant legal principles, potential risks, contractual obligations, and key considerations. "
            "Provide structured, objective, and analytical explanations. Always include an appropriate legal disclaimer "
            "that your answers are for informational purposes only and do not constitute formal legal counsel."
        ),
        keywords=[
            "law", "legal", "contract", "liability", "clause", "statute", "court", "compliance",
            "rights", "damages", "jurisdiction", "agreement", "regulation", "tort", "attorney",
            "prosecution", "litigation", "claim", "plaintiff", "defendant", "settlement",
            "landlord", "tenant", "lease", "deposit", "eviction", "dispute", "sue", "lawsuit"
        ],
        anchor_phrase="legal questions, contract law, regulations, legal rights, liability, statutes, and legal compliance",
        subject_clause_format=", specializing in {subject}",
    ),
    "teacher": PersonaTemplate(
        id="teacher",
        name="Teacher",
        description="Step-by-step academic instruction, pedagogical analogies, and conceptual breakdowns.",
        requires_subject=True,
        subject_prompt="What subject or topic would you like to learn (e.g. Physics, History, Calculus)?",
        system_prompt=(
            "You are an encouraging, expert teacher and tutor{subject_clause}. "
            "Break down complex concepts into clear, intuitive step-by-step explanations. Use vivid analogies, "
            "examples, and check-for-understanding questions to reinforce learning. Be patient, pedagogical, and adaptive."
        ),
        keywords=[
            "teach", "explain", "learn", "study", "homework", "lesson", "concept", "formula",
            "example", "problem", "solve", "quiz", "definition", "tutorial", "practice"
        ],
        anchor_phrase="educational explanations, teaching concepts, academic learning, study questions, and step-by-step lessons",
        subject_clause_format=" in {subject}",
    ),
}


def get_persona_templates() -> list[PersonaTemplate]:
    return list(PERSONA_TEMPLATES.values())


def get_persona(persona_id: str | None) -> PersonaTemplate | None:
    if not persona_id or not persona_id.strip():
        return None
    cleaned = persona_id.strip()
    key = cleaned.lower()
    if key in PERSONA_TEMPLATES:
        return PERSONA_TEMPLATES[key]

    # Custom persona fallback: create lightweight PersonaTemplate dynamically
    display_name = cleaned.title() if cleaned.islower() else cleaned
    return PersonaTemplate(
        id=key,
        name=display_name,
        description=f"Custom persona: {display_name}",
        system_prompt=f"You are acting as a {cleaned}{{subject_clause}}. Stay within that domain, be accurate, and be direct.",
        requires_subject=False,
        subject_prompt="",
        keywords=[],
        anchor_phrase="",
        subject_clause_format=", specializing in {subject}",
    )


