# Annotation guidelines

## safe
Normal requests. Includes hard negatives: harmless messages that contain words like
"ignore", "instructions" or "system prompt", and security-education questions.

## jailbreak
Goal: make the model drop its safety rules or adopt an unrestricted persona.
Examples: DAN-style personas, "no restrictions" or "unrestricted mode",
role-play as an amoral AI, hypotheticals used to get banned content.

## injection
Goal: override or hijack the model's instructions, or extract protected information.
Examples: "ignore previous instructions", asking for the system prompt, secrets or
confidential data, claiming to be a developer or admin, threats to force compliance,
instructions hidden inside data.

## Tie-break
If a message fits both, label by the primary mechanism:
- persona or rule-removal -> jailbreak
- instruction override or information extraction -> injection
Mark such rows ambiguous=True so we can measure them separately.
