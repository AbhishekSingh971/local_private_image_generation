"""Prompt enhancement and size presets for Stable Diffusion 1.5.

The user's prompt is authoritative. Explicit full-body framing requests must remain
active for reference-image generation; img2img strength still controls how strongly
the source composition is preserved.
"""
from __future__ import annotations

import re
import secrets

IMAGE_DIMENSIONS = {
    "portrait": (512, 768),
    "square": (512, 512),
    "landscape": (768, 512),
}

IMAGE_PRESETS = {"standard", "photorealistic", "full_body_photo"}
MAX_GENERATION_SEED = (2 ** 31) - 1

# The browser accepts a 1,200-character user prompt. Prompt planning may add
# reviewed quality, framing, anatomy and identity constraints before it reaches
# the GPU worker, so remote transport limits must cover the expanded prompt too.
MAX_REMOTE_PROMPT_CHARS = 2400
MAX_REMOTE_NEGATIVE_PROMPT_CHARS = 1200


def validate_remote_generation_prompts(prompt: str, negative_prompt: str = "") -> tuple[str, str]:
    """Validate the expanded prompts sent from the app to a remote GPU worker."""
    positive = str(prompt or "").strip()
    negative = str(negative_prompt or "").strip()
    if not positive:
        raise ValueError(
            "The cloud GPU worker received an empty prompt. Check that the Prompt field is filled and submitted."
        )
    if len(positive) > MAX_REMOTE_PROMPT_CHARS:
        raise ValueError(
            f"The expanded prompt exceeds the remote limit of {MAX_REMOTE_PROMPT_CHARS} characters."
        )
    if len(negative) > MAX_REMOTE_NEGATIVE_PROMPT_CHARS:
        raise ValueError(
            f"The expanded negative prompt exceeds the remote limit of {MAX_REMOTE_NEGATIVE_PROMPT_CHARS} characters."
        )
    return positive, negative



def fit_sd15_prompt(tokenizer, prompt: str, max_tokens: int = 77) -> str:
    """Fit a prompt to Stable Diffusion 1.5's CLIP context without position errors.

    SD 1.5 CLIP has a 77-token context (including special tokens). The function
    preserves the prompt's leading instructions and trims only the tail. Callers
    should put identity, subject, and pose constraints before optional style detail.
    """
    prompt = str(prompt or "").strip()
    if not prompt:
        return prompt

    try:
        model_limit = int(getattr(tokenizer, "model_max_length", max_tokens))
    except (TypeError, ValueError):
        model_limit = max_tokens
    if model_limit <= 2 or model_limit > 1000:
        model_limit = max_tokens
    limit = min(int(max_tokens), model_limit)

    try:
        tokens = tokenizer.tokenize(prompt)
        special_count = int(tokenizer.num_special_tokens_to_add(pair=False))
        if len(tokens) + special_count <= limit:
            return prompt

        keep = max(0, limit - special_count)
        while keep > 0:
            candidate = tokenizer.convert_tokens_to_string(tokens[:keep]).strip()
            if candidate and len(tokenizer.tokenize(candidate)) + special_count <= limit:
                return candidate
            keep -= 1
        return ""
    except Exception as exc:
        raise ValueError(f"Could not fit prompt to the SD 1.5 CLIP context: {exc}") from exc

def resolve_generation_seed(seed: object = None) -> int:
    """Return a safe 31-bit seed, creating one when the caller leaves it blank."""
    if seed is None or str(seed).strip() == "":
        return secrets.randbelow(MAX_GENERATION_SEED + 1)
    try:
        value = int(str(seed).strip())
    except (TypeError, ValueError) as exc:
        raise ValueError("Seed must be an integer between 0 and 2147483647.") from exc
    if not 0 <= value <= MAX_GENERATION_SEED:
        raise ValueError("Seed must be an integer between 0 and 2147483647.")
    return value


# Explicit style requests always win over a photo-oriented preset.
_NON_PHOTO_STYLE = re.compile(
    r"\b(?:anime|manga|cartoon|comic(?:[ -]?book)?|illustration|illustrated|"
    r"drawing|sketch|watercolor|watercolour|oil painting|acrylic painting|"
    r"painting|digital art|pixel art|3d render|cgi|claymation|stop motion)\b",
    re.IGNORECASE,
)
_EXPLICIT_FRAMING = re.compile(
    r"\b(?:extreme close[ -]?up|close[ -]?up|headshot|bust shot|waist[ -]?up|"
    r"half[ -]?body|full[ -]?body|head[ -]?to[ -]?toe|wide shot|long shot|"
    r"macro shot|aerial view|overhead view|landscape composition|portrait crop)\b",
    re.IGNORECASE,
)
_HUMAN_SUBJECT = re.compile(
    r"\b(?:person|people|woman|women|man|men|girl|boy|model|portrait|face|"
    r"character|subject|bride|groom|actor|actress|dancer|athlete)\b",
    re.IGNORECASE,
)

_FACE_FEATURE_PATTERNS = {
    "age": re.compile(r"\b(?:young adult|adult|middle[- ]aged|mature|elderly|older)\b", re.I),
    "face_shape": re.compile(r"\b(?:oval|round|heart[- ]shaped|square|diamond[- ]shaped|long|angular)\s+face\b", re.I),
    "eyes": re.compile(r"\b(?:(?:almond[- ]shaped|round|hooded|monolid|wide[- ]set|close[- ]set|large|small)\s+)?(?:brown|hazel|blue|green|grey|gray|dark|light)?\s*eyes?\b", re.I),
    "eyebrows": re.compile(r"\b(?:thick|thin|arched|straight|soft|defined)\s+eyebrows?\b", re.I),
    "nose": re.compile(r"\b(?:straight|small|wide|narrow|button|upturned|rounded|defined)\s+nose\b", re.I),
    "lips": re.compile(r"\b(?:(?:full|thin|wide|soft|defined|natural)\s+)?lips?\b", re.I),
    "skin": re.compile(r"\b(?:(?:fair|light|medium|olive|tan|brown|deep|dark|warm|cool)\s+)?skin(?:\s+tone)?\b", re.I),
    "hair": re.compile(r"\b(?:(?:short|long|curly|wavy|straight)\s+)?(?:black|brown|dark|blonde|blond|red|auburn|burgundy|grey|gray)(?:\s+(?:curly|wavy|straight))?\s+hair\b", re.I),
    "expression": re.compile(r"\b(?:neutral expression|gentle smile|smiling|laughing|serious expression|surprised expression)\b", re.I),
    "view": re.compile(r"\b(?:front[- ]facing|three[- ]quarter|left profile|right profile|side profile|looking upward|looking down)\b", re.I),
}


def extract_face_prompt_matrix(prompt: str) -> dict:
    """Extract only facial traits explicitly present in a user's natural-language prompt."""
    source = str(prompt or "").strip()
    features = {}
    for key, pattern in _FACE_FEATURE_PATTERNS.items():
        match = pattern.search(source)
        if match:
            features[key] = match.group(0).strip()
    subject_detected = bool(_HUMAN_SUBJECT.search(source))
    face_requested = subject_detected and not bool(re.search(
        r"\b(?:back view|view from behind|face hidden|face covered|covered face|helmet covering the face)\b",
        source, re.I,
    ))
    return {
        "subject_detected": subject_detected,
        "face_requested": face_requested,
        "features": features,
        "feature_count": len(features),
    }


def build_face_identity_prompt(
    prompt: str,
    *,
    profile_token: str | None = None,
    profile_description: str = "",
) -> tuple[str, dict]:
    """Prepare a trained-identity trigger or a prompt-derived facial-feature matrix.

    A trained profile is activated with the exact trigger phrase used by FaceDataset.
    Without a trained profile, only neutral anatomy/rendering guidance is added; no
    ethnicity, age, eye colour, face shape or other identity traits are invented.
    """
    source = str(prompt or "").strip()
    if not source:
        raise ValueError("Enter a prompt describing the image you want.")
    matrix = extract_face_prompt_matrix(source)

    token = str(profile_token or "").strip()
    if token:
        if not re.fullmatch(r"[A-Za-z0-9_-]{1,48}", token):
            raise ValueError("The selected face profile has an invalid trigger token.")
        # Match the training dataset's exact trigger phrase: a photo of {token} person.
        trigger = f"a photo of {token} person"
        if re.search(rf"(?<![A-Za-z0-9_-]){re.escape(token)}(?![A-Za-z0-9_-])", source):
            effective = source
            if not effective.lower().startswith(trigger.lower()):
                effective = trigger + ", " + effective
        else:
            effective = trigger + ", " + source
        description_words = re.findall(r"\S+", str(profile_description or "").strip())
        # Training rotates 18-word description chunks. Use a short first chunk at
        # inference so the user's scene and pose instructions survive CLIP truncation.
        description = " ".join(description_words[:10])
        if description and description.lower() not in effective.lower():
            effective = trigger + ", " + description + ", " + source
        matrix.update({
            "mode": "trained_lora",
            "trigger_token": token,
            "profile_description_used": bool(description),
            "identity_constraints": ["trained profile trigger", "preserve trained facial identity"],
        })
        return effective, matrix

    if matrix["face_requested"]:
        face_guard = (
            "clear, coherent natural human face, balanced facial proportions, "
            "naturally aligned eyes, anatomically well-formed nose and lips, "
            "consistent jaw and cheek structure, realistic skin texture"
        )
        effective = face_guard + ", " + source
        matrix.update({
            "mode": "prompt_matrix",
            "trigger_token": None,
            "profile_description_used": False,
            "identity_constraints": ["use only facial traits stated in the prompt", "natural coherent facial anatomy"],
        })
        return effective, matrix

    matrix.update({
        "mode": "base_prompt",
        "trigger_token": None,
        "profile_description_used": False,
        "identity_constraints": [],
    })
    return source, matrix


_PHOTO_PREFIX = (
    "photorealistic photograph, natural skin and material texture, lifelike details, "
    "realistic lighting and shadows, natural colors, "
)
_FULL_BODY_PREFIX = (
    "photorealistic full-body photograph, entire person visible from head to feet, "
    "feet inside the frame, centered subject with space above the head and below the feet, "
    "camera pulled back, natural anatomy and realistic textures, "
)
_PHOTO_NEGATIVE = (
    "blurry, low detail, distorted face, asymmetrical eyes, deformed anatomy, "
    "missing limbs, missing arms, missing shoulders, extra limbs, duplicate body parts, "
    "malformed hands, fused fingers, extra fingers, missing fingers, twisted joints, "
    "disconnected limbs, unnatural torso, distorted hips, unnatural body proportions"
)
_FULL_BODY_NEGATIVE = (
    "cropped body, cut off feet, cut off head, out of frame, incomplete body, "
    "missing arm, missing leg, missing shoulder, extra arm, extra leg"
)
_ANATOMY_DETAIL = (
    "anatomically coherent adult human body, complete shoulders and limbs, "
    "natural joint placement, balanced shoulders and hips, coherent torso and pelvis, "
    "realistic facial structure, symmetrical eyes, natural skin texture, "
)


def resolve_image_dimensions(aspect_ratio: str = "portrait") -> tuple[int, int]:
    """Return a supported SD 1.5 canvas size for the requested composition."""
    key = (aspect_ratio or "portrait").strip().lower()
    try:
        return IMAGE_DIMENSIONS[key]
    except KeyError as exc:
        raise ValueError("Image aspect ratio must be portrait, square or landscape.") from exc


def validate_image_preset(preset: str = "full_body_photo") -> str:
    key = (preset or "full_body_photo").strip().lower()
    if key not in IMAGE_PRESETS:
        raise ValueError("Image preset must be standard, photorealistic or full_body_photo.")
    return key


def build_generation_prompts(
    prompt: str,
    negative_prompt: str = "",
    preset: str = "full_body_photo",
    has_reference_image: bool = False,
) -> tuple[str, str]:
    """Enhance realism while honoring explicit composition requests.

    A supplied reference normally keeps its original framing. However, an explicit
    full-body request is a deliberate request to reframe and must receive full-body
    framing and anti-cropping constraints even in img2img mode. The caller's
    denoising strength determines how much the source composition can change.
    Explicit art styles and user prompt/negative prompt wording remain authoritative.
    """
    key = validate_image_preset(preset)
    clean_prompt = (prompt or "").strip()
    clean_negative = (negative_prompt or "").strip()
    if key == "standard" or not clean_prompt:
        return clean_prompt, clean_negative

    if _NON_PHOTO_STYLE.search(clean_prompt):
        return clean_prompt, clean_negative

    is_human = bool(_HUMAN_SUBJECT.search(clean_prompt))
    explicit_full_body = bool(re.search(
        r"\b(?:full[ -]?body|head[ -]?to[ -]?toe|entire body|full length)\b",
        clean_prompt, re.IGNORECASE,
    ))
    # An explicit full-body instruction takes precedence over the source photo's
    # original crop. Without an explicit instruction, keep reference framing intact.
    use_full_body_framing = (
        key == "full_body_photo"
        and is_human
        and (
            explicit_full_body
            or (not has_reference_image and not _EXPLICIT_FRAMING.search(clean_prompt))
        )
    )
    use_full_body_constraints = use_full_body_framing

    prefix = _FULL_BODY_PREFIX if use_full_body_framing else _PHOTO_PREFIX
    if key in {"photorealistic", "full_body_photo"} and is_human:
        prefix += _ANATOMY_DETAIL
    # Give multi-person beach scenes explicit composition guidance. SD 1.5 often
    # merges nearby people or drops limbs when a group scene is described only
    # in natural language, so make the spatial arrangement and framing explicit.
    beach_group_scene = (
        bool(re.search(r"\bbeach\b|\bseaside\b|\bshoreline\b", clean_prompt, re.IGNORECASE))
        and bool(re.search(r"\bwoman\b|\bwomen\b|\bfemale\b", clean_prompt, re.IGNORECASE))
        and bool(re.search(r"\b(?:two|2)\s+(?:(?:adult|naked|shirtless)\s+)*(?:men|males)\b", clean_prompt, re.IGNORECASE))
    )
    if beach_group_scene:
        prefix = (
            "photorealistic wide full-body beach photograph, three separate adult people "
            "visible head to toe, woman centered with two adult men positioned separately "
            "one on each side, clear space between each person, all faces and limbs visible, "
            "tropical trees clearly visible in the background, natural daylight, "
            "realistic camera perspective, "
        ) + prefix

    negative_prefix = _PHOTO_NEGATIVE
    if use_full_body_constraints:
        negative_prefix = _FULL_BODY_NEGATIVE + ", " + negative_prefix
    if beach_group_scene:
        negative_prefix += (
            ", extra people, duplicated people, merged bodies, fused torsos, overlapping "
            "limbs, hidden arms, hidden legs, cropped feet, cropped heads, out of frame, "
            "extra hands, malformed fingers, distorted faces, text, watermark"
        )

    enhanced_prompt = clean_prompt if clean_prompt.lower().startswith(prefix.rstrip(", ").lower()) else prefix + clean_prompt
    enhanced_negative = ", ".join(part for part in (negative_prefix, clean_negative) if part)
    return enhanced_prompt, enhanced_negative
