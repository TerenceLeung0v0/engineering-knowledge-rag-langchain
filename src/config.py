from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]

# Data directories
DATA_DIR = PROJECT_ROOT / "data"
CATALOG_DIR = DATA_DIR / "catalog"
RAW_DOCS_DIR = DATA_DIR / "raw_docs"
RAW_DOCS_PDF_DIR = RAW_DOCS_DIR / "pdf"
RAW_DOCS_HTML_DIR = RAW_DOCS_DIR / "html"
RAW_DOCS_MD_DIR = RAW_DOCS_DIR / "md"
CURATED_QA_DIR =  DATA_DIR / "curated_qa"
LORA_TRAIN_DIR=  DATA_DIR / "lora_train"
CI_FIXTURES_DIR = DATA_DIR / "ci_fixtures"
CI_FIXTURES_RAW_DOCS_DIR = CI_FIXTURES_DIR / "raw_docs"

DATA_DIRS = [
    DATA_DIR,
    CATALOG_DIR,
    RAW_DOCS_DIR,
    RAW_DOCS_PDF_DIR,
    RAW_DOCS_HTML_DIR,
    RAW_DOCS_MD_DIR,
    CURATED_QA_DIR,
    LORA_TRAIN_DIR,
    CI_FIXTURES_DIR,
    CI_FIXTURES_RAW_DOCS_DIR
]

# Artifacts directories
ARTIFACTS_DIR = PROJECT_ROOT / "artifacts"
VECTORSTORE_DIR = ARTIFACTS_DIR / "vectorstore"
CI_VECTORSTORE_DIR = ARTIFACTS_DIR / "ci_vectorstore"
LORA_ADAPTER_DIR = ARTIFACTS_DIR / "lora_adapter"
TAG_INDEX_DIR = ARTIFACTS_DIR / "tag_index"

ARTIFACTS_DIRS = [
    VECTORSTORE_DIR,
    CI_VECTORSTORE_DIR,
    LORA_ADAPTER_DIR,
    TAG_INDEX_DIR
]

# Required directories exist before any execution
REQUIRED_DIRS = DATA_DIRS + ARTIFACTS_DIRS

# Ambiguity
AMBIGUITY_CONFIG = {
    "keep_ambiguous_for_generic_queries": True,
    "generic_query_patterns": [
        r"\boverall\b",
        r"\bin general\b",
        r"\bgenerally\b",
        r"^\s*how does\b",
        r"^\s*what is\b",
    ],
    "facet_query_patterns": [
        # IoT Jobs facets
        r"\bqueued\b",
        r"\bin[_\s-]?progress\b",
        r"\bsucceeded\b|\bfailed\b|\brejected\b|\btimed[_\s-]?out\b|\bcanceled\b",
        r"\bjob\s+execution(s)?\b",
        r"\bjob\s+document(s)?\b",
        r"\brollout(\s+configuration)?\b",
        r"\btimeout(s)?\b|\bstep\s+timer\b|\bin[_\s-]?progress\s+timer\b",
        r"\bthing\s+group(s)?\b|\bjob\s+target(s)?\b|\btargets?\b",

        # MQTT “standards definition” style facets
        r"\baccording\s+to\b",
        r"\bv3\.?1\.?1\b|\bv5(\.0)?\b",
        r"\bserver\b|\bclient\b|\btopic\s+name\b|\btopic\s+filter\b",
        r"\bqos\b|\bretain(ed)?\b|\blast\s+will\b|\blwt\b",
        r"\bpublish/subscribe\b|\bpub\s*/\s*sub\b|\bpubsub\b",
    ],
}

# Retrieval
RETRIEVAL_CONFIG = {
    "fetch_k": 50,          # Fetch from FAISS for gating
    "final_k": 4,           # Feed to the prompt
    "max_l2": 0.8,         # L2 distance gate
    "soft_max_l2": 1.05,     # L2 soft distance gate
    "min_keep": 1,          # Min. docs qty after gating
    "min_gap": 0.015,         # Confidence gap gate (ambiguous)
    "max_options": 3,        # Options when hit ambiguous
    "min_group_gap": 0.05,

    # embedding tie-breaker
    "strict_sig": False,

    # query-aware signature embedding tie-breaker
    "enable_sig_tiebreak": True,
    "min_sig_sim": 0.30,
    "min_sig_sim_gap": 0.015,

    # anchor-content embedding tie-breaker
    "enable_anchor_tiebreak": True,
    "min_anchor_sim": 0.3,
    "min_anchor_sim_gap": 0.01,

    # entity-aware
    "enable_entity_resolve": True,
    "require_full_entity_coverage": False,

    # Out-of-domain (OOD)
    "ood": {
        "enabled": True,

        # Positive signals (in-domain keywords / patterns)
        "allow_patterns": [
            # Core IoT / networking protocols
            r"\bmqtt\b",
            r"\bcoap\b",
            r"\bzigbee\b",
            r"\bthread\b",
            r"\bmatter\b",
            r"\bble\b|\bbluetooth\b",

            # General IoT / AWS IoT
            r"\biot\b",
            r"\baws\s*iot\b|\biot\s*core\b|\baws\s*iot\s*core\b",
            r"\bthing\s*shadow\b|\bdevice\s*shadow\b",

            # MQTT concepts
            r"\btopic\b|\btopic\s*filter\b",
            r"\bpublish\b|\bsubscribe\b|\bpub\s*/\s*sub\b|\bpubsub\b",
            r"\bqos\s*[012]\b",
            r"\bbroker\b|\bclient\b|\bretain(ed)?\b|\bretained\s+message\b",
            r"\blast\s+will\b|\blwt\b",

            # AWS IoT Jobs (avoid generic "job" false positives by requiring Jobs-ish context)
            r"\baws\s*iot\s*jobs?\b|\biot\s*jobs?\b",
            r"(?=.*\bjob(s)?\b)(?=.*\b(execution(s)?|document(s)?|device(s)?|rollout|timeout|status|lifecycle|target|deployment|cancel)\b)",
            r"\bjob\s*execution(s)?\b",
            r"\bjob\s*document(s)?\b",
            r"\bjob\s*status\b|\bstatus\s*detail(s)?\b",
            r"\brollout\b.*\bjob(s)?\b|\bjob(s)?\b.*\brollout\b",
            r"\btimeout\b.*\bjob(s)?\b|\bjob(s)?\b.*\btimeout\b",
            r"\bin-?progress\s+timer\b",
            r"\bstep\s+timer\b",
            r"(?=.*\bjob(s)?\b)(?=.*\bretry\b)",
            r"(?=.*\b(threshold|percentage)\b)(?=.*\b(device|devices|thing|things)\b)(?=.*\b(fail|failure)\b)",
        ],
        # Hard OOD triggers (obvious out-of-domain)
        "deny_patterns": [
            r"\bstock(s)?\b|\bshare\s*price\b|\bticker\b|\bearnings\b|\bportfolio\b|\bdividend\b|\bmarket\s*cap\b",
            r"\bweather\b|\bforecast\b|\btemperature\b|\brain\b|\bhumidity\b|\buv\b",
            r"\bmovie\b|\bnetflix\b|\banime\b|\bcelebrity\b|\bdrama\b|\bseason\s*\d+\b",
            r"\brecipe\b|\bcook(ing)?\b|\bcalories\b|\bprotein\b|\bcarb(s)?\b",
            r"\btravel\b|\bhotel\b|\bflight\b|\bvisa\b|\bitinerary\b|\bairbnb\b",
            r"\bfootball\b|\bsoccer\b|\bnba\b|\bnfl\b|\bmlb\b|\bscore\b|\bstandings\b",
            r"\bdiagnos(is|e)\b|\bmedicine\b|\bskin\b|\bacne\b|\bdermatitis\b|\brash\b|\bsteroid\b",
        ],
    },
    "coverage": {
        "enabled": True,

        "compare_markers": [
            r"\bvs\.?\b",
            r"\bversus\b",
            r"\bcompare\b",
            r"\bcomparison\b",
            r"\bdifference\s+between\b",
            r"\bpros?\s+and\s+cons?\b",
            r"\btrade-?offs?\b",
        ],

        "generic_markers": [
            r"\bbest\s+practice(s)?\b",
            r"\boverall\b",
            r"\bhigh-?level\b",
            r"\bexplain\b",
            r"\bguide\b",
            r"\bhow\s+to\b",
            r"\bwhat\s+is\b",
        ],

        # Query-side entity detection
        "entity_aliases": {
            "mqtt": [
                r"\bmqtt\b", r"\bbroker\b", r"\bqos\b", r"\btopic\b",
                r"\bpublish\b", r"\bsubscribe\b", r"\bretain(ed)?\b",
                r"\blwt\b|\blast\s+will\b",
            ],
            "http": [
                r"\bhttp\b", r"\bhttps\b", r"\brest\b",
                r"\bheader(s)?\b", r"\bstatus\s*code(s)?\b",
                r"\brequest\b", r"\bresponse\b",
            ],
            "kafka": [
                r"\bkafka\b",
                r"\bproducer\b", r"\bconsumer\b",
                r"\bpartition(s)?\b", r"\boffset(s)?\b",
                r"\bconsumer\s*group(s)?\b",
            ],
            "aws_iot": [
                r"\baws\s*iot\b", r"\biot\s*core\b", r"\baws\s*iot\s*core\b",
            ],
            "aws_iot_jobs": [
                r"\baws\s*iot\s*jobs?\b|\biot\s*jobs?\b",
                r"\bjob\s*execution\b",
                r"\bjob\s*document\b",
                r"\bjob\s*status\b",
                r"\brollout\b",
                r"\btimeout\b",
            ],
            "firmware_update": [
                r"\bfirmware\b",
                r"\bota\b|\bover-?the-?air\b",
                r"\bbootloader\b",
                r"\bdfu\b",
                r"\bupdate\b|\bupgrade\b",
            ],
        },

        # DOC-side tagging patterns
        "entity_doc_aliases": {
            "mqtt": {
                "min_hits": 1,
                "patterns": [
                    r"\bmqtt\b",
                    r"(?=.*\bmqtt\b)(?=.*\b(qos|topic|publish|subscribe|broker)\b)",
                    r"\bqos\s*[012]\b",
                    r"\btopic\s*filter\b",
                    r"\bconnect\s+packet\b|\bconnack\b|\bpublish\s+packet\b|\bsubscribe\s+packet\b",
                ],
            },
            "kafka": {
                "min_hits": 2,
                "patterns": [
                    r"\bApache\s+Kafka\b",
                    r"\bconsumer\s+group\b",
                    r"\boffset\s+commit\b",
                    r"\bpartition\s+leader\b|\bleader\s+election\b",
                    r"\bKafka\s+broker(s)?\b",
                ],
            },
            "http": {
                "min_hits": 2,
                "patterns": [
                    r"\bHTTP/1\.1\b|\bHTTP/2\b|\bHTTP/3\b",
                    r"\bStatus\s*Code\b|\bstatus\s+code\s+[1-5]\d\d\b",
                    r"\bRequest\s+Header\b|\bResponse\s+Header\b",
                    r"\bContent-Type\b|\bAuthorization\b|\bUser-Agent\b",
                ],
            },
            "aws_iot": {
                "min_hits": 1,
                "patterns": [
                    r"\bAWS\s+IoT\s+Core\b",
                    r"\bThing\s+Shadow\b|\bDevice\s+Shadow\b",
                    r"\bAWS\s+IoT\s+Jobs\b",
                ],
            },
            "aws_iot_jobs": {
                "min_hits": 1,
                "patterns": [
                    r"\bAWS\s+IoT\s+Jobs?\b",
                    r"\bIoT\s+Jobs?\b",
                    r"\bJob\s+Execution\b",
                    r"\bJob\s+Document\b",
                    r"\bJob\s+Status\b",
                    r"\bRollout\b",
                    r"\bTimeout\b",
                    r"\bCreate\s+Job\b|\bUpdate\s+Job\b|\bDescribe\s+Job\b|\bCancel\s+Job\b",
                ],
            },
            "firmware_update": {
                "min_hits": 2,
                "patterns": [
                    r"\bOver-?the-?Air\b|\bOTA\b",
                    r"\bfirmware\s+update\b|\bfirmware\s+upgrade\b",
                    r"\bbootloader\b",
                    r"\bDFU\b|\bDevice\s+Firmware\s+Update\b",
                ],
            },
        },
    },
}

# Text splitting
TEXT_SPLITTING_CONFIG = {
    "chunk_size": 500,
    "chunk_overlap": 100
}

# Embeddings (local)
EMBEDDING_CONFIG = {
    "backend": "huggingface",
    "model_name": "sentence-transformers/all-MiniLM-L6-v2",
    "batch_size": 64,
    "show_progress": True
}

# LLM (local)
LLM_CONFIG = {
    "backend": "ollama",
    "model_name": "llama3",
    "temperature": 0
}

# Prompt context formatting
PROMPT_CONFIG = {
    "max_chars_per_chunk": 1800,
}

RANDOM_STATE = 42

# Debug configuration
DEBUG_CONFIG = {
    "rag": {
        "ingest": {
            "print_diagnostics": True,
            "print_metadata": True
        },
        "tiebreakers": {
            "print_score": True
        },
        "chain": {
            "print_refusal_reason": True
        },
        "gating": {
            "print_absolute": True,
            "print_gap": True,
            "print_density": True
        },
        "retriever": {
            "print_ambiguous_options": True
        },
        "ood": {
            "print_ood": True
        },
        "coverage": {
            "print_coverage": True
        }
    }
}
