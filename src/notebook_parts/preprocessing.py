# Extracted state and preprocessing helpers. Execute after setup.py and tools.py
# in the same notebook namespace. Routing nodes are in a later batch.
class UnifiedAgentState(TypedDict):
    """State object passed through the LangGraph pipeline."""
    raw_rows: list[dict]
    shipment_id: str
    consolidated_event: dict
    customer_profile: dict
    customer_profile_full: dict
    locker_availability: list[dict]
    playbook_context: list[dict]
    escalation_signals: dict
    noise_override: bool
    guardrail_triggered: bool
    resolution_output: dict
    critic_resolution_output: dict
    resolution_revision_count: int
    critic_feedback: str
    communication_output: dict
    critic_communication_output: dict
    next_agent: str
    max_loops: int
    escalated: bool
    escalation_reasons: list[str]
    tool_calls_log: list[str]
    trajectory_log: list[str]
    start_time: Optional[float]
    latency_sec: Optional[float]
    final_actions: list[dict]

INJECTION_KEYWORDS = ['ignore previous instructions', 'ignore all instructions', 'ignore above instructions', 'you are now a', 'act as a', 'system:', 'override your rules', 'override the rules', 'bypass your filters', 'bypass the safety', 'jailbreak', 'pretend you', 'roleplay as', 'dan mode', 'do not follow your rules', 'disregard your instructions']

def scan_for_injection(text: str) -> bool:
    """Returns True if prompt injection keywords are detected in the text."""
    if not text or not isinstance(text, str):
        return False
    text_lower = text.lower()
    return any(keyword in text_lower for keyword in INJECTION_KEYWORDS)

def deduplicate_rows(raw_rows: list[dict]) -> list[dict]:
    """Remove duplicate scan events."""
    return [r for r in raw_rows if r.get('is_duplicate_scan', 'False') != 'True']

def consolidate_event(unique_rows: list[dict], raw_rows: list[dict]) -> dict:
    """Consolidate shipment events using the highest attempt number."""
    if unique_rows:
        primary = max(unique_rows, key=lambda r: int(r.get('attempt_number', 0)))
    else:
        primary = raw_rows[0]
    prior_notes = [f"Attempt {r['attempt_number']}: {r['status_description']}" for r in unique_rows if r is not primary]
    return {
        'shipment_id': primary['shipment_id'],
        'timestamp': primary['timestamp'],
        'status_code': primary['status_code'],
        'status_description': primary['status_description'],
        'customer_id': primary['customer_id'],
        'delivery_address': primary['delivery_address'],
        'package_type': primary['package_type'],
        'package_size': primary['package_size'],
        'attempt_number': int(primary['attempt_number']),
        'prior_attempt_notes': prior_notes,
        'total_rows': len(raw_rows),
        'duplicates_removed': len(raw_rows) - len(unique_rows)
    }

def scan_inputs_for_injection(consolidated: dict, raw_rows: list[dict]) -> bool:
    """Scan delivery notes for known prompt-injection patterns."""
    texts = [consolidated['status_description']]
    texts.extend(row.get('status_description', '') for row in raw_rows)
    return any(scan_for_injection(text) for text in texts)

def scan_chunks_for_injection(playbook_context: list[dict]) -> bool:
    """Scan retrieved RAG chunks for known prompt-injection patterns."""
    return any(scan_for_injection(chunk.get('content', '')) for chunk in playbook_context)

STATUS_TO_PLAYBOOK_TOPIC = {
    'ATTEMPTED': 'failed delivery attempt redelivery attempt number',
    'ADDRESS_ISSUE': 'address mismatch building or street not found verify address',
    'DAMAGED': 'damaged package assessment minor moderate severe fragile',
    'REFUSED': "refused delivery return to sender didn't order it",
    'WEATHER_DELAY': 'weather delay perishable packages estimated delay 4 hours'
}

def retrieve_playbook_context(consolidated: dict, customer_tier: str, tool_log: list[str]) -> list[dict]:
    """Retrieve exception rules, escalation policy and applicable locker rules."""
    status = consolidated['status_code']
    queries = [
        f"{STATUS_TO_PLAYBOOK_TOPIC.get(status, status)} {consolidated['package_type']} {consolidated['status_description'][:100]}",
        f"escalation triggers {customer_tier} customer {status.replace('_', ' ').lower()}"
    ]
    if status == 'ATTEMPTED':
        queries.append('locker reroute eligibility capacity FULL LIMITED package size')
    seen, chunks = set(), []
    for q in queries:
        for c in search_playbook.invoke({'query': q}):
            if c['content'] not in seen:
                seen.add(c['content'])
                chunks.append(c)
        tool_log.append(f"TOOL: search_playbook('{q[:60]}')")
    return chunks

def fetch_context(consolidated: dict, tool_log: list[str]) -> dict:
    """Fetch customer profiles, lockers, playbook and escalation signals."""
    customer_id = consolidated['customer_id']
    customer_profile = lookup_customer_profile.invoke({'customer_id': customer_id, 'include_pii': False})
    tool_log.append(f'TOOL: lookup_customer_profile({customer_id}, pii=False)')
    customer_profile_full = lookup_customer_profile.invoke({'customer_id': customer_id, 'include_pii': True})
    tool_log.append(f'TOOL: lookup_customer_profile({customer_id}, pii=True)')
    address_parts = consolidated['delivery_address'].split(',')
    zip_code = address_parts[-1].strip() if address_parts else ''
    locker_availability = check_locker_availability.invoke({'zip_code': zip_code, 'package_size': consolidated['package_size']})
    tool_log.append(f"TOOL: check_locker_availability({zip_code}, {consolidated['package_size']})")
    playbook_context = retrieve_playbook_context(consolidated, customer_profile.get('tier', 'STANDARD'), tool_log)
    escalation_signals = check_escalation_rules.invoke({
        'customer_tier': customer_profile.get('tier', 'STANDARD'),
        'exceptions_last_90d': customer_profile.get('exceptions_last_90d', 0),
        'attempt_number': consolidated['attempt_number'],
        'package_type': consolidated['package_type'],
        'status_code': consolidated['status_code'],
        'status_description': consolidated['status_description']
    })
    tool_log.append(f"TOOL: check_escalation_rules(...) -> {escalation_signals['trigger_count']} trigger(s)")
    return {
        'customer_profile': customer_profile,
        'customer_profile_full': customer_profile_full,
        'locker_availability': locker_availability,
        'playbook_context': playbook_context,
        'escalation_signals': escalation_signals
    }

def check_noise_override(consolidated: dict) -> bool:
    """Flag routine status codes with no anomaly indicators."""
    routine_codes = {'DELIVERED', 'IN_TRANSIT', 'OUT_FOR_DELIVERY', 'SCANNED'}
    if consolidated['status_code'] not in routine_codes:
        return False
    anomaly_indicators = ['damage', 'wrong', 'suspicious', 'overdue', 'missing', 'unexpected', 'misroute', 'lost', 'stolen', 'abandoned', 'leak', 'crush', 'broke', 'delay', 'late', 'fraud']
    desc = consolidated['status_description'].lower()
    return not any(indicator in desc for indicator in anomaly_indicators)
