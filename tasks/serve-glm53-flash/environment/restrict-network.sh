#!/bin/bash
# Outbound firewall (anti-cheat): allow ONLY DNS, loopback, docker-internal networks, and the LLM API
# endpoints the agent needs. Everything else is dropped — in particular pypi, github, and huggingface,
# so the agent cannot fetch a serving engine's implementation of this model (`pip install sglang`,
# `git clone sgl-project/sglang`, HF) and copy it. The model is bind-mounted read-only and agent CLIs are
# pre-installed, so no package index is needed at run time.
#
# Run it as the container ENTRYPOINT, before the CMD. iptables rules are network-namespace-wide, so they
# also apply to the agent's `docker exec` commands. Requires cap_add: NET_ADMIN.
set -uo pipefail

# FAIL CLOSED. These tasks are scored on the assumption that the agent cannot reach pypi/github/HF --
# offline is what stops it fetching the very thing it is asked to build. A warning protects nothing, so
# refuse to start rather than produce a score that looks like a pass and is not comparable to one.
# ALLOW_UNFIREWALLED=1 overrides deliberately (local debugging); it is logged loudly so a run that used
# it is identifiable afterwards from the container output alone.
fw_fail() {
    echo "restrict-network: FATAL -- $1" >&2
    if [ "${ALLOW_UNFIREWALLED:-}" = "1" ]; then
        echo "restrict-network: ALLOW_UNFIREWALLED=1 set -- CONTINUING WITHOUT A FIREWALL." >&2
        echo "restrict-network: any score from this run is NOT comparable to a firewalled run." >&2
        return 0
    fi
    echo "restrict-network: refusing to start unfirewalled. Grant the container cap_add: NET_ADMIN," >&2
    echo "restrict-network: or set ALLOW_UNFIREWALLED=1 to override deliberately." >&2
    exit 1
}

# LLM API endpoints the agent may reach. Extend at launch via ALLOWED_HOSTS=host1,host2 (no scheme).
API_HOSTS=(
    openrouter.ai openrouter.io
    api.openai.com api.anthropic.com generativelanguage.googleapis.com
    api.together.xyz integrate.api.nvidia.com api.moonshot.cn api.kimi.com
)
# Auto-allow hosts from any *_BASE_URL / *_ENDPOINT env var (ANTHROPIC_BASE_URL, OPENAI_BASE_URL, …).
CUSTOM=()
while IFS='=' read -r n v; do
    case "$n" in
        *_BASE_URL|*_URL|*_ENDPOINT)
            [ -n "$v" ] && CUSTOM+=("$(echo "$v" | sed 's|.*://||; s|[:/].*||')") ;;
    esac
done < <(env)
if [ -n "${ALLOWED_HOSTS:-}" ]; then IFS=',' read -ra E <<< "$ALLOWED_HOSTS"; CUSTOM+=("${E[@]}"); fi
ALL_HOSTS=("${API_HOSTS[@]}" ${CUSTOM[@]+"${CUSTOM[@]}"})

# BEDROCK_MODE=1 => the agent's LLM is Amazon Bedrock via the EC2 instance role, reached
# through IMDS (169.254.169.254) for credentials and the bedrock-runtime/sts endpoints for
# inference. Bedrock's endpoint DNS rotates ~8 IPs/resolve, so a startup-only allowlist would
# eventually block a mid-run request; a background re-resolver re-adds new IPs every 30s.
# The default path (BEDROCK_MODE unset) is unchanged.
BEDROCK_MODE="${BEDROCK_MODE:-}"
AWS_REGION="${AWS_REGION:-us-east-1}"
BEDROCK_HOSTS=(
    "bedrock-runtime.${AWS_REGION}.amazonaws.com"
    "bedrock.${AWS_REGION}.amazonaws.com"
    "sts.${AWS_REGION}.amazonaws.com"
)

allow_host_ips() {  # resolve $1, ACCEPT any not-yet-allowed IP; echo count added
    local h="$1" added=0 ip
    for ip in $(getent ahosts "$h" 2>/dev/null | awk '{print $1}' | sort -u); do
        if ! iptables -C OUTPUT -d "$ip" -j ACCEPT 2>/dev/null; then
            iptables -I OUTPUT -d "$ip" -j ACCEPT && added=$((added+1))
        fi
    done
    echo "$added"
}

if iptables -L OUTPUT -n >/dev/null 2>&1; then
    echo "restrict-network: applying outbound allowlist firewall..."
    iptables -F OUTPUT 2>/dev/null || true
    iptables -A OUTPUT -o lo -j ACCEPT
    iptables -A OUTPUT -m state --state ESTABLISHED,RELATED -j ACCEPT
    iptables -A OUTPUT -p udp --dport 53 -j ACCEPT
    iptables -A OUTPUT -p tcp --dport 53 -j ACCEPT
    iptables -A OUTPUT -d 10.0.0.0/8 -j ACCEPT
    iptables -A OUTPUT -d 172.16.0.0/12 -j ACCEPT
    iptables -A OUTPUT -d 192.168.0.0/16 -j ACCEPT
    iptables -A OUTPUT -d 127.0.0.11 -j ACCEPT
    if [ -n "$BEDROCK_MODE" ]; then
        iptables -A OUTPUT -d 169.254.169.254 -j ACCEPT   # IMDSv2 instance-role creds
        for h in "${BEDROCK_HOSTS[@]}"; do
            for ip in $(getent ahosts "$h" 2>/dev/null | awk '{print $1}' | sort -u); do
                iptables -A OUTPUT -d "$ip" -j ACCEPT
            done
        done
    fi
    for h in "${ALL_HOSTS[@]}"; do
        for ip in $(getent ahosts "$h" 2>/dev/null | awk '{print $1}' | sort -u); do
            iptables -A OUTPUT -d "$ip" -j ACCEPT
        done
    done
    iptables -A OUTPUT -j REJECT --reject-with icmp-admin-prohibited \
        || fw_fail "could not install the final REJECT rule"
    # Everything above only carves out exceptions; this REJECT is the rule that actually blocks
    # anything. Confirm it is in the table rather than trusting an exit status, so the success line
    # below cannot announce a firewall that was never installed.
    iptables -C OUTPUT -j REJECT --reject-with icmp-admin-prohibited 2>/dev/null \
        || fw_fail "the final REJECT rule is not present after applying the ruleset"
    echo "restrict-network: outbound locked to DNS + LLM endpoints; pypi/github/HF blocked."
    if [ -n "$BEDROCK_MODE" ]; then
        ( while true; do
              sleep 30
              tot=0; for h in "${BEDROCK_HOSTS[@]}"; do n=$(allow_host_ips "$h"); tot=$((tot+n)); done
              [ "$tot" -gt 0 ] && echo "[reresolver] +$tot new Bedrock IP(s) at $(date +%H:%M:%S)"
          done ) &
        echo "restrict-network: BEDROCK_MODE — IMDS+bedrock allowed; re-resolver PID $!"
    fi
else
    fw_fail "iptables is unavailable (the container needs cap_add: NET_ADMIN)"
fi

exec "$@"
