"""Generate a Rust fixture pinning burn-deltanet's delta rule to the
flash-linear-attention reference implementation."""
import os, sys, torch

OUT = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
    "src/delta/tests/reference.rs",
)
sys.path.insert(0, '/shared/claude/flash-linear-attention')
from fla.ops.delta_rule.naive import delta_rule_recurrence
from fla.ops.gated_delta_rule.naive import naive_recurrent_gated_delta_rule
from fla.ops.gdn2.naive import naive_recurrent_gdn2

torch.manual_seed(20260826)
B, T, H, K, V = 2, 9, 2, 4, 6

def l2(x, eps=1e-6):
    return x / torch.sqrt((x * x).sum(-1, keepdim=True) + eps)

def flat(t):
    return ", ".join(f"{v:.10e}" for v in t.double().reshape(-1).tolist())

def emit(name, tensors):
    out = []
    for k, t in tensors.items():
        out.append(f"/// `{list(t.shape)}`\npub const {name}_{k.upper()}: &[f64] = &[{flat(t)}];\n")
    return "\n".join(out)

# ── inputs, in [B, T, H, ·] layout (what burn-deltanet's DeltaInput takes) ──
q = l2(torch.randn(B, T, H, K, dtype=torch.float64))
k = l2(torch.randn(B, T, H, K, dtype=torch.float64))
v = torch.randn(B, T, H, V, dtype=torch.float64)
beta = torch.rand(B, T, H, dtype=torch.float64) * 0.9 + 0.05
g = -(torch.rand(B, T, H, dtype=torch.float64) * 0.5 + 0.01)
s0 = torch.randn(B, H, K, V, dtype=torch.float64) * 0.2
# GDN-2's channel-wise gates: erase on the key axis, write on the value axis,
# and a per-key-channel log decay.
b = torch.rand(B, T, H, K, dtype=torch.float64) * 0.9 + 0.05
w = torch.rand(B, T, H, V, dtype=torch.float64) * 0.9 + 0.05
gk = -(torch.rand(B, T, H, K, dtype=torch.float64) * 0.5 + 0.01)

# ── ungated: delta_rule_recurrence wants [B, H, T, ·] ──
tr = lambda x: x.transpose(1, 2).contiguous()
y_u, s_u = delta_rule_recurrence(tr(q), tr(k), tr(v), tr(beta), initial_state=s0.clone())
y_u = y_u.transpose(1, 2).contiguous()

# ── gated: naive_recurrent_gated_delta_rule wants [B, T, H, ·] ──
y_g, s_g = naive_recurrent_gated_delta_rule(
    q.clone(), k.clone(), v.clone(), beta.clone(), g.clone(),
    initial_state=s0.clone(), output_final_state=True,
)

# ── GDN-2: naive_recurrent_gdn2 wants [B, T, H, ·] ──
y_2, s_2 = naive_recurrent_gdn2(
    q.clone(), k.clone(), v.clone(), gk.clone(), b.clone(), w.clone(),
    initial_state=s0.clone(), output_final_state=True,
)

header = f"""//! Reference values captured from `flash-linear-attention`'s naive delta-rule
//! implementations (`fla/ops/delta_rule/naive.py::delta_rule_recurrence`,
//! `fla/ops/gated_delta_rule/naive.py::naive_recurrent_gated_delta_rule` and
//! `fla/ops/gdn2/naive.py::naive_recurrent_gdn2`) at float64, for a fixed
//! pseudo-random input.
//!
//! This is what makes the test suite a *port* check rather than only an
//! internal-consistency check: everything else in the crate proves the chunked
//! path equals the recurrent one, and this proves the recurrent one is the
//! published recurrence.
//!
//! Regenerate with `python3 scripts/gen_fixture.py`. `q`/`k` are already L2-normalised
//! (`x/sqrt(sum(x²)+1e-6)`, the reference's own epsilon), since the reference
//! kernels take them that way.

/// `[batch, sequence, nheads, head_k_dim]` = {[B, T, H, K]}
pub const DIMS: [usize; 5] = [{B}, {T}, {H}, {K}, {V}];
"""

body = emit("IN", dict(q=q, k=k, v=v, beta=beta, g=g, state=s0, b=b, w=w, gk=gk))
body += emit("UNGATED", dict(y=y_u, state=s_u))
body += emit("GATED", dict(y=y_g, state=s_g))
body += emit("GDN2", dict(y=y_2, state=s_2))

open(OUT, 'w').write(header + "\n" + body)
print("wrote fixture:", y_u.shape, s_u.shape, y_g.shape, s_g.shape, y_2.shape, s_2.shape)
