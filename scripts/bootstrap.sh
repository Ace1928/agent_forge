#!/usr/bin/env bash
# Eidos E3 bootstrap: scaffold dirs, seed configs, prep venv. Idempotent.

set -euo pipefail

root_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

say() { printf "[bootstrap] %s\n" "$*"; }

need_cmd() { command -v "$1" >/dev/null 2>&1 || { echo "Missing: $1"; exit 1; }; }

main() {
  need_cmd bash
  need_cmd python3

  say "Creating base directories"
  mkdir -p "$root_dir"/{cfg,core,planners/templates,actuators,instruments,reflect,ui,psyche,memory,learners,meta,embodiment,protocols/keys,net,sensors/camera/calib_db,sensors/{imu,lidar,audio,power,env},actuators_hw/{base,manip,io},control,safety,teleop,sim,datasets/sessions,ops,curriculum/{syllabus,generators,grader,bank},arenas,oracles/{property_tests,contracts,fuzzers},curiosity,forge/{dsl,catalog},model_lab/{adapters,distillers,calibration},atlas/{maps,roadmaps,chronicle},introspection,charter/audits,social,economy,population/genealogy,backups/parity,creative/{stylebook,moodboard},telemetry/dashboards,provenance/repro,legal,rituals,grants/{proposals,audits},state/{events,vector_store,weights,adapters,snaps}}

  say "Seeding minimal config files (only if absent)"
  seed_cfg

  say "Making Python venv"
  if [ ! -d "$root_dir/.venv" ]; then
    python3 -m venv "$root_dir/.venv"
  fi
  # shellcheck disable=SC1091
  source "$root_dir/.venv/bin/activate"
  python -m pip install -U pip wheel >/dev/null
  python -m pip install --quiet pyyaml

  say "Touch state files"
  : >"$root_dir/state/events/.keep"
  : >"$root_dir/state/vector_store/.keep"
  : >"$root_dir/state/weights/.keep"
  : >"$root_dir/state/adapters/.keep"
  : >"$root_dir/state/snaps/.keep"

  say "Done. Next: add core/state.py migrations and core/config.py loader."
}

seed_cfg() {
  # cfg/self.yaml
  if [ ! -f "$root_dir/cfg/self.yaml" ]; then
cat >"$root_dir/cfg/self.yaml" <<'YAML'
name: Eidos
workspace: .
axioms: ["Verify with tests", "Keep diffs small"]
non_negotiables: ["workspace-only unless scope extends"]
temperament: { novelty_bias: 0.25, risk_aversion: 0.6, patience: 0.7 }
commit_style: "conventional"
YAML
  fi

  # cfg/drives.yaml
  if [ ! -f "$root_dir/cfg/drives.yaml" ]; then
cat >"$root_dir/cfg/drives.yaml" <<'YAML'
drives:
  progress:   { metric: git.prs_per_week,    target: 7,   weight: 0.35 }
  integrity:  { metric: tests.pass_rate,     target: 0.98, weight: 0.30 }
  efficiency: { metric: mean.mins_per_ticket,target: 45,  weight: 0.15, direction: decrease }
  ergonomics: { metric: lint.warnings,       target: 0,   weight: 0.10, direction: decrease }
  stability:  { metric: git.rollbacks_30d,   target: 0,   weight: 0.10, direction: decrease }
activation: { min_tension: 0.08, satiety_hysteresis: 0.03 }
YAML
  fi

  # cfg/budgets.yaml
  if [ ! -f "$root_dir/cfg/budgets.yaml" ]; then
cat >"$root_dir/cfg/budgets.yaml" <<'YAML'
global: { wall_clock_day_min: 60, cpu_time_day_sec: 18000, energy_kj_day: 1200, max_parallel_tasks: 2 }
scopes:
  hygiene:   { wall_clock: 10, cpu_time: 600,  retry_limit: 1 }
  refactor:  { wall_clock: 30, cpu_time: 2400, retry_limit: 3 }
  feature:   { wall_clock: 45, cpu_time: 3600, retry_limit: 3 }
  migration: { wall_clock: 90, cpu_time: 7200, retry_limit: 4 }
YAML
  fi

  # cfg/policies.yaml
  if [ ! -f "$root_dir/cfg/policies.yaml" ]; then
cat >"$root_dir/cfg/policies.yaml" <<'YAML'
autonomy_ladder:
  - { name: observe,     act: false, require_approvals: none,     min_competence: 0.0, max_risk: 1.0 }
  - { name: propose,     act: false, require_approvals: none,     min_competence: 0.3, max_risk: 0.8 }
  - { name: act_limited, act: true,  require_approvals: commands, min_competence: 0.5, max_risk: 0.6 }
  - { name: act_full,    act: true,  require_approvals: none,     min_competence: 0.7, max_risk: 0.4 }
risk_model:
  weights: { surface_area: 0.35, reversibility: 0.25, blast_radius: 0.25, novelty: 0.15 }
  cutoffs: { red: 0.7, amber: 0.4 }
confidence: { min_brier_improvement: 0.02, decay_per_week: 0.01 }
approvals:
  safe_commands: ["pnpm test -w", "pnpm lint --fix", "ruff --fix", "pytest -q", "go test ./..."]
  path_allowlist: ["./src", "./tests", "./docs", "./tools"]
  deny_patterns: ["|", ">", "<(", "curl", "nc", "scp", "ssh", "/etc", "/var", "~"]
YAML
  fi

  # cfg/skills.yaml
  if [ ! -f "$root_dir/cfg/skills.yaml" ]; then
cat >"$root_dir/cfg/skills.yaml" <<'YAML'
competencies:
  js_refactor:    { level: 0.50, brier: 0.25 }
  python_tooling: { level: 0.50, brier: 0.25 }
  e2e_playwright: { level: 0.40, brier: 0.30 }
  sql_migrations: { level: 0.45, brier: 0.28 }
YAML
  fi
}

main "$@"
