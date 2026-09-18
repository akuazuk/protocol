#!/usr/bin/env bash
# Budget alerts for the billed Gemini Developer API project.
# Gemini spend lives on gen-lang-client-0274478609, not on protocol-home-e1 GCE.
#
# Usage:
#   bash deploy/gcp-llm/ensure_gemini_spend_budget.sh
set -euo pipefail

GEMINI_PROJECT="${GEMINI_BILLING_PROJECT:-gen-lang-client-0274478609}"
BILLING_ACCOUNT="${GCP_BILLING_ACCOUNT:-01D5C2-ECFF77-88FFEC}"
SOFT_USD="${GEMINI_BUDGET_SOFT_USD:-25}"
HARD_USD="${GEMINI_BUDGET_HARD_USD:-40}"

echo "project=${GEMINI_PROJECT} billing=${BILLING_ACCOUNT} soft=${SOFT_USD} hard=${HARD_USD}"

gcloud services enable billingbudgets.googleapis.com --project="${GEMINI_PROJECT}" --quiet || true

create_budget() {
  local name="$1"
  local amount="$2"
  local percent="$3"
  gcloud billing budgets create \
    --billing-account="${BILLING_ACCOUNT}" \
    --display-name="${name}" \
    --budget-amount="${amount}USD" \
    --filter-projects="projects/${GEMINI_PROJECT}" \
    --threshold-rule="percent=${percent}" \
    --quiet
}

if ! create_budget "gemini-mo-soft-${SOFT_USD}" "${SOFT_USD}" 1.0; then
  echo "WARN: soft budget not created (already exists or no billing.budgets permission)" >&2
fi
if ! create_budget "gemini-mo-hard-${HARD_USD}" "${HARD_USD}" 1.0; then
  echo "WARN: hard budget not created (already exists or no billing.budgets permission)" >&2
fi

echo "OK: check https://console.cloud.google.com/billing/${BILLING_ACCOUNT}/budgets"
echo "Gemini usage: https://aistudio.google.com/usage  project ${GEMINI_PROJECT}"
