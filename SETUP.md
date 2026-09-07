# Phase 0 setup notes

## What's automated (already done on this branch)

- Project layout: `infra/`, `lambdas/{fetch,extract,embed,threshold,query_api,skill_gap,dlq_redrive}/`,
  `ml/`, `tests/`, `.github/workflows/`.
- Root Python 3.13 virtualenv at `.venv/`, with dependencies pinned via `pip-tools`:
  - `requirements.in` → `requirements.txt` (boto3, scikit-learn, xgboost, shap, numpy, pandas, rank-bm25)
  - `requirements-dev.in` → `requirements-dev.txt` (pytest, moto, ruff, pip-tools)
  - Recompile after editing the `.in` files: `pip-compile requirements.in -o requirements.txt`
- CDK app initialized in `infra/` (`cdk init app --language python`, run via `npx aws-cdk@2`
  since the CDK CLI isn't installed globally). It has its own venv at `infra/.venv/` for the
  CDK toolkit/library deps (`aws-cdk-lib`, `constructs`), pinned by the generated
  `infra/requirements.txt`. Stack renamed from the default `InfraStack` to `JobPulseStack`
  (`infra/jobpulse_infra/jobpulse_stack.py`), entry point `infra/app.py`.
  - Verified `cdk synth` succeeds (empty stack, no resources yet — that's expected until Phase 2+).
- `.gitignore` covering venvs, `cdk.out/`, `.env`, ML model artifacts.

## What you need to do manually (Phase 0, step 3)

This step requires your own AWS account and credentials, so it isn't something that can be
done on your behalf:

1. Install the AWS CLI v2 if you don't have it: https://docs.aws.amazon.com/cli/latest/userguide/getting-started-install.html
2. Run `aws configure` (or `aws configure sso` if your org uses SSO) and set up a named
   profile for this project's dev account, e.g. `jobpulse-dev`.
3. Pick a dev region (e.g. `us-east-1`) and confirm **Amazon Bedrock model access** is
   enabled for Claude and Titan Embeddings in that region/account (Bedrock model access is
   opt-in per account and must be requested in the console before Phase 3/4 will work).
   The extraction Lambda (Phase 3) defaults to `BEDROCK_MODEL_ID=anthropic.claude-3-5-sonnet-20241022-v2:0`,
   and the embedding Lambda (Phase 4) defaults to
   `BEDROCK_EMBEDDING_MODEL_ID=amazon.titan-embed-text-v2:0` (both in
   `infra/jobpulse_infra/jobpulse_stack.py`) — if either model isn't enabled/available in
   your account/region, request access to it or update the corresponding env var.
4. Export or reference the profile when running CDK, e.g.:
   ```
   export AWS_PROFILE=jobpulse-dev
   cd infra && source .venv/bin/activate
   cdk bootstrap   # one-time per account/region
   ```
5. Also install the CDK CLI globally if you'd rather not prefix every command with `npx`:
   `npm install -g aws-cdk`.

Once credentials are configured and the account is bootstrapped, Phase 1+ can start adding
real stacks/resources to `infra/jobpulse_infra/jobpulse_stack.py`.

## What you need to do manually (Phase 7 — alerting)

Amazon SES starts every account in a sandbox: it will only send email **to and from
verified identities**, and the threshold Lambda's re-ranker model must exist before you
can even `cdk synth`/`cdk deploy` this stack. Both are things only you can do:

1. Train the re-ranker so `ml/artifacts/reranker_inference.json` exists (the CDK stack
   raises a clear error naming this file if it's missing):
   ```
   python -m ml.train
   ```
2. In the SES console (same region as the rest of the stack), verify a sender email
   identity (the address `alert_email` sends *from*) and, while still in the SES
   sandbox, a recipient identity too (the address you want alerts delivered *to*) —
   sandbox accounts can't email unverified addresses at all.
3. Override the placeholder addresses at deploy time via CDK context, e.g.:
   ```
   cdk deploy -c senderEmail=you-verified-sender@example.com -c recipientEmail=you-verified-recipient@example.com
   ```
   Left unset, the stack falls back to `alerts@example.com`/`candidate@example.com` —
   placeholders that will never actually deliver anything.
4. The fit threshold (`FIT_THRESHOLD`, default `0.7`) that decides when an alert fires
   is also just a stack constant in `jobpulse_stack.py` — adjust it there if 0.7 is too
   strict/loose once you're watching real scores come through.
