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

## Smoke-testing the query API (Phase 8)

After `cdk deploy`, the stack outputs (`cdk deploy` prints them, or `aws cloudformation
describe-stacks --stack-name JobPulseStack`) include the API Gateway's invoke URL and
the query API's API key ID. Retrieve the actual key value and call the API:

```
API_KEY=$(aws apigateway get-api-key --api-key <key-id-from-stack-output> --include-value --query value --output text)

curl -H "x-api-key: $API_KEY" "https://<api-id>.execute-api.<region>.amazonaws.com/prod/?min_score=0.7&remote_status=remote&limit=10"
```

Expect `{"postings": [...], "count": N}` on success, `{"error": "..."}` with a 400
status for a malformed query param (e.g. non-numeric `min_score`), and a 403 from API
Gateway itself (before the Lambda even runs) if `x-api-key` is missing or wrong.

## Viewing observability (Phase 9)

No manual setup needed — CloudWatch and X-Ray are enabled entirely by the CDK stack.
After `cdk deploy`, the dashboard is in the CloudWatch console under **Dashboards ->
JobPulse-Pipeline**; per-Lambda structured logs (with `posting_id` correlation) are in
each Lambda's own log group (**CloudWatch -> Log groups -> /aws/lambda/&lt;function
name&gt;**); traces are under **X-Ray -> Traces** (each Lambda invocation is its own
trace — X-Ray doesn't automatically link the separate async invocations one posting
passes through across fetch/extract/embed/threshold into a single end-to-end trace, so
use each Lambda's structured logs' `posting_id` field to correlate across stages
instead).

## What you need to do manually (Phase 10 — CI/CD)

`.github/workflows/ci-cd.yml`'s `test` job runs on every PR/push with no setup needed.
The two deploy jobs (`deploy-staging`, `deploy-production`) need real AWS accounts and
GitHub repo configuration only you can provide — a workflow file alone can't create
either:

1. **Per environment** (once for a staging AWS account, once for a separate production
   AWS account), set up OIDC so GitHub Actions can assume a role without long-lived
   access keys:
   - In each AWS account, create an IAM OIDC identity provider for
     `token.actions.githubusercontent.com` (skip if one already exists in that account).
   - Create an IAM role in each account trusting that provider, scoped to this repo
     (condition on `token.actions.githubusercontent.com:sub` matching
     `repo:<your-github-username>/serverless-job-match-pipeline:*`), with permissions
     for whatever this stack needs to create/update (Lambda, S3, DynamoDB, SNS, SES,
     API Gateway, IAM, CloudWatch, EventBridge, CloudFormation) — or attach
     `AdministratorAccess` for a personal dev/staging account if you'd rather not
     hand-scope a policy for this project.
   - `cdk bootstrap` each account/region once (see the Phase 0 section above), using
     credentials with sufficient privileges (this can be your own, one-time, not the
     CI role).
2. In this repo's **Settings -> Environments**, create two environments named exactly
   `staging` and `production`:
   - On each, add an environment secret `AWS_DEPLOY_ROLE_ARN` (that environment's IAM
     role ARN from step 1) and an environment variable `AWS_REGION`. Same secret/
     variable *names* in both environments, different *values* — the workflow
     references `secrets.AWS_DEPLOY_ROLE_ARN`/`vars.AWS_REGION` once, and GitHub
     resolves them from whichever environment the running job declared.
   - On **`production` only**, add a required-reviewers protection rule (yourself, or
     anyone else who should approve a prod deploy). This is the actual manual-approval
     gate PLAN.md Phase 10.3 asks for — the workflow's `environment: production` line
     only *requests* the gate; this repo setting is what *enforces* it. Without it,
     `deploy-production` runs automatically right after `deploy-staging` succeeds.
3. Push to `main` (or merge a PR) to trigger `deploy-staging` — `deploy-production`
   then waits for the approval configured in step 2 before running.
