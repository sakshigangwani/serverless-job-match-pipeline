# JobPulse — Implementation Plan

Source spec: `Docs/JobPulse_Project_Spec.pdf`

This plan sequences the build so every technology in the spec's tech stack and every
feature (core + all advanced extensions in section 5, not just a subset) gets implemented.
Phases are ordered so each one produces something runnable before the next begins.

Scope: Phases 0–11 (the full core pipeline) plus exactly three advanced features —
SQS Dead-Letter Queue + retry logic (§5.1), SHAP explainability (§5.2), and skill-gap
analysis (§5.2). All other extensions from spec section 5 are out of scope.

Tech checklist (everything below must appear in the final system):
AWS Lambda (Python 3.12) · EventBridge · S3 · DynamoDB · Bedrock (Claude + Titan Embeddings) ·
scikit-learn / XGBoost · SNS · SES · API Gateway · AWS CDK (Python) · GitHub Actions ·
CloudWatch (Logs/Metrics/Alarms) + X-Ray · SQS (DLQ) · SHAP.

---

## Phase 0 — Repo & environment setup

1. Set up project layout:
   ```
   /infra          → AWS CDK app (Python)
   /lambdas        → one folder per Lambda (fetch, extract, embed, threshold, query_api, skill_gap, dlq_redrive)
   /ml             → offline training scripts, notebooks, saved model artifacts
   /tests          → unit + integration tests
   /.github/workflows
   ```
2. Create a Python virtualenv (3.12) with `pip-tools` or `poetry`; pin dependencies.
3. Configure AWS CLI profile / credentials (or GCP if you choose the GCP path in section 2's
   "GCP equivalent") and an AWS account/region for dev.
4. Initialize `AWS CDK` app (`cdk init app --language python`) in `/infra`. This is the IaC
   backbone every later phase adds stacks/constructs to.
5. Add `.gitignore` for `cdk.out/`, `.env`, model artifacts, `node_modules`.

## Phase 1 — Candidate profile & data contracts

1. Define the candidate profile schema (resume text, parsed sections: skills, experience,
   projects) — this is the input the whole pipeline scores against, and must be pluggable
   (not hardcoded to one person) per the spec's "general-purpose tool" framing.
2. Define the structured posting schema that extraction must produce: `title, company,
   comp_min, comp_max, seniority, required_skills[], remote_status, visa_status,
   source_url, posting_hash, ingested_at`.
3. Define the DynamoDB item shape (partition key `posting_id`, sort/GSI for score-ranked
   queries) and the S3 raw-posting key convention (`raw/{source}/{date}/{posting_hash}.json`).

## Phase 2 — Ingestion Lambda + EventBridge + S3 (core pipeline, stage 1)

1. Write `lambdas/fetch/`: pulls new postings from a job board API or RSS feed (pick one
   real source, e.g. RemoteOK, Adzuna, or an RSS feed) behind a small adapter interface
   (a `JobSource` base class) so more sources could be added later without changing the
   Lambda handler.
2. Lambda writes raw JSON responses to **S3** (`raw postings` bucket) untouched — this is
   the audit trail / immutable landing zone.
3. Create an **EventBridge** cron rule (e.g. `rate(6 hours)`) in CDK that triggers the fetch
   Lambda.
4. Deploy via `cdk deploy` and confirm a scheduled run lands raw JSON in S3.

## Phase 3 — Extraction Lambda (Bedrock LLM)

1. Write `lambdas/extract/`: triggered by an S3 `ObjectCreated` event under `raw/`, calls
   **Amazon Bedrock** (Claude) with a structured extraction prompt (JSON schema for
   title/company/comp/seniority/skills/remote/visa).
2. Validate/repair LLM JSON output (retry once on malformed JSON or a schema violation).
3. Reuse the `posting_hash` already computed by the fetch Lambda (Phase 2) as the
   `posting_id`, so dedup (Phase 12) works off one consistent identity.

## Phase 4 — Embeddings Lambda (Bedrock Titan Embeddings)

1. Extend (or add a second) Lambda that embeds:
   - The posting's extracted requirements/description text.
   - The candidate profile (whole-resume embedding, plus per-section embeddings for
     skills/experience/projects) — computed once and cached, not per posting.
2. Use **Bedrock Titan Embeddings** as the embedding model.
3. Store posting embeddings alongside the structured record.

## Phase 5 — Structured store (DynamoDB)

1. Create the **DynamoDB** table in CDK with the schema from Phase 1, plus a GSI for
   querying top-scored postings.
2. Extraction writes the structured record (DynamoDB `PutItem`, replacing Phase 3's
   interim S3 write); embedding merges in the packed embedding vector (compact
   float32/Binary, not base64 text) and the embedding-similarity baseline (DynamoDB
   `UpdateItem`, replacing Phase 4's interim S3 write). `score` is left unset until
   Phase 7 — DynamoDB simply excludes an item from the `ScoreIndex` GSI until it has one,
   so unscored postings don't show up in ranked queries yet, which is the correct
   "not ready to rank" behavior rather than a placeholder value.

## Phase 6 — Match-scoring model (offline ML, the rigor component)

1. **Baseline**: implement keyword/BM25 matching as the naive baseline (`ml/baselines.py`,
   using `rank_bm25`).
2. **Embedding similarity**: cosine similarity between candidate and posting embeddings
   (already computed in Phase 4/5) as the second baseline (`ml/baselines.py`).
3. **Labeling**: manually label 150–300 postings as good-fit / not-a-fit for a candidate
   profile → held-out test set (`ml/data/labels.csv`). Real manual labeling requires a
   human's real resume against real scraped postings, which can't be fabricated — a
   synthetic-but-structurally-realistic dataset (`ml/synthetic_data.py`) stands in so the
   rest of the pipeline is built and proven correct end-to-end; `ml/README.md` documents
   the real-data workflow this gets swapped for once postings have accumulated.
4. **Re-ranker**: engineer features (skill overlap count, seniority match, comp-range fit,
   remote/visa match, embedding similarity — `ml/features.py`, shared with Phase 7's
   inference path to avoid training/serving skew) and train a classifier with
   **scikit-learn** / **XGBoost** (logistic regression + gradient-boosted trees, compare
   both — `ml/train.py`, keeping whichever wins on held-out ROC-AUC).
5. The trained model is saved as a self-describing joblib artifact
   (`ml/artifacts/reranker.joblib`, gitignored/regenerated, not committed). Packaging it
   as an actual **Lambda layer** is Phase 7's job, since that's the CDK construct that
   naturally belongs with the Lambda that loads it.
6. **Evaluation**: compute Precision@K, Recall@K, ROC-AUC for each of the three approaches
   (keyword-only, embedding-only, embedding+re-ranker) and record the improvement delta in
   `ml/evaluation_report.md` (`ml/evaluate.py`) — using the synthetic dataset until real
   labels replace it; the report explicitly flags itself as not resume-bullet-ready yet.

## Phase 7 — Scoring + threshold + alerting Lambda (core pipeline, stage 2)

1. Write `lambdas/threshold/`: triggered by a DynamoDB Streams MODIFY event once embed
   has set a posting's embedding (a finer-grained filter than Phase 5's INSERT-only one,
   since both embed's and threshold's own writes are MODIFYs — the filter keys on
   `embedding` existing and `score` still being NULL). Scores each posting with the
   re-ranker and writes the final score + `alert_sent` flag back to DynamoDB. The
   re-ranker itself is deployed as a pure-Python inference module
   (`lambdas/threshold/reranker.py` + a small exported weights/intercept JSON), not a
   scikit-learn/xgboost Lambda layer — bundling those plus numpy/scipy measured at
   ~345MB unzipped, over Lambda's 250MB function+layers limit (see `ml/train.py`).
2. If score clears the fit threshold, publish to **SNS** (`AlertsTopic`), which fans out
   to a Lambda subscriber (`lambdas/alert_email/`) that calls **SES** to send the email —
   SNS has no native SES subscription protocol, so a small subscriber Lambda is what
   realizes "SNS fans out to SES." This also sets up "add another subscriber later"
   (e.g. a future Slack/Discord webhook, currently out of scope) as a free extension
   point: it would attach to the same topic without touching either existing Lambda.
3. Wire SNS → Lambda → SES in CDK; verify a sender identity (and, in SES sandbox, a
   recipient identity too) in the SES console for dev testing (see `SETUP.md`).

## Phase 8 — Query API (on-demand path)

1. Write `lambdas/query_api/`: reads the ScoreIndex GSI (score descending, so only
   already-scored postings ever appear), supports filtering (`min_score`,
   `remote_status`, `seniority`) and pagination-aware limiting (DynamoDB's
   `FilterExpression` applies after `Limit`, so filtered results page with
   `LastEvaluatedKey` until the requested count is reached), and returns ranked
   postings as JSON.
2. Expose it through **API Gateway** — a REST API specifically (not the newer HTTP
   API), since REST API has native API key + usage plan support for simple access
   gating; a Cognito authorizer was the spec's other option but is unnecessary
   overhead for a single-candidate demo API.
3. Smoke-test with `curl`/Postman against the deployed endpoint (see `SETUP.md`).

## Phase 9 — Observability

1. Add **CloudWatch** log groups (automatic per Lambda), structured JSON logging in every
   Lambda (`common/logging_utils.py`; `posting_id` correlation id on every log line that
   has one — fetch logs per-item as it writes, since one invocation handles a batch).
2. Add **CloudWatch Metrics** (`common/metrics.py`, hand-rolled CloudWatch Embedded
   Metric Format rather than the `aws-embedded-metrics` package or `PutMetricData` calls
   — printing the right JSON shape to stdout is enough, no extra dependency or IAM
   permission needed): postings ingested, extraction failures, average score (CloudWatch's
   own Average statistic over many emitted per-posting values, not computed here), alerts
   sent. **CloudWatch Alarms**: error rate (as a percentage of invocations, not a raw
   count) and p99 duration (vs. 80% of each Lambda's own timeout) per core-pipeline
   Lambda. DLQ depth is intentionally not alarmed on yet — added in Phase 12 once the
   queue exists.
3. Enable **X-Ray** tracing on all Lambdas and the API Gateway stage. This gives each
   Lambda invocation and the AWS SDK calls it makes their own trace segments — it does
   **not** automatically link separate async invocations (an S3 event, a DynamoDB Stream
   event) into one true end-to-end trace across fetch → extract → embed → score → alert;
   that would require manually propagating a trace/correlation id through each event
   payload, which is what `posting_id` in the structured logs (point 1) does instead for
   cross-stage correlation.
4. Build a CloudWatch dashboard (via CDK) summarizing pipeline health: per-Lambda
   invocations/errors/p99 duration, DynamoDB capacity + throttles, and the custom metrics
   from point 2.

## Phase 10 — CI/CD

1. **GitHub Actions** workflow: on PR — lint (`ruff`/`flake8`), type-check, run unit tests
   for each Lambda, run the ML evaluation script against a fixed sample and assert
   metrics don't regress below a floor.
2. On merge to `main` — `cdk diff` then `cdk deploy` to a dev/staging AWS account.
3. Add a manual-approval gate (GitHub Environments) before deploying to prod.

## Phase 11 — Core pipeline validation

1. Run the full pipeline end-to-end on real data for several scheduled cycles.
2. Confirm S3 audit trail, DynamoDB records, alert emails, and query API all work together.
3. This closes out the "core capabilities" list from spec section 1. The three phases below
   are the only additive depth in scope.

---

## Phase 12 — SQS Dead-Letter Queue + retry logic (spec §5.1)

1. Add an **SQS** DLQ to the extraction and embedding Lambdas (via Lambda destinations, or
   event-source-mapping/on-failure DLQ config in CDK).
2. Configure exponential backoff retry (Lambda's built-in async retry, or a Step
   Functions/SQS visibility-timeout-based backoff) before a failed message lands in the DLQ.
3. Add a CloudWatch alarm on DLQ depth so failures are visible (ties back into Phase 9).
4. Write a small redrive/reprocessing script or Lambda to replay DLQ messages after a fix.

## Phase 13 — Explainability (SHAP) (spec §5.2)

1. Add `shap` to the re-ranker's dependencies (Phase 6's model layer).
2. At inference time in the threshold/scoring Lambda, compute SHAP values for the
   re-ranker's engineered features (skill overlap, seniority match, comp-range fit,
   remote/visa match, embedding similarity) for each scored posting.
3. Store the top contributing features (name + SHAP value) alongside the score in
   DynamoDB, and expose them through the query API (Phase 8) so each ranked posting shows
   *why* it scored high instead of a black-box number.

## Phase 14 — Skill-gap analysis (spec §5.2)

1. Write a Lambda or scheduled script that scans DynamoDB for postings where the candidate
   scored below the fit threshold.
2. Aggregate `required_skills` across those low-scoring postings and rank the most
   frequently missing skills relative to the candidate profile.
3. Surface the ranked skill-gap list through a new query API endpoint (or a scheduled
   report written to S3/email) as a learning recommendation.

---

## Phase 15 — Final validation & write-up

1. Run the full system (core pipeline + DLQ/retry + SHAP explainability + skill-gap
   analysis) end-to-end for several scheduled cycles; confirm DLQ redrive and the two ML
   additions work against real data.
2. Finalize `ml/evaluation_report.md` with real measured Precision@K/Recall@K/ROC-AUC
   numbers and the baseline comparison table.
3. Update `README.md` with architecture diagram, setup/deploy instructions, and the
   resume-bullet framing from spec section 6 (general-purpose tool, pluggable candidate
   profile) using your actual measured numbers.

---

## Coverage map (spec → phase)

| Spec item | Phase |
|---|---|
| EventBridge, S3, Lambda fetch | 2 |
| Bedrock LLM extraction | 3 |
| Bedrock Titan Embeddings | 4 |
| DynamoDB | 5 |
| Match-scoring model, labeling, metrics | 6 |
| SNS / SES alerting | 7 |
| API Gateway query path | 8 |
| CloudWatch + X-Ray | 9 |
| CDK IaC | 0 (scaffold) + all phases |
| GitHub Actions CI/CD | 10 |
| SQS DLQ & retries (§5.1) | 12 |
| Explainability / SHAP (§5.2) | 13 |
| Skill-gap analysis (§5.2) | 14 |
