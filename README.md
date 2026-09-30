# Lumr Feedback Worker

A Cloudflare Worker that stores shape records and feedback from the Lumr Studio plugin.

## What it stores

- Shape records: timing, editing decisions, and context for audio edits.
- Feedback: free-text comments, plus an optional name and email if the sender types them.

Nothing else is stored. No IP addresses, user agents, request headers, or device info.

## Deploy

1. Install Node: `brew install node`
2. Log in to Cloudflare: `npx wrangler login`
3. Create the R2 bucket: `npx wrangler r2 bucket create lumr-feedback-sends`
4. Deploy: `npx wrangler deploy`. It needs wrangler 4.36.0 or newer, which `npx` fetches.
5. Take note of your deployed URL (e.g., `https://lumr-feedback.yourname.workers.dev`).

## Set the admin token (needed to delete anything)

Deleting a send or a feedback message needs an admin token. Until you set one, every delete is refused.

1. Make a token: `openssl rand -hex 32`
2. Set it: `npx wrangler secret put ADMIN_TOKEN`, then paste the token.
3. Keep it somewhere safe. Delete requests send it as `Authorization: Bearer <ADMIN_TOKEN>`.

## Read sends and feedback

List shape records:
```bash
npx wrangler r2 object list lumr-feedback-sends --prefix sends/
```

List feedback:
```bash
npx wrangler r2 object list lumr-feedback-sends --prefix feedback/
```

Or use the Cloudflare dashboard: R2 bucket `lumr-feedback-sends`.

## Delete a send

```bash
curl -X DELETE https://<worker-url>/v1/sends/<send_id> \
  -H "Authorization: Bearer <ADMIN_TOKEN>"
```

## Delete feedback

```bash
curl -X DELETE https://<worker-url>/v1/feedback/<feedback_id> \
  -H "Authorization: Bearer <ADMIN_TOKEN>"
```

## API endpoints

- `POST /v1/sends`: Store shape records (returns 201 with `send_id` and record count).
- `DELETE /v1/sends/<send_id>`: Delete a send (requires admin token).
- `POST /v1/feedback`: Store feedback (returns 201 with `feedback_id`).
- `DELETE /v1/feedback/<feedback_id>`: Delete feedback (requires admin token).

Rate limit: 20 requests per 60 seconds per IP address.

## Run the tests

```sh
node --test test/validate.test.mjs
```
