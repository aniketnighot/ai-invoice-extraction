# Simple AI Server Task

## Overview

Implement an AI server that provides chat completion capabilities and email data extraction functionality.
Your solution should be fully tested.

## Requirements

### Server Setup

- Server should listen on `localhost:8090`
- Implement two main endpoints as described below
- Use FastAPI

### 1. Chat Completions Endpoint

**Path**: `/v1/chat/completions`

- Must be compatible with OpenAI's ChatCompletion API
- Should serve at least:
  - `gpt-5-mini` from OpenAI
  - One free model from OpenRouter (examples: `deepseek/deepseek-r1-0528:free`, `moonshotai/kimi-k2:free`, `qwen/qwen3-235b-a22b:free`)

### 2. Prefill Endpoint

**Path**: `/v1/prefill`

Extract payment information from email text and save to `data.csv`.

#### Request Format

```http
POST /v1/prefill
Body: {
  "email_text": "...",
  "model": "..."
}
```

#### Response Format

```json
Success: {"success": true, "message": "data extracted and written"}
Error: {"success": false, "message": "something went wrong"}
```

#### Data Fields to Extract

From the email text, extract and save these fields:

- amount
- currency
- due_date
- description
- company
- contact

## API Keys & Limits

You'll be provided with:

- **OpenAI API Key**: 1 million token budget for large models (e.g., `gpt-5`), 10 million tokens for small models (e.g., `gpt-5-mini`)
- **OpenRouter API Key**: For accessing free tier models

We recommend using the free open router models for heavy testing as the OpenAI key is budgeted!

## Time Expectation

This task is designed to take less than 3 hours to implement.

## Implementation Freedom

You have complete freedom in your technical choices for this challenge - pick your favorite framework and architecture patterns. We're interested in seeing your problem-solving approach and coding style, not checking boxes on specific technologies. Feel free to add creative touches or bonus features if you're having fun with it and time permits. The core requirements are intentionally straightforward, leaving room for you to showcase what makes your code unique. Whether you prefer a minimalist elegant solution or want to add some interesting extras, we're excited to see your take on this challenge!

## Submission

Once complete, either:

- Upload to GitHub and share the repository link, **OR**
- Zip the solution folder as if you would publish it to git and send it back
