# API

The API client is a fairly standard Redux Toolkit Query (RTK-Query) setup.

It defines a simple base query with special handling for OpenAPI schema queries and endpoints: invokeai/frontend/web/src/services/api/index.ts

## Types

The API provides an OpenAPI schema and we generate TS types from it. They are stored in: invokeai/frontend/web/src/services/api/schema.ts

We use https://github.com/openapi-ts/openapi-typescript/ to generate the types.

- Python script to outut the OpenAPI schema: scripts/generate_openapi_schema.py
- Node script to call openapi-typescript and generate the TS types: invokeai/frontend/web/scripts/typegen.js

Pipe the output of the python script to the node script to update the types. There is a `make` target that does this in one fell swoop (after activating venv): `make frontend-typegen`

Alternatively, start the ptyhon server and run `pnpm typegen`.

The schema.ts file is pushed to the repo, and a CI check ensures it is up to date.

### ⚠️ Important: DO NOT Manually Edit schema.ts

**The `schema.ts` file is auto-generated and should NEVER be manually edited.**

Manual modifications to this file:
- Break the single source of truth (the Python backend's OpenAPI schema)
- Will be silently overwritten the next time the generation script runs
- Can cause the CI checks to fail
- May introduce inconsistencies between frontend and backend

If you need to add or modify types in the API schema:
1. Make the changes in the Python backend code (pydantic models, invocations, etc.)
2. Run `make frontend-typegen` to regenerate the schema.ts file
3. Commit both the backend changes AND the regenerated schema.ts file together
