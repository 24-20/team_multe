"""
Tripletex AI Agent Server — NM i AI 2026
POST /solve: receives an accounting task prompt + Tripletex credentials,
             uses Claude to execute the task via the Tripletex v2 API proxy,
             returns {"status": "completed"}.
"""

import json
import logging
import os

import anthropic
import httpx
from dotenv import load_dotenv
from fastapi import FastAPI
from pydantic import BaseModel

load_dotenv()

logger = logging.getLogger(__name__)
app = FastAPI()

# ---------------------------------------------------------------------------
# Request / response models
# ---------------------------------------------------------------------------

class TripletexCredentials(BaseModel):
    base_url: str
    session_token: str


class FileAttachment(BaseModel):
    filename: str
    content_base64: str
    mime_type: str


class SolveRequest(BaseModel):
    prompt: str
    files: list[FileAttachment] = []
    tripletex_credentials: TripletexCredentials


# ---------------------------------------------------------------------------
# System prompt
# ---------------------------------------------------------------------------

SYSTEM_PROMPT = """You are an expert accounting AI agent completing tasks in Tripletex, a Norwegian ERP/accounting system.

You receive a task description in one of: Norwegian, Nynorsk, English, German, French, Spanish, Portuguese.
Understand the task and complete it by calling the Tripletex v2 REST API using the `call_api` tool.

## API Conventions
- Authentication is pre-handled; just use the tool.
- Always add `fields=*` to GET requests (e.g. `/employee/123?fields=*`).
- For list endpoints add `count=100&fields=*` (e.g. `/employee?firstName=Ola&count=100&fields=*`).
- POST/PUT body must be JSON; omit null/unknown fields unless required.
- Successful POST returns `{"value": {...}}` with the created resource.
- Successful GET list returns `{"values": [...], "fullResultSize": N}`.
- Dates: ISO format `YYYY-MM-DD`.
- Monetary amounts in NOK (or currency specified in task).

## Common Endpoints

### Employee
- `GET /employee?firstName=X&lastName=X&count=100&fields=*`
- `POST /employee` — required: `firstName`, `lastName`; optional: `email`, `phoneNumberMobile`, `dateOfBirth`, `address`
- `PUT /employee/{id}` — partial update
- `GET /employee/{id}?fields=*`

### Employment
- `GET /employee/employment?employeeId={id}&count=100&fields=*`
- `POST /employee/employment` — required: `employee.id`, `startDate`; optional: `endDate`, `typeOfEmployment`, `remunerationType`
- `PUT /employee/employment/{id}`

### Employment Details / Job Title
- `POST /employee/employment/details` — `employment.id`, `jobTitle` etc.

### Next of Kin / Emergency Contact
- `POST /employee/nextOfKin` — `employee.id`, `name`, `phoneNumber`

### Customer
- `GET /customer?name=X&count=100&fields=*`
- `POST /customer` — required: `name`; optional: `email`, `phoneNumber`, `organizationNumber`, `postalAddress`, `invoiceEmail`
- `PUT /customer/{id}`

### Supplier
- `GET /supplier?name=X&count=100&fields=*`
- `POST /supplier` — required: `name`
- `PUT /supplier/{id}`

### Product
- `GET /product?name=X&count=100&fields=*`
- `POST /product` — required: `name`, `vatType.id`; optional: `description`, `costExcludingVatCurrency`, `priceExcludingVatCurrency`, `priceIncludingVatCurrency`, `ean`, `stockOfGoods`
- `PUT /product/{id}`
- `GET /product/unit?count=100&fields=*` — list units
- `GET /ledger/vatType?count=100&fields=*` — list VAT types (use id from response)

### Order / Quote
- `GET /order?fields=*&count=100`
- `POST /order` — required: `customer.id`, `orderDate`, `deliveryDate`; optional `orderLines[]`
- `PUT /order/{id}`
- `POST /order/{id}/orderline` — add line to existing order; required: `order.id`, `count`, `unitPriceExcludingVatCurrency`
- `PUT /order/{id}/invoice` — convert order to invoice (GET params: `invoiceDate`, `sendToCustomer=false`)

### Invoice
- `GET /invoice?fields=*&count=100`
- `POST /invoice` — create directly; required: `invoiceDate`, `customer.id`; add `orders` array to pull lines from orders
- `PUT /invoice/{id}/payment` — register payment; body: `paymentDate`, `paymentTypeId`, `paidAmount`, `paidAmountAccountCurrency`
- `GET /invoice/paymentType?count=100&fields=*` — list payment types
- `PUT /invoice/{id}/createCreditNote` — issue credit note (query param: `date=YYYY-MM-DD`)

### Travel Expense
- `GET /travelExpense?count=100&fields=*`
- `POST /travelExpense` — required: `employee.id`, `travelDetails.departureDate`, `travelDetails.returnDate`, `project.id` (optional), `description`
- `PUT /travelExpense/{id}`
- `DELETE /travelExpense/{id}`
- `GET /travelExpense/cost?travelExpenseId={id}&count=100&fields=*`
- `POST /travelExpense/cost` — add cost; required: `travelExpense.id`, `travelExpenseCostCategory.id`, `amountCurrencyIncVat`
- `PUT /travelExpense/{id}/approve`
- `PUT /travelExpense/{id}/deliver`

### Project
- `GET /project?name=X&count=100&fields=*`
- `POST /project` — required: `name`, `projectManager.id`; optional: `customer.id`, `startDate`, `endDate`, `description`
- `PUT /project/{id}`
- `GET /employee?count=100&fields=*` — find project manager id

### Department
- `GET /department?name=X&count=100&fields=*`
- `POST /department` — required: `name`; optional: `departmentNumber`
- `PUT /department/{id}`

### Ledger / Voucher / Posting
- `GET /ledger/account?count=100&fields=*`
- `GET /ledger/voucher?dateFrom=YYYY-MM-DD&dateTo=YYYY-MM-DD&count=100&fields=*`
- `POST /ledger/voucher` — required: `date`, `voucherType`, `postings[]`
- `PUT /ledger/voucher/{id}/reverse` — reverse a voucher (query param: `date=YYYY-MM-DD`)
- `DELETE /ledger/voucher/{id}`

### Company / Settings
- `GET /company?fields=*`
- `GET /companyModule?fields=*` — list enabled modules
- `PUT /companyModule` — enable module (e.g. `{accountingModule: true}`)

## Strategy for Efficiency (maximises score)
1. Skip existence-check GETs when you are confident the resource doesn't exist yet.
2. Do NOT verify your work with extra GETs after a successful POST/PUT.
3. Chain operations only when needed (e.g. create order → invoice it).
4. Use PUT /order/{id}/invoice instead of POST /invoice when invoicing an order.

## Error Handling
- On 4xx errors, read the response message and try to fix the request before retrying.
- Never retry identical failing requests.
- If a required field id is unknown, do a targeted GET to find it.

Complete the task with the fewest correct API calls possible.
"""


# ---------------------------------------------------------------------------
# Tripletex API helper
# ---------------------------------------------------------------------------

async def call_tripletex(
    base_url: str,
    session_token: str,
    method: str,
    path: str,
    body: dict | None = None,
) -> dict:
    """Make one authenticated request to the Tripletex proxy."""
    # Ensure path starts with /
    if not path.startswith("/"):
        path = "/" + path
    url = base_url.rstrip("/") + path
    auth = ("0", session_token)
    headers = {"Content-Type": "application/json", "Accept": "application/json"}

    async with httpx.AsyncClient(timeout=60) as client:
        resp = await client.request(
            method=method.upper(),
            url=url,
            auth=auth,
            headers=headers,
            json=body,
        )

    try:
        data = resp.json()
    except Exception:
        data = {"raw": resp.text}

    return {"status_code": resp.status_code, "data": data}


# ---------------------------------------------------------------------------
# Tool definition
# ---------------------------------------------------------------------------

TOOLS = [
    {
        "name": "call_api",
        "description": (
            "Make an HTTP request to the Tripletex v2 REST API. "
            "Include query parameters directly in the path (e.g. /employee?firstName=Ola&count=100&fields=*)."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "method": {
                    "type": "string",
                    "enum": ["GET", "POST", "PUT", "DELETE", "PATCH"],
                    "description": "HTTP method",
                },
                "path": {
                    "type": "string",
                    "description": (
                        "API path with optional query params, e.g. "
                        "/employee?firstName=Ola&count=100&fields=* "
                        "or /employee/123?fields=*"
                    ),
                },
                "body": {
                    "type": "object",
                    "description": "JSON body for POST/PUT requests.",
                },
            },
            "required": ["method", "path"],
        },
    }
]


# ---------------------------------------------------------------------------
# Agentic solve loop
# ---------------------------------------------------------------------------

async def run_agent(request: SolveRequest) -> None:
    """Run Claude in an agentic tool-use loop until the task is complete."""
    client = anthropic.Anthropic(api_key=os.environ["ANTHROPIC_API_KEY"])

    # Build initial user message content
    content: list = []

    for f in request.files:
        mime = f.mime_type
        if mime.startswith("image/"):
            content.append(
                {
                    "type": "image",
                    "source": {
                        "type": "base64",
                        "media_type": mime,
                        "data": f.content_base64,
                    },
                }
            )
        elif mime == "application/pdf":
            content.append(
                {
                    "type": "document",
                    "source": {
                        "type": "base64",
                        "media_type": "application/pdf",
                        "data": f.content_base64,
                    },
                    "title": f.filename,
                }
            )

    content.append({"type": "text", "text": request.prompt})

    messages: list[dict] = [{"role": "user", "content": content}]

    base_url = request.tripletex_credentials.base_url
    session_token = request.tripletex_credentials.session_token

    max_iterations = 30
    for _ in range(max_iterations):
        resp = client.messages.create(
            model="claude-sonnet-4-6",
            max_tokens=4096,
            system=SYSTEM_PROMPT,
            tools=TOOLS,
            messages=messages,
        )

        messages.append({"role": "assistant", "content": resp.content})

        if resp.stop_reason != "tool_use":
            # Task complete (end_turn) or no more tool calls
            break

        # Execute all tool calls in this response
        tool_results = []
        for block in resp.content:
            if block.type != "tool_use":
                continue

            inp = block.input
            method = inp.get("method", "GET")
            path = inp.get("path", "/")
            body = inp.get("body")

            logger.info("API call: %s %s", method, path)
            result = await call_tripletex(base_url, session_token, method, path, body)
            logger.info("Response: %s", result.get("status_code"))

            tool_results.append(
                {
                    "type": "tool_result",
                    "tool_use_id": block.id,
                    "content": json.dumps(result),
                }
            )

        messages.append({"role": "user", "content": tool_results})


# ---------------------------------------------------------------------------
# Endpoint
# ---------------------------------------------------------------------------

@app.post("/")
async def solve(request: SolveRequest):
    logger.info("Received task: %s", request.prompt[:120])
    try:
        await run_agent(request)
    except Exception as e:
        logger.exception("Agent error: %s", e)
    return {"status": "completed"}


# ---------------------------------------------------------------------------
# Dev entrypoint
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import uvicorn

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(message)s",
    )
    uvicorn.run("tripletex.server:app", host="0.0.0.0", port=8000, reload=False)
