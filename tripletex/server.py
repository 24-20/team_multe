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

## CRITICAL: Pre-populated Sandbox
Each sandbox account is pre-populated with employees, customers, products, departments, and projects.
- NEVER create an employee, customer, supplier, or product that the task says already exists (e.g. "Jonas Hauge", "Tindra AS").
- ALWAYS search for referenced people/companies/products first, then use their id.
- Only create NEW resources when the task explicitly asks you to create something new.

## Search Before Referencing
When a task says "project manager is Jonas Hauge" or "customer is Tindra AS (org.nr 886715536)":
1. Search: `GET /employee?firstName=Jonas&lastName=Hauge&count=100&fields=*`
2. Use the `id` from the result in your POST body.
Do the same for customers, suppliers, products, departments.

## API Conventions
- Authentication is pre-handled; just use the tool.
- Always add `fields=*` to GET requests.
- For list endpoints add `count=100&fields=*`.
- POST/PUT body is JSON; omit fields you don't know unless required.
- Successful POST returns `{"value": {...}}` with the created resource.
- Successful GET list returns `{"values": [...], "fullResultSize": N}`.
- Dates: ISO format `YYYY-MM-DD`.
- Amounts in NOK unless stated otherwise.

## Endpoints Reference

### Employee
- `GET /employee?firstName=X&lastName=X&count=100&fields=*` — search
- `POST /employee` — required: `firstName`, `lastName`; optional: `email`, `phoneNumberMobile`, `phoneNumberHome`, `dateOfBirth`
- `PUT /employee/{id}` — update fields

### Employment record
- `GET /employee/employment?employeeId={id}&count=100&fields=*`
- `POST /employee/employment` — required: `employee.id`, `startDate`; optional: `endDate`, `typeOfEmployment` ("ORDINARY","MARITIME","FREELANCE"), `remunerationType`
- `PUT /employee/employment/{id}`

### Employment details (job title, position)
- `POST /employee/employment/details` — body: `employment.id`, `jobTitle`, `employmentType`
- `PUT /employee/employment/details/{id}`

### Emergency contact / next of kin
- `POST /employee/nextOfKin` — body: `employee.id`, `name`, `phoneNumber`, `typeOfRelationship`

### Customer
- `GET /customer?name=X&count=100&fields=*` — or search by `organizationNumber=X`
- `POST /customer` — required: `name`; optional: `organizationNumber`, `email`, `phoneNumber`, `invoiceEmail`, `postalAddress`
- `PUT /customer/{id}`

### Supplier
- `GET /supplier?name=X&count=100&fields=*`
- `POST /supplier` — required: `name`; optional: `organizationNumber`, `email`
- `PUT /supplier/{id}`

### Product
- `GET /product?name=X&count=100&fields=*`
- `GET /ledger/vatType?count=100&fields=*` — get VAT type ids FIRST before creating product
- `POST /product` — required: `name`, `vatType.id`; optional: `description`, `priceExcludingVatCurrency`, `priceIncludingVatCurrency`, `costExcludingVatCurrency`
- `PUT /product/{id}`

### Order
- `POST /order` — required: `customer.id`, `orderDate`, `deliveryDate`; optional: `orderLines[]`
  - orderLine fields: `product.id` (optional), `description`, `count`, `unitPriceExcludingVatCurrency`, `vatType.id`
- `PUT /order/{id}/invoice?invoiceDate=YYYY-MM-DD&sendToCustomer=false` — invoice an order (PUT with no body)

### Invoice
- `GET /invoice?id=X&fields=*` or `GET /invoice?count=100&fields=*`
- `PUT /invoice/{id}/payment` — body: `paymentDate`, `paymentTypeId`, `paidAmount`, `paidAmountAccountCurrency`
- `GET /invoice/paymentType?count=100&fields=*` — get paymentTypeId (use id of "Innbetaling" or first result)
- `PUT /invoice/{id}/createCreditNote?date=YYYY-MM-DD` — credit note (PUT with no body)

### Travel Expense
- `GET /travelExpense?count=100&fields=*`
- `POST /travelExpense` — required: `employee.id`, `travelDetails.departureDate`, `travelDetails.returnDate`, `isCompleted`; optional: `description`, `project.id`
  - travelDetails fields: `departureDate`, `returnDate`, `departureFrom`, `destination`
- `DELETE /travelExpense/{id}`
- `GET /travelExpense/cost?travelExpenseId={id}&count=100&fields=*`
- `GET /travelExpense/costCategory?count=100&fields=*` — list cost categories
- `POST /travelExpense/cost` — body: `travelExpense.id`, `travelExpenseCostCategory.id`, `amountCurrencyIncVat`, `currency.id` (optional)
- `PUT /travelExpense/{id}/deliver` — deliver/submit expense report

### Project
- `GET /project?name=X&count=100&fields=*`
- `POST /project` — required: `name`, `projectManager.id`; optional: `customer.id`, `startDate`, `endDate`, `description`, `number`
- `PUT /project/{id}`

### Department
- `GET /department?name=X&count=100&fields=*`
- `POST /department` — required: `name`; optional: `departmentNumber`, `manager.id`
- `PUT /department/{id}`

### Ledger / Voucher
- `GET /ledger/account?count=100&fields=*`
- `GET /ledger/voucher?dateFrom=YYYY-MM-DD&dateTo=YYYY-MM-DD&count=100&fields=*`
- `PUT /ledger/voucher/{id}/reverse?date=YYYY-MM-DD` — reverse a voucher (PUT with no body)
- `DELETE /ledger/voucher/{id}`

### Company modules
- `GET /companyModule?fields=*`
- `PUT /companyModule` — body: e.g. `{"moduleAccountingReports": true}`

## Common Workflows

**Create project linked to existing customer + manager:**
1. `GET /customer?name=X&count=100&fields=*` → get customer id
2. `GET /employee?firstName=X&lastName=Y&count=100&fields=*` → get manager id
3. `POST /project` with both ids

**Create invoice and register payment:**
1. `GET /ledger/vatType?count=100&fields=*` → get vat type id
2. `POST /order` with customer.id, orderDate, deliveryDate, orderLines
3. `PUT /order/{id}/invoice?invoiceDate=YYYY-MM-DD&sendToCustomer=false` → get invoice id from response
4. `GET /invoice/paymentType?count=100&fields=*` → get paymentTypeId
5. `PUT /invoice/{id}/payment` with paymentDate, paymentTypeId, paidAmount, paidAmountAccountCurrency

**Create employee with role:**
1. `POST /employee` → get employee id
2. `POST /employee/employment` with employee.id, startDate, typeOfEmployment

**Register travel expense:**
1. `GET /employee?firstName=X&count=100&fields=*` → get employee id (if referencing existing employee)
2. `POST /travelExpense` with employee.id, travelDetails
3. `GET /travelExpense/costCategory?count=100&fields=*` → if adding costs
4. `POST /travelExpense/cost` for each cost item

## Efficiency Rules (unlock bonus score)
1. Search for existing resources ONLY when you need their id — do it in one targeted GET.
2. Do NOT verify your work with extra GETs after a successful POST/PUT.
3. Do NOT retry identical requests that succeed.
4. Zero 4xx errors = efficiency bonus. If you get a 4xx, read the error and fix before retrying.

Complete the task correctly with as few API calls as possible.
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
