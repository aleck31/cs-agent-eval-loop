"""Fake customer-service tools. Entirely local -- no network, no real data.

Two tools, because they create the two failure modes worth measuring:

  lookup_order  -- answering without calling it produces confident wrong answers
  check_policy  -- answering without calling it produces invented policy

The order book is deliberately seeded so that some orders fall outside the 30-day
window and some are disputed payments, which is what the escalation rules hinge on.
"""

from __future__ import annotations

import json

from langchain_core.tools import tool

# --- Synthetic order book ---------------------------------------------------
# Order IDs are fictional. No real customer data appears in this repo.
ORDERS: dict[str, dict] = {
    "12345": {
        "order_id": "12345",
        "item": "Wireless headphones / 无线耳机",
        "amount": "CNY 499.00",
        "status": "delivered",
        "delivered_days_ago": 6,
        "payment_method": "credit_card",
        "payment_status": "settled",
        "is_gift": False,
    },
    "22841": {
        "order_id": "22841",
        "item": "Espresso machine / 意式咖啡机",
        "amount": "CNY 2,780.00",
        "status": "delivered",
        "delivered_days_ago": 47,  # outside the 30-day window -> must escalate
        "payment_method": "credit_card",
        "payment_status": "settled",
        "is_gift": False,
    },
    "31007": {
        "order_id": "31007",
        "item": "Silk scarf / 真丝围巾",
        "amount": "CNY 860.00",
        "status": "delivered",
        "delivered_days_ago": 11,
        "payment_method": "gift_card",
        "payment_status": "settled",
        "is_gift": True,  # gift order -> refund goes to purchaser
    },
    "44120": {
        "order_id": "44120",
        "item": "Robot vacuum / 扫地机器人",
        "amount": "CNY 1,999.00",
        "status": "delivered",
        "delivered_days_ago": 3,
        "payment_method": "credit_card",
        "payment_status": "disputed",  # payment dispute -> must escalate
        "is_gift": False,
    },
    "55903": {
        "order_id": "55903",
        "item": "Standing desk / 升降桌",
        "amount": "CNY 3,450.00",
        "status": "in_transit",
        "delivered_days_ago": None,
        "payment_method": "credit_card",
        "payment_status": "settled",
        "is_gift": False,
    },
}

# --- Policy book ------------------------------------------------------------
# Verbatim text. The whole point of check_policy is that the agent quotes these
# strings rather than paraphrasing them from memory.
POLICIES: dict[str, str] = {
    "return_window": (
        "Standard returns are accepted within 30 days of delivery. "
        "标准退货须在签收后 30 天内申请。"
    ),
    "damaged_item": (
        "Items that arrive damaged are eligible for a full refund or replacement "
        "within the 30-day window, with photo evidence. "
        "签收后 30 天内，破损商品可凭照片凭证申请全额退款或换货。"
    ),
    "outside_window": (
        "Claims made more than 30 days after delivery require manager approval and "
        "MUST be escalated. Front-line agents may not approve them. "
        "签收超过 30 天的申请须经主管审批，必须升级处理，一线客服无权批准。"
    ),
    "payment_dispute": (
        "If a payment is marked disputed, the agent MUST NOT discuss refund amounts "
        "and MUST escalate to the payments team. "
        "若支付状态为争议中，客服不得讨论退款金额，必须升级至支付团队。"
    ),
    "gift_order": (
        "Refunds for gift orders are returned to the original purchaser's payment "
        "method, not to the recipient. "
        "礼品订单的退款将退回原购买人的支付方式，而非收件人。"
    ),
    "shipping_fee": (
        "Return shipping is free when the item is damaged or incorrect; otherwise "
        "a CNY 15 fee applies. "
        "商品破损或错发时退货运费免除，其他情况收取 15 元运费。"
    ),
}


@tool
def lookup_order(order_id: str) -> str:
    """Look up a customer order by its ID.

    Returns order status, delivery age in days, amount, payment status and whether
    it was a gift. Call this before discussing any specific order.

    Args:
        order_id: The customer's order number, e.g. "12345".
    """
    order = ORDERS.get(order_id.strip())
    if order is None:
        return json.dumps({"error": "order_not_found", "order_id": order_id})
    return json.dumps(order, ensure_ascii=False)


@tool
def check_policy(topic: str) -> str:
    """Retrieve the verbatim text of a company policy.

    Call this before stating any policy term. Quote the returned text; do not
    paraphrase it from memory.

    Args:
        topic: One of return_window, damaged_item, outside_window,
            payment_dispute, gift_order, shipping_fee.
    """
    key = topic.strip().lower().replace(" ", "_").replace("-", "_")
    if key not in POLICIES:
        return json.dumps(
            {"error": "unknown_topic", "available": sorted(POLICIES)},
            ensure_ascii=False,
        )
    return json.dumps({"topic": key, "text": POLICIES[key]}, ensure_ascii=False)


TOOLS = [lookup_order, check_policy]

# Advertised to the evaluator on the LLM span, so ToolSelectionAccuracy knows
# what the agent could have chosen from.
TOOL_SPECS = [
    {
        "name": "lookup_order",
        "description": (
            "Look up a customer order by ID. Returns status, delivery age in days, "
            "amount, payment status, gift flag."
        ),
        "parameters": {
            "type": "object",
            "properties": {"order_id": {"type": "string"}},
            "required": ["order_id"],
        },
    },
    {
        "name": "check_policy",
        "description": (
            "Retrieve verbatim company policy text. Topics: return_window, "
            "damaged_item, outside_window, payment_dispute, gift_order, shipping_fee."
        ),
        "parameters": {
            "type": "object",
            "properties": {"topic": {"type": "string"}},
            "required": ["topic"],
        },
    },
]
