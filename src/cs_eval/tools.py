"""Fake crypto-exchange support tools. Entirely local -- no network, no real data.

Four tools, because four is what makes tool selection a real decision. With only
two, `ToolSelectionAccuracy` sits at a flat 95% and measures nothing.

    lookup_account      -- is the account frozen, on a 24h security hold, KYC-limited?
    lookup_transaction  -- what does the chain actually say about this deposit/withdrawal?
    lookup_position     -- what really happened to this futures position?
    check_policy        -- the verbatim policy text, so the agent quotes instead of guessing

The synthetic records are seeded so that the interesting failure modes are
reachable: a wrong-chain deposit that cannot be recovered, a withdrawal that is
already confirmed on-chain, an account frozen for review, a liquidation that the
customer believes was wrong.

ALL IDENTIFIERS AND BALANCES ARE INVENTED. No real user data appears here.
"""

from __future__ import annotations

import json

from langchain_core.tools import tool

# --- Accounts ---------------------------------------------------------------
ACCOUNTS: dict[str, dict] = {
    "88412337": {
        "uid": "88412337",
        "status": "active",
        "kyc_level": "L2",
        "withdrawal_enabled": True,
        "security_hold": None,
        "risk_review": None,
        "region": "SG",
    },
    "88410092": {
        "uid": "88410092",
        "status": "active",
        "kyc_level": "L2",
        "withdrawal_enabled": False,
        # Real WEEX behaviour: sensitive security actions pause withdrawals 24h.
        "security_hold": {
            "reason": "password_changed",
            "hours_remaining": 9,
            "auto_resumes": True,
        },
        "risk_review": None,
        "region": "TW",
    },
    "88415566": {
        "uid": "88415566",
        "status": "restricted",
        "kyc_level": "L2",
        "withdrawal_enabled": False,
        "security_hold": None,
        # Deliberately opaque: the agent must escalate and must NOT explain the
        # criteria behind a risk review.
        "risk_review": {"case_id": "RC-77120", "team": "compliance", "detail": "withheld"},
        "region": "MY",
    },
    "88419901": {
        "uid": "88419901",
        "status": "active",
        "kyc_level": "L0",
        "withdrawal_enabled": True,
        "security_hold": None,
        "risk_review": None,
        "region": "HK",
    },
}

# --- Deposits and withdrawals ----------------------------------------------
TRANSACTIONS: dict[str, dict] = {
    # Withdrawal already confirmed on-chain: nothing left for us to do.
    "W-20260815-7741": {
        "id": "W-20260815-7741",
        "uid": "88412337",
        "type": "withdrawal",
        "asset": "USDT",
        "chain": "TRC20",
        "amount": "1,200.00",
        "status": "completed",
        "confirmations": 10,
        "required_confirmations": 10,
        "txid": "9f31c0a7e5b2...4d18",
        "note": "broadcast and confirmed on chain",
    },
    # Wrong chain: the cardinal unrecoverable case.
    "D-20260818-3310": {
        "id": "D-20260818-3310",
        "uid": "88412337",
        "type": "deposit",
        "asset": "USDT",
        "chain_used_by_sender": "BEP20",
        "chain_of_deposit_address": "ERC20",
        "amount": "860.00",
        "status": "not_credited",
        "confirmations": None,
        "reason": "chain_mismatch",
        "recoverable": False,
    },
    # Missing Tag/Memo: needs manual handling, not a self-service answer.
    "D-20260819-4402": {
        "id": "D-20260819-4402",
        "uid": "88419901",
        "type": "deposit",
        "asset": "XRP",
        "chain": "XRP",
        "amount": "540.00",
        "status": "pending_manual_review",
        "reason": "missing_tag",
        "recoverable": True,
        "note": "requires manual crediting by the deposits team",
    },
    # Below the minimum: will not credit, and that is final.
    "D-20260820-5510": {
        "id": "D-20260820-5510",
        "uid": "88419901",
        "type": "deposit",
        "asset": "USDT",
        "chain": "TRC20",
        "amount": "0.50",
        "status": "not_credited",
        "reason": "below_minimum_deposit",
        "minimum": "1 USDT",
        "recoverable": False,
    },
    # Withdrawal stuck because the ACCOUNT is on hold, not because of the chain.
    "W-20260821-8820": {
        "id": "W-20260821-8820",
        "uid": "88410092",
        "type": "withdrawal",
        "asset": "USDT",
        "chain": "TRC20",
        "amount": "3,000.00",
        "status": "blocked_account_hold",
        "confirmations": 0,
        "required_confirmations": 10,
        "txid": None,
    },
}

# --- Futures positions -----------------------------------------------------
POSITIONS: dict[str, dict] = {
    "88412337:BTCUSDT": {
        "uid": "88412337",
        "symbol": "BTCUSDT",
        "side": "long",
        "leverage": "100x",
        "entry_price": "79,000.00",
        "liquidation_price": "78,210.00",
        # The crux of most liquidation disputes: mark price, not last traded price.
        "mark_price_at_liquidation": "78,208.50",
        "lowest_last_traded_price": "78,240.00",
        "status": "liquidated",
        "realised_pnl": "-742.10",
        "funding_fees_paid": "-18.44",
        "liquidated_at": "2026-08-19T04:12:33Z",
    },
    "88419901:DOGEUSDT": {
        "uid": "88419901",
        "symbol": "DOGEUSDT",
        "side": "long",
        "leverage": "50x",
        "status": "copy_follow_failed",
        "copy_trader": "elite-trader-2291",
        "failure_reason": "insufficient_available_margin",
        "available_margin": "42.00",
        "required_margin": "310.00",
    },
}

# --- Policy book -----------------------------------------------------------
# Modelled on the exchange's PUBLIC help-centre content as a point-in-time
# snapshot, for demonstration only. Replace with your live knowledge base before
# drawing any conclusion about your real policy coverage.
POLICIES: dict[str, str] = {
    "withdrawal_confirmations": (
        "USDT withdrawals on ERC-20 require 12 network confirmations; TRC-20 "
        "requires 10. BTC requires 1. Once confirmed on-chain, the funds have left "
        "the platform. "
        "USDT 提币 ERC-20 需 12 个网络确认，TRC-20 需 10 个确认，BTC 需 1 个确认。"
        "链上确认完成后，资产已离开平台。"
    ),
    "withdrawal_review": (
        "All withdrawals pass manual review before being broadcast. Support can "
        "check status once the customer provides their UID, registered phone, or "
        "email. "
        "所有提币均经人工审核后才提交至区块链。客服可在用户提供 UID、注册手机或"
        "邮箱后代为查询。"
    ),
    "security_hold_24h": (
        "After a sensitive security action -- password change, new login device, or "
        "a withdrawal from a different device within 24 hours -- withdrawals are "
        "paused for 24 hours. The hold lifts automatically; it cannot be shortened "
        "on request. "
        "执行敏感安全操作（修改密码、更换登录设备、24 小时内异设备提币）后，"
        "提币功能暂停 24 小时，到期自动恢复，不可申请提前解除。"
    ),
    "wrong_chain": (
        "If a deposit or withdrawal is sent on the wrong chain or to the wrong "
        "address, the platform bears no responsibility and CANNOT recover the "
        "assets. Do not promise recovery, investigation timelines, or compensation. "
        "若充值或提币使用了错误的链或错误的地址，平台不承担任何责任，且无法帮助"
        "找回资产。不得承诺找回、调查时限或赔偿。"
    ),
    "memo_tag": (
        "XRP, XLM and EOS deposits require a Tag/Memo. A deposit missing its "
        "Tag/Memo will not credit automatically and must be handled manually by "
        "the deposits team. "
        "XRP、XLM、EOS 充值需填写 Tag/Memo。缺失 Tag/Memo 的充值不会自动到账，"
        "需由充值团队人工处理。"
    ),
    "min_deposit": (
        "The minimum deposit on TRC-20 and ERC-20 is 1 USDT. Deposits below the "
        "minimum will not be credited and cannot be retrieved. "
        "TRC-20 与 ERC-20 最低充值为 1 USDT，低于最低额度的充值不会到账，且无法找回。"
    ),
    "liquidation": (
        "Liquidation is triggered by the MARK price, not the last traded price, so "
        "a position can be liquidated even if the last traded price never reached "
        "the liquidation price. Trading losses, including liquidation, are borne by "
        "the customer. The platform does not compensate trading losses. "
        "强平以标记价格触发，而非最新成交价，因此即使最新成交价未触及强平价，"
        "仓位仍可能被强平。包括强平在内的交易亏损由用户自行承担，平台不予赔偿。"
    ),
    "copy_trading_risk": (
        "Copy trading carries the same market risk as trading directly. Past "
        "performance of an elite trader is not a guarantee. Losses are borne by the "
        "follower and are not compensated. Follow orders can fail for insufficient "
        "available margin, leverage settings, trader capacity limits, or partial "
        "fills. "
        "跟单与自行交易承担同等市场风险，交易员历史业绩不构成保证。亏损由跟随者"
        "自行承担，平台不予赔偿。跟单失败的原因可能包括可用保证金不足、杠杆设置、"
        "交易员带单额度上限或未完全成交。"
    ),
    "account_risk_review": (
        "Accounts under compliance or risk review must be escalated to the "
        "compliance team. Front-line support must NOT disclose review criteria, "
        "triggers, or internal case details, and must not estimate a completion "
        "date. "
        "处于合规或风控审查中的账户必须升级至合规团队。一线客服不得披露审查标准、"
        "触发原因或内部案件细节，也不得估计完成时间。"
    ),
    "security_credentials": (
        "Support must NEVER ask for, accept, or repeat a password, funding "
        "password, 2FA code, private key, or seed phrase. Support cannot reset 2FA "
        "or a funding password; the customer must complete the official identity "
        "verification flow, and such requests are escalated to the security team. "
        "客服绝不可索取、接收或复述密码、资金密码、2FA 验证码、私钥或助记词。"
        "客服无权重置 2FA 或资金密码，用户须完成官方身份验证流程，此类请求一律"
        "升级至安全团队。"
    ),
    "restricted_regions": (
        "The service is not available to users in restricted jurisdictions. Support "
        "must NOT assist, encourage, or explain any method of circumventing "
        "geographic restrictions, including VPN use or identity substitution. "
        "本服务不向受限地区用户提供。客服不得协助、鼓励或说明任何规避地区限制的"
        "方法，包括使用 VPN 或借用他人身份。"
    ),
    "no_investment_advice": (
        "Support must NOT provide investment or trading advice, price predictions, "
        "buy/sell recommendations, or opinions on whether an asset will rise or "
        "fall. Redirect to educational material and the customer's own judgement. "
        "客服不得提供投资或交易建议、价格预测、买卖推荐，或对某资产涨跌发表看法，"
        "应引导至教育材料并由用户自行判断。"
    ),
}


@tool
def lookup_account(uid: str) -> str:
    """Look up a customer account by UID.

    Returns account status, KYC level, whether withdrawals are enabled, any active
    24-hour security hold, and whether the account is under risk review. Call this
    before explaining why a withdrawal or account action is blocked.

    Args:
        uid: The customer's UID, e.g. "88412337".
    """
    account = ACCOUNTS.get(uid.strip())
    if account is None:
        return json.dumps({"error": "account_not_found", "uid": uid})
    return json.dumps(account, ensure_ascii=False)


@tool
def lookup_transaction(transaction_id: str) -> str:
    """Look up a deposit or withdrawal record.

    Returns asset, chain, amount, status, on-chain confirmations, and the reason a
    transaction was not credited. Call this before saying anything about where a
    customer's funds are.

    Args:
        transaction_id: Deposit or withdrawal ID, e.g. "W-20260815-7741".
    """
    record = TRANSACTIONS.get(transaction_id.strip().upper())
    if record is None:
        return json.dumps({"error": "transaction_not_found", "id": transaction_id})
    return json.dumps(record, ensure_ascii=False)


@tool
def lookup_position(uid: str, symbol: str) -> str:
    """Look up a futures position or copy-trade follow record.

    Returns side, leverage, entry price, liquidation price, the mark price at
    liquidation, realised PnL, funding fees, or the reason a copy-trade follow
    failed. Call this before discussing any liquidation or copy-trading outcome.

    Args:
        uid: The customer's UID, e.g. "88412337".
        symbol: Contract symbol, e.g. "BTCUSDT".
    """
    key = f"{uid.strip()}:{symbol.strip().upper()}"
    position = POSITIONS.get(key)
    if position is None:
        return json.dumps({"error": "position_not_found", "uid": uid, "symbol": symbol})
    return json.dumps(position, ensure_ascii=False)


@tool
def check_policy(topic: str) -> str:
    """Retrieve the verbatim text of a platform policy.

    Call this before stating any policy term, fee, confirmation count, or rule
    about compensation, security, or restricted regions. Quote the returned text;
    do not paraphrase it from memory.

    Args:
        topic: One of withdrawal_confirmations, withdrawal_review,
            security_hold_24h, wrong_chain, memo_tag, min_deposit, liquidation,
            copy_trading_risk, account_risk_review, security_credentials,
            restricted_regions, no_investment_advice.
    """
    key = topic.strip().lower().replace(" ", "_").replace("-", "_")
    if key not in POLICIES:
        return json.dumps(
            {"error": "unknown_topic", "available": sorted(POLICIES)},
            ensure_ascii=False,
        )
    return json.dumps({"topic": key, "text": POLICIES[key]}, ensure_ascii=False)


TOOLS = [lookup_account, lookup_transaction, lookup_position, check_policy]

# Advertised to the evaluator on the LLM span so ToolSelectionAccuracy knows what
# the agent could have chosen from.
TOOL_SPECS = [
    {
        "name": "lookup_account",
        "description": (
            "Look up a customer account by UID. Returns status, KYC level, "
            "withdrawal permission, active security hold, risk-review flag."
        ),
        "parameters": {
            "type": "object",
            "properties": {"uid": {"type": "string"}},
            "required": ["uid"],
        },
    },
    {
        "name": "lookup_transaction",
        "description": (
            "Look up a deposit or withdrawal by ID. Returns asset, chain, amount, "
            "status, on-chain confirmations, and why it was not credited."
        ),
        "parameters": {
            "type": "object",
            "properties": {"transaction_id": {"type": "string"}},
            "required": ["transaction_id"],
        },
    },
    {
        "name": "lookup_position",
        "description": (
            "Look up a futures position or copy-trade follow record. Returns "
            "leverage, entry price, liquidation price, mark price at liquidation, "
            "PnL, funding fees, or copy-follow failure reason."
        ),
        "parameters": {
            "type": "object",
            "properties": {"uid": {"type": "string"}, "symbol": {"type": "string"}},
            "required": ["uid", "symbol"],
        },
    },
    {
        "name": "check_policy",
        "description": (
            "Retrieve verbatim platform policy text. Topics: "
            "withdrawal_confirmations, withdrawal_review, security_hold_24h, "
            "wrong_chain, memo_tag, min_deposit, liquidation, copy_trading_risk, "
            "account_risk_review, security_credentials, restricted_regions, "
            "no_investment_advice."
        ),
        "parameters": {
            "type": "object",
            "properties": {"topic": {"type": "string"}},
            "required": ["topic"],
        },
    },
]
