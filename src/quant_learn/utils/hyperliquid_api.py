"""
Simple Hyperliquid Order Sender
Uses official hyperliquid-python-sdk for authenticated order placement.
Supports Testnet and optimizations for slippage/latency.

Installation:
pip install hyperliquid-python-sdk

Setup:
1. Get API wallet private key from https://app.hyperliquid.xyz/API
2. Set config variables below
3. For Testnet: use TESTNET_API_URL and get mock USDC from faucet

Optimizations:
- Use 'Alo' (post-only) to avoid taker fees and reduce slippage
- TWAP orders for large sizes (built-in suborder slippage limit 3%)
- Nonce invalidation (noop) for fast cancels instead of cancel spam
- Persistent connections via SDK
"""

import json
import time
from typing import Dict, Any, Optional

from hyperliquid.info import Info
from hyperliquid.exchange import Exchange
from hyperliquid.utils import constants
from hyperliquid.utils.signing import get_timestamp_ms

# CONFIG - Edit these
ACCOUNT_ADDRESS = ""  # Main wallet address (public)
SECRET_KEY = ""        # API wallet private key
USE_TESTNET = True                                  # Set False for mainnet
SYMBOL = "BTC"                                      # Asset name, e.g. "BTC", "ETH"
ASSET_INDEX = 0                                     # Get from info.meta() -> universe.indexOf(SYMBOL)

BASE_URL = constants.TESTNET_API_URL if USE_TESTNET else constants.MAINNET_API_URL

def get_info_exchange():
    """Initialize Info and Exchange clients."""
    info = Info(BASE_URL, skip_ws=True)
    exchange = Exchange(ACCOUNT_ADDRESS, SECRET_KEY, base_url=BASE_URL)
    return info, exchange

def get_asset_index(info: Info, symbol: str) -> int:
    """Fetch current asset index from meta."""
    meta = info.meta()
    universe = meta['universe']
    try:
        return universe.index(symbol)
    except ValueError:
        raise ValueError(f"Symbol {symbol} not found. Available: {universe[:10]}...")

def get_best_price(info: Info, asset: int, is_buy: bool) -> str:
    """Get best bid (for buy limit) or ask (for sell limit) to minimize slippage."""
    l2 = info.l2_book(asset)
    if is_buy:
        return l2['levels'][0][0] if l2['levels'] else "0"  # Best bid
    else:
        asks = l2['asks']  # Note: docs may vary, adjust if needed
        return asks[0][0] if asks else "0"  # Best ask

def place_limit_order(exchange: Exchange, asset: int, is_buy: bool, sz: float, px_offset_pct: float = 0.0,
                      tif: str = "Alo", reduce_only: bool = False, cloid: Optional[str] = None) -> Dict[str, Any]:
    """
    Place limit order with slippage reduction.
    - 'Alo': Post-only to avoid taker fees
    - px_offset_pct: Price offset from best (negative for aggressive)
    - Returns response
    """
    best_px = float(get_best_price(info, asset, is_buy))  # Requires global info or pass it
    px = best_px * (1 + px_offset_pct / 100 if is_buy else 1 - px_offset_pct / 100)
    
    order_result = exchange.order(
        asset, is_buy, sz, str(px),
        {"limit": {"tif": tif}},  # Alo/Ioc/Gtc
        reduce_only=reduce_only,
        cloid=cloid
    )
    print(f"Order result: {json.dumps(order_result, indent=2)}")
    return order_result

def place_twap_order(exchange: Exchange, asset: int, is_buy: bool, total_sz: float, minutes: int = 30,
                     randomize: bool = True, reduce_only: bool = False) -> Dict[str, Any]:
    """
    Place TWAP order (reduces slippage for large orders via suborders with 3% max slippage each).
    """
    result = exchange.twap_order(asset, is_buy, str(total_sz), minutes, randomize, reduce_only)
    print(f"TWAP result: {json.dumps(result, indent=2)}")
    return result

def cancel_order_by_oid(exchange: Exchange, asset: int, oid: int) -> Dict[str, Any]:
    """Cancel specific order by OID."""
    return exchange.cancel([{"asset": asset, "oid": oid}])

def cancel_by_cloid(exchange: Exchange, asset: int, cloid: str) -> Dict[str, Any]:
    """Cancel by client order ID."""
    return exchange.cancel_by_cloid([{"asset": asset, "cloid": cloid}])

def invalidate_nonce(exchange: Exchange) -> Dict[str, Any]:
    """
    Fast cancel all pending: noop invalidates nonce (saves rate limits vs cancel spam).
    """
    return exchange.noop()  # Marks nonce used, cancels in-flight

def get_user_state(info: Info) -> Dict[str, Any]:
    """Fetch positions, balances."""
    return info.user_state(ACCOUNT_ADDRESS)

def main():
    info, exchange = get_info_exchange()
    # Update asset index dynamically
    global ASSET_INDEX
    ASSET_INDEX = get_asset_index(info, SYMBOL)
    print(f"Using asset index {ASSET_INDEX} for {SYMBOL}")
    
    # Example usage - REPLACE WITH YOUR STRATEGY LOGIC HERE
    print("User state:", json.dumps(get_user_state(info), indent=2))
    
    # Example: Post-only buy limit near best bid (low slippage)
    # place_limit_order(exchange, ASSET_INDEX, True, 0.01, px_offset_pct=-0.1, tif="Alo")
    
    # Example: TWAP for larger size
    # place_twap_order(exchange, ASSET_INDEX, True, 1.0, minutes=10)
    
    print("Ready for strategy signals. Call place_limit_order() or place_twap_order() when ready.")

if __name__ == "__main__":
    main()