
import os
import time
import math
import argparse
import datetime as dt
from typing import List, Optional, Dict, Any

try:
    from dotenv import load_dotenv
    load_dotenv()
except Exception:
    pass

from binance.client import Client
from binance.exceptions import BinanceAPIException
import pandas as pd


def parse_date(s: str) -> int:
    """Parse YYYY-MM-DD or ISO-like date to milliseconds epoch."""
    try:
        # Try date
        d = dt.datetime.strptime(s, "%Y-%m-%d")
    except ValueError:
        # Try full ISO
        d = dt.datetime.fromisoformat(s)
    return int(d.replace(tzinfo=dt.timezone.utc).timestamp() * 1000)


def ms_to_iso(ms: int) -> str:
    return dt.datetime.fromtimestamp(ms/1000, tz=dt.timezone.utc).isoformat()


def chunk_ranges(start_ms: int, end_ms: int, days_per_chunk: int = 3):
    """Yield (chunk_start, chunk_end) in ms to respect API windows."""
    step = days_per_chunk * 24 * 60 * 60 * 1000
    cur = start_ms
    while cur < end_ms:
        nxt = min(end_ms, cur + step - 1)
        yield cur, nxt
        cur = nxt + 1


def make_client() -> Client:
    api_key = os.getenv("BINANCE_API_KEY") or os.getenv("API_KEY") or os.getenv("BINANCE_KEY")
    api_secret = os.getenv("BINANCE_API_SECRET") or os.getenv("API_SECRET") or os.getenv("BINANCE_SECRET")
    if not api_key or not api_secret:
        raise SystemExit("API key/secret tidak ditemukan. Set BINANCE_API_KEY dan BINANCE_API_SECRET di .env.")
    # python-binance Client covers Spot+Futures. We'll use the UM Futures endpoints via methods prefixed with futures_*.
    cli = Client(api_key=api_key, api_secret=api_secret, requests_params={"timeout": int(os.getenv("REQUEST_TIMEOUT", "20"))})
    return cli


def safe_call(fn, *args, **kwargs):
    """Retry with basic backoff for 429/418/5xx."""
    base = float(os.getenv("HTTP_BACKOFF", "1.5"))
    tries = int(os.getenv("REQUEST_RETRIES", "5"))
    for i in range(tries):
        try:
            return fn(*args, **kwargs)
        except BinanceAPIException as e:
            if e.status_code in (418, 429) or 500 <= e.status_code < 600:
                sleep_s = base ** (i + 1)
                time.sleep(min(sleep_s, 30))
                continue
            raise
        except Exception:
            # Transient
            sleep_s = base ** (i + 1)
            time.sleep(min(sleep_s, 30))
    raise RuntimeError("Gagal memanggil API setelah retry.")


def export_trades(cli: Client, symbol: str, start_ms: int, end_ms: int) -> pd.DataFrame:
    """
    Export user trades (fills) for a symbol in a time range.
    Endpoint: GET /fapi/v1/userTrades  (python-binance: futures_account_trades)
    """
    rows = []
    for a, b in chunk_ranges(start_ms, end_ms, days_per_chunk=3):
        # futures_account_trades supports startTime/endTime params.
        data = safe_call(cli.futures_account_trades, symbol=symbol, startTime=a, endTime=b, limit=1000)
        for t in data or []:
            t["time_iso"] = ms_to_iso(int(t["time"]))
            rows.append(t)
        # polite sleep to avoid bans
        time.sleep(0.2)
    df = pd.DataFrame(rows)
    return df


def export_income(cli: Client, start_ms: int, end_ms: int, income_type: Optional[str] = None) -> pd.DataFrame:
    """
    Export income history (PnL, funding, commission, transfer).
    Endpoint: GET /fapi/v1/income  (python-binance: futures_income_history)
    income_type can be one of: REALIZED_PNL, FUNDING_FEE, COMMISSION, TRANSFER, WELCOME_BONUS, INSURANCE_CLEAR...
    """
    rows = []
    for a, b in chunk_ranges(start_ms, end_ms, days_per_chunk=7):
        params = {"startTime": a, "endTime": b, "limit": 1000}
        if income_type:
            params["incomeType"] = income_type
        data = safe_call(cli.futures_income_history, **params)
        for t in data or []:
            t["time_iso"] = ms_to_iso(int(t["time"]))
            rows.append(t)
        time.sleep(0.2)
    df = pd.DataFrame(rows)
    return df


def export_orders(cli: Client, symbol: str, start_ms: int, end_ms: int) -> pd.DataFrame:
    """
    Export all orders (including filled/canceled) for a symbol.
    Endpoint: GET /fapi/v1/allOrders  (python-binance: futures_get_all_orders)
    """
    rows = []
    for a, b in chunk_ranges(start_ms, end_ms, days_per_chunk=7):
        data = safe_call(cli.futures_get_all_orders, symbol=symbol, startTime=a, endTime=b, limit=1000)
        for t in data or []:
            t["updateTime_iso"] = ms_to_iso(int(t.get("updateTime") or t.get("time") or 0))
            rows.append(t)
        time.sleep(0.2)
    return pd.DataFrame(rows)


def main():
    ap = argparse.ArgumentParser(description="Export Binance USDM Futures history to CSV")
    ap.add_argument("--symbols", type=str, default="BTCUSDT,ETHUSDT", help="Daftar simbol dipisah koma")
    ap.add_argument("--start", type=str, required=True, help="Tanggal mulai (YYYY-MM-DD atau ISO) UTC")
    ap.add_argument("--end", type=str, required=True, help="Tanggal akhir (YYYY-MM-DD atau ISO) UTC (eksklusif dianjurkan +1 hari)")
    ap.add_argument("--outdir", type=str, default="./exports_futures")
    ap.add_argument("--what", type=str, default="trades,income,orders", help="Pilih data: trades,income,orders (kombinasi dipisah koma)")
    ap.add_argument("--income-type", type=str, default="", help="Filter income: REALIZED_PNL,FUNDING_FEE,COMMISSION,TRANSFER,... (opsional)")
    args = ap.parse_args()

    os.makedirs(args.outdir, exist_ok=True)

    start_ms = parse_date(args.start)
    end_ms = parse_date(args.end)

    cli = make_client()

    syms: List[str] = [s.strip().upper() for s in args.symbols.split(",") if s.strip()]
    tasks = [w.strip().lower() for w in args.what.split(",") if w.strip()]

    if "income" in tasks:
        df_income = export_income(cli, start_ms, end_ms, args.income_type or None)
        p = os.path.join(args.outdir, f"income_{args.start}_{args.end}.csv".replace(":","-"))
        if not df_income.empty:
            df_income.to_csv(p, index=False)
            print(f"[OK] income → {p} ({len(df_income)} baris)")
        else:
            print("[INFO] income kosong di rentang waktu ini.")

    for sym in syms:
        if "trades" in tasks:
            df_tr = export_trades(cli, sym, start_ms, end_ms)
            p = os.path.join(args.outdir, f"trades_{sym}_{args.start}_{args.end}.csv".replace(":","-"))
            if not df_tr.empty:
                df_tr.to_csv(p, index=False)
                print(f"[OK] trades {sym} → {p} ({len(df_tr)} baris)")
            else:
                print(f"[INFO] trades {sym} kosong di rentang waktu ini.")

        if "orders" in tasks:
            df_od = export_orders(cli, sym, start_ms, end_ms)
            p = os.path.join(args.outdir, f"orders_{sym}_{args.start}_{args.end}.csv".replace(":","-"))
            if not df_od.empty:
                df_od.to_csv(p, index=False)
                print(f"[OK] orders {sym} → {p} ({len(df_od)} baris)")
            else:
                print(f"[INFO] orders {sym} kosong di rentang waktu ini.")


if __name__ == "__main__":
    main()
