import numpy as np
import numpy_financial as npf
import os

def compute_irr(cash_flows):
    """
    Returns IRR as decimal (e.g. 0.14 == 14%).
    """
    irr = npf.irr(cash_flows)
    if irr is None or np.isnan(irr):
        return 0.0
    return irr

def save_summary_md(filename, stock, horizon, equity_val, exit_pe, implied_irr, irr_stream, fcfe_stream, tv):
    """
    Saves the key analysis outputs to a clean Markdown file.
    """
    moic = sum(irr_stream[1:]) / abs(irr_stream[0])
    
    with open(filename, "w") as f:
        f.write(f"# Implied IRR Analysis: {stock.upper()}\n")
        f.write(f"**Date:** December 2025 | **Horizon:** {horizon} Years\n\n")
        
        f.write("## 1. Valuation Summary\n")
        f.write(f"- **Implied Levered IRR:** `{implied_irr:.2%}`\n")
        f.write(f"- **Exit Multiple (P/FCF):** {exit_pe:.2f}x\n")
        f.write(f"- **Gross Multiple of Money (MOIC):** {moic:.2f}x\n\n")
        
        f.write("## 2. Key Assumptions\n")
        f.write("| Metric | Value |\n")
        f.write("| :--- | :--- |\n")
        f.write(f"| Initial Equity Value | {equity_val:,.0f} |\n")
        f.write(f"| Terminal Value (Yr {horizon}) | {tv:,.0f} |\n")
        f.write(f"| Avg FCFE (Forecast) | {np.mean(fcfe_stream):,.0f} |\n\n")

        f.write("## 3. Cash Flow Stream (for IRR)\n")
        f.write("| Year | Cash Flow (FCFE) |\n")
        f.write("| :--- | :---: |\n")
        f.write(f"| 0 (Investment) | ({abs(irr_stream[0]):,.0f}) |\n")
        
        # Intermediate years
        for i, cf in enumerate(irr_stream[1:-1], 1):
            f.write(f"| {i} | {cf:,.0f} |\n")
            
        # Final year
        f.write(f"| {horizon} (Incl. Exit) | {irr_stream[-1]:,.0f} |\n")
    
    print(f"\n[SUCCESS] Summary saved to: {os.path.abspath(filename)}")

# -----------------------------
# 0) CONTROLS
# -----------------------------
STOCK = "lulu"        # "nike", "lulu", or "atz_cn"
HORIZON_YEARS = 5

print(f"Analysis for: {STOCK}")
print(f"Forecast Horizon: {HORIZON_YEARS} years")

# -----------------------------
# 1) DATA INPUTS
# -----------------------------

# FCFF consensus (unlevered FCF), in absolute currency units
fcff_dict = {
    "nike":   [2_635.2e6, 3_765.5e6, 4_572.1e6, 4_965.6e6, 5_502.7e6],
    "lulu":   [1_152.6e6, 1_245.6e6, 1_405.3e6, 1_400.1e6, 1_503.7e6],
    "atz_cn": [ 384.9e6,   601.5e6,   679.1e6,   811.1e6,   956.7e6],
}

# Current Equity Value (Market Cap), same currency as FCFF
equity_value_dict = {
    "nike":   88_692_081_760.00,
    "lulu":   24_681_182_826.00,
    "atz_cn": 13_432_629_145.00,
}

# Interest expense (LTM) and effective tax rate (%)
debt_params = {
    "nike":   {"interest_exp": -67.0e6,  "tax_rate": 17.14},  
    "lulu":   {"interest_exp": 0.0,     "tax_rate": 29.56},  
    "atz_cn": {"interest_exp": 69.6e6,  "tax_rate": 28.46},  
}

# Exit P/FCF multiples from median of RV
exit_multiples = {
    "nike":   26.08,
    "lulu":   24.51,
    "atz_cn": 22.36,
}

# -----------------------------
# 2) FCFE UNDER CONSTANT-DEBT ASSUMPTION
# -----------------------------

fcff_stream = np.array(fcff_dict[STOCK][:HORIZON_YEARS], dtype=float)

params = debt_params[STOCK]
tax = params["tax_rate"] / 100.0               
interest = params["interest_exp"]              
after_tax_interest = interest * (1.0 - tax)    

# Constant net debt: FCFE = FCFF - after-tax interest
fcfe_stream = fcff_stream - after_tax_interest

print("\n--- Cash Flow Adjustments ---")
print(f"Avg FCFF: {np.mean(fcff_stream):,.0f}")
print(f"Less: After-tax interest/yr: {after_tax_interest:,.0f}")

# -----------------------------
# 3) TERMINAL VALUE (EXIT MULTIPLE)
# -----------------------------

exit_pe = exit_multiples[STOCK]
final_year_fcfe = fcfe_stream[-1]
terminal_value = final_year_fcfe * exit_pe

print("\n--- Terminal Value (Exit Multiple) ---")
print(f"Exit multiple (P/FCFE proxy): {exit_pe:.2f}x")
print(f"Terminal value at horizon: {terminal_value:,.0f}")

# -----------------------------
# 4) IRR COMPUTATION
# -----------------------------

equity_investment = -equity_value_dict[STOCK]

# Cash-flow stream: t=0 outlay + HORIZON_YEARS FCFE (with TV in final year)
irr_stream = [equity_investment] + list(fcfe_stream[:-1]) + [fcfe_stream[-1] + terminal_value]

implied_irr = compute_irr(irr_stream)

print("\n========================================")
print(f"IMPLIED LEVERED IRR: {implied_irr:.2%}")
print("========================================")

# -----------------------------
# 5) SAVE TO MARKDOWN
# -----------------------------
output_filename = f"{STOCK}_IRR_Summary.md"
save_summary_md(
    output_filename, 
    STOCK, 
    HORIZON_YEARS, 
    -equity_investment, 
    exit_pe, 
    implied_irr, 
    irr_stream, 
    fcfe_stream, 
    terminal_value
)